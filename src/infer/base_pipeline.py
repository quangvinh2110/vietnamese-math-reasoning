from typing import Any, Optional, List, Union
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from ..utils.stopping_criteria import StopSequenceCriteria
from ..utils.constants import USER_PROMPT_TEMPLATE
from ..utils.preprocess import preprocess


class BasePipeline:

    def __init__(
        self,
        model_path: Optional[str] = None,
        quantize: Optional[str] = None,
        adapter_path: Optional[str] = None,
        merge_adapter: bool = False,
        use_vllm: bool = False,
        system_prompt: str = "",
        assistant_prompt: str = "",
        stop: List[str] = [],
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        if stop:
            self.stop = list(set(stop+[self.tokenizer.eos_token]))
        else:
            self.stop = [self.tokenizer.eos_token]
        self.use_vllm = use_vllm
        self.adapter_path = adapter_path
        if use_vllm:
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                max_num_batched_tokens=16384,
                max_context_len_to_capture=4096,
                max_model_len=4096,
                max_num_seqs=32,
                dtype=torch.bfloat16,
                enable_lora=True if adapter_path else False,
                max_lora_rank=256,
            )
            self.generation_config = SamplingParams(
                n=1,
                best_of=1,
                use_beam_search=False,
                max_tokens=1024,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                stop=self.stop
            )
        else:
            bnb_config = None
            if quantize == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(  
                model_path, 
                config=config, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True, 
                quantization_config=bnb_config,
                device_map="auto"
            )
            if adapter_path:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                if merge_adapter:
                    self.model = self.model.merge_and_unload()
            self.model.eval()

            self.stopping_criteria = StopSequenceCriteria(
                self.stop, 
                self.tokenizer
            )
            self.generation_config = GenerationConfig(
                max_new_tokens=1024,
                repetition_penalty=1.0,
                eos_token_id=[self.tokenizer.eos_token_id],
                pad_token_id=[self.tokenizer.pad_token_id],
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
        self.tokenizer.add_eos_token = False
        self.tokenizer.padding_side = "left"
        self.system_prompt = system_prompt
        self.assistant_prompt = assistant_prompt


    def remove_stop_sequences(self, text):
        for stop_sequence in self.stop:
            text = text.removesuffix(stop_sequence)
        return text

    
    def prepare_prompts(
        self,
        instruction_list: List[str],
        question_list: List[str], 
        choices_list: List[list],
    ) -> List[str]:
        prompts = []
        for instruction, question, choices in zip(instruction_list, question_list, choices_list):
            choices = "\n".join(choices)
            user_prompt = USER_PROMPT_TEMPLATE.format(
                instruction=instruction,
                question=question,
                choices=choices
            )
            user_prompt = preprocess(user_prompt)
            prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": self.assistant_prompt}
            ], tokenize=False).strip()
            prompt = self.remove_stop_sequences(prompt)
            prompts.append(prompt)
        return prompts


    def get_answers(
        self, 
        instruction_list: List[str],
        question_list: List[str], 
        choices_list: List[list],
    ) -> List[str]:
        prompts = self.prepare_prompts(
            instruction_list=instruction_list,
            question_list=question_list, 
            choices_list=choices_list,
        )
        return self.generate_batch(prompts)


    def generate_batch(
        self, 
        prompts: List[list],
    ) -> List[str]:
        if self.use_vllm:
            outputs = self.model.generate(
                prompts, 
                self.generation_config,
                lora_request=LoRARequest("math_adapter", 1, self.adapter_path) \
                                 if self.adapter_path else None
            )
            outputs = [output.prompt + output.outputs[0].text for output in outputs]
        else:
            _inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = _inputs["input_ids"].to(self.model.device)
            attention_mask = _inputs["attention_mask"].to(self.model.device)
            with torch.no_grad():
                # streamer = TextStreamer(self.tokenizer, skip_prompt=True)
                generated = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    # streamer=streamer,
                    stopping_criteria=self.stopping_criteria
                )
            gen_tokens = generated["sequences"].cpu()
            outputs = self.tokenizer.batch_decode(gen_tokens)
            outputs = [output.replace(self.tokenizer.pad_token, "") for output in outputs]
        return outputs 


    def compute_logit_for_choices(self, prompt, choices):
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        choices_token_id = [self.tokenizer.convert_tokens_to_ids(choice.strip()[0]) for choice in choices]
        with torch.no_grad():
            logits = self.model(input_ids).logits
        return [logits[:, -1, choice_token_id].item() for choice_token_id in choices_token_id]

    
    def select_answer_for_multiple_choices(
        self,
        generated_answer: str, 
        choices: list,
        prediction_prefix: str="Kết luận: đáp án đúng là ",
    ):
        prompt = generated_answer.removesuffix(self.tokenizer.eos_token) + "\n\n" + prediction_prefix
        logits = self.compute_logit_for_choices(
            prompt=prompt, 
            choices=choices
        )
        max_logit = max(logits)
        max_logit_id = logits.index(max_logit)
        # print("Logits: " + str(logits))
        return f"Predict prompt: {prompt}", chr(max_logit_id+65)
