from typing import Any, Optional, List
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from ..utils.constants import USER_PROMPT_TEMPLATE

from ..utils.preprocess import preprocess


class BasePipeline:

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: int = 0,
        assistant_prompt: str = None
    ):
        if (not tokenizer or not model) and model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(  
                model_path, 
                config=config, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True, 
                device_map={"": device}
                # device_map="auto"
            )
        elif tokenizer and model:
            self.tokenizer = tokenizer
            self.model = model
        else:
            raise ValueError()
        self.model.eval()
        self.tokenizer.add_eos_token = False
        self.tokenizer.padding_side = "left"
        self.assistant_prompt = assistant_prompt

    
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
            if self.assistant_prompt:
                prompt = self.tokenizer.apply_chat_template([
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": self.assistant_prompt}
                ], tokenize=False).strip().removesuffix(self.tokenizer.eos_token)
            else:
                prompt = self.tokenizer.apply_chat_template([
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_prompt},
                ], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        return prompts


    def generate_batch(
        self, 
        instruction_list: List[str],
        question_list: List[str], 
        choices_list: List[list],
        max_new_tokens: int = 1024
    ):
        prompts = self.prepare_prompts(
            instruction_list=instruction_list,
            question_list=question_list, 
            choices_list=choices_list,
        )
        _inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = _inputs["input_ids"].to(self.model.device)
        attention_mask = _inputs["attention_mask"].to(self.model.device)
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.0,
                # temperature=0.7,
                # top_p=0.95,
                # top_k=20,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # eos_token_id=0, # for open-end generation.
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            # streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            generated = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                # streamer=streamer,
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
        get_prediction_prefix: str="Kết luận: đáp án đúng là ",
    ):
        prompt = generated_answer.removesuffix(self.tokenizer.eos_token) + "\n\n" + get_prediction_prefix
        logits = self.compute_logit_for_choices(
            prompt=prompt, 
            choices=choices
        )
        max_logit = max(logits)
        max_logit_id = logits.index(max_logit)
        # print("Logits: " + str(logits))
        return prompt, chr(max_logit_id+65)
