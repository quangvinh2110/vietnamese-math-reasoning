import re
import traceback
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

from .base_pipeline import BasePipeline

from ..utils.constants import USER_PROMPT_TEMPLATE
from ..utils.preprocess import preprocess
from ..utils.utils import add_notes
from ..utils.python_executor import PythonExecutor


CODE_PATTERN = re.compile(r"```python([\s\S]*)```")
COMPARISON_PATTERN = re.compile("if ([\w]+==.*|[\w]+ ==.*):")


def is_float(string: str):
    try:
        float(string)
        return True
    except ValueError:
        return False


class CodePipeline(BasePipeline):

    def __init__(
        self,
        model_path: Optional[str] = None,
        quantize: Optional[str] = None,
        adapter_path: Optional[str] = None,
        merge_adapter: bool = False,
        use_vllm: bool = False,
        system_prompt: str = "",
        assistant_prompt: str = ""
    ):
        super().__init__(
            model_path=model_path,
            quantize=quantize,
            adapter_path=adapter_path,
            merge_adapter=merge_adapter,
            use_vllm=use_vllm,
            system_prompt=system_prompt,
            assistant_prompt=assistant_prompt
        )
        self.executor = PythonExecutor(get_answer_from_stdout=True)


    def _extract_code(self, s: str):
        return CODE_PATTERN.findall(s)[0]

    
    def _fix_rounding_error(self, code: str):
        comparisons = COMPARISON_PATTERN.findall(code)
        if len(comparisons) == 0:
            return code
        answer_variable = comparisons[0].split("==")[0]
        answer_value = self.executor.apply(
            code + f"\nprint({answer_variable})"
        )[0]
        if is_float(answer_value):
            for comparison in comparisons:
                choice_variable = comparison.split("==")[-1]
                str_to_replace = f"abs({answer_variable}-{choice_variable}) < 1e-8"
                code = code.replace(comparison, str_to_replace)
        
        return code


    def _execute_python_code(self, code: str):
        code = self._fix_rounding_error(code)
        output = self.executor.apply(code)
        if output[1] == "Done":
            return output[0]
        return ""
    

    def prepare_prompts(
        self,
        instruction_list: List[str],
        question_list: List[str], 
        choices_list: List[list],
    ) -> List[str]:
        prompts = []
        for instruction, question, choices in zip(instruction_list, question_list, choices_list):
            choices = "\n".join(choices)
            notes = add_notes(question=question, choices=choices)
            user_prompt = USER_PROMPT_TEMPLATE.format(
                instruction=instruction,
                question=question,
                choices=choices
            )
            if notes:
                user_prompt+=("\nNotes:\n"+"\n".join(notes))
            user_prompt = preprocess(user_prompt)
            prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": self.assistant_prompt}
            ], tokenize=False).strip().removesuffix(self.tokenizer.eos_token)
            prompts.append(prompt)
        return prompts


    def select_answer_for_multiple_choices(
        self,
        generated_answer: str, 
        choices: list,
        prediction_prefix: str="Conclusion: The right choice is ",
    ):
        try:
            code = self._extract_code(generated_answer)
            python_interperter_output = self._execute_python_code(code) if code else ""
        except:
            code = ""
            python_interperter_output = ""
            # print(str(traceback.format_exc()))

        if code:
            prompt = generated_answer.removesuffix(self.tokenizer.eos_token) + f"\n\n```output\n{python_interperter_output}\n```\n\n" + prediction_prefix
        else:
            prompt = generated_answer.removesuffix(self.tokenizer.eos_token) + "\n\n" + prediction_prefix
        logits = self.compute_logit_for_choices(
            prompt=prompt, 
            choices=choices
        )
        max_logit = max(logits)
        max_logit_id = logits.index(max_logit)
        # print("Logits: " + str(logits))
        return f"Predict prompt: {prompt}\n\nCode: {code}", chr(max_logit_id+65)


# if __name__ == "__main__":

#     public_test = []
#     with open("./data/zalo/test/public_test.json", "r") as f:
#         public_test = json.loads(f.read())["data"]
    
#     with open(f"./data/submissions/toracode-13b_v{version}.{subversion}.jsonl", "a") as f:    
#         for s in tqdm(public_test):
#             start = time.time()
#             question = s["question"] + "\n" + "\n".join(s["choices"])
#             output = get_answer(question)
#             try:
#                 code = extract_code(output)
#                 _, prediction = execute_python_code(code)
#             except:
#                 prediction = ""
#             infer_time = time.time() - start
#             d = json.dumps({
#                 "out": prediction, "time": infer_time
#             }, ensure_ascii=False) + "\n"
#             f.write(d)
