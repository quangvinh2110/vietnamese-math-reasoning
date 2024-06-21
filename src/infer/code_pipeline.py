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
from ..utils.python_executor import PythonExecutor


CODE_PATTERN = re.compile(r"```python([\s\S]*?)```")


DEFAULT_SYS_PROMPT = """
Với vai trò là giáo viên dạy kèm môn toán, nhiệm vụ của bạn là giúp học sinh ở mọi cấp độ nắm bắt và giải quyết các vấn đề toán học. Tuy nhiên, bạn có thể thấy khó khăn với các phép toán số học cơ bản. Để đối phó với vấn đề này, bạn sẽ cung cấp hướng dẫn từng bước và viết mã Python theo hướng dẫn đó để thực hiện tất cả các phép tính cần thiết và cung cấp kết quả cuối cùng mỗi khi một câu hỏi đòi hỏi phép tính toán.
Hãy đưa ra câu trả lời theo định dạng: ```guidance
# your guidance
```

(Optional) ```python
# your python code if necessary
```
```output
# output of the above code
```
# your conclusion
""".strip()


class CodePipeline(BasePipeline):

    def __init__(
        self,
        model_path: Optional[str] = None,
        quantize: Optional[str] = None,
        adapter_path: Optional[str] = None,
        merge_adapter: bool = False,
        use_vllm: bool = False,
        system_prompt: str = DEFAULT_SYS_PROMPT,
        assistant_prompt: str = "",
        stop: List[str] = [],
    ):
        if stop:
            stop = list(set(stop+["```output"]))
        else:
            stop = ["```output"]
        super().__init__(
            model_path=model_path,
            quantize=quantize,
            adapter_path=adapter_path,
            merge_adapter=merge_adapter,
            use_vllm=use_vllm,
            system_prompt=system_prompt,
            assistant_prompt=assistant_prompt,
            stop=stop,
        )
        self.executor = PythonExecutor(get_answer_from_stdout=True)


    def _extract_code(self, s: str):
        codes = CODE_PATTERN.findall(s)
        if len(codes) > 1:
            return codes[-1]
        return ""


    def _execute_python_code(self, code: str):
        output = self.executor.apply(code)
        if output[1] == "Done":
            return output[0]
        return "no output"


    def select_answer_for_multiple_choices(
        self,
        generated_answer: str, 
        choices: list,
        prediction_prefix: str="Vậy áp án đúng là ",
    ):
        generated_answer = self.remove_stop_sequences(generated_answer)
        code = self._extract_code(generated_answer)
        try:
            python_interperter_output = self._execute_python_code(code) if code else ""
        except:
            python_interperter_output = ""
            # print(str(traceback.format_exc()))

        prompt = generated_answer
        if code:
            prompt += f"\n\n```output\n{python_interperter_output}\n```\n\n" 
            prompt += prediction_prefix
        else:
            prompt += ("\n" + prediction_prefix)
        logits = self.compute_logit_for_choices(
            prompt=prompt, 
            choices=choices
        )
        max_logit = max(logits)
        max_logit_id = logits.index(max_logit)
        # print("Logits: " + str(logits))
        return f"Predict prompt: {prompt}\n\nCode: {code}", chr(max_logit_id+65)
