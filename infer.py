import json
import re
import time
import torch
import unicodedata
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer
)
from peft import PeftModel


from src.utils.utils import add_notes, construct_prompt
from src.utils.python_executor import PythonExecutor


version = 3
subversion = 1
model_path = "./pretrained_models/toracode-13b"
adapter_path = f"./checkpoints/toracode-13b_v{version}.{subversion}"
PROMPT_TYPE = "zalo"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=AutoConfig.from_pretrained(model_path),
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True
)
model = PeftModel.from_pretrained(model, adapter_path)
# model = model.merge_and_unload()
executor = PythonExecutor(get_answer_from_stdout=True)


CODE_PATTERN = re.compile(r"```python([\s\S]*)```")

def extract_code(s: str):
    return CODE_PATTERN.findall(s)[0]


def is_float(string: str):
    try:
        float(string)
        return True
    except ValueError:
        return False


COMPARISON_PATTERN = re.compile("if ([\w]+==.*|[\w]+ ==.*):")

def fix_rounding_error(code: str):
    comparisons = COMPARISON_PATTERN.findall(code)
    if len(comparisons) == 0:
        return code
    answer_variable = comparisons[0].split("==")[0]
    answer_value = executor.apply(
        code + f"\nprint({answer_variable})"
    )[0]
    if is_float(answer_value):
        for comparison in comparisons:
            choice_variable = comparison.split("==")[-1]
            str_to_replace = f"abs({answer_variable}-{choice_variable}) < 1e-8"
            code = code.replace(comparison, str_to_replace)
    
    return code


def execute_python_code(code: str):
    output = executor.apply(code)
    if output[1] == 'Done':
        if output[0] == '':
            return code, "Missing print function!!!!"
        if output[0].strip() not in ["A", "B", "C", "D", "E"]:
            code = fix_rounding_error(code)
            output = executor.apply(code)        

    return code, output[0]


def normalize(s: str):
    return unicodedata.normalize('NFC', s)


def get_answer(question: str, max_new_tokens: int=1024):
    question = normalize(question)
    question = add_notes(question)
    prompt = construct_prompt(question, PROMPT_TYPE)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            # temperature=0.4,
            # top_p=0.95,
            # top_k=20,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )
    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()


if __name__ == "__main__":

    public_test = []
    with open("./data/zalo/test/public_test.json", "r") as f:
        public_test = json.loads(f.read())["data"]
    
    with open(f"./data/submissions/toracode-13b_v{version}.{subversion}.jsonl", "a") as f:    
        for s in tqdm(public_test):
            start = time.time()
            question = s["question"] + "\n" + "\n".join(s["choices"])
            output = get_answer(question)
            try:
                code = extract_code(output)
                _, prediction = execute_python_code(code)
            except:
                prediction = ""
            infer_time = time.time() - start
            d = json.dumps({
                "out": prediction, "time": infer_time
            }, ensure_ascii=False) + "\n"
            f.write(d)