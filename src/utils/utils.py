import os
import json
import random
import json
import re
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any
import unicodedata
import string

from .constants import unit_list


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print("Saved to", save_path)


def convert_unit(unit_1: str, unit_2: str):
    pass


def add_unit_conversion(question: str):
    pass


def add_definitions(question: str):
    original_question = question+''
    question = question.lower()
    if original_question[-1] != "\n":
        original_question += "\n"
    # minus
    if ("số bị trừ" in question) or ("số trừ" in question):
        note = (
            "NOTE: "
            "the term `số bị trừ` refers to the minuend; "
            "the term `số trừ` refers to the subtrahend; "
            "the term `hiệu` refers to the difference\n"
        )
        original_question += note
    # add
    if "số hạng" in question:
        note = (
            "Note: "
            "the term `số hạng` refers to the addend; "
            "the term `tổng` refers to the sum\n"
        )
        original_question += note
    # multiply
    if "thừa số" in question: 
        note = (
            "NOTE: "
            "the term `thừa số` refers to the factor; "
            "the term `tích` refers to the product\n"
        )
        original_question += note
    # divide
    if ("số bị chia" in question) or ("số chia" in question) or ("số dư" in question):
        note = (
            "NOTE: "
            "the term `số bị chia` refers to the dividend; "
            "the term `số chia` refers to the divisor; "
            "the term `thương` refers to the quotient; "
            "the term `số dư` refers to the remainder\n"
        )
        original_question += note
    # property
    if ("tính chất giao hoán" in question) or ("giao hoán" in question) or ("tính giao hoán" in question):
        note = "NOTE: the term `tính chất giao hoán` refers to the commutative property\n"
        original_question += note
    if ("tính chất kết hợp" in question) or ("tính kết hợp" in question):
        note = "NOTE: the term `tính chất kết hợp` refers to the associative property\n"
        original_question += note
    # numbers
    if "số tròn chục" in question:
        note = "NOTE: The term `số tròn chục` refers to numbers that are greater than or equal to 10 and divisible by 10\n"
        original_question += note
    if "số tròn trăm" in question:
        note = "NOTE: The term `số tròn trăm` refers to numbers that are greater than or equal to 100 and divisible by 100\n"
        original_question += note
    if "số tròn nghìn" in question:
        note = "NOTE: The term `số tròn nghìn` refers to numbers that are greater than or equal to 1000 and divisible by 1000\n"
        original_question += note

    return original_question

trans_table = str.maketrans({p: f"\{p}" for p in string.punctuation})
unit_list = list(map(lambda u: u.translate(trans_table), unit_list))
UNIT_PATTERN = re.compile(r"\d[\s]*({})".format("|".join(unit_list)))

def add_notes(question: str):

    choices = "\n".join(question.split("\n")[1:])
    if UNIT_PATTERN.search(choices):
        question += "NOTE: **You should convert the final result and all the choices to the same unit**\n"

    question = add_definitions(question)
    if "hình tròn" in question or "vòng tròn" in question:
        question += "NOTE: The value of pi is 3.14. You should use this value to compute the result"
    return question



# def load_prompt(data_name, prompt_type):
#     if data_name in ['gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']:
#         data_name = "gsm8k"
#     if prompt_type in ['platypus_fs', 'wizard_zs']:
#         prompt_type = "cot"
#     prompt_path = "./prompts/{}/{}.md".format(prompt_type, data_name)
#     if not os.path.exists(prompt_path):
#         prompt_path = "./prompts/{}.md".format(prompt_type)
#     if os.path.exists(prompt_path):
#         with open(prompt_path, 'r', encoding='utf-8') as fp:
#             prompt = fp.read().strip() + "\n\n"
#     else:
#         print(f"Error: prompt file {prompt_path} not found")
#         prompt = ""
#     return prompt

# def construct_prompt(args, example):
#     demo_prompt = load_prompt(args.data_name, args.prompt_type)
#     if args.use_train_prompt_format:
#         full_prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n"
#     elif "tora" in args.prompt_type or "pot" in args.prompt_type:
#         context = f"Question: {example['question']}\n\nSolution:"
#         full_prompt = demo_prompt + context
#     elif args.prompt_type in ["direct", "cot"]:
#         context = f"Question: {example['question']}\nAnswer:"
#         full_prompt = demo_prompt + context
#     elif args.prompt_type == "pal":
#         context = f"Question: {example['question']}"
#         full_prompt = demo_prompt + context
#     elif args.prompt_type == "wizard_zs":
#         full_prompt = (
#             "Below is an instruction that describes a task. "
#             "Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
#         )
#         full_prompt = full_prompt.format(instruction=example['question'])
#     elif args.prompt_type == "platypus_fs":
#         full_prompt = (
#             "Below is an instruction that describes a task. "
#             "Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n{instruction}\n\n### Response:\n"
#         )
#         full_prompt = full_prompt.format(instruction=demo_prompt + f"Question: {example['question']}\nAnswer:")
#     else:
#         raise NotImplementedError(args.prompt_type)
#     return full_prompt


def load_static_prompt(prompt_type: str):
    if prompt_type=="dynamic" or prompt_type=="zalo":
        return ""
    current_path = os.path.realpath(__file__)
    prompt_path = "/".join(current_path.split("/")[:-2])+f"/prompts/{prompt_type}.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as fp:
            prompt = fp.read().strip() + "\n\n"
    else:
        print(f"Error: prompt file {prompt_path} not found")
        prompt = ""
    return prompt


def load_dynamic_prompt(question: str):
    pass


def is_multiple_choices(question: str):
    return len(question.split("\n")) > 1

MULTIPLE_CHOICES_TEMPLATE = "Solve the following multiple-choices problem: {question}\n"
NORMAL_TEMPLATE = "Solve the following problem: {question}\n"
DYNAMIC_PROMPT_TEMPLATE = f"""

"""


def construct_prompt(question: str, prompt_type: str):

    demo_prompt = load_static_prompt(prompt_type)
    if prompt_type == "zalo":
        if is_multiple_choices(question):
            full_prompt = "".join([
                "<|user|>\n",
                MULTIPLE_CHOICES_TEMPLATE.format(question=question),
                "<|assistant|>\n"
            ])
        else:
            full_prompt = "".join([
                "<|user|>\n",
                NORMAL_TEMPLATE.format(question=question),
                "<|assistant|>\n"
            ])
    elif prompt_type == "tora":
        if is_multiple_choices(question):
            context = "".join([
                "<|user|>\n",
                MULTIPLE_CHOICES_TEMPLATE.format(question=question),
                "<|assistant|>\n"
            ])
        else:
            context = "".join([
                "<|user|>\n",
                NORMAL_TEMPLATE.format(question=question),
                "<|assistant|>\n"
            ])
        full_prompt = demo_prompt + context
    elif prompt_type == "pal":
        pass
    elif prompt_type == "wizard_zs":
        pass
    elif prompt_type == "platypus_fs":
        pass
    elif prompt_type == "dynamic":
        pass
    else:
        raise NotImplementedError(args.prompt_type)
    
    return full_prompt


def show_sample(sample):
    print("=="*20)
    print("idx:", sample['idx'])
    for key in ["type", "level"]:
        if key in sample:
            print("{}: {}".format(key, sample[key]))
    print("question:", sample['question'])
    if 'code' in sample:
        for code in sample['code']:
            print('-'*20)
            print("code:", code)
        print("execution", sample['report'])
    for key in ["pred", "gt", "score", "unit", "gt_cot"]:
        if key in sample:
            print("{}: {}".format(key, sample[key]))
    print()


# testing
if __name__ == "__main__":

    question = "Số bị trừ là số liền sau của số tròn chục lớn nhất có hai chữ số, số trừ là tổng của 37 và 25. Vậy hiệu là:\nA. 29\nB. 7\nC. 9\nD. 6"
    question = unicodedata.normalize("NFC", question)
    print(add_definitions(question))