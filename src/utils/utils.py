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

from rank_bm25 import BM25Okapi

from .constants import unit_list
from .preprocess import preprocess


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
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def convert_unit(unit_1: str, unit_2: str):
    pass


def add_unit_conversion(question: str):
    pass


def add_definitions(question: str):
    definitions =[]
    # minus
    if ("số bị trừ" in question) or ("số trừ" in question):
        definitions.append("".join([
            "the term `số bị trừ` refers to the minuend; ",
            "the term `số trừ` refers to the subtrahend; ",
            "the term `hiệu` refers to the difference."
        ]))
    # add
    if "số hạng" in question:
        definitions.append("".join([
            "the term `số hạng` refers to the addend; ",
            "the term `tổng` refers to the sum."
        ]))
    # multiply
    if "thừa số" in question: 
        definitions.append("".join([
            "the term `thừa số` refers to the factor; "
            "the term `tích` refers to the product."
        ]))
    # divide
    if ("số bị chia" in question) or ("số chia" in question) or ("số dư" in question):
        definitions.append("".join([
            "the term `số bị chia` refers to the dividend; ",
            "the term `số chia` refers to the divisor; ",
            "the term `thương` refers to the quotient; ",
            "the term `số dư` refers to the remainder."
        ]))
    # property
    if ("tính chất giao hoán" in question) or ("giao hoán" in question) or ("tính giao hoán" in question):
        definitions.append("the term `tính chất giao hoán` refers to the commutative property.")
    if ("tính chất kết hợp" in question) or ("tính kết hợp" in question):
        definitions.append("the term `tính chất kết hợp` refers to the associative property.")
    # numbers
    if "số tròn chục" in question:
        definitions.append("the term `số tròn chục` refers to numbers that are greater than or equal to 10 and divisible by 10.")
    if "số tròn trăm" in question:
        definitions.append("the term `số tròn trăm` refers to numbers that are greater than or equal to 100 and divisible by 100.")
    if "số tròn nghìn" in question:
        definitions.append("the term `số tròn nghìn` refers to numbers that are greater than or equal to 1000 and divisible by 1000.")

    return definitions

trans_table = str.maketrans({p: f"\{p}" for p in string.punctuation})
unit_list = list(map(lambda u: u.translate(trans_table), unit_list))
UNIT_PATTERN = re.compile(r"\d[\s]*({})".format("|".join(unit_list)))


def add_notes(question: str, choices: str) -> list:
    notes = []
    if UNIT_PATTERN.search("\n".join(choices)):
        notes.append(
            "**you should convert the final result and all the choices to the same unit**"
        )

    notes += add_definitions(question)
    if "hình tròn" in question or "vòng tròn" in question:
        notes.append("the value of pi is 3.14. You should use this value to compute the result")
    return notes
