import os
import re
import json
import argparse
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_from_disk
from src.infer.base_pipeline import BasePipeline
from src.infer.code_pipeline import CodePipeline
from tqdm import tqdm

MODEL_LIST = [
    "metamath-mistral-7b",
    "qwen1-5-7b-chat",
    "tora-code-7b-v1-0",
    "vinallama-7b-chat",
    "vistral-7b-chat",
    "wizardcoder-python-7b-v1-0",
    "wizardmath-7b-v1-1",
    "zephyr-7b-beta"
]

def normalize_name(name: str) -> str:
    return re.sub(r"[\W]+", "-", name)

def read_jsonl(filepath: str):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_model_name_from_adapter_name(adapter_name: str) -> str:
    for model_name in MODEL_LIST:
        if model_name in adapter_name:
            return model_name

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="base",
        help=""
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help=""
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="",
        help=""
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--do_quantize",
        action="store_true",
        help=""
    )
    parser.add_argument(
        "--use_adapter",
        action="store_true",
        help=""
    )
    parser.add_argument(
        "--merge_adapter",
        action="store_true",
        help=""
    )
    args = parser.parse_args()

    # Sanity checks
    if not (args.adapter_name or args.model_name) or not args.dataset_name:
        raise ValueError("")
    if args.use_adapter and not args.adapter_name:
        raise ValueError("")
    
    return args


def prepare_pipeline(args):
    Pipeline = BasePipeline if args.pipeline=="base" else CodePipeline
    pipeline = Pipeline(
        model_path=args.model_path,
        quantize="4bit" if args.do_quantize else None,
        adapter_path=args.adapter_path if args.use_adapter else None,
        merge_adapter=args.merge_adapter,
    )
    return pipeline


if __name__ == "__main__":

    print("="*63)
    print("="*20+" VIMATHQA RIGHT CHOICE "+"="*20)
    print("="*63)
    args = parse_args()
    base_models_hub = "/workspace/home/vinhnq29/zac2023-main/models_hub/"
    merged_models_hub = "/workspace/home/vinhnq29/zac2023-main/checkpoints/full_models/"
    adapters_hub = "/workspace/home/vinhnq29/zac2023-main/checkpoints/adapters/"
    # args.adapter_path = adapters_hub + args.adapter_name
    if "lora" in args.model_name or "qlora" in args.model_name:
        args.model_path = merged_models_hub + args.model_name
        args.adapter_name = ""
        args.adapter_path = ""
        args.use_adapter = False
        args.merge_adapter = False
    elif args.adapter_name:
        args.model_name = get_model_name_from_adapter_name(args.adapter_name)
        args.model_path = base_models_hub + args.model_name
        args.adapter_path = adapters_hub + args.adapter_name
    else:
        args.model_path = base_models_hub + args.model_name
        args.adapter_name = ""
        args.adapter_path = ""
        args.use_adapter = False
        args.merge_adapter = False
    dataset_name = args.dataset_name
    result_folder_hub = "/workspace/home/vinhnq29/zac2023-main/results/"
    if args.adapter_name:
        result_folder_path = result_folder_hub+normalize_name(args.adapter_name)
    else: 
        result_folder_path = result_folder_hub+normalize_name(args.model_name)
    result_filepath = result_folder_path+"/"+normalize_name(dataset_name)+".jsonl"
    print(f"MODEL PATH: {args.model_path}")
    print(f"ADAPTER PATH: {args.adapter_path}")
    print(f"RESULT PATH: {result_filepath}")
    print(f"DO QUANTIZE: {args.do_quantize}")
    print(f"USE ADAPTER: {args.use_adapter}")
    print(f"MERGE ADAPTER: {args.merge_adapter}")
    print("="*63)

    pipeline = prepare_pipeline(args)
    
    infer_result = read_jsonl(result_filepath)
    output_file = open(result_folder_path+"/"+normalize_name(dataset_name)+"-pred_choice.jsonl", "w")
    for sample in tqdm(infer_result):
        get_pred_prompt, pred_choice = pipeline.select_answer_for_multiple_choices(
            generated_answer=sample["generated_answer"],
            choices=sample["choices"],
        )
        sample["get_pred_prompt"] = get_pred_prompt
        sample["pred_choice"] = pred_choice
        output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    output_file.close()
