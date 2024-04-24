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

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--peft_model_name",
        type=str,
        default=None,
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
    if not (args.peft_model_name and args.dataset_name):
        raise ValueError("")
    
    return args

if __name__ == "__main__":

    print("="*61)
    print("="*20+" VIMATHQA GET ANSWER "+"="*20)
    print("="*61)
    args = parse_args()
    models_hub = "/workspace/home/vinhnq29/zac2023-main/models_hub/"
    model_path = ""
    peft_models_hub = "/workspace/home/vinhnq29/zac2023-main/checkpoints/adapters/"
    peft_model_name = args.peft_model_name
    peft_model_path = peft_models_hub + peft_model_name
    for model_name in MODEL_LIST:
        if model_name in peft_model_name:
            model_path = models_hub + model_name
    dataset_name = args.dataset_name
    result_folder_hub = "/workspace/home/vinhnq29/zac2023-main/results/"
    result_folder_path = result_folder_hub+normalize_name(peft_model_name)
    result_filepath = result_folder_path+"/"+normalize_name(dataset_name)+".jsonl"
    print(f"MODEL PATH: {model_path}")
    print(f"PEFT MODEL PATH: {peft_model_path}")
    print(f"RESULT PATH: {result_filepath}")
    print(f"DO QUANTIZE: {args.do_quantize}")
    print(f"USE ADAPTER: {args.use_adapter}")
    print(f"MERGE ADAPTER: {args.merge_adapter}")
    print("="*61)

    if args.do_quantize:
        bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(  
            model_path, 
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config, 
            # load_in_4bit=True,
            trust_remote_code=True, 
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(  
            model_path, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True, 
            device_map="auto"
        )
    if args.use_adapter:
        model = PeftModel.from_pretrained(model, peft_model_path)
    if args.use_adapter and args.merge_adapter:
        model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pipeline = BasePipeline(
        model=model,
        tokenizer=tokenizer
    )
    
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
