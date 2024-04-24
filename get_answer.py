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
        "--batch_size",
        type=int,
        default=8,
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
    parser.add_argument(
        "--use_cot_prompt",
        action="store_true",
        help=""
    )
    args = parser.parse_args()

    # Sanity checks
    if not (args.peft_model_name and args.dataset_name):
        raise ValueError("")
    
    return args

if __name__ == "__main__":

    print("="*56)
    print("="*20+" VIMATHQA INFER "+"="*20)
    print("="*56)
    args = parse_args()
    models_hub = "/workspace/home/vinhnq29/zac2023-main/models_hub/"
    model_path = ""
    peft_models_hub = "/workspace/home/vinhnq29/zac2023-main/checkpoints/adapters/"
    peft_model_name = args.peft_model_name
    peft_model_path = peft_models_hub + peft_model_name
    for model_name in MODEL_LIST:
        if model_name in peft_model_name:
            model_path = models_hub + model_name
    test_datasets_hub = "/workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/"
    dataset_name = args.dataset_name
    dataset_path = test_datasets_hub + dataset_name
    batch_size = args.batch_size
    print(f"MODEL PATH: {model_path}")
    print(f"PEFT MODEL PATH: {peft_model_path}")
    print(f"DATASET PATH: {dataset_path}")
    print(f"BATCH SIZE: {batch_size}")
    print(f"DO QUANTIZE: {args.do_quantize}")
    print(f"USE ADAPTER: {args.use_adapter}")
    print(f"MERGE ADAPTER: {args.merge_adapter}")
    print(f"USE COT PROMPT: {args.use_cot_prompt}")
    print("="*56)

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
    if args.use_cot_prompt:
        pipeline.assistant_prompt = "Trước hết hãy phân tích câu hỏi một cách cẩn thận và suy luận từng bước một."
    
    test_dataset = load_from_disk(dataset_path)

    result_folder_hub = "/workspace/home/vinhnq29/zac2023-main/results/"
    result_folder_path = result_folder_hub+normalize_name(peft_model_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    result_filepath = result_folder_path+"/"+normalize_name(dataset_name)+".jsonl"
    result_file = open(result_filepath, "w")
    for batch in tqdm(test_dataset.iter(batch_size=batch_size)):
        generated_answers = pipeline.generate_batch(
            instruction_list=batch["instruction"],
            question_list=batch["question"], 
            choices_list=batch["choices"],
        )
        for instruction, question, choices, answer, right_choice, generated_answer in zip(batch["instruction"], batch["question"], batch["choices"], batch["answer"], batch["right_choice"], generated_answers):
            result_file.write(json.dumps({
                'instruction': instruction,
                'question': question,
                'choices': choices,
                'answer': answer,
                'right_choice': right_choice,
                'generated_answer': generated_answer,
            }, ensure_ascii=False) + "\n")
    result_file.close()
