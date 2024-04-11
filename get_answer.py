import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--peft_model_name",
        type=str,
        default=None,
        help=""
    )
    args = parser.parse_args()

    # Sanity checks
    if not args.peft_model_name:
        raise ValueError("")
    
    return args

if __name__ == "__main__":

    print("="*54)
    print("="*20+" VIMATH INFER "+"="*20)
    print("="*54)
    args = parse_args()
    models_hub = "/workspace/home/vinhnq29/zac2023-main/models_hub/"
    model_path = ""
    peft_models_hub = "/workspace/home/vinhnq29/zac2023-main/checkpoints/"
    peft_model_name = args.peft_model_name
    peft_model_path = peft_models_hub + peft_model_name
    for model_name in MODEL_LIST:
        if model_name in peft_model_name:
            model_path = models_hub + model_name
    print(f"MODEL_PATH: {model_path}")
    print(f"PEFT MODEL_PATH: {peft_model_path}")

    batch_size = 8
    print(f"BATCH SIZE: {batch_size}")
    
    model = AutoModelForCausalLM.from_pretrained(  
        model_path, 
        # quantization_config=bnb_config, 
        load_in_4bit=True,
        trust_remote_code=True, 
        device_map="auto"
    )
    model.load_adapter(
        peft_model_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pipeline = BasePipeline(
        model=model,
        tokenizer=tokenizer
    )
    
    test_dataset = load_from_disk("/workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/test_v1")

    print("="*54)

    result_folder_path = "/workspace/home/vinhnq29/zac2023-main/results/"+"/".join(peft_model_name.split("/")[:-1])
    result_filepath = f"{peft_model_name.split("/")[-1]}.jsonl"
    result_file = open(f"{result_folder_path}/{result_filepath}", "w")
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
