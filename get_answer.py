import os
import re
import json
import argparse
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
        default="",
        help=""
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=""
    )
    # for qlora checkpoint
    parser.add_argument(
        "--do_quantize",
        action="store_true",
        help=""
    )
    # for lora/qlora checkpoint
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
    parser.add_argument(
        "--use_vllm",
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
    if args.use_vllm:
        pipeline = Pipeline(
            model_path=args.model_path,
            use_vllm=True,
            adapter_path=args.adapter_path if args.use_adapter else None,
        )
    else:        
        pipeline = Pipeline(
            model_path=args.model_path,
            quantize="4bit" if args.do_quantize else None,
            adapter_path=args.adapter_path if args.use_adapter else None,
            merge_adapter=args.merge_adapter,
        )

    if args.use_cot_prompt:
        # pipeline.assistant_prompt = "Phân tích và suy luận: "
        # pipeline.assistant_prompt = "STEP"
        pipeline.assistant_prompt = "```python"

    # pipeline.system_prompt = """Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.
# Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."""
    # pipeline.system_prompt = "Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác."
    
    return pipeline


if __name__ == "__main__":

    print("="*61)
    print("="*20+" VIMATHQA GET ANSWER "+"="*20)
    print("="*61)
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
    elif args.use_adapter and args.adapter_name:
        for model_name in MODEL_LIST:
            if model_name in args.adapter_name:
                args.model_name = model_name
                args.model_path = base_models_hub + model_name
                break
        args.adapter_path = adapters_hub + args.adapter_name
    else:
        args.model_path = base_models_hub + args.model_name
        args.adapter_name = ""
        args.adapter_path = ""
        args.use_adapter = False
        args.merge_adapter = False
    test_datasets_hub = "/workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/"
    dataset_name = args.dataset_name
    dataset_path = test_datasets_hub + dataset_name
    batch_size = args.batch_size
    print(f"MODEL PATH: {args.model_path}")
    print(f"ADAPTER PATH: {args.adapter_path}")
    print(f"DATASET PATH: {dataset_path}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"DO QUANTIZE: {args.do_quantize}")
    print(f"USE ADAPTER: {args.use_adapter}")
    print(f"MERGE ADAPTER: {args.merge_adapter}")
    print(f"USE COT PROMPT: {args.use_cot_prompt}")
    print(f"USE VLLM: {args.use_vllm}")
    print("="*61)

    pipeline = prepare_pipeline(args)
    
    test_dataset = load_from_disk(dataset_path)

    result_folder_hub = "/workspace/home/vinhnq29/zac2023-main/results/"
    if args.adapter_name:
        result_folder_path = result_folder_hub+normalize_name(args.adapter_name)
    else:
        result_folder_path = result_folder_hub+normalize_name(args.model_name)
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
