from transformers import AutoTokenizer
import argparse
import random
from datasets import load_dataset
import json
from string import punctuation

# random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=""
    )
    args = parser.parse_args()

    # Sanity checks
    if not args.model_name or not args.output_file or not args.input_file:
        raise ValueError()
    
    return args


def normalize_name(name: str):
    return name.translate(
        str.maketrans(punctuation, "_"*len(punctuation))
    ).lower()

vi_instruction_multiple_choices_prompts = [
    "Giải bài toán trắc nghiệm sau:",
    "Hãy từng bước giải quyết câu hỏi trắc nghiệm sau:",
    "Suy luận từng bước để trả lời câu hỏi trắc nghiệm dưới đây:",
    "Trả lời câu hỏi sau:",
    "Giải bài toán dưới đây:",
    "Hãy từng bước giải quyết bài toán dưới đây:",
    "Sử dụng kiến thức của bạn để giải bài toán sau:",
    "Suy luận từng bước để trả lời câu hỏi dưới đây:"
    "Suy luận từng bước để giải bài toán sau đây:",
    "Giải bài toán dưới đây bằng cách suy luận từng bước:"
]

USER_PROMPT_TEMPLATE = """
{instruction}
{question}
{choices}
"""
    
    
    

if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("/path/to/your/models_hub/"+args.model_name)
    if args.chat_template and not tokenizer.chat_template:
        tokenizer.chat_template = args.chat_template
        
    train_data = []
    with open(args.input_file) as f:
        for line in f:
            train_data.append(json.loads(line))
        
    input_output_data  = []
    for sample in train_data:
        input_output_data.append({"segments": [
            {
                "label": False,
                "text": tokenizer.apply_chat_template([
                    {"role": "system", "content": ""},
                    {
                        "role": "user", 
                        "content": USER_PROMPT_TEMPLATE.format(
                            instruction=random.choice(vi_instruction_multiple_choices_prompts),
                            question=sample["md_question"],
                            choices="\n".join(sample["md_choices"])
                        ).strip()
                    },
                ], tokenize=False, add_generation_prompt=False)
            },
            {
                "label": True,
                "text": tokenizer.apply_chat_template([
                    {"role": "assistant", "content": sample["md_answer"]},
                ], tokenize=False, add_generation_prompt=False)
            }
        ]})

    with open(args.output_file, "w") as f:
        f.write(json.dumps(
            input_output_data,
            indent=4,
            ensure_ascii=False
        ))

    # input_output_dataset = load_dataset(
    #     "json", 
    #     data_files=args.output_file,
    #     split="train"
    # )

    # input_output_dataset.push_to_hub(
    #     "vinhnq29/ViMathQA",
    #     "train_v1", split=f"input_output_{normalize_name(args.model_name)}",
    #     token="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"
    # )
