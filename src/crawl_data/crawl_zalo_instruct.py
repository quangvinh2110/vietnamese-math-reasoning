import g4f
import json
import pandas as pd
from tqdm import tqdm
import argparse

from ..utils.utils import add_definitions
from prompts import zalo_instruct_system_prompt

g4f.debug.logging = True  # Enable logging
g4f.check_version = False  # Disable automatic version checking
print(g4f.version)  # Check version
print(g4f.Provider.Ails.params)  # Supported args

def parse_args():
    parser = argparse.ArgumentParser(description="Crawl step-by-step instructions to solve multi-choices math problems.")
    parser.add_argument(
        "--question_path",
        type=str,
        default=None,
        help="The absolute path to datasets containing questions and choices."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The absolute path to output file"
    )
    args = parser.parse_args()

    # Sanity checks
    if args.question_path is None or args.output_path is None:
        raise ValueError("Need both a question corpus and a output file")
    
    return args

# Define system prompt
system_prompt = zalo_instruct_system_prompt


def main():
    args = parse_args()
    train_question = pd.read_csv(args.question_path)
    questions = train_question["question"]
    with open(args.output_path, "w") as outfile:
        for question in tqdm(questions):
            question = add_definitions(question)
            user_prompt = (
                f"Solve the following multiple-choices problem: {question}\n"
            )
            # Streamed completion
            response = g4f.ChatCompletion.create(
                model="gpt-3.5-turbo",
                provider=g4f.Provider.GPTalk,
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}],
                stream=False,
            )
            d = json.dumps({"question": question, "instruction": response}, ensure_ascii=False)+"\n"
            outfile.write(d)


if __name__ == "__main__":
    main()