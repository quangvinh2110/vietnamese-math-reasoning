import g4f
import json
import pandas as pd
from tqdm import tqdm
import argparse

from prompts import gms8k_instruct_system_prompt

g4f.debug.logging = True  # Enable logging
g4f.check_version = False  # Disable automatic version checking
print(g4f.version)  # Check version
print(g4f.Provider.Ails.params)  # Supported args

def parse_args():
    parser = argparse.ArgumentParser(description="Crawl step-by-step instructions to solve math problems.")
    parser.add_argument(
        "--question_path",
        type=str,
        default=None,
        help="The absolute path to datasets containing questions"
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
system_prompt = gms8k_instruct_system_prompt


def main():
    args = parse_args()
    train_question = pd.read_csv(args.question_path)
    questions = train_question["question"]
    count = 0
    with open(args.output_path, "a") as outfile:
        for question in tqdm(questions):
            if count < 1310:
                count+=1
                continue
            user_prompt = (
                f"Solve the following problem: {question}\n"
            )
            # Streamed completion
            response = None
            while not response:
                try:
                    response = g4f.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        provider=g4f.Provider.GPTalk,
                        messages=[{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}],
                        stream=False,
                    )
                except:
                    response = None
            d = json.dumps({"question": question, "instruction": response}, ensure_ascii=False)+"\n"
            outfile.write(d)

if __name__ == "__main__":
    main()
