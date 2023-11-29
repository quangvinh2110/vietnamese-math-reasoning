import g4f
import json
from tqdm import tqdm
import argparse

from prompts import zalo_code_user_prompt_prefix

g4f.debug.logging = True  # Enable logging
g4f.check_version = False  # Disable automatic version checking
print(g4f.version)  # Check version
print(g4f.Provider.Ails.params)  # Supported args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crawl code to solve multi-choices math problems."
    )
    parser.add_argument(
        "--question_path",
        type=str,
        default=None,
        help="The absolute path to datasets containing questions and instructions."
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


user_prompt_prefix = zalo_code_user_prompt_prefix

def main():
    args = parse_args()
    f = open(args.question_path, "r")
    with open(args.output_path, "w") as outfile:
        for line in tqdm(f):
            line = json.loads(line)
            if line["instruction"][:5] != "STEP\n":
                line["instruction"] = "STEP\n"+line["instruction"]
            question = line["question"] + line["instruction"]
            user_prompt_postfix = (
                f"Solve the following multiple-choices problem: {question}"
            )
            user_prompt = user_prompt_prefix + user_prompt_postfix
            # Streamed completion
            response = None
            while not response:
                try:
                    response = g4f.ChatCompletion.create(
                        model=g4f.models.gpt_4,
                        provider=g4f.Provider.Bing,
                        messages=[{"role": "user", "content": user_prompt}],
                        stream=False,
                    )
                except:
                    response = None
            d = json.dumps({"question": line["question"], "instruction": line["instruction"], "code": response}, ensure_ascii=False)+"\n"
            outfile.write(d)



if __name__ == "__main__":
    main()
