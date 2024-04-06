import json
import argparse

from bs4 import BeautifulSoup

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="The absolute path to input file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The absolute path to output file"
    )
    args = parser.parse_args()

    # Sanity checks
    if args.input_path is None or args.output_path is None:
        raise ValueError("Need both a input file and a output file")
    
    return args


def main():
    args = parse_args()
    input_file = open(args.input_path, "r")
    output_file = open(args.output_path, "w")
    for line in tqdm(input_file):
        sample = json.loads(line)
        soup = BeautifulSoup(sample["content"], "html.parser")
        choices = soup.find_all("div", {"class": "option-choices js-answer"})
        choices.append(soup.find("div", {"class": "option-choices js-answer answer-correct"}))
        question = soup.find("h1", {"class": "title-question overflow-x-el"})
        answers = [soup.find("div", {"class": "result"}), soup.find("div", {"class": "option-choices js-answer answer-correct"})]
        


        choices = [choice.prettify() if choice else "" for choice in choices] if choices else [""]
        question = question.prettify() if question else ""
        answers = [answer.prettify() if answer else "" for answer in answers] if answers else [""]
        d = json.dumps(
            {"url": sample["url"],
             "question": question, 
             "choices": choices, 
             "answer": answers}, 
            ensure_ascii=False
        )+"\n"
        output_file.write(d)
    input_file.close()
    output_file.close()

if __name__ == "__main__":
    main()