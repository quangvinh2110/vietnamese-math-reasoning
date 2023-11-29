import json
import re
import argparse

import requests
from bs4 import BeautifulSoup

from tqdm import tqdm

close_tag_patterns = ["</span>", "</div>", "</p>", "</h[\d]>"]
open_tag_patterns = ["<div.*?>", "<span.*?>", "<p.*?>", "<h.*?>"]
CLOSE_TAG = re.compile("|".join(close_tag_patterns))
OPEN_TAG = re.compile("|".join(open_tag_patterns))

def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--links_path",
        type=str,
        default=None,
        help="The absolute path to link file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The absolute path to output file"
    )
    args = parser.parse_args()

    # Sanity checks
    if args.links_path is None or args.output_path is None:
        raise ValueError("Need both a links file and a output file")
    
    return args

def cleanhtml(raw_html):
    text = CLOSE_TAG.sub("", raw_html)
    text = OPEN_TAG.sub("", text)
    return text


def main():
    args = parse_args()
    math_5_links_file = open(args.links_path, "r")
    math_5_questions_file = open(args.output_path, "a")
    for link in tqdm(math_5_links_file):
        page = requests.get(link.strip())
        soup = BeautifulSoup(page.content, "html.parser")
        choices = soup.find_all("div", {"class": "answer-content"})
        question = soup.find("div", {"class": "question-content"})
        answer = soup.find("div", {"class": "question-reason"})
        if len(choices)==0 or not question or not answer:
            continue
        choices = list(map(lambda t: t.prettify(), choices))
        question = question.prettify()
        answer = answer.prettify()
        if "<img" in question or "<img" in answer or "<img" in choices:
            continue
        choices = "########".join(choices)
        choices = cleanhtml(choices).strip()
        question = cleanhtml(question).strip()
        answer = cleanhtml(answer).strip()
        d = json.dumps(
            {"question": question, "choices": choices, "answer": answer}, 
            ensure_ascii=False
        )+"\n"
        math_5_questions_file.write(d)

    math_5_links_file.close()
    math_5_questions_file.close()


if __name__ == "__main__":
    main()