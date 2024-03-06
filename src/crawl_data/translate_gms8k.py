import requests
import time
import json
import argparse

from tqdm import tqdm


URL = "https://nlu-translate.vinai.io/translate"
HEADERS = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Connection': 'keep-alive',
    'Content-Type': 'application/json',
    'Origin': 'https://vinai-translate.vinai.io',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Linux"',
    'Cookie': 'TS01e39374=01dcc7e0b05336d78b3734f0dee5c36ea465956cb7339701f272e13b4003df2130b0e393fdba4272e884d80ca24ecdd737d4e2d38f'
}
PAYLOAD = {
    "mt_mode": "en2vi",
    "mt_input": ""
}
CHUNK_SIZE = 10
SPLITTER = "\n"


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


def divide_chunks(data: list): 
      
    # looping till length l 
    for i in range(0, len(data), CHUNK_SIZE):  
        yield data[i:i + CHUNK_SIZE] 



def main(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as f:
        data = f.read().split("\n")

    output_file = open(output_path, "w", encoding="utf-8")
    for chunk in tqdm(divide_chunks(data)):
        PAYLOAD["mt_input"] = SPLITTER.join(chunk)
        payload = json.dumps(PAYLOAD)
        try:
            response = requests.request("POST", URL, headers=HEADERS, data=payload).json()["mt_output"].split(SPLITTER)
        except Exception as err:
            response = [str(err)]*CHUNK_SIZE
        for ques, ques_translated in zip(chunk, response):
            d = json.dumps({
                "en": ques, "vi": ques_translated
            }, ensure_ascii=False) + "\n"
            output_file.write(d)
        time.sleep(3)
        


if __name__ == "__main__":
    args = parse_args()
    main(args.input_path, args.output_path)
