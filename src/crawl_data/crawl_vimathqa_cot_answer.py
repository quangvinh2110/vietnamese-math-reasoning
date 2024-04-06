import requests
import json
from tqdm import tqdm


with open("/home/vinhnq29/Public/zalo_challenge_2023/data/ViMathQA/user_msg_example", "r") as f:
    user_msg_prefix = f.read()
user_msg_postfix = """
Question: {question}
{choices}

Original solution: {answer}

Converted solution:
""".strip()
url = "https://www.dongstop.vip/api/openai/v1/chat/completions"
payload = {
    "stream": True,
    "model": "gpt-3.5-turbo",
    "temperature": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "top_p": 1
}
headers = {
    'accept': 'application/json, text/event-stream',
    'accept-language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'content-type': 'application/json',
    # 'cookie': '__vtins__3GZORVyxiPPrdcYf=%7B%22sid%22%3A%20%22c068b8d8-bbaf-5961-8192-8fbd70a772f4%22%2C%20%22vd%22%3A%201%2C%20%22stt%22%3A%200%2C%20%22dr%22%3A%200%2C%20%22expires%22%3A%201712156849432%2C%20%22ct%22%3A%201712155049432%7D; __51uvsct__3GZORVyxiPPrdcYf=1; __51vcke__3GZORVyxiPPrdcYf=919b2255-1823-58aa-b665-3699e2be6353; __51vuft__3GZORVyxiPPrdcYf=1712155049435; _dd_s=rum=0&expire=1712155953523',
    'origin': 'https://www.dongstop.vip',
    'referer': 'https://www.dongstop.vip/',
    'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Linux"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
}
n_attempts = 5


def read_jsonl(filepath: str) -> list:
    result = []
    with open(filepath) as f:
        for line in f:
            result.append(json.loads(line))
    return result


def main():
    data = read_jsonl("/home/vinhnq29/Public/zalo_challenge_2023/data/ViMathQA/data.jsonl")
    with open("/home/vinhnq29/Public/zalo_challenge_2023/data/ViMathQA/cot_data.jsonl", "a") as outfile:
        for sample in tqdm(data[80:]):
            user_msg = user_msg_prefix + user_msg_postfix.format(
                question=sample["md_question"],
                choices="\n".join(sample["md_choices"]),
                answer=sample["md_answer"]
            )
            messages = [
                {
                    "role": "system",
                    "content": "\nYou are ChatGPT, a large language model trained by OpenAI."
                },
                {
                    "role": "user",
                    "content": user_msg
                }
            ]
            payload["messages"] = messages
            # Streamed completion
            response_text = None
            for i in range(n_attempts):
                try:
                    response = requests.request(
                        "POST", url, headers=headers, data=json.dumps(payload)
                    )
                    response_tokens_list = []
                    for line in response.iter_lines():
                        if line:
                            try:
                                response_tokens_list.append(
                                    json.loads(line.decode("utf-8")[6:])["choices"][0]["delta"]["content"]
                                )
                            except:
                                pass
                    response_text = "".join(response_tokens_list)
                    break
                except Exception as e:
                    pass
            sample["cot_answer"] = response_text
            d = json.dumps(sample, ensure_ascii=False)+"\n"
            outfile.write(d)
        
        
if __name__ == "__main__":
    main()