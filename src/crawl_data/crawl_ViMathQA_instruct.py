import json
import asyncio

from tgi_crawler import TgiCrawler
from vllm_crawler import VllmCrawler

SYS_PROMPT = """
Given a question and a corresponding answer, your task is to rewrite provided answer to format: first-reasoning-then-conclusion and make it more detail if necessary. 
Please return your final answer in format ```
# your Vietnamese rewritten answer
```

Example 1:
Question: Natalia đã bán kẹp tóc cho 48 người bạn của cô ấy vào tháng 4, và sau đó cô ấy đã bán nửa số lượng kẹp tóc đó vào tháng 5. Natalia đã bán tổng cộng bao nhiêu kẹp tóc trong tháng 4 và tháng 5?

Provided answer: Đáp án là 72. Natalia đã bán 24 kẹp trong tháng 5.
Natalia đã bán tổng cộng 72 kẹp trong tháng 4 và tháng 5.

step-by-step answer: ```
Vào tháng 4, Natalia đã bán kẹp tóc cho 48 người bạn của cô. Sau đó, vào tháng 5, cô đã bán nửa số lượng kẹp tóc mà cô đã bán trong tháng trước, tức là 48/2 = 24 kẹp tóc. Do đó, tổng số kẹp tóc mà Natalia đã bán trong hai tháng này là 48 (tháng 4) + 24 (tháng 5) = 72 kẹp tóc. Vậy đáp án là 72
```

Question: Một cửa hàng trong hai tháng bán được 3 450 m vải. Tháng thứ nhất bán được ít hơn tháng thứ hai là 170 m vải. Vậy tháng thứ nhất cửa hàng đó bán được số mét vải là:
A. 170 m vải
B. 1 640 m vải
C. 1 810 m vải
D. 1 725 m vải.

Provided answer: Đáp án đúng là: B Tháng thứ nhất bán được số mét vải là: (3 450 − 170) : 2 = 1 640 (m) Đáp số: 1 640 mét vải.

step-by-step answer: ```
Cửa hàng đã bán tổng cộng 3450 mét vải trong hai tháng. Trong đó, tháng thứ hai bán được nhiều hơn tháng thứ nhất 170 mét vải. Vì vậy, để tìm ra số mét vải mà cửa hàng đã bán trong tháng thứ nhất, chúng ta cần trừ 170 mét vải từ tổng số vải đã bán, sau đó chia đều cho hai tháng. Tức là (3450 - 170) : 2 = 1640 mét vải. Do đó, cửa hàng đã bán được 1640 mét vải trong tháng thứ nhất. Vậy đáp án là B: 1640 mét vải.
```
""".strip()

USER_PROMPT_TEMPLATE = """
Question: {question}
{choices}

Provided answer: {answer}
""".strip()


# class Gsm8kTgiCrawler(TgiCrawler):
#     def extract_sample(self, sample) -> dict:
#         question = sample["question"]
#         explanation = sample["explanation"]
#         return {
#             "question": question, 
#             "explanation": explanation, 
#         }


class VimathqaVllmCrawler(VllmCrawler):
    def extract_sample(self, sample) -> dict:
        question = sample["md_question"]
        choices = sample["md_choices"]
        answer = sample["md_answer"]
        return {
            "question": question, 
            "choices": "\n".join(choices), 
            "answer": answer
        }


if __name__ == "__main__":


    crawler = VimathqaVllmCrawler(
        endpoint_ip = "http://10.254.138.192:9002",
        model_name = "Meta-Llama-3-70B-Instruct",
        eos_token = "<|eot_id|>",
        system_prompt = SYS_PROMPT, 
        user_prompt_template = USER_PROMPT_TEMPLATE, 
        assistant_prompt_prefix = "```\n"
    )

    dataset = []
    with open("/workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/train_v1/train_v1.jsonl") as f:
        for line in f:
            dataset.append(json.loads(line))

    asyncio.run(crawler.generate_async(
        dataset=dataset,
        output_field="rewritten_answer",
        output_file="/workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/train_v2/train_2.jsonl"
    ))
