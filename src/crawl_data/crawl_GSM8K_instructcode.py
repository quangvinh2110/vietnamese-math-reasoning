import json
import asyncio

from tgi_crawler import TgiCrawler
from vllm_crawler import VllmCrawler

SYS_PROMPT = """
You are a world-class math tutor who helps students of all levels understand and solve mathematical problems. 
However, you are really bad at arithmetic operations like addition, subtraction, multiplication, division, comparison, exponential,...
So if the question requires any calculation, you will write a Python code to calculate. 
Please provide STEP-BY-STEP explanations and guidance of the problem in Vietnamese based on provided answer first.
Then write a Python code (if necessary) to return the final result.
Use clear language to make complex concepts easier to grasp. 
NOTE: **Do NOT do any calculations or provide any result in the guidance. You should let the Python code do that part for you. If you calculate anything in the guidance, someone might die.**
Please return your final answer in format: ```guidance
# your guidance
```

(Optional) ```python
# your python code if necessary
```
```output
# output of the above code
```
# conclusion

Example 1:
Question: Vịt của Janet đẻ được 16 quả trứng mỗi ngày. Cô ấy ăn 3 quả vào bữa sáng và nướng bánh muffin cho bạn bè mỗi ngày với 4 quả. Cô ấy bán số trứng còn lại tại chợ nông sản hàng ngày với giá $2 mỗi quả trứng vịt tươi. Cô ấy kiếm được bao nhiêu đô la mỗi ngày tại chợ nông sản?

Provided answer: Janet bán 9 quả trứng vịt mỗi ngày. Cô ấy kiếm được 18 đô la mỗi ngày tại chợ nông sản.

Answer: ```guidance
Bước 1: Xác định tổng số trứng đẻ ra mỗi ngày. Trong trường hợp này, đó là 16 quả trứng.
Bước 2: Xác định số trứng Janet tiêu thụ hoặc sử dụng hàng ngày. Cô ấy ăn 3 quả và dùng 4 quả để nướng bánh, tổng cộng là 7 quả.
Bước 3: Tính số trứng còn lại để bán. Để làm điều này, trừ số trứng đã sử dụng từ tổng số trứng.
Bước 4: Nhân số trứng còn lại với giá bán để tìm ra số tiền Janet kiếm được hàng ngày.
```

```python
total_eggs = 16
eggs_eaten = 3
eggs_baked = 4
remaining_eggs = total_eggs - eggs_eaten - eggs_baked
price_per_egg = 2
janet_earnings = remaining_eggs * price_per_egg
print(janet_earnings)
```
```output
18
```
Vậy Janet kiếm được 18 đô la mỗi ngày tại chợ nông sản. Đáp án là 18.


Example 2:
Question: Lựa chọn đáp án đúng nhất: h : k : g = ...............
B. h : (k : g)
C. h : (k – g)
D. h : (k + g)
A. h : (k × g)

Provided answer: Dựa vào quy tắc chia 1 số cho 1 tích ta có: Khi chia một số cho một tích hai thừa số, ta có thể chia số đó cho một thừa số, rồi lấy kết quả tìm được chia tiếp cho thừa số kia Ngược lại: khi chia một số lần lượt cho hai số, ta có thể lấy số bị chia chia cho tích của hai số còn lại Ta có: h : k : g = h : (k × g) Vậy đáp án đúng là h : (k × g). Chọn A

Answer: ```guidance
Bước 1: Nhớ lại quy tắc chia một số cho một tích hai thừa số: Khi chia một số cho một tích hai thừa số, chúng ta có thể chia số đó cho một thừa số, rồi lấy kết quả tìm được chia tiếp cho thừa số kia.
Bước 2: Áp dụng quy tắc vào biểu thức đã cho: h : k : g
Bước 3: Chia h cho tích của k và g: h : (k × g)
Bước 4: So sánh kết quả với các lựa chọn và chọn đáp án đúng.
```
Ghi chú: Không cần thực hiện bất kỳ phép tính nào
Vậy đáp án đúng là A
""".strip()

USER_PROMPT_TEMPLATE = """
Question: {question}

Provided answer: {explanation}
""".strip()


# class Gsm8kTgiCrawler(TgiCrawler):
#     def extract_sample(self, sample) -> dict:
#         question = sample["question"]
#         explanation = sample["explanation"]
#         return {
#             "question": question, 
#             "explanation": explanation, 
#         }


class Gsm8kVllmCrawler(VllmCrawler):
    def extract_sample(self, sample) -> dict:
        question = sample["question"]
        explanation = sample["explanation"]
        return {
            "question": question, 
            "explanation": explanation, 
        }


if __name__ == "__main__":

    # crawler = Gsm8kTgiCrawler(
    #     endpoint_ip = "http://10.254.138.189:9002",
    #     eos_token = "<|eot_id|>",
    #     system_prompt = SYS_PROMPT, 
    #     user_prompt_template = USER_PROMPT_TEMPLATE, 
    #     assistant_prompt_prefix = "```guidance\n"
    # )

    crawler = Gsm8kVllmCrawler(
        endpoint_ip = "http://10.254.138.192:9002",
        model_name = "Meta-Llama-3-70B-Instruct",
        eos_token = "<|eot_id|>",
        system_prompt = SYS_PROMPT, 
        user_prompt_template = USER_PROMPT_TEMPLATE, 
        assistant_prompt_prefix = "```guidance\n"
    )

    dataset = []
    with open("/workspace/home/vinhnq29/zac2023-main/data_hub/GSM8K/vi_train.json") as f:
        dataset = json.load(f)

    asyncio.run(crawler.generate_async(
        dataset=dataset[3000:],
        output_field="instruct_code",
        output_file="/workspace/home/vinhnq29/zac2023-main/data_hub/MathIntructCode/instructcode_vigsm8k_1.jsonl"
    ))
