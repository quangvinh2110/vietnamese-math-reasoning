import json
import os
import aiohttp
import asyncio
import requests
import traceback
from transformers import AutoTokenizer

from tqdm.asyncio import tqdm

from abc import ABC, abstractmethod

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


class TgiCrawler(ABC):

    model_hub = "/workspace/home/NLP_CORE/HUB_LLM"

    def __init__(
        self,
        tgi_ip: str = "",
        eos_token: str = None,
        system_prompt: str = "",
        user_prompt_template: str = "",
        assistant_prompt_prefix: str = "",
    ):
        self.tgi_ip = tgi_ip

        self.tokenizer = self.get_tokenizer(tgi_ip)
        self.tokenizer.add_eos_token = False
        self.tokenizer.padding_side = "left"
        if eos_token:
            self.tokenizer.eos_token = eos_token

        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.assistant_prompt_prefix = assistant_prompt_prefix


    def get_tokenizer(self, tgi_ip: str):
        tgi_info = requests.get(tgi_ip.strip("/")+"/info").json()
        model_name = tgi_info["model_id"].split("/")[-1]
        model_path = os.path.join(self.model_hub, model_name)
        # print(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        return tokenizer
    

    @abstractmethod
    def extract_sample(sample) -> dict:
        pass


    async def get_answer(
            self, 
            session, 
            sample, 
            max_new_tokens=768):
        
        user_prompt = self.user_prompt_template.format(
            **self.extract_sample(sample)
        )
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": self.assistant_prompt_prefix}
            ], tokenize=False, add_generation_prompt=False
        ).strip().removesuffix(self.tokenizer.eos_token)
        # print(prompt)

        headers = {
            "Content-Type": "application/json",
        }
        data = {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': max_new_tokens,
                'repetition_penalty': 1.0,
                'do_sample': False,
                'use_cache': True,
                'stop': [self.tokenizer.eos_token],
            },
        }
        try:
            async with session.post(self.tgi_ip.strip("/")+'/generate', headers=headers, json=data, timeout=600000) as resp:
                resp = await resp.json()
                return prompt, resp["generated_text"]
        except:
            return prompt, "Failed: " + str(traceback.format_exc())
    

    async def generate_async(
        self, 
        dataset: list,
        max_new_tokens: int = 768,
        output_file: str = "tmp.jsonl",
        output_field: str = "output",
    ):
        async with asyncio.BoundedSemaphore(128):
            session_timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                tasks = []
                for sample in dataset:
                    tasks.append(asyncio.ensure_future(
                        self.get_answer(
                            session, 
                            sample,
                            max_new_tokens)
                    ))
                results = await tqdm.gather(*tasks)

        with open(output_file, "w") as f:
            for sample, (input_prompt, output) in zip(dataset, results):
                sample[output_field] = self.assistant_prompt_prefix+output
                # sample["input_prompt"] = input_prompt
                f.write(json.dumps(sample, ensure_ascii=False)+"\n")

        return 
