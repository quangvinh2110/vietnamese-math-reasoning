from typing import Any, Optional, List
import os
import aiohttp
import asyncio
import traceback
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
)

from tqdm.asyncio import tqdm

from .base_pipeline import BasePipeline


class TgiPipeline(BasePipeline):

    def __init__(
        self,
        tgi_ip: str = None,
        model_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        self.tgi_ip = tgi_ip
        if not tokenizer and model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif tokenizer:
            self.tokenizer = tokenizer
        else:
            raise ValueError()
        self.tokenizer.add_eos_token = False
        self.tokenizer.padding_side = "left"


    async def get_answer(self, session, prompt, max_new_tokens=768):
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
            async with session.post(self.tgi_ip.strip("/")+'/generate', headers=headers, json=data, timeout=600) as resp:
                resp = await resp.json()
                return prompt, resp["generated_text"]
        except:
            return prompt, "Failed: " + str(traceback.format_exc())
    

    async def generate_async(
        self, 
        instruction_list: List[str],
        question_list: List[str], 
        choices_list: List[list],
        max_new_tokens: int = 768
    ):
        prompts = self.prepare_prompts(
            instruction_list=instruction_list,
            question_list=question_list, 
            choices_list=choices_list,
        )
        async with asyncio.BoundedSemaphore(int(os.getenv("SEMAPHORE_LIMIT"))):
            session_timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                tasks = []
                for prompt in prompts:
                    tasks.append(asyncio.ensure_future(
                        self.get_answer(session, prompt, max_new_tokens)
                    ))
                results = await tqdm.gather(*tasks)
        return ["".join(result) for result in results] 
