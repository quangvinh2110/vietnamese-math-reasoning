import os
import time
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer
)
from peft import PeftModel


from ..utils.utils import add_notes, construct_prompt


model_path = "./pretrained_models/toracode-13b"
adapter_path = "./checkpoints/toracode-13b_v1.1"
PROMPT_TYPE = "zalo"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=AutoConfig.from_pretrained(model_id),
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True
)
model = PeftModel.from_pretrained(model, adapter_path)
# model = model.merge_and_unload()


def get_answer(prompt: str, max_new_tokens: int=1024):
    start_time = time.time()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            # temperature=0.4,
            # top_p=0.95,
            # top_k=20,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )
    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    infer_time = time.time() - start_time
    return output.strip(), infer_time


if __name__ == "__main__":

    question = (
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
    )
    question = add_notes(question)
    prompt = construct_prompt(question, PROMPT_TYPE)
    _, infer_time = get_answer(prompt)
    # print("Output:")
    # print(output)
    print(f"Inference time: {infer_time}")