import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import PeftModel, PeftModelForCausalLM

def get_model_path_from_adapter_path(adapter_path: str) -> str:
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        model_path = json.load(f)["base_model_name_or_path"]
    return model_path

peft_models_hub = "/workspace/home/vinhnq29/zac2023-main/checkpoints/adapters/"
peft_model_name = "qwen2-7b-instruct-lora-2024-06-26-05-23-50/checkpoint-456"
peft_model_path = os.path.join(peft_models_hub, peft_model_name)
model_path = get_model_path_from_adapter_path(peft_model_path)


quantize = None
bnb_config = None
if quantize == "4bit":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config, 
    trust_remote_code=True, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = PeftModel.from_pretrained(model, peft_model_path)

model = model.merge_and_unload()

import re

def normalize_name(name: str) -> str:
    return re.sub(r"[\W]+", "-", name)

full_checkpoints_hub = "/workspace/home/vinhnq29/zac2023-main/checkpoints/full_models/"
full_checkpoint_path = full_checkpoints_hub + normalize_name(peft_model_name)
model.save_pretrained(full_checkpoint_path, safe_serialization=True)
tokenizer.save_pretrained(full_checkpoint_path)
