#!/bin/bash#
set -ex


# python src/utils/format_input_output_instruct.py \
# --model_name vistral-7b-chat \
# --input_file /workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/train_v2/train_2.jsonl \
# --output_file /workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/train_v2/input_output_vistral.json \
# # --eos_token "<|eot_id|>"


python src/utils/format_input_output_instructcode.py \
--model_name qwen2-7b-instruct \
--input_file /workspace/home/vinhnq29/zac2023-main/data_hub/ViMathIntructCode/instructcode_vimathqa.jsonl \
--output_file /workspace/home/vinhnq29/zac2023-main/data_hub/ViMathIntructCode/input_output_qwen.json \
# --eos_token "<|eot_id|>"
