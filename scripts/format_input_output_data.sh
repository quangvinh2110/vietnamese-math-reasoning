#!/bin/bash#
set -ex


# python src/utils/format_input_output_instruct.py \
# --model_name deepseek-math-7b-rl \
# --input_file /workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/train_v2/train_2.jsonl \
# --output_file /workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/train_v2/input_output_dsmath.json \
# # --eos_token "<|eot_id|>"


python src/utils/format_input_output_instructcode.py \
--model_name Llama-3-8B-Instruct \
--input_file /workspace/home/vinhnq29/zac2023-main/data_hub/ViMathIntructCode/train-2024-06-27-27-01-15/vimathqa_llama_qwen.jsonl \
--output_file /workspace/home/vinhnq29/zac2023-main/data_hub/ViMathIntructCode/train-2024-06-27-27-01-15/input_output_llama3.json \
--eos_token "<|eot_id|>"
