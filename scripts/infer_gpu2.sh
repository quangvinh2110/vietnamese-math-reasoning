#!/bin/bash#
set -ex

DEVICE=2
# TEST_DATASET=test_v1/final_code_models_test
TEST_DATASET=test_v1/final_base_models_test
BATCH_SIZE=4

CUDA_VISIBLE_DEVICES=${DEVICE} python get_answer.py --peft_model_name "metamath-mistral-7b-lora-2024-04-23-08-29-47" --dataset_name ${TEST_DATASET} --batch_size ${BATCH_SIZE} --use_adapter --use_cot_prompt

CUDA_VISIBLE_DEVICES=${DEVICE} python get_answer.py --peft_model_name "metamath-mistral-7b-lora-2024-04-23-08-29-47" --dataset_name ${TEST_DATASET} --batch_size ${BATCH_SIZE} --use_adapter

CUDA_VISIBLE_DEVICES=${DEVICE} python get_answer.py --peft_model_name "metamath-mistral-7b-lora-2024-04-23-08-29-47/checkpoint-211" --dataset_name ${TEST_DATASET} --batch_size ${BATCH_SIZE} --use_adapter --use_cot_prompt

CUDA_VISIBLE_DEVICES=${DEVICE} python get_answer.py --peft_model_name "metamath-mistral-7b-lora-2024-04-23-08-29-47/checkpoint-211" --dataset_name ${TEST_DATASET} --batch_size ${BATCH_SIZE} --use_adapter

# CUDA_VISIBLE_DEVICES=${DEVICE} python  -m pdb -c continue get_right_choice.py --peft_model_name "metamath-mistral-7b-qlora-2024-04-09-03-30-47/checkpoint-211" --dataset_name ${TEST_DATASET}

# CUDA_VISIBLE_DEVICES=${DEVICE} python get_right_choice.py --peft_model_name "metamath-mistral-7b-qlora-2024-04-09-03-30-47/checkpoint-211" --dataset_name ${TEST_DATASET}

# CUDA_VISIBLE_DEVICES=${DEVICE} python get_right_choice.py --peft_model_name "qwen1-5-7b-chat-qlora-2024-04-09-04-12-55" --dataset_name ${TEST_DATASET}

# CUDA_VISIBLE_DEVICES=${DEVICE} python get_right_choice.py --peft_model_name "wizardmath-7b-v1-1-qlora-2024-04-09-03-35-12/checkpoint-211" --dataset_name ${TEST_DATASET}

# CUDA_VISIBLE_DEVICES=${DEVICE} python get_right_choice.py --peft_model_name "zephyr-7b-beta-qlora-2024-04-09-02-51-25/checkpoint-211" --dataset_name ${TEST_DATASET}
