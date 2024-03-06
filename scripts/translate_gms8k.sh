#!/bin/bash  
set -ex

python src/crawl_data/translate_gms8k.py \
    --input_path /home/vinhnq29/Public/zalo_challenge_2023/data/OpenMathInstruct-1/correct_solutions/train_questions.txt \
    --output_path /home/vinhnq29/Public/zalo_challenge_2023/data/OpenMathInstruct-1/correct_solutions/train_questions_translated.jsonl