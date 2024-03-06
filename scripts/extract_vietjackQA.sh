#!/bin/bash  
set -ex

python src/crawl_data/extract_vietjackQA.py \
    --input_path /home/vinhnq29/Public/zalo_challenge_2023/data/vietjack/vietjack_questions.jsonl \
    --output_path /home/vinhnq29/Public/zalo_challenge_2023/data/ViMathQA/raw_vietjack_questions.jsonl