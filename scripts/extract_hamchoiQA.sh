#!/bin/bash  
set -ex

python src/crawl_data/extract_hamchoiQA.py \
    --input_path /home/vinhnq29/Public/zalo_challenge_2023/data/hamchoi/hamchoi_questions.jsonl \
    --output_path /home/vinhnq29/Public/zalo_challenge_2023/data/ViMathQA/raw_hamchoi_questions.jsonl