#!/bin/bash  
set -ex

python src/crawl_data/crawl_MathQA.py \
    --links_path /home/vinhnq29/Public/zalo_challenge_2023/data/vietjack/toan5_links.txt \
    --output_path /home/vinhnq29/Public/zalo_challenge_2023/data/vietjack/toan5_questions.jsonl