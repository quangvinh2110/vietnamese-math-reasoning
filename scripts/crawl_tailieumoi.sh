set -ex

INPUT_FILE="/workspace/vinhnq29/zalo_challenge_2023/data/tailieumoi/toan5_links.txt"
OUTPUT_FILE="/workspace/vinhnq29/zalo_challenge_2023/data/tailieumoi/toan5_questions.jsonl"


python src/crawl_data/crawl_tailieumoi.py \
--links_path $INPUT_FILE \
--output_path $OUTPUT_FILE