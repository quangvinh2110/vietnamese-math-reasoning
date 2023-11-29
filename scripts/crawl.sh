set -ex

INPUT_FILE=workspace/vinhnq29/zalo_challenge_2023/data/zalo/train/math_code_v2.jsonl
OUTPUT_FILE=workspace/vinhnq29/zalo_challenge_2023/data/zalo/train/math_instruct_v2.jsonl


python src/crawl_data/crawl_math_code.py
    --question_path $INPUT_FILE \
    --output_path $OUTPUT_FILE