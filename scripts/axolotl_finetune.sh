#!/bin/bash#
set -ex

DEVICE=6
MODEL_NAME=vinallama-7b-chat
CONFIG_FILE=/workspace/home/vinhnq29/zac2023-main/config/${MODEL_NAME}/lora_v1.yml
OUTPUT_DIR=/workspace/home/vinhnq29/zac2023-main/checkpoints/${MODEL_NAME}

cat ${CONFIG_FILE}
HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${DEVICE} \
    accelerate launch -m axolotl.cli.train ${CONFIG_FILE}

cp ${CONFIG_FILE} ${OUTPUT_DIR}
