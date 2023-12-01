#!/bin/bash

VER=
SUB_VER=

MODEL_NAME=toracode-13b
MODEL_PATH=pretrained_models/${MODEL_NAME}
TRAIN_FILE=data/gsm8k_zalo/train_v${VER}.jsonl
OUTPUT_DIR=checkpoints/${MODEL_NAME}_v${VER}.${SUB_VER}
LOGGING_DIR=logs/${MODEL_NAME}_v${VER}.${SUB_VER}


NUM_TRAIN_EPOCHS=5
TOTAL_BATCH_SIZE=128
BATCH_SIZE_PER_GPU=1
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$BATCH_SIZE_PER_GPU))
LEARNING_RATE=1e-4


python3 src/train/run_clm_lora_nvidia.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --bf16 True \
    --do_train \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --logging_steps 1 \
    --report_to tensorboard \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 10 \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type constant \
    --warmup_ratio  0.03 \
    --weight_decay 0. \
    --max_grad_norm  0.3 \
    --full_finetune False \
    --lora_rank 256 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules "from_mapping" \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --ddp_bucket_cap_mb 50 \
    --low_cpu_mem_usage True \
    --train_on_inputs False \
    --mask_prompt
