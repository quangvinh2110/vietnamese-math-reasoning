#!/bin/bash#
set -ex

DEVICE=4
CURRENT_TIME=$( date '+%F-%H-%M-%S' )
MODEL_NAME=zephyr-7b-beta
MODEL_TYPE=MistralForCausalLM
TOKENIZER_TYPE=LlamaTokenizer
MAX_SEQ_LEN=4096
LORA_RANK=256
LORA_ALPHA=128
GLOBAL_BATCH_SIZE=16
MICRO_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$((${GLOBAL_BATCH_SIZE}/${MICRO_BATCH_SIZE}))
EPOCHS=2
LR_SCHEDULER=cosine
LR=2e-4
CONFIG_FILE=/workspace/home/vinhnq29/zac2023-main/config/${MODEL_NAME}-qlora-${CURRENT_TIME}.yml
OUTPUT_DIR=/workspace/home/vinhnq29/zac2023-main/checkpoints/${MODEL_NAME}-qlora-${CURRENT_TIME}

cat > ${CONFIG_FILE} << EOF
base_model: /workspace/home/vinhnq29/zac2023-main/models_hub/${MODEL_NAME}
model_type: ${MODEL_TYPE}
tokenizer_type: ${TOKENIZER_TYPE}

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: /workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/train_v1/input_output_zephyr-00000-of-00001.parquet
    type: input_output
dataset_prepared_path:
val_set_size: 0.05
eval_sample_packing: false
output_dir: ${OUTPUT_DIR}

sequence_len: ${MAX_SEQ_LEN}
sample_packing: false
pad_to_sequence_len: false

adapter: lora
lora_model_dir:
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:
lora_target_modules:
  - gate_proj
  - down_proj
  - up_proj
  - q_proj
  - v_proj
  - k_proj
  - o_proj

wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
micro_batch_size: ${MICRO_BATCH_SIZE}
num_epochs: ${EPOCHS}
optimizer: adamw_bnb_8bit
lr_scheduler: ${LR_SCHEDULER}
learning_rate: ${LR}

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: false
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention: 
flash_attention: true
s2_attention:

loss_watchdog_threshold: 5.0
loss_watchdog_patience: 3

warmup_steps: 10
evals_per_epoch: 1
eval_table_size:
eval_max_new_tokens: 128
saves_per_epoch: 2
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
EOF

CUDA_VISIBLE_DEVICES=${DEVICE} accelerate launch -m axolotl.cli.train ${CONFIG_FILE}

cp ${CONFIG_FILE} ${OUTPUT_DIR}
