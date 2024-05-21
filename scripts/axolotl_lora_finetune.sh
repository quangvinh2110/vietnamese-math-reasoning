#!/bin/bash#
set -ex

DEVICE=0
CURRENT_TIME=$( date '+%F-%H-%M-%S' )
MODEL_NAME=deepseek-coder-6.7b-instruct
MODEL_TYPE=AutoModelForCausalLM
TOKENIZER_TYPE=AutoTokenizer
FLASH_ATTENTION=true
declare -a DATASETS_PATH=("/workspace/home/vinhnq29/zac2023-main/data_hub/MathIntructCode/input_output_llamacode.jsonl" "/workspace/home/vinhnq29/zac2023-main/data_hub/ViMathQA/train_v1/input_output_vistral-00000-of-00001.parquet")
DATASET_TYPE=input_output
DATASETS=$(printf $"  - path: %s\n    type: ${DATASET_TYPE}\n" "${DATASETS_PATH[@]}")
MAX_SEQ_LEN=4096
ADAPTER=lora
PRETRAINED_ADAPTER_DIR=
LORA_RANK=256
LORA_ALPHA=128
GLOBAL_BATCH_SIZE=16
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$((${GLOBAL_BATCH_SIZE}/${MICRO_BATCH_SIZE}))
EPOCHS=4
OPTIMIZER=adamw_torch
LR_SCHEDULER=cosine
LR=2e-4
CONFIG_FILE=/workspace/home/vinhnq29/zac2023-main/config/${MODEL_NAME}-${ADAPTER}-${CURRENT_TIME}.yml
OUTPUT_DIR=/workspace/home/vinhnq29/zac2023-main/checkpoints/adapters/${MODEL_NAME}-${ADAPTER}-${CURRENT_TIME}

cat > ${CONFIG_FILE} << EOF
base_model: /workspace/home/vinhnq29/zac2023-main/models_hub/${MODEL_NAME}
model_type: ${MODEL_TYPE}
tokenizer_type: ${TOKENIZER_TYPE}

load_in_8bit: false
load_in_4bit: false
strict: false
 
datasets:
${DATASETS}
dataset_prepared_path:
val_set_size: 0.05
eval_sample_packing: false
output_dir: ${OUTPUT_DIR}

sequence_len: ${MAX_SEQ_LEN}
sample_packing: false
pad_to_sequence_len: false

adapter: ${ADAPTER}
lora_model_dir: ${PRETRAINED_ADAPTER_DIR}
lora_r: ${LORA_RANK}
lora_alpha: ${LORA_ALPHA}
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:

wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
micro_batch_size: ${MICRO_BATCH_SIZE}
num_epochs: ${EPOCHS}
optimizer: ${OPTIMIZER}
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
flash_attention: ${FLASH_ATTENTION}
s2_attention:

loss_watchdog_threshold: 5.0
loss_watchdog_patience: 3

warmup_steps: 10
evals_per_epoch: 5
eval_table_size:
eval_max_new_tokens: 128
saves_per_epoch: 2
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"    
EOF

mkdir ${OUTPUT_DIR}
cp ${CONFIG_FILE} ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=${DEVICE} accelerate launch -m axolotl.cli.train ${CONFIG_FILE} --debug
