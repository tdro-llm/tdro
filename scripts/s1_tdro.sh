#!/bin/bash
#
# Stage 1: tDRO Optimization
#
# @Author  :   Ma (Ma787639046@outlook.com)
#
BASE_DIR=$(dirname "$PWD")      # Get folder path of `tdro`

DOMAIN_CONFIG_PATH=$BASE_DIR/config/uniform_sampling_s0.json    # Path to Dataset config, here we use uniform sampling
TRAIL_NAME=${0%.*}      # Use filename as model's output dir name

# Multi-GPU Settings
NNODES=1                # Num of Nodes
RANK=0                  # Node rank
NPROC_PER_NODE=8        # Num of GPU per node
MASTER_ADDR=127.0.0.1   # Master addr
MASTER_PORT=1234        # Master Port

# Batch Size, LR, Total Steps, Num of Passages
TOTAL_BATCH_SIZE=$((2048))      # Global batch size
GRAD_ACCU_STEP=4                # Gradient Accumulation Steps
REAL_BATCH_SIZE_PER_GPU=$(($TOTAL_BATCH_SIZE/$GRAD_ACCU_STEP/$NPROC_PER_NODE/$NNODES))  # Batch size per GPU
LR=1e-4                 # Learning rate
TRAIN_N_PASSAGES=8      # How many passages corespoding to one query. The num of negatives will be `TRAIN_N_PASSAGES-1`
MAX_STEPS=1000          # Max steps
SAVE_STEPS=20000        # Intervals to save. Set larger than `MAX_STEPS` will not save intermediate
SAVE_TOTAL_LIMIT=10     # Max limits of saved intermediates

# Model path, Save path, Log path
MODEL_PATH=Qwen/Qwen1.5-0.5B                # Where to load the model
OUTPUT_DIR=$BASE_DIR/results/$TRAIL_NAME    # Where to save the model
LOG_DIR=$BASE_DIR/logs/$TRAIL_NAME/dpr      # where to dump the logs
mkdir -p $LOG_DIR

# Global Model Arguments
MODEL_KWARGS=""
MODEL_KWARGS+=" --model_type EncoderModel "     # Only support EncoderModel for now
MODEL_KWARGS+=" --pooling_strategy lasttoken "  # Last token (</eos>) pooling. Make sure tokenizer appends a </eos> token
MODEL_KWARGS+=" --score_function cos_sim "      # Cosine similarity
MODEL_KWARGS+=" --q_max_len 128 "               # Query max length
MODEL_KWARGS+=" --p_max_len 512 "               # Passage max length
MODEL_KWARGS+=" --bf16 "                        # Bfloat16 training / inferencing (Mix-precision w/ auto-cast)
MODEL_KWARGS+=" --add_prompt "                  # Whether to add prompt in front of the queries
MODEL_KWARGS+=" --prompt_type e5 "              # Here we follow the prompt settings of Mistral-E5

##########################
# Common Fine-tuning Args
##########################
# Distributed Command
CMD="accelerate launch "
CMD+=" --num_machines ${NNODES} "
CMD+=" --machine_rank ${RANK} "
CMD+=" --num_processes $((NNODES*NPROC_PER_NODE)) "
if [ $NNODES -gt 1 ]; then
  CMD+=" --main_process_ip ${MASTER_ADDR} "
  CMD+=" --main_process_port ${MASTER_PORT} "
fi

# ** DDP / DS / FSDP: Choose based on your GPU MEM **
CMD+=" --config_file $BASE_DIR/config/ddp.yaml "
# CMD+=" --config_file $BASE_DIR/config/ds_stage0.yaml "
# CMD+=" --config_file $BASE_DIR/config/ds_stage1.yaml "
# CMD+=" --config_file $BASE_DIR/config/ds_stage2.yaml "
# CMD+=" --config_file $BASE_DIR/config/ds_stage3_w_config.yaml "
# CMD+=" --config_file $BASE_DIR/config/fsdp_shard_grad_op.yaml "
# CMD+=" --config_file $BASE_DIR/config/fsdp_hybrid_shard.yaml "
# CMD+=" --config_file $BASE_DIR/config/fsdp_full_shard.yaml "

CMD+=" -m tdro.dro.fit "     # Entry for tDRO

# Data Arguments
DATA_ARGS=""
DATA_ARGS+=" --domain_config_path $DOMAIN_CONFIG_PATH "             # Path to Dataset config
DATA_ARGS+=" --preprocessed_dir $BASE_DIR/data/retrieval/dedup "    # Folder Path to all jsonl datasets
# DATA_ARGS+=" --homogenous_batch "       # Do NOT use homogenous batching. tDRO needs to compare different domains in one batch

# Training Arguments
TRAIN_ARGS=""
TRAIN_ARGS+=" --do_train "
TRAIN_ARGS+=" --save_steps $SAVE_STEPS "                # Intervals to save.
TRAIN_ARGS+=" --save_total_limit $SAVE_TOTAL_LIMIT "    # Max limits of saved intermediates
TRAIN_ARGS+=" --logging_steps 2 "                       # Intervals to log
TRAIN_ARGS+=" --warmup_steps 100 "                      # Warmup steps
TRAIN_ARGS+=" --per_device_train_batch_size $REAL_BATCH_SIZE_PER_GPU "  # Batch size per GPU
TRAIN_ARGS+=" --gradient_accumulation_steps $GRAD_ACCU_STEP "
TRAIN_ARGS+=" --learning_rate $LR "                     # Learning Rate
TRAIN_ARGS+=" --min_lr_ratio 0.1 "                      # Min Learning Rate Ratio
TRAIN_ARGS+=" --lr_scheduler_type cosine "              # Cosine Learning Rate Scheduler
TRAIN_ARGS+=" --max_steps $MAX_STEPS "                  # Max steps
TRAIN_ARGS+=" --temperature 0.002 "                     # Contrastive Learning Temperature
TRAIN_ARGS+=" --train_n_passages $TRAIN_N_PASSAGES "    # How many passages corespoding to one query.
# TRAIN_ARGS+=" --negatives_x_device "                    # Do NOT use Cross-batch negatives
TRAIN_ARGS+=" --seed 42 "                               # Seed
TRAIN_ARGS+=" --dataloader_num_workers 4 "              # Num of processes for PyTorch Dataloader
TRAIN_ARGS+=" --optim adamw_torch_fused "               # Fused AdamW Optimizer
TRAIN_ARGS+=" --weight_decay 0.1 "                      # Weight decay for AdamW
TRAIN_ARGS+=" --gradient_checkpointing "                # Activation checkpointing (Crucial for reducing GPU memory)

# ** GroupDRO **
TRAIN_ARGS+=" --dro_type DROModelv2 "
TRAIN_ARGS+=" --ref_model_name_or_path $BASE_DIR/results/s0_train_baseline_model "
TRAIN_ARGS+=" --reweight_eta 2e-2 "    # Learning rate for group weights
TRAIN_ARGS+=" --normalize_weights_on_every_update "    # Whether to use softmax to normalize the log-train_domain_weights (log-alpha) on every update steps.

# Core Argorithm
TRAIN_ARGS+=" --normalize_group_loss_scale_with_ref_loss "  # lm_loss / ref_loss
TRAIN_ARGS+=" --dro_only_hn " # Only use hard negatives for loss computation with DRO Optimization.

# => DROModelv2 Optim Args
## Optimizer
TRAIN_ARGS+=" --dro_optimizer sgd "
## Grad Norm
TRAIN_ARGS+=" --dro_apply_grad_norm "
TRAIN_ARGS+=" --dro_max_grad_norm 1.0 "
# ## Scheduler
TRAIN_ARGS+=" --dro_lr_scheduler_type constant "    # linear, cosine, constant, constant_with_warmup, ...
TRAIN_ARGS+=" --dro_warmup_ratio 0.1 "
TRAIN_ARGS+=" --dro_min_lr_ratio 0.1 "

set -ex
##########################
# tDRO Optimize
##########################
$CMD \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    $DATA_ARGS \
    $TRAIN_ARGS \
    $MODEL_KWARGS \
    --report_to tensorboard \
    --run_name ${TRAIL_NAME} \
    |& tee $LOG_DIR/finetune_rank${RANK}-${NNODES}.log
