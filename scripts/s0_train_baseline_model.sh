#!/bin/bash
#
# Stage 0: Fine-tuning a Baseline Retriever (which is also the Reference Model)
# Dataset sampling config: Uniformly sampling (uniform_sampling_s0.json)
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
REAL_BATCH_SIZE_PER_GPU=$(($TOTAL_BATCH_SIZE/$NPROC_PER_NODE/$NNODES))  # Batch size per GPU
LR=1e-4                 # Learning rate
TRAIN_N_PASSAGES=2      # How many passages corespoding to one query. The num of negatives will be `TRAIN_N_PASSAGES-1`
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

CMD+=" -m tdro.finetune.fit "     # Entry for Contrastive Fine-tuning a model

# Data Arguments
DATA_ARGS=""
DATA_ARGS+=" --domain_config_path $DOMAIN_CONFIG_PATH "             # Path to Dataset config
DATA_ARGS+=" --preprocessed_dir $BASE_DIR/data/retrieval/dedup "    # Folder Path to all jsonl datasets
DATA_ARGS+=" --homogenous_batch "       # Yeilds a homogenous batch from one dataset at each iteration

# Training Arguments
TRAIN_ARGS=""
TRAIN_ARGS+=" --do_train "
TRAIN_ARGS+=" --save_steps $SAVE_STEPS "                # Intervals to save.
TRAIN_ARGS+=" --save_total_limit $SAVE_TOTAL_LIMIT "    # Max limits of saved intermediates
TRAIN_ARGS+=" --logging_steps 2 "                       # Intervals to log
TRAIN_ARGS+=" --warmup_steps 100 "                      # Warmup steps
TRAIN_ARGS+=" --per_device_train_batch_size $REAL_BATCH_SIZE_PER_GPU "  # Batch size per GPU
TRAIN_ARGS+=" --learning_rate $LR "                     # Learning Rate
TRAIN_ARGS+=" --min_lr_ratio 0.1 "                      # Min Learning Rate Ratio
TRAIN_ARGS+=" --lr_scheduler_type cosine "              # Cosine Learning Rate Scheduler
TRAIN_ARGS+=" --max_steps $MAX_STEPS "                  # Max steps
TRAIN_ARGS+=" --temperature 0.002 "                     # Contrastive Learning Temperature
TRAIN_ARGS+=" --train_n_passages $TRAIN_N_PASSAGES "    # How many passages corespoding to one query.
TRAIN_ARGS+=" --negatives_x_device "                    # Use Cross-batch negatives
TRAIN_ARGS+=" --seed 42 "                               # Seed
TRAIN_ARGS+=" --dataloader_num_workers 4 "              # Num of processes for PyTorch Dataloader
TRAIN_ARGS+=" --optim adamw_torch_fused "               # Fused AdamW Optimizer
TRAIN_ARGS+=" --weight_decay 0.1 "                      # Weight decay for AdamW
TRAIN_ARGS+=" --gradient_checkpointing "                # Activation checkpointing (Crucial for reducing GPU memory)

# ** GradCache: Activate this if GPU OOM **
# TRAIN_ARGS+=" --grad_cache "                 # Use GradCache to chunking query/passage inputs
# TRAIN_ARGS+=" --no_sync_except_last "        # Only trigger grad sync on the last step of mini-batch forward-backwards
# TRAIN_ARGS+=" --gc_q_chunk_size 64 "         # Chunk size of query for GradCache
# TRAIN_ARGS+=" --gc_p_chunk_size 64 "         # Chunk size of passage for GradCache

set -ex
##########################
# Fine-tuning 
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
