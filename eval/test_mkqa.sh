#!/bin/bash
BASE_DIR=$(dirname "$PWD")
TRAIL_NAME=$1
CKPT_NAME=$2
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2048}

LANGS_ARGS=${LANGS_ARGS:-" --eval_all_langs "}

if [ "$CKPT_NAME" == "" ]; then
  MODEL_PATH=$BASE_DIR/results/$TRAIL_NAME
  OUTPUT_PATH=$BASE_DIR/outputs/${TRAIL_NAME}/mteb
  LOG_DIR=$BASE_DIR/logs/${TRAIL_NAME}/mteb
else
  MODEL_PATH=$BASE_DIR/results/$TRAIL_NAME/$CKPT_NAME
  OUTPUT_PATH=$BASE_DIR/outputs/${TRAIL_NAME}_${CKPT_NAME}/mteb
  LOG_DIR=$BASE_DIR/logs/${TRAIL_NAME}_${CKPT_NAME}/mteb
fi

mkdir -p $OUTPUT_PATH
mkdir -p $LOG_DIR

# Distributed Command
CMD="torchrun "
CMD+=" --nnodes ${NNODES:-1} "
CMD+=" --nproc_per_node ${NPROC_PER_NODE:-8} "
if [ $NNODES -gt 1 ]; then
  CMD+=" --node_rank ${NODE_RANK:-0} "
  CMD+=" --master_addr ${MASTER_ADDR:-127.0.0.1} "
  CMD+=" --master_port ${MASTER_PORT:-1234} "
fi

set -x

# Test BEIR
$CMD evaluate_mkqa.py \
  --model_name_or_path $MODEL_PATH \
  $MODEL_KWARGS \
  $LANGS_ARGS \
  --output_dir $OUTPUT_PATH \
  --batch_size $EVAL_BATCH_SIZE \
  |& tee -a $LOG_DIR/mkqa.log

