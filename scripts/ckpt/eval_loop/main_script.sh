#!/bin/bash

TASKS=$1
NUM_FEWSHOT=$2
MODEL_PATH=$3

# avoid issue "sqlite3.OperationalError: database is locked"
export OUTLINES_CACHE_DIR=/tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID/$MODEL_PATH

echo "Evaluating model $3 with $2 few-shots on the following tasks $1."
mkdir -p LM_EVAL_OUTPUT_PATH

srun accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True\
    --tasks $TASKS \
    --output_path $LM_EVAL_OUTPUT_PATH/$SLURM_JOB_ID/$SLURM_ARRAY_JOB_ID/eval_results/ \
    --use_cache $LM_EVAL_OUTPUT_PATH/$SLURM_JOB_ID/$SLURM_ARRAY_JOB_ID/eval_cache/ \
    --batch_size $BATCH_SIZE \
    --num_fewshot $NUM_FEWSHOT \
    --trust_remote_code \
    --wandb_args project=lm-eval-harness-integration,name="$SLURM_ARRAY_JOB_ID-$MODEL_PATH"