#!/bin/bash

# social_iqa =>
# ValueError: The repository for social_i_qa contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/social_i_qa.
# Please pass the argument `trust_remote_code=True` to allow custom code to be run.
TASKS=$1
NUM_FEWSHOT=$2
MODEL_PATH=$3

# avoid issue "sqlite3.OperationalError: database is locked"
export OUTLINES_CACHE_DIR=/tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID/

echo "Evaluating model $3 with $2 few-shots on the following tasks $1."

srun accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True\
    --tasks $TASKS \
    --output_path $SCRATCH/eval_results/ \
    --use_cache $SCRATCH/eval_cache/ \
    --batch_size $BATCH_SIZE \
    --num_fewshot $NUM_FEWSHOT \
    --wandb_args project=lm-eval-harness-integration,name=$SLURM_JOB_ID/$MODEL_PATH,tag=$SLURM_JOB_ID