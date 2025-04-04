#!/bin/bash

TASKS=$1
NUM_FEWSHOT=$2
MODEL_ROOT=$3

# avoid issue "sqlite3.OperationalError: database is locked"
export OUTLINES_CACHE_DIR=/tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID/$MODEL_ROOT

echo "Evaluating models in folder $3 with $2 few-shots on the following tasks $1."
mkdir -p $LM_EVAL_OUTPUT_PATH

# gets all files "model.safetensors" defined in the current path and launch evaluation for each of them
mapfile -t SAFETENSOR_FILES < <(find "$MODEL_ROOT" -type f -name "model.safetensors")

# Check if any files were found
if [ ${#SAFETENSOR_FILES[@]} -eq 0 ]; then
    echo "No model.safetensors files found in $MODEL_ROOT."
    exit 0
fi

echo "Found ${#SAFETENSOR_FILES[@]} model.safetensors files:"

# TODO trust_remote_code=True does not work properly right now

# Loop through and print each file path
for FILE_PATH in "${SAFETENSOR_FILES[@]}"; do
  MODEL_PATH=$(dirname "$FILE_PATH")
  echo "Evaluate $MODEL_PATH"
  accelerate launch -m lm_eval --model hf \
      --model_args pretrained=$MODEL_PATH,trust_remote_code=True\
      --tasks $TASKS \
      --output_path $LM_EVAL_OUTPUT_PATH/$SLURM_ARRAY_JOB_ID/$SLURM_JOB_ID/eval_results/ \
      --use_cache $LM_EVAL_OUTPUT_PATH/$SLURM_ARRAY_JOB_ID/$SLURM_JOB_ID/eval_cache/ \
      --batch_size $BATCH_SIZE \
      --num_fewshot $NUM_FEWSHOT \
      --trust_remote_code \
      --wandb_args project=lm-eval-harness-integration,name="$SLURM_ARRAY_JOB_ID-$MODEL_PATH"
done


