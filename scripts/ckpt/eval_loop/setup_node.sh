WORK=/leonardo_work/EUHPC_E03_068
python -m venv $WORK/$USER/openeurollm-eval
source $WORK/$USER/openeurollm-eval/bin/activate

# install lm-eval and accelerate
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
pushd lm-evaluation-harness
python -m pip install -e .
python -m pip install accelerate

python -m pip install wandb
python -m pip install sentencepiece
# Download datasets
export HF_HOME=/leonardo_scratch/large/userexternal/$USER/HF_cache
TASKS="commonsense_qa,piqa,winogrande,arc_challenge,arc_easy,mmlu,hellaswag,copa,openbookqa,lambada_openai,winogrande,boolq,mmlu_pro"

lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks $TASKS \
    --output_path $SCRATCH/eval_results/ \
    --use_cache $SCRATCH/eval_cache/ \
    --batch_size $BATCH_SIZE \
    --limit 1 \
    --device cpu