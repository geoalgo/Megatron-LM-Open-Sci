import os
from pathlib import Path
from slurmpilot import JobCreationInfo, SlurmPilot, unify

jobname = "openeurollm/eval/iteration"

# List of tasks to evaluate
n_fewshot_to_tasks = {
    0: ["copa,openbookqa,lambada_openai,winogrande,social_iqa"],
    5: ["mmlu", "mmlu_pro"],
    10: ["commonsense_qa", "piqa", "arc_challenge", "arc_easy", "hellaswag", "boolq"],
}

models_file = "models.txt"
with open(models_file, "r") as f:
    model_paths = f.readlines()

# List of models to evaluate
# model_paths = [
#     # "/leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf/open-sci-ref_model-1.3b_data-HPLT-2.0_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13686118",
#     # "/leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf/open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13977373",
#     # "/leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf/open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13715533",
#     # "/leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf/open-sci-ref_model-1.3b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13661750",
#     # "/leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf/open-sci-ref_model-1.7b_data-HPLT-2.0_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO",
#     # "/leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf/open-sci-ref_model-1.7b_data-HPLT-2.0_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13686312",
#     # "/leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf/open-sci-ref_model-1.3b_data-HPLT-2.0_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13686118",
#     # "/leonardo_work/EUHPC_E03_068/tcarsten/converted_checkpoints/hf/open-sci-ref_model-1.7b_data-HPLT-2.0_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13624630",
#     # "/leonardo_work/EUHPC_E03_068/tcarsten/converted_checkpoints/hf/open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13977373",
#     # "HuggingFaceFW/ablation-model-c4",
#     # "Qwen/Qwen2.5-1.5B",
#     # "HuggingFaceFW/ablation-model-fineweb-edu",
#     # "HuggingFaceTB/SmolLM-1.7B",
#     # "HuggingFaceTB/SmolLM2-1.7B",
# ]


# remove "\n" at the end of each string
model_paths = [x.strip() for x in model_paths]

# loop over all models first, then all tasks
python_args = [
    f"{','.join(tasks)} {n_fewshot} {model_path}"
    for model_path in model_paths
    for n_fewshot, tasks in n_fewshot_to_tasks.items()
]


# we set things here that depends on $USER which is known at runtime as opposed to other env vars
bash_setup_command = """
source /leonardo_work/EUHPC_E03_068/$USER/openeurollm-eval/bin/activate
export HF_HOME=/leonardo_scratch/large/userexternal/$USER/HF_cache
export LM_EVAL_OUTPUT_PATH="/leonardo_scratch/large/userexternal/$USER"
"""

#python_args = python_args[:4]

# scheduled [0,500] => 14709396
# scheduled [500,800] => 14709409
# scheduled [800,1100] => TODO
print(f"{len(python_args)} jobs.")


# 14798246

job = JobCreationInfo(
    cluster="leonardo",
    partition="boost_usr_prod",
    jobname=unify(jobname),
    account="EUHPC_E03_068",
    entrypoint="main_script.sh",
    src_dir=str(Path(__file__).parent),
    python_binary="bash",
    python_args=python_args,
    bash_setup_command=bash_setup_command,
    n_gpus=1,
    n_concurrent_jobs=min(len(python_args), 32),
    max_runtime_minutes=24 * 60 - 1,
    env={
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
        "BATCH_SIZE": "auto:4",
    },
    # sbatch_arguments="--dependency=afterok:14709409",
)
api = SlurmPilot(clusters=["leonardo"])
api.schedule_job(job_info=job)