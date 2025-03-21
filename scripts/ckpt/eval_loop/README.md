# Evaluation loop

This script launches evaluations on a list of datasets and checkpoints with a slurm job array.

## Setup

### Node setup

You should first make sure you have a valid python environments and datasets downloaded as internet is not available
on Slurm nodes.

You can do the following:
```
ssh leonardo
git clone https://github.com/geoalgo/Megatron-LM-Open-Sci.git
git checkout eval_loop
bash scripts/ckpt/eval_loop/setup_node.sh
```

which
1) creates an environment at /leonardo_work/EUHPC_E03_068/$USER/openeurollm-eval
2) install dependencies
3) download datasets in /leonardo_scratch/large/userexternal/$USER/HF_cache
 

### Launching evaluations

You can now launch the experiments, first install slurmpilot and then call:
```
pip install "slurmpilot[extra] @ git+https://github.com/geoalgo/slurmpilot.git"
cd scripts/ckpt/eval_loop/
python launch_eval.py
```

which will launch all evaluations.

Results will be logged in wandb but you will have to sync them as nodes are cut from internet.
Once results are in WANDB, you can do the following to get the table of all results (you should update the list of 
jobids manually).

```
cd scripts/ckpt/eval_loop/
python print_results.py
```

If you want you can also show the results for a certain group of Slurm jobs, 
`python print_results --jobids 14140172 14141165 14147553 14170824`.