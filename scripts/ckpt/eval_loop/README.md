# Evaluation loop

This script launches evaluations on a list of datasets and checkpoints with a slurm job array.

## Setup

### Node setup

You should first make sure you have a valid python environments and datasets downloaded as internet is not available
on Slurm nodes.

You can do the following:
```
ssh leonardo
bash setup_node.sh
```

which
1) creates an environment at /leonardo_work/EUHPC_E03_068/$USER/openeurollm-eval
2) install dependencies
3) download datasets in /leonardo_scratch/large/userexternal/$USER/HF_cache
 

### Launching evaluations

You can now launch the experiments, first install slurmpilot and then call:
```
pip install slurmpilot==0.1.5-dev0
python launch_eval.py
```

which will launch all evaluations.

Results will be logged in wandb but you will have to sync them as nodes are cut from internet.