from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import wandb


regenerate = True
if regenerate:
    api = wandb.Api()
    jobids = [14080774]
    runs = api.runs(
        path="geoalgo-university-of-freiburg/lm-eval-harness-integration",
        #filters={"tags": {"$in": [f"JOB-{jobid}"]}}
    )

    selected_runs = []
    model_evals = defaultdict(dict)
    for run in tqdm(runs):
        for jobid in jobids:
            if str(jobid) in run.name:
                selected_runs.append(run)
                splits = run.name.split("-")
                slurmjobid, model = splits[0], "-".join(splits[1:])
                model_evals[model].update(run.summary)
    df = pd.DataFrame(model_evals).T
    df.index.name = "checkpoint"
    df.to_csv("results.csv", index=True)
    print(len(selected_runs))


df = pd.read_csv("results.csv", index_col="checkpoint")

cols = [
    "mmlu/acc",
    "mmlu_pro/exact_match,custom-extract",
    "copa/acc",
    "lambada_openai/acc",
    "openbookqa/acc",
    "winogrande/acc",
    "arc_challenge/acc",
    "boolq/acc",
    "commonsense_qa/acc",
    # "hellaswag/acc",
    "hellaswag/acc_norm",
    # "piqa/acc",
    "piqa/acc_norm",
]

#df.columns = [col.split("/")[0] for col in df.columns]
# whacky
#cols = [col.split("/")[0] for col in cols]

# keep only numerics
df = df.select_dtypes(include=['number'])

#df.index = [x.replace("leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf/open-sci-ref_model-", "") for x in df.index]
print(df[cols].to_string(float_format='%.2f'))

df[cols].to_csv("results-filtered.csv", index=True)
