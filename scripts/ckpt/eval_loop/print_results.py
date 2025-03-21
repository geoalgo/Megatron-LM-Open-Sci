import argparse
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import wandb


def main(jobids: list[str]):
    regenerate = True
    if regenerate:
        api = wandb.Api()
        #jobids = [14140172, 14141165, 14147553, 14170824]
        runs = api.runs(
            path="geoalgo-university-of-freiburg/lm-eval-harness-integration",
            # filters={"tags": {"$in": [f"JOB-{jobid}"]}}
        )

        selected_runs = []
        model_evals = defaultdict(dict)
        for run in tqdm(runs):
            add_run = False
            if not jobids:
                add_run = True
            else:
                for jobid in jobids:
                    if str(jobid) in run.name:
                        add_run = True
            if add_run:
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

    # df.columns = [col.split("/")[0] for col in df.columns]
    # whacky
    # cols = [col.split("/")[0] for col in cols]

    # keep only numerics
    df = df.select_dtypes(include=['number'])
    df = df[cols]
    df["Average"] = df.mean(axis=1)

    cols.append("Average")
    # df.index = [x.replace("leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf/open-sci-ref_model-", "") for x in df.index]
    print(df[cols].sort_values(by="Average", ascending=False).to_string(float_format='%.2f'))

    df[cols].to_csv("results-filtered.csv", index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Print results",
        description='Download results from WANDB previously generated with `launch_eval.py` and stores and display them.',
    )
    # TODO list all local jobs and all remote jobs
    parser.add_argument(
        "--jobids",
        nargs='*',
        default=None,
        help='List of Slurm job id to include, default to include all. For instance `--jobids 123 124 125`',
    )
    args = parser.parse_args()
    jobids = args.jobids

    if jobids:
        print(f"Only considering jobs whose ids are {jobids}.")
    main(jobids)