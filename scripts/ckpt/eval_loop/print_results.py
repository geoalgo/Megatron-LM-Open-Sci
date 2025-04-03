import argparse
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import wandb


def main(jobids: list[str]):
    regenerate = False
    if regenerate:
        api = wandb.Api()
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

        # add iteration column, detect it from checkpoint name
        def extract_iteration_from_checkpoint(checkpoint_name: str):
            from pathlib import Path
            # /.../iter_0106000 => 106000
            name = Path(checkpoint_name).stem
            if name.startswith("iter_"):
                number = name[len("iter_"):]
                try:
                    return int(str(number).lstrip('0'))
                except ValueError:
                    return None
            else:
                return None
        df["iteration"] = [extract_iteration_from_checkpoint(checkpoint_name) for checkpoint_name in df.index]

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
        "iteration",
    ]

    # keep only numerics
    df = df.select_dtypes(include=['number'])
    df = df[cols]
    df["Average"] = df.mean(axis=1)
    index = df["Average"].dropna().index

    cols.append("Average")
    print(df.loc[index, cols].sort_values(by="Average", ascending=False).to_string(float_format='%.2f'))

    df.loc[index, cols].to_csv("results-filtered.csv", index=True)


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