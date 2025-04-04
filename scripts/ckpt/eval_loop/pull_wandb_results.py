import argparse
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import wandb


def main(jobids: list[str]):

    api = wandb.Api()
    filter_queries = [{"display_name": {"$regex": s}} for s in jobids]
    filter_query = {"$or": filter_queries}
    runs = api.runs(
        path="geoalgo-university-of-freiburg/lm-eval-harness-integration",
        filters=filter_query,
    )

    selected_runs = []
    rows = []
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
            row = {"model_path": model}
            row.update(run.summary)
            rows.append(row)
    print(f"Found {len(rows)} jobs.")
    df = pd.DataFrame(rows).set_index("model_path")
    df.to_csv(f'results{"-".join(jobids)}.csv', index=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Pull results from wandb",
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