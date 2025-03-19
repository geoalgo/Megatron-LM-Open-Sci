import wandb

api = wandb.Api()
runs = api.runs(
    path="geoalgo-university-of-freiburg/lm-eval-harness-integration",
    filters={"tags": {"$in": [tag]}}
)
