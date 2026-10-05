import wandb

with wandb.init(entity='pedrosidra',project='diss',id="ut10kmjo", resume="must") as run:
    artifact = wandb.Artifact("model_last", "model")
    artifact.add_file("exp/scannet/diss/pretrain-msc-fulldgx/model/model_last.pth")
    run.log_artifact(artifact)