from pointcept.engines.defaults import  default_setup
from pathlib import Path
import os
import yaml
import wandb
import pandas as pd
import tempfile
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from copy import deepcopy

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
)
import pandas as pd
from pypcd.pypcd import pandas_to_pypcd, encode_rgb_for_pcl
from pointcept.engines.train import TRAINERS
import os
import sculpting
from copy import deepcopy
import torch
import numpy as np
import sys
import wandb
from wandb_test import save_run_config, download_temp_weights

WANDB_ENTITY = "pedrosidra"
WANDB_PROJECT = "diss"

api=wandb.Api()

run_ids = sys.argv[1:]
for run_id in run_ids:
    # config_file = save_run_config(api, WANDB_ENTITY,WANDB_PROJECT, run_id)
    config_file="/workspaces/Sculpting/configs/scannet/semseg-spunet-sidra-efficient-lr1.py"
    run_dir=f"./run_data/{run_id}"
    run_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}"
    run = api.run(run_path)
    run_name = run.name
    # weight = sys.argv[2]

    # 3. Download weights temporarily and run the script
    with download_temp_weights(api, WANDB_ENTITY, WANDB_PROJECT, run_id) as weight_path:
        run_dir = os.path.join(run_dir, run_id)
        print(run_dir, config_file, weight_path)
        

        args = default_argument_parser().parse_args(
            f"--config-file {config_file} --num-gpus 1 --options weight={weight_path} num_workers=1 num_worker_per_gpu=1 batch_size=1 gradient_accumulation_steps=1".split()
        )
        cfg = default_config_parser(args.config_file, args.options)
        cfg = default_setup(cfg)
        cfg.num_worker_per_gpu=0
        trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))

        # train_loader = trainer.train_loader
        # i, b = next(enumerate(train_loader))

        trainer.before_train()
        trainer.model.eval()
        for i, input_dict in enumerate(trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            # segment = input_dict["segment"]
            input_dict['pred']=pred
            break

        feat_key = 'feat'
        coord_key = 'coord'
        seg_key = 'pred'

        scene_name = Path(trainer.val_loader.dataset.data_list[i]).stem

        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cpu()

        colors = input_dict[feat_key][:,:3]
        c= (colors-colors.min())/(colors.max()-colors.min())
        coord=input_dict[coord_key]

        rgb = encode_rgb_for_pcl((255*c).numpy().astype(np.uint8))


        pc_data = pd.DataFrame(dict(
            x=coord[:,0],
            y=coord[:,1],
            z=coord[:,2],
            rgb=rgb,
            label=input_dict['segment'].numpy().astype(np.int32),
            pred=input_dict['pred'].numpy().astype(np.int32),
        ))

        pandas_to_pypcd(
            pc_data
        ).save_pcd(
            f"samples/sample_{run_name}_{scene_name}.pcd",
            compression="binary_compressed"
        )

        confusion = pd.crosstab(pc_data['label'], pc_data['pred'])
        confusion.to_csv(
            f"samples/confusion_{run_id}.csv",
        )