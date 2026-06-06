from pointcept.engines.defaults import  default_setup
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
from pypcd.pypcd import pandas_to_pypcd
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

run_id = sys.argv[1]
config_file = save_run_config(api, WANDB_ENTITY,WANDB_PROJECT, run_id)
run_dir=f"./run_data/{run_id}"
# weight = sys.argv[2]

# 3. Download weights temporarily and run the script
with download_temp_weights(api, WANDB_ENTITY, WANDB_PROJECT, run_id) as weight_path:
    run_dir = os.path.join(run_dir, run_id)
    print(run_dir, config_file, weight_path)
    

    args = default_argument_parser().parse_args(
        f"--config-file {config_file} --num-gpus 0 --options weight={weight_path} num_workers=0 num_worker_per_gpu=0 batch_size=1 gradient_accumulation_steps=1".split()
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

    # breakpoint()
    feat_key = 'feat'
    coord_key = 'coord'
    seg_key = 'pred'

    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cpu()

    c = (1+input_dict[feat_key])/2
    coord=input_dict[coord_key]
    seg_color=np.zeros_like(coord)
    colors={
        0: [1,0,0],
        1: [0,1,0],
        2: [0,0,1],
    }
    for s in np.unique(input_dict[seg_key]):
        seg_color[input_dict[seg_key]==s] = colors.get(s, [0,0,0])

    pandas_to_pypcd(
        pd.DataFrame(dict(
        x=coord[:,0],
        y=coord[:,1],
        z=coord[:,2],
        red=seg_color[:,0]*255,
        green=seg_color[:,1]*255,
        blue=seg_color[:,2]*255,
        ))
    ).save("sample_seg.pcd")

    pandas_to_pypcd(
        pd.DataFrame(dict(
        x=coord[:,0],
        y=coord[:,1],
        z=coord[:,2],
        red=c[:,0]*255,
        green=c[:,1]*255,
        blue=c[:,2]*255,
        label=input_dict['segment'],
        pred=input_dict['pred']
        ))
    ).save("sample.pcd")