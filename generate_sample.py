from pointcept.engines.defaults import  default_setup
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
)
from pointcept.engines.train import TRAINERS
import sculpting
import torch
import numpy as np
import sys

config_file = sys.argv[1]

args = default_argument_parser().parse_args(
    f"--config-file {config_file} --num-gpus 0 --options num_workers=0 num_worker_per_gpu=0 batch_size=1".split()
)
cfg = default_config_parser(args.config_file, args.options)

cfg = default_setup(cfg)
cfg.num_worker_per_gpu=0
trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))

train_loader = trainer.train_loader
i, b = next(enumerate(train_loader))

b

import pandas as pd

c = (1+b['feat'])/2
coord=b['coord']

pd.DataFrame(dict(
    x=coord[:,0],
    y=coord[:,1],
    z=coord[:,2],
    red=c[:,0]*255,
    green=c[:,1]*255,
    blue=c[:,2]*255,
    label=b['segment']
)).to_csv("test.csv")