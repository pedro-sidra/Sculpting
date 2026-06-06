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

if __name__=="__main__":
    config_file = sys.argv[1]

    args = default_argument_parser().parse_args(
        f"--config-file {config_file} --num-gpus 0 --options num_workers=0 num_worker_per_gpu=0 batch_size=2".split()
    )
    cfg = default_config_parser(args.config_file, args.options)

    cfg = default_setup(cfg)
    cfg.num_worker_per_gpu=0
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))

    train_loader = trainer.train_loader
    i, b = next(enumerate(train_loader))

    b

    import pandas as pd
    from pypcd.pypcd import pandas_to_pypcd

    # breakpoint()
    feat_key = 'feat'
    coord_key = 'coord'
    seg_key = 'segment'

    c = (1+b[feat_key])/2
    coord=b[coord_key]
    seg_color=np.zeros_like(coord)
    colors={
        0: [1,0,0],
        1: [0,1,0],
        2: [0,0,1],
    }
    for s in np.unique(b[seg_key]):
        seg_color[b[seg_key]==s] = colors.get(s, [0,0,0])

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
        ))
    ).save("sample_rgb.pcd")