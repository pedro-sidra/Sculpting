from pointcept.engines.defaults import  default_setup
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
)
import pandas as pd
from pypcd.pypcd import pandas_to_pypcd, encode_rgb_for_pcl

from pointcept.engines.train import TRAINERS
import sculpting
import torch
import numpy as np
import sys

if __name__=="__main__":
    config_file = sys.argv[1]

    args = default_argument_parser().parse_args(
        f"--config-file {config_file} --num-gpus 0 --options num_workers=0 num_worker_per_gpu=0 batch_size=1".split()
    )
    cfg = default_config_parser(args.config_file, args.options)

    cfg = default_setup(cfg)
    cfg.num_worker_per_gpu=0
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))

    train_loader = trainer.train_loader
    i, input_dict = next(enumerate(train_loader))

    # breakpoint()
    feat_key = 'feat'
    coord_key = 'coord'
    seg_key = 'segment'

    colors = input_dict[feat_key][:,:3]
    c= (colors-colors.min())/(colors.max()-colors.min())
    coord=input_dict[coord_key]

    rgb = encode_rgb_for_pcl((255*c).numpy().astype(np.uint8))

    pc_data = pd.DataFrame(dict(
        x=coord[:,0],
        y=coord[:,1],
        z=coord[:,2],
        rgb=rgb,
        segment=input_dict['segment'].numpy().astype(np.int32),
        **{f'feat_{i}':f for i,f in enumerate(input_dict[feat_key].T)}
    ))

    pandas_to_pypcd(
        pc_data
    ).save_pcd(
        f"sample.pcd",
        compression="binary_compressed"
    )