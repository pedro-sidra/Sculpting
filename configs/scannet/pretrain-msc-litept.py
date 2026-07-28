_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 32  # bs: total bs in all gpus
num_worker = 32
mix_prob=0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = False

# model settings
model = dict(
    type="MSC-v1m1",
    backbone=dict(
        type="LitePT-v1",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    backbone_in_channels=6,
    backbone_out_channels=72,
    mask_grid_size=0.1,
    mask_rate=0.4,
    view1_mix_prob=0.8,
    view2_mix_prob=0,
    matching_max_k=8,
    matching_max_radius=0.03,
    matching_max_pair=8192,
    nce_t=0.4,
    contrast_weight=1,
    reconstruct_weight=1,
    reconstruct_color=True,
    reconstruct_normal=True,
)

epoch = 200
base_lr = 0.004
lr_decay = 0.9  # layer-wise lr decay

base_wd = 0.04  # wd scheduler enable in hooks
final_wd = 0.2  # wd scheduler enable in hooks

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    train=dict(
        type=dataset_type,
        split=["train",
                # "val",
                #   "test"
                  ],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(type="SphereCrop", point_max=150_000, mode="random"),
            dict(
                type="ContrastiveViewsGenerator",
                view_keys=("coord", "color", "normal", "origin_coord"),
                view_trans_cfg=[
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=1),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=1),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(
                        type="RandomColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.02,
                        p=0.8,
                    ),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    dict(type="Update",
                         keys_dict={ "index_valid_keys": [ "origin_coord", "coord", "color", "normal" ] }
                    ),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.6, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                ],
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "view1_origin_coord",
                    "view1_grid_coord",
                    "view1_coord",
                    "view1_color",
                    "view1_normal",
                    "view2_origin_coord",
                    "view2_grid_coord",
                    "view2_coord",
                    "view2_color",
                    "view2_normal",
                ),
                offset_keys_dict=dict(
                    view1_offset="view1_coord", view2_offset="view2_coord"
                ),
                view1_feat_keys=("view1_color", "view1_normal"),
                view2_feat_keys=("view2_color", "view2_normal"),
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaverWandb", save_freq=None),
]
