_base_ = ["../_base_/default_runtime.py"]

# wandb_off = 1
enable_wandb = False

# Sculpting params
sculpting_transform = dict(
    type="AdditiveMasking",
    mode="sculpting",  # "Sculpting" or "Trimming"
    sampling="chessboard",  # "chessboard" or "random" or "random_rotate"
    density_factor='rand',
    mask_size_min=0.4,
    mask_size_max=0.4,
    cell_size=0.02,
    mask_dictname="mask"
)

voxelize_transform = dict(
    type="VoxelizeAggJIT",
    grid_size=0.02,
    hash_type="fnv",
    mode="train",
    return_grid_coord=True,
    follow_ref='mask',
    how_to_agg_feats=dict(
        coord="mean",
        color="mean",
        # mask="max",
        segment="max",
        normal="mean",
    ),
)

update_index_keys = dict(
    type="Update",
    keys_dict={
        "index_valid_keys": [
            "coord",
            "grid_coord",
            "color",
            "normal",
            "mask",
        ]
    },
)

## ===== MODEL DEFINITION

# misc custom setting
batch_size = 98  # bs: total bs in all gpus
num_worker = 98
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = False
evaluate = False
find_unused_parameters = False


# model settings
model = dict(
    type="SonataSculptor-v1m1",
    # backbone - student & teacher
    backbone=dict(
        type="LitePT-v1",
        in_channels=9,
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
        enc_mode=True,
    ),
    teacher_custom=dict(),
    head_in_channels=900,
    head_hidden_channels=256,
    head_embed_channels=256,
    head_num_prototypes=1024,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.4,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.1,
    mask_ratio_base=0.4,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 12,
    roll_mask_loss_weight=2 / 12,
    unmask_loss_weight=4 / 12,
    # sculpt_loss_weight=4 / 12,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
)

# scheduler settings
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

transform = [
    update_index_keys,
    sculpting_transform,
    voxelize_transform,
    # dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord","normal", "color", "mask"),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="NormalizeColor"),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="NormalizeColor"),
        ],
        max_size=65536,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.02}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_normal",
            "global_color",
            "global_offset",
            "global_mask",
            "local_origin_coord",
            "local_coord",
            "local_color",
            "local_normal",
            "local_offset",
            "local_mask",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord","global_color","global_normal"),
        local_feat_keys=("local_coord","local_color","local_normal"),
    ),
]

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            # ScanNet
            dict(
                type="ScanNetDataset",
                split=["train", "val", "test"],
                data_root="data/scannet",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            # ScanNet++
            dict(
                type="ScanNetPPDataset",
                split=[
                    "train_grid1mm_chunk6x6_stride3x3",
                    "val_grid1mm_chunk6x6_stride3x3",
                    "test_grid1mm_chunk6x6_stride3x3",
                ],
                data_root="data/scannetpp",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            # S3DIS
            dict(
                type="S3DISDataset",
                split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
                data_root="data/s3dis",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            # ArkitScenes
            dict(
                type="DefaultDataset",
                split=["Training", "Validation"],
                data_root="data/arkitscenes",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            # HM3D
            dict(
                type="HM3DDataset",
                split=["train", "val"],
                data_root="data/hm3d",
                transform=transform,
                test_mode=False,
                force_label=False,
                loop=1,
            ),
            # Structured3D
            dict(
                type="Structured3DDataset",
                split=["train", "val", "test"],
                data_root="data/structured3d",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
        ],
    ),
)

hooks = [
    dict(type="CheckpointLoaderAllowMismatch", strict=False),
    # dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="AdditiveMaskSizeScheduler",
        mask_size_start=0.1,
        mask_size_base=0.4,
        mask_size_end=0.4,
        mask_size_warmup_ratio=0.25,
        mask_ratio_start=0.8,
        mask_ratio_base=1.0,
        mask_ratio_end=1.5,
        mask_ratio_warmup_ratio=0.5,
    ),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaverWandb", save_freq=5),
    dict(type="MaskBalanceLoggingHook"),
    # dict(type="PreciseEvaluator", test_last=False),
]