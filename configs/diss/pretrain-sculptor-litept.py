_base_ = ["../_base_/default_runtime.py"]

epoch = 100
# misc custom setting
batch_size = 32  # bs: total bs in all gpus
num_worker = 32
mix_prob = 0.0
clip_grad = 1.0
empty_cache = False
enable_amp = False
find_unused_parameters = False
evaluate=False

hooks = [
    dict(type="CheckpointLoaderAllowMismatch"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # dict(type="AdditiveMaskSizeScheduler",
    #     mask_size_start=0.1,
    #     mask_size_base=0.4,
    #     mask_size_end=0.4,
    #     mask_size_warmup_ratio=0.25,
    #     mask_ratio_start=0.3,
    #     mask_ratio_base=1.0,
    #     mask_ratio_end=1.5,
    #     mask_ratio_warmup_ratio=0.5,
    # ),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaverWandb", save_freq=5),
    dict(type="MaskBalanceLoggingHook"),
    # dict(type="PreciseEvaluator", test_last=False),
]

optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# Sculpting params
sculpting_transform = dict(
    type="AdditiveMasking",
    mode="trimming",  # "Sculpting" or "Trimming"
    sampling="chessboard",  # "chessboard" or "random" or "random_rotate"
    density_factor="1",
    mask_size_min=0.1,
    mask_size_max=0.4,
    cell_size=0.02,
    mask_feature_mode_1="null",
    mask_feature_mode_2=None,
    mask_dictname="segment",
)

voxelize_transform = dict(
    type="VoxelizeAggJIT",
    grid_size=0.02,
    hash_type="fnv",
    mode="train",
    return_grid_coord=True,
    follow_ref='segment',
    how_to_agg_feats=dict(
        coord="follow-1",
        color="follow-2",
        # mask="max",
        segment="max",
        normal="follow-2",
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
            "segment",
        ]
    },
)

tta_identity = [
    [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
]

sculpting_data_base_configs = dict(
    num_classes=3,
    ignore_index=0,
    names=[
        "original",
        "occluded",
        "masked",
    ],
)

model = dict(
    type="Sculptor-v1m1",
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
        enc_mode=False,
    ),
    #criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=0)],
    head_in_channels=72,
    sculpt_loss_weight=0.5,
    reconstruct_loss_weight=0.5,
    sculpt_original_point_weight=0,
    sculpt_block_point_weight=1,
    sculpt_mask_point_weight=1,
)

# dataset settings
# dataset_type = "ScanNetDataset"
# data_root = "data/scannet"

transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(
            #    type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            # ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.2),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.2),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="SphereCrop", point_max=150000, mode="random"),
            update_index_keys,
            sculpting_transform,
            voxelize_transform,
            dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment","mask_balance"),
                feat_keys=("coord","color","normal"),
            ),
        ]
data = dict(
    **sculpting_data_base_configs,
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
