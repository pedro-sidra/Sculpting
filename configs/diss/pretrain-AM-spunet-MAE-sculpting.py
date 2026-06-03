_base_ = ["pretrain-AM-spunet-base.py"]

hooks = [
    dict(type="CheckpointLoaderAllowMismatch"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="MaskSizeScheduler",
        mask_size_start=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
    ),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaverWandb", save_freq=5),
    dict(type="MaskBalanceLoggingHook"),
    # dict(type="PreciseEvaluator", test_last=False),
]

# Sculpting params
sculpting_transform = dict(
    type="AdditiveMasking",
    mode="sculpting",  # "Sculpting" or "Trimming"
    sampling="chessboard",  # "chessboard" or "random" or "random_rotate"
    density_factor=1.0,
    mask_size_min=0.4,
    mask_size_max=0.4,
    cell_size=0.02,
)

model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=3,
        num_classes=3,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=0)],
)

sculpting_data_base_configs = dict(
    num_classes=3,
    ignore_index=0,
    names=[
        "original",
        "occluded",
        "masked",
    ],
)
voxelize_transform = dict(
    type="VoxelizeAgg",
    grid_size=0.02,
    hash_type="fnv",
    mode="train",
    return_grid_coord=True,
    how_to_agg_feats=dict(
        coord="mean",
        color="mean",
        segment="max",
        normal="first",
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
            "superpoint",
            "strength",
            "segment",
        ]
    },
)

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    **sculpting_data_base_configs,
    train=dict(
        type=dataset_type,
        split=[
            "train",
        ],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.2),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.2),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="SphereCrop", point_max=150000, mode="random"),
            update_index_keys,
            sculpting_transform,
            voxelize_transform,
            dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment","mask_balance"),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split=[
            "val",
            # "arkit",
        ],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            update_index_keys,
            sculpting_transform,
            voxelize_transform,
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment","mask_balance"),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
    ),
)
