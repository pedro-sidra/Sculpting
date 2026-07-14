"""
Sonata v1m1 Base

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from itertools import chain
from packaging import version
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_

import pointops
from pointcept.models.sonata.sonata_v1m1_base import Sonata, OnlineCluster
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler
from pointcept.datasets.transform import TRANSFORMS

from timm.layers import trunc_normal_

from pointcept.models.modules import PointModel


@MODELS.register_module("Sculptor-v1m1")
class Sculptor(PointModel):
    def __init__(
        self,
        backbone,
        head_in_channels,
        head_hidden_channels=256,
        # Sculpting
        sculpt_loss_weight=1 / 2,
        reconstruct_loss_weight=1 / 2,
        sculpt_original_point_weight=1,
        sculpt_block_point_weight=1,
        sculpt_mask_point_weight=1,
    ):
        super(Sculptor, self).__init__()

        self.backbone = build_model(backbone)

        # Sculpting additions
        self.features_to_reconstruct = backbone["in_channels"]
        self.sculpt_loss_weight = sculpt_loss_weight
        self.reconstruct_loss_weight = reconstruct_loss_weight
        self.mask_token = nn.Parameter(torch.zeros(1, self.features_to_reconstruct))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        self.sculpt_head = nn.Sequential(
            nn.Linear(head_in_channels, head_hidden_channels),
            nn.GELU(),
            # Changed to 3 classes (0=Original, 1=Block, 2=Masked)
            nn.Linear(head_hidden_channels, 3), 
        )

        self.reconstruct_head = nn.Sequential(
            nn.Linear(head_in_channels, head_hidden_channels),
            nn.GELU(),
            nn.Linear(head_hidden_channels, self.features_to_reconstruct),
        )

        # Register weights as a buffer so they automatically move to the correct device
        self.sculpt_loss_fn = nn.CrossEntropyLoss(weight=1.0*torch.tensor([
            sculpt_original_point_weight,
            sculpt_block_point_weight,
            sculpt_mask_point_weight,
        ]))
        self.reconstruct_loss_fn= nn.MSELoss()


    def forward(self, data_dict):

        # prepare global_point, mask_global_point, local_point
        with torch.no_grad():

            # global_point & masking
            feat = data_dict["feat"]
            mask = data_dict["segment"]
            coord = data_dict["coord"]
            grid_coord = data_dict["grid_coord"]
            batch = offset2batch(data_dict["offset"])

            masked_feats = feat.clone()
            masked_feats[mask != 0] = self.mask_token  # zero-out when masked or cube

            mask_global_point = Point(
                feat=masked_feats,
                coord=coord,
                offset=data_dict["offset"],
                mask=mask,  # masked points
                grid_coord=grid_coord
            )

            # create result dictionary for return
            result_dict = dict(loss=[])

        backbone_out = self.backbone(mask_global_point)
        
        # SpUNet returns a raw tensor, while other backbones might return a Point object
        backbone_feat = backbone_out if isinstance(backbone_out, torch.Tensor) else backbone_out.feat

        sculpt_pred = self.sculpt_head(backbone_feat)

        # Compatible with Pointcept DefaultSegmentor for validation evaluation

        reconstruct_pred = self.reconstruct_head(backbone_feat)

        # 1. Sculpt Loss (Standard CrossEntropy over the array weights)
        result_dict["sculpt_loss"] = self.sculpt_loss_fn(sculpt_pred, mask_global_point.mask.long())
        result_dict["loss"].append(result_dict["sculpt_loss"] * self.sculpt_loss_weight)

        # 2. Reconstruct Loss (L2 loss applied ONLY to points with mask == 2)
        mask_2_idx = (mask_global_point.mask == 2)
        if mask_2_idx.any():
            result_dict["reconstruct_loss"] = self.reconstruct_loss_fn(reconstruct_pred[mask_2_idx], feat[mask_2_idx])
        else:
            # Dummy loss to prevent DDP unused parameter errors if no points match
            result_dict["reconstruct_loss"] = reconstruct_pred.sum() * 0.0

        result_dict["loss"].append(result_dict["reconstruct_loss"] * self.reconstruct_loss_weight)
        
        # Ensure mask_token always receives gradients (prevents DDP crash if no points are masked in a batch)
        result_dict["loss"].append(self.mask_token.sum() * 0.0)

        result_dict["loss"] = sum(result_dict["loss"])

        if self.training:
            pass
        else:
            result_dict["seg_logits"] = sculpt_pred


        return result_dict