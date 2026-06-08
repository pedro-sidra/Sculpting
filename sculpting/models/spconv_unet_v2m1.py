"""
SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.layers import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from addict import Dict
import torch_scatter


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
        re_serialization=False,
        serialization_order="z",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

        self.re_serialization = re_serialization
        self.serialization_order = serialization_order

    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[gird_coord] or [coord, grid_size] should be include in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        if "origin_coord" in point.keys():
            point_dict["origin_coord"] = torch_scatter.segment_csr(
                point.origin_coord[indices], idx_ptr, reduce="mean"
            )
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride
        if "mask" in point.keys():
            point_dict["mask"] = (
                torch_scatter.segment_csr(
                    point.mask[indices].float(), idx_ptr, reduce="mean"
                )
                > 0.5
            )

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)

        if self.re_serialization:
            point.serialization(
                order=self.serialization_order, shuffle_orders=self.shuffle_orders
            )
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
        mode="cat"
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())

        self.traceable = traceable
        self.mode = mode

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse

        up_feat = self.proj(point).feat[inverse]

        if self.mode == "cat":
            parent.feat = torch.cat([up_feat, parent.feat], dim=1)
        elif self.mode == "add":
            parent.feat = parent.feat + up_feat
        elif self.mode == "replace":
            parent.feat = up_feat

        if hasattr(parent, "sparse_conv_feat") and parent.sparse_conv_feat is not None:
            parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            parent["unpooling_parent"] = point
            parent["unpooling_inverse"] = inverse
        return parent


@MODELS.register_module("SpUNet-v3m1")
class SpUNetBase(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        enc_mode=False,
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.enc_mode = enc_mode

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = PointSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList() if not self.enc_mode else None
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.enc_mode else None

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                GridPooling(
                    in_channels=enc_channels,
                    out_channels=channels[s],
                    stride=2,
                    norm_layer=norm_fn,
                    act_layer=nn.ReLU,
                )
            )
            self.enc.append(
                PointSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            if not self.enc_mode:
                # decode num_stages
                self.up.append(
                    GridUnpooling(
                        in_channels=channels[len(channels) - s - 2],
                        out_channels=dec_channels,
                        norm_layer=norm_fn,
                        act_layer=nn.ReLU,
                        mode="cat"
                    )
                )
                self.dec.append(
                    PointSequential(
                        OrderedDict(
                            [
                                (
                                    f"block{i}",
                                    block(
                                        dec_channels + enc_channels if i == 0 else dec_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                                for i in range(layers[len(channels) - s - 1])
                            ]
                        )
                    )
                )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        final_in_channels = (
            channels[-1] if not self.enc_mode else channels[self.num_stages - 1]
        )
        self.final = PointSequential(
            spconv.SubMConv3d(
                final_in_channels, num_classes, kernel_size=1, padding=1, bias=True
            )
            if num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        point = Point(input_dict)
        if "grid_coord" not in point.keys():
            grid_size = point.get("grid_size", 0.02)
            min_coord = scatter(point.coord, point.batch.long(), dim=0, reduce="min")
            grid_coord = torch.div(point.coord - min_coord[point.batch.long()], grid_size, rounding_mode="floor").int()
            point.grid_coord = grid_coord

        point.sparsify()
        point = self.conv_input(point)
        
        # enc forward
        for s in range(self.num_stages):
            point = self.down[s](point)
            point = self.enc[s](point)
            
        if not self.enc_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                point = self.up[s](point)
                point = self.dec[s](point)

        point = self.final(point)
        
        if self.enc_mode:
            return point
            
        return point.feat


@MODELS.register_module("SpUNetNoSkip-v3m1")
class SpUNetNoSkipBase(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = PointSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                GridPooling(
                    in_channels=enc_channels,
                    out_channels=channels[s],
                    stride=2,
                    norm_layer=norm_fn,
                    act_layer=nn.ReLU,
                )
            )
            self.enc.append(
                PointSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )

            # decode num_stages
            self.up.append(
                GridUnpooling(
                    in_channels=channels[len(channels) - s - 2],
                    out_channels=dec_channels,
                    norm_layer=norm_fn,
                    act_layer=nn.ReLU,
                    mode="replace"
                )
            )
            self.dec.append(
                PointSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                block(
                                    dec_channels,
                                    dec_channels,
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s}",
                                ),
                            )
                            for i in range(layers[len(channels) - s - 1])
                        ]
                    )
                )
            )
            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        self.final = PointSequential(
            spconv.SubMConv3d(
                channels[-1], out_channels, kernel_size=1, padding=1, bias=True
            )
            if out_channels > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data_dict):
        point = Point(data_dict)
        if "grid_coord" not in point.keys():
            grid_size = point.get("grid_size", 0.01)
            min_coord = scatter(point.coord, point.batch.long(), dim=0, reduce="min")
            grid_coord = torch.div(point.coord - min_coord[point.batch.long()], grid_size, rounding_mode="floor").int()
            point.grid_coord = grid_coord

        point.sparsify()
        point = self.conv_input(point)
        
        # enc forward
        for s in range(self.num_stages):
            point = self.down[s](point)
            point = self.enc[s](point)
            
        # dec forward
        for s in reversed(range(self.num_stages)):
            point = self.up[s](point)
            point = self.dec[s](point)

        point = self.final(point)
        return point.feat