from pointcept.engines.train import Trainer, TRAINERS
import json
from pointcept.engines.test import SemSegTester, TESTERS
from pointcept.utils import comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.engines.defaults import worker_init_fn
from functools import partial
import torch
from pointcept.engines.hooks import HOOKS, HookBase

import os
import time
import numpy as np
from collections import OrderedDict
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from pointcept.engines.defaults import create_ddp_model
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)


def dict_to_cuda(input_dict):
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)
        
import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

@TRAINERS.register_module()
class SidraTrainer(Trainer):
    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=self.cfg.pin_memory,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=self.cfg.num_worker_per_gpu > 0,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                persistent_workers=self.cfg.num_worker_per_gpu > 0,
                pin_memory=self.cfg.pin_memory,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

@HOOKS.register_module()
class MaskBalanceLoggingHook(HookBase):
    """
    Hook to log the average mask_balance (ratio of added points to original points)
    for each batch to TensorBoard.
    """
    def after_step(self):
        trainer=self.trainer
        # Pointcept typically stores the current batch in trainer.data_dict (or trainer.input_dict in some forks)
        batch_data = self.trainer.comm_info["input_dict"]
        
        if batch_data is not None and "mask_balance" in batch_data:
            mask_balance = batch_data["mask_balance"]
            
            # Calculate the average balance for the current batch
            if isinstance(mask_balance, torch.Tensor):
                avg_balance = mask_balance.float().mean().item()
            elif isinstance(mask_balance, (list, tuple)):
                avg_balance = float(sum(mask_balance)) / len(mask_balance)
            else:
                avg_balance = float(mask_balance)
            
            # Log using Pointcept's EventStorage system (which automatically handles TensorBoard and smoothing)
            if hasattr(trainer, "storage"):
                trainer.storage.put_scalar("train/mask_balance", avg_balance)
            # Fallback if a direct writer is attached instead
            if hasattr(trainer, "writer") and trainer.writer is not None:
                step = getattr(trainer, "global_step", getattr(trainer, "step", 0))
                trainer.writer.add_scalar("train/mask_balance", avg_balance, step)

@TESTERS.register_module()
class SidraTester(SemSegTester):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = self.cfg.save_path
        make_dirs(save_path)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad():
                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        batch_idx=i,
                        batch_num=len(fragment_list),
                    )
                )
            if self.cfg.data.test.type == "ScanNetPPDataset":
                pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
            else:
                pred = pred.max(1)[1].data.cpu().numpy()
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            metrics = {}
            metrics['mIoU']=mIoU
            metrics['mAcc']=mAcc
            metrics['allAcc']=allAcc
            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                idx=i
                name=self.cfg.data.names[i]
                metrics[f"mIoU_{idx}_{name}"]=iou_class[i]
                metrics[f"Acc_{idx}_{name}"]=accuracy_class[i]
            with open(f"{save_path}/results.json","w") as f:
                json.dump(metrics, f)
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")