from pointcept.engines.train import Trainer, TRAINERS
from pointcept.utils import comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.engines.defaults import worker_init_fn
from functools import partial
import torch
from pointcept.engines.hooks import HOOKS, HookBase

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