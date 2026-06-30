import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from .sculpting_ops import  get_pointgrid
from .trimming import get_perlin

from pointcept.engines.hooks import HOOKS, HookBase
from pointcept.utils.scheduler import CosineScheduler
from pointcept.datasets.transform import TRANSFORMS

def get_sculpting_block(num_cells, cell_size):
    return get_pointgrid(int(num_cells)) * cell_size


def get_trimming_block(num_cells, cell_size):
    return get_perlin(
        noise_num_cells=np.ones(3) * 2 * (int(num_cells) // 2),
        noise_cell_size=cell_size,
    ).T

@HOOKS.register_module()
class AdditiveMaskSizeScheduler(HookBase):
    def __init__(
        self,
        mask_size_start=0.1,
        mask_size_end=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_end=0.1,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        density_factor_start=None,
        density_factor_end=None,
        density_factor_base=None,
        density_factor_warmup_ratio=0.05,
    ):
        # masking and scheduler
        self.mask_size_start = mask_size_start
        self.mask_size_end = mask_size_end
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio
        self.mask_size_scheduler = None
        
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio
        self.mask_ratio_scheduler = None
        
        # density factor scheduler
        self.density_factor_start = density_factor_start
        self.density_factor_end = density_factor_end
        self.density_factor_base = density_factor_base
        self.density_factor_warmup_ratio = density_factor_warmup_ratio
        self.density_factor_scheduler = None

    def find_additive_masking(self, transform):
        """Recursively search for the AdditiveMasking transform inside Pointcept's Compose object."""
        if isinstance(transform, AdditiveMasking):
            return transform
        if hasattr(transform, "transforms"):
            for t in transform.transforms:
                res = self.find_additive_masking(t)
                if res is not None:
                    return res
        return None

    def before_train(self):
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        
        # 1. Main process mirror schedulers (used exclusively for Tensorboard logging)
        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_end,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step

        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_end,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step
        
        if self.density_factor_start is not None:
            self.density_factor_scheduler = CosineScheduler(
                start_value=self.density_factor_start,
                base_value=self.density_factor_base,
                final_value=self.density_factor_end,
                warmup_iters=int(total_steps * self.density_factor_warmup_ratio),
                total_iters=total_steps,
            )
            self.density_factor_scheduler.iter = curr_step

        # 2. Worker schedulers (injected into the transform before DataLoader workers spawn)
        dataset = self.trainer.train_loader.dataset
        transform = self.find_additive_masking(dataset.transform)
        
        if transform is not None:
            batch_size = getattr(self.trainer.train_loader, "batch_size", 1)
            num_workers = max(1, getattr(self.trainer.train_loader, "num_workers", 1))
            
            # Workers process 1 sample at a time. This calculates the total samples each worker will handle.
            samples_per_step_per_worker = batch_size / num_workers
            worker_total_iters = int(total_steps * samples_per_step_per_worker)
            worker_curr_iter = int(curr_step * samples_per_step_per_worker)

            transform.mask_size_scheduler = CosineScheduler(
                start_value=self.mask_size_start,
                base_value=self.mask_size_base,
                final_value=self.mask_size_end,
                warmup_iters=int(worker_total_iters * self.mask_size_warmup_ratio),
                total_iters=worker_total_iters,
            )
            transform.mask_size_scheduler.iter = worker_curr_iter
            
            transform.mask_ratio_scheduler = CosineScheduler(
                start_value=self.mask_ratio_start,
                base_value=self.mask_ratio_base,
                final_value=self.mask_ratio_end,
                warmup_iters=int(worker_total_iters * self.mask_ratio_warmup_ratio),
                total_iters=worker_total_iters,
            )
            transform.mask_ratio_scheduler.iter = worker_curr_iter
            
            if self.density_factor_start is not None:
                transform.density_factor_scheduler = CosineScheduler(
                    start_value=self.density_factor_start,
                    base_value=self.density_factor_base,
                    final_value=self.density_factor_end,
                    warmup_iters=int(worker_total_iters * self.density_factor_warmup_ratio),
                    total_iters=worker_total_iters,
                )
                transform.density_factor_scheduler.iter = worker_curr_iter

    def before_step(self):
        # We step the main process logging schedulers. 
        # Since workers advance their own schedulers locally, we don't need to push this down!
        mask_size = self.mask_size_scheduler.step()
        mask_ratio = self.mask_ratio_scheduler.step()

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/mask_size",
                mask_size,
                self.mask_size_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/mask_ratio",
                mask_ratio,
                self.mask_ratio_scheduler.iter,
            )
            
            if self.density_factor_scheduler is not None:
                density_factor = self.density_factor_scheduler.step()
                self.trainer.writer.add_scalar(
                    "params/density_factor",
                    density_factor,
                    self.density_factor_scheduler.iter,
                )


@TRANSFORMS.register_module()
class AdditiveMasking(object):
    """
    Applies additive masking/occlusion to point clouds.
    It samples K centerpoints and adds procedural blocks (sculpting or trimming) at those locations.
    
    Labels generated in the output `mask`:
    - 0: Original point, unmasked.
    - 1: Added procedural block point (the occlusion itself).
    - 2: Original point, masked (occluded because it falls within the area of an added block).
    """
    def __init__(
        self,
        mode="Sculpting",  # "Sculpting", "Trimming", or "rand"
        sampling="chessboard",  # "chessboard", "random", or "random_rotate"
        npoint_frac=None,
        mask_size_min=0.1,
        mask_size_max=0.5,
        cell_size=0.02,
        density_factor=0.1,  # float or "rand"
        mask_dictname="segment",
        mask_ratio=1,
        remove_masked_points=False,
        mask_feature_mode=None, # backwards compat, overrides 1 and 2 if not None
        mask_feature_mode_1="point", # for mask == 1: "point", "random", "null", or "rand"
        mask_feature_mode_2=None, # for mask == 2: "point", "random", "null", "rand", or None
    ):
        self.mode = mode
        self.sampling = sampling
        self.npoint_frac = npoint_frac
        self.mask_size_min = mask_size_min
        self.mask_size_max = mask_size_max
        self.cell_size = cell_size
        self.density_factor = density_factor
        self.mask_dictname = mask_dictname
        self.mask_ratio = mask_ratio
        self.remove_masked_points = remove_masked_points
        
        if mask_feature_mode is not None:
            self.mask_feature_mode_1 = mask_feature_mode
            self.mask_feature_mode_2 = mask_feature_mode
        else:
            self.mask_feature_mode_1 = mask_feature_mode_1
            self.mask_feature_mode_2 = mask_feature_mode_2

        # Schedulers are safely injected by AdditiveMaskSizeScheduler hook right before DataLoader spawning
        self.mask_size_scheduler = None
        self.mask_ratio_scheduler = None
        self.density_factor_scheduler = None

        # Load both block generators to support 'rand' mode swapping
        self._get_trimming_block_fn = get_trimming_block
        self._get_sculpting_block_fn = get_sculpting_block

    def balance_npoint_frac(self, current_ncells_min, current_ncells_max, active_mode, active_density_factor):
        if active_mode.lower().startswith("t"):
            actual_density_factor = 0.15
        else:
            actual_density_factor = active_density_factor

        if self.sampling.startswith("c"): # chessboard
            self.npoint_frac = self.mask_ratio / (actual_density_factor * max(1, current_ncells_max**3))
        else: # random or random_rotate
            self.npoint_frac = self.mask_ratio * 8 / (actual_density_factor * max(1, (current_ncells_max+current_ncells_min+1)**3))

    def __call__(self, data_dict):
        """
        Executes the masking augmentation on the point cloud dictionary.
        """
        coord = data_dict["coord"]
        color = data_dict.get("color", None)
        normal = data_dict.get("normal", None)

        # --- INDEPENDENT WORKER SCHEDULING ---
        # The worker tracks how many individual point clouds it processes and updates itself seamlessly.
        if self.mask_size_scheduler is not None:
            self.mask_size_max = self.mask_size_scheduler.step()
        if self.mask_ratio_scheduler is not None:
            self.mask_ratio = self.mask_ratio_scheduler.step()
        if self.density_factor_scheduler is not None:
            self.density_factor = self.density_factor_scheduler.step()

        # Resolve mode dynamically for this sample
        if self.mode.lower() == "rand":
            active_mode = "Trimming" if np.random.rand() > 0.5 else "Sculpting"
        else:
            active_mode = self.mode

        # Resolve feature modes dynamically for this sample
        if self.mask_feature_mode_1.lower() == "rand":
            active_mask_feature_mode_1 = ["point", "random", "null"][np.random.randint(0, 3)]
        else:
            active_mask_feature_mode_1 = self.mask_feature_mode_1
            
        if self.mask_feature_mode_2 is None:
            active_mask_feature_mode_2 = None
        elif self.mask_feature_mode_2.lower() == "rand":
            active_mask_feature_mode_2 = ["point", "random", "null"][np.random.randint(0, 3)]
        else:
            active_mask_feature_mode_2 = self.mask_feature_mode_2

        # Resolve density factor and block function
        if active_mode.lower().startswith("t"):
            active_get_mask = self._get_trimming_block_fn
            active_density_factor = 1.0
        else:
            active_get_mask = self._get_sculpting_block_fn
            if self.density_factor == "rand":
                # Uniformly pick density factor between 0.1 and 1.0
                active_density_factor = np.random.uniform(0.1, 1.0)
            else:
                active_density_factor = float(self.density_factor)

        # --- SCENE-LEVEL SIZE CALCULATION ---
        # Generate a single random block size for the entire scene before balancing and the loop
        scene_mask_size = self.mask_size_min + np.random.rand() * (self.mask_size_max - self.mask_size_min)
        scene_ncells = int(scene_mask_size // self.cell_size)

        self.balance_npoint_frac(
            current_ncells_min=scene_ncells,
            current_ncells_max=scene_ncells,
            active_mode=active_mode,
            active_density_factor=active_density_factor
        )

        # 1. Choose K based on npoint_frac and point cloud size
        K = max(1, int(self.npoint_frac * len(coord)))

        # 2. Sample K coordinates for centerpoints
        if self.sampling == "chessboard":
            min_coord = np.min(coord, axis=0)
            grid_coord = np.floor((coord - min_coord) / scene_mask_size).astype(np.int32)
            
            unique_cells, clusters, count = torch.unique(
                torch.tensor(grid_coord), dim=0, return_inverse=True, return_counts=True
            )
            
            ncells = unique_cells.shape[0]
            picked_cells = np.random.randint(low=0, high=ncells, size=(K,))
            
            # Derived centerpoints
            centerpoints = unique_cells[picked_cells].numpy() * scene_mask_size + min_coord
            
            # Setup original labels: 0 for untouched, 2 for masked cells
            orig_mask = np.isin(clusters.numpy(), picked_cells).astype(np.int32) * 2
            
            # Map features based on the first point in the chosen cell
            first_point_idx = np.cumsum(np.insert(count.numpy(), 0, 0)[0:-1])
            feat_idxs = first_point_idx[picked_cells]

        elif self.sampling in ["random", "random_rotate"]:
            picked_idxs = np.random.randint(0, len(coord), size=(K,))
            
            # Directly sample centerpoints from the point cloud
            centerpoints = coord[picked_idxs]
            
            # Initialize to 0; we will update masked points to 2 dynamically during block generation
            orig_mask = np.zeros(len(coord), dtype=np.int32)
            feat_idxs = picked_idxs
            
        else:
            raise ValueError(f"Unknown sampling technique: {self.sampling}")

        # 3. Add K offsetted blocks
        _block_coords = []
        _block_colors = []
        _block_normals = []
        cached_blocks = {}

        for i in range(K):
            num_cells = scene_ncells
            
            # Retrieve basic block at origin (cached for Sculpting mode to save time)
            if active_mode.lower().startswith("s"):
                if num_cells not in cached_blocks:
                    cached_blocks[num_cells] = active_get_mask(num_cells, self.cell_size)
                block = cached_blocks[num_cells]
            else:
                block = active_get_mask(num_cells, self.cell_size)

            if len(block) == 0:
                continue

            # Apply random Z rotation if strategy is random_rotate
            if self.sampling == "random_rotate":
                rotation = R.from_euler(
                    "z", np.random.rand() * np.pi, degrees=False
                ).as_matrix()
                # Center the block before rotating it around its local origin
                block = block - block.mean(axis=0)
                block = (rotation @ block.T).T
                
            # Offset it to centerpoint
            block = block + centerpoints[i]
            
            # For random sampling, dynamically mask original points that fall within the block's bounding box
            if self.sampling in ["random", "random_rotate"]:
                b_min = block.min(axis=0)
                b_max = block.max(axis=0)
                # Label original points within this bounding box as masked (2)
                in_box = np.all((coord >= b_min) & (coord <= b_max), axis=1)
                orig_mask[in_box] = 2
            
            _block_coords.append(block)
            
            num_pts = len(block)
            if color is not None:
                if active_mask_feature_mode_1 == "point":
                    _block_colors.append(np.tile(color[feat_idxs[i]], (num_pts, 1)))
                else:
                    _block_colors.append(np.zeros((num_pts, color.shape[1])))
            if normal is not None:
                if active_mask_feature_mode_1 == "point":
                    _block_normals.append(np.tile(normal[feat_idxs[i]], (num_pts, 1)))
                else:
                    _block_normals.append(np.zeros((num_pts, normal.shape[1])))

        # Remove points that were flagged as masked (2) if the flag is active
        if self.remove_masked_points:
            keep_mask = orig_mask != 2
            coord = coord[keep_mask]
            orig_mask = orig_mask[keep_mask]
            if color is not None:
                color = color[keep_mask]
            if normal is not None:
                normal = normal[keep_mask]

            # In case no blocks were generated, we still update the base dict now
            data_dict["coord"] = coord
            if color is not None: data_dict["color"] = color
            if normal is not None: data_dict["normal"] = normal

        if _block_coords:
            block_coords = np.vstack(_block_coords)
            
            if color is not None:
                block_colors = np.vstack(_block_colors)
            if normal is not None:
                block_normals = np.vstack(_block_normals)

            # Optimization: Uniformly subsample the aggregated blocks
            if active_density_factor is not None and active_density_factor < 1.0:
                num_total_added = len(block_coords)
                num_to_keep = max(1, int(num_total_added * active_density_factor))
                choices = np.random.choice(num_total_added, num_to_keep, replace=False)
                
                block_coords = block_coords[choices]
                if color is not None: block_colors = block_colors[choices]
                if normal is not None: block_normals = block_normals[choices]

            # Calculate mask_balance (ratio of added points to original points)
            mask_balance = len(block_coords) / max(1, len(coord))

            data_dict["coord"] = np.vstack([coord, block_coords]).astype(np.float32)
            
            # 4. Set labels of added blocks to 1
            final_mask = np.hstack([orig_mask, np.ones(len(block_coords), dtype=np.int32)])
            data_dict[self.mask_dictname] = final_mask
            if color is not None:
                final_color = np.vstack([color, block_colors]).astype(np.float32)
                
                # Apply changes independently for block points (1) and masked points (2)
                mask_1 = (final_mask == 1)
                if active_mask_feature_mode_1 == "random":
                    final_color[mask_1] = np.random.rand(np.sum(mask_1), final_color.shape[1]).astype(np.float32)
                elif active_mask_feature_mode_1 == "null":
                    final_color[mask_1] = 1.0
                    
                mask_2 = (final_mask == 2)
                if active_mask_feature_mode_2 == "random":
                    final_color[mask_2] = np.random.rand(np.sum(mask_2), final_color.shape[1]).astype(np.float32)
                elif active_mask_feature_mode_2 == "null":
                    final_color[mask_2] = 1.0
                    
                data_dict["color"] = final_color
                
            if normal is not None:
                final_normal = np.vstack([normal, block_normals]).astype(np.float32)
                
                mask_1 = (final_mask == 1)
                if active_mask_feature_mode_1 == "random":
                    rand_n = np.random.rand(np.sum(mask_1), final_normal.shape[1]).astype(np.float32) * 2.0 - 1.0
                    norms = np.linalg.norm(rand_n, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    final_normal[mask_1] = rand_n / norms
                elif active_mask_feature_mode_1 == "null":
                    final_normal[mask_1] = 1.0

                mask_2 = (final_mask == 2)
                if active_mask_feature_mode_2 == "random":
                    rand_n = np.random.rand(np.sum(mask_2), final_normal.shape[1]).astype(np.float32) * 2.0 - 1.0
                    norms = np.linalg.norm(rand_n, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    final_normal[mask_2] = rand_n / norms
                elif active_mask_feature_mode_2 == "null":
                    final_normal[mask_2] = 1.0
                    
                data_dict["normal"] = final_normal

            # Pass mask_balance via data_dict for Pointcept's main writer to log safely
            data_dict["mask_balance"] = float(mask_balance)
        else:
            data_dict[self.mask_dictname] = orig_mask
            data_dict["mask_balance"] = 0.0

            # Even if no blocks were added, we might need to overwrite masked (2) point features
            if active_mask_feature_mode_2 == "random":
                mask_2 = (orig_mask == 2)
                if color is not None and np.any(mask_2):
                    data_dict["color"][mask_2] = np.random.rand(np.sum(mask_2), color.shape[1]).astype(np.float32)
                if normal is not None and np.any(mask_2):
                    rand_n = np.random.rand(np.sum(mask_2), normal.shape[1]).astype(np.float32) * 2.0 - 1.0
                    norms = np.linalg.norm(rand_n, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    data_dict["normal"][mask_2] = rand_n / norms
            elif active_mask_feature_mode_2 == "null":
                mask_2 = (orig_mask == 2)
                if color is not None and np.any(mask_2):
                    data_dict["color"][mask_2] = 1.0
                if normal is not None and np.any(mask_2):
                    data_dict["normal"][mask_2] = 1.0

        return data_dict