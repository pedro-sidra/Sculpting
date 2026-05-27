import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from .sculpting_ops import  get_pointgrid
from .trimming import get_perlin

from pointcept.datasets.transform import TRANSFORMS

def get_sculpting_block(num_cells, cell_size):
    return get_pointgrid(int(num_cells)) * cell_size


def get_trimming_block(num_cells, cell_size):
    return get_perlin(
        noise_num_cells=np.ones(3) * 2 * (int(num_cells) // 2),
        noise_cell_size=cell_size,
    ).T


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
        mode="Sculpting",  # "Sculpting" or "Trimming"
        sampling="chessboard",  # "chessboard", "random", or "random_rotate"
        npoint_frac=None,
        mask_size_min=0.1,
        mask_size_max=0.5,
        cell_size=0.02,
        density_factor=0.1,
    ):
        self.mode = mode
        self.sampling = sampling
        self.npoint_frac = npoint_frac
        self.mask_size_min = mask_size_min
        self.mask_size_max = mask_size_max
        self.cell_size = cell_size
        self.density_factor = density_factor

        if mode.lower().startswith("t"):
            self.get_mask = get_trimming_block
            self.density_factor = 1.0
        else:
            self.get_mask = get_sculpting_block

        if self.sampling.startswith("c"): # chessboard
            # For chessboard sampling, we want a fixed block size
            self.mask_size_min = mask_size_max 
        else:
            pass
        

    def balance_npoint_frac(self, current_ncells_min, current_ncells_max):
        if self.sampling.startswith("c"): # chessboard
            # assert current_ncells_min == current_ncells_max, "For chessboard sampling, block size should be fixed."
            self.npoint_frac=1/(self.density_factor * current_ncells_max**3)
        else: # random or random_rotate
            self.npoint_frac=8/(self.density_factor * (current_ncells_max+current_ncells_min+1)**3)

    def __call__(self, data_dict):
        """
        Executes the masking augmentation on the point cloud dictionary.
        """
        coord = data_dict["coord"]
        color = data_dict.get("color", None)
        normal = data_dict.get("normal", None)

        self.mask_size_max = data_dict.get("mask_size", self.mask_size)

        self.balance_npoint_frac(
            current_ncells_min=int(self.mask_size_min // self.cell_size),
            current_ncells_max=int(self.mask_size_max // self.cell_size),
        )

        # 1. Choose K based on npoint_frac and point cloud size
        K = max(1, int(self.npoint_frac * len(coord)))

        # 2. Sample K coordinates for centerpoints
        if self.sampling == "chessboard":
            mask_size = self.mask_size_min
            min_coord = np.min(coord, axis=0)
            grid_coord = np.floor((coord - min_coord) / mask_size).astype(np.int32)
            
            unique_cells, clusters, count = torch.unique(
                torch.tensor(grid_coord), dim=0, return_inverse=True, return_counts=True
            )
            
            ncells = unique_cells.shape[0]
            picked_cells = np.random.randint(low=0, high=ncells, size=(K,))
            
            # Derived centerpoints
            centerpoints = unique_cells[picked_cells].numpy() * mask_size + min_coord
            
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
            if self.sampling == "chessboard":
                b_size = mask_size
            else:
                b_size = self.mask_size_min + np.random.rand() * (self.mask_size_max - self.mask_size_min)
                
            num_cells = int(b_size // self.cell_size)
            
            # Retrieve basic block at origin (cached for Sculpting mode to save time)
            if self.mode.lower().startswith("s"):
                if num_cells not in cached_blocks:
                    cached_blocks[num_cells] = self.get_mask(num_cells, self.cell_size)
                block = cached_blocks[num_cells]
            else:
                block = self.get_mask(num_cells, self.cell_size)

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
                _block_colors.append(np.tile(color[feat_idxs[i]], (num_pts, 1)))
            if normal is not None:
                _block_normals.append(np.tile(normal[feat_idxs[i]], (num_pts, 1)))

        if _block_coords:
            block_coords = np.vstack(_block_coords)
            
            if color is not None:
                block_colors = np.vstack(_block_colors)
            if normal is not None:
                block_normals = np.vstack(_block_normals)

            # Optimization: Uniformly subsample the aggregated blocks
            if self.density_factor is not None and self.density_factor < 1.0:
                num_total_added = len(block_coords)
                num_to_keep = max(1, int(num_total_added * self.density_factor))
                choices = np.random.choice(num_total_added, num_to_keep, replace=False)
                
                block_coords = block_coords[choices]
                if color is not None: block_colors = block_colors[choices]
                if normal is not None: block_normals = block_normals[choices]

            # Calculate mask_balance (ratio of added points to original points)
            mask_balance = len(block_coords) / len(coord)

            data_dict["coord"] = np.vstack([coord, block_coords]).astype(np.float32)
            
            if color is not None:
                data_dict["color"] = np.vstack([color, block_colors]).astype(np.float32)
                
            if normal is not None:
                data_dict["normal"] = np.vstack([normal, block_normals]).astype(np.float32)
                
            # 4. Set labels of added blocks to 1
            final_mask = np.hstack([orig_mask, np.ones(len(block_coords), dtype=np.int32)])
            data_dict["segment"] = final_mask
            
            # Pass mask_balance via data_dict for Pointcept's main writer to log safely
            data_dict["mask_balance"] = float(mask_balance)
        else:
            data_dict["segment"] = orig_mask
            data_dict["mask_balance"] = 0.0

        return data_dict