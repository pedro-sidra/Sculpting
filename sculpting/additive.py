import numpy as np
import torch
from perlyn import generate_perlin_noise

from pointcept.datasets.transform import TRANSFORMS


def get_pointgrid(ncells=[10, 10, 10]):

    if isinstance(ncells, int):
        ncells = [ncells, ncells, ncells]

    points_x = np.arange(0, ncells[0])
    points_y = np.arange(0, ncells[1])
    points_z = np.arange(0, ncells[2])

    x, y, z = np.meshgrid(points_x, points_y, points_z)

    return np.stack(
        [
            x.flatten(),
            y.flatten(),
            z.flatten(),
        ]
    ).T

def get_perlin(noise_num_cells, noise_cell_size):
    noise = generate_perlin_noise(
        noise_num_cells, # need to get multiple of 2
          (2, 2, 2), tileable=(False, False, False)
    )
    i, j, k = np.indices(noise.shape)

    noise = noise.flatten()

    threshold = 0.8/noise_num_cells[0]
    if np.random.rand()>0.5:
        locs = np.bitwise_and(noise < threshold, noise > -threshold)
    else:
        locs = np.bitwise_and(noise > 1e-6, noise < 2*threshold)

    i = noise_cell_size * i.flatten()[locs]
    j = noise_cell_size * j.flatten()[locs]
    k = noise_cell_size * k.flatten()[locs]
    return np.vstack([i, j, k])

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
        sampling="chessboard",  # "chessboard" or "random"
        npoint_frac=0.05,
        mask_size_min=0.1,
        mask_size_max=0.5,
        cell_size=0.02,
    ):
        self.mode = mode
        self.sampling = sampling
        self.npoint_frac = npoint_frac
        self.mask_size_min = mask_size_min
        self.mask_size_max = mask_size_max
        self.cell_size = cell_size

        if mode.lower().startswith("t"):
            self.get_mask = get_trimming_block
        else:
            self.get_mask = get_sculpting_block

    def __call__(self, data_dict):
        """
        Executes the masking augmentation on the point cloud dictionary.
        """
        coord = data_dict["coord"]
        color = data_dict.get("color", None)
        normal = data_dict.get("normal", None)

        # 1. Choose K based on npoint_frac and point cloud size
        K = max(1, int(self.npoint_frac * len(coord)))

        # 2. Sample K coordinates for centerpoints
        if self.sampling == "chessboard":
            mask_size = self.mask_size_min + np.random.rand() * (self.mask_size_max - self.mask_size_min)
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

        elif self.sampling == "random":
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

        for i in range(K):
            if self.sampling == "chessboard":
                b_size = mask_size
            else:
                b_size = self.mask_size_min + np.random.rand() * (self.mask_size_max - self.mask_size_min)
                
            num_cells = int(b_size // self.cell_size)
            
            # Retrieve basic block at origin
            block = self.get_mask(num_cells, self.cell_size)
            if len(block) == 0:
                continue
                
            # Offset it to centerpoint
            block = block + centerpoints[i]
            
            # For random sampling, dynamically mask original points that fall within the block's bounding box
            if self.sampling == "random":
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
            data_dict["coord"] = np.vstack([coord, block_coords]).astype(np.float32)
            
            if color is not None:
                block_colors = np.vstack(_block_colors)
                data_dict["color"] = np.vstack([color, block_colors]).astype(np.float32)
                
            if normal is not None:
                block_normals = np.vstack(_block_normals)
                data_dict["normal"] = np.vstack([normal, block_normals]).astype(np.float32)
                
            # 4. Set labels of added blocks to 1
            final_mask = np.hstack([orig_mask, np.ones(len(block_coords), dtype=np.int32)])
            data_dict["mask"] = final_mask
        else:
            data_dict["mask"] = orig_mask

        # Clean up legacy labels
        data_dict.pop("segment", None)
        data_dict.pop("instance", None)

        return data_dict