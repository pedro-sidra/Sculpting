import numpy as np
import torch
from .sculpting_ops import (
    get_random_colored_cubes_on_pts,
    array_mode,
    array_rand_choice,
    array_choice,
    get_pointgrid,
)
from copy import deepcopy


from perlyn import (
    generate_perlin_noise,
)

# from perlin_numpy import (
#     generate_fractal_noise_2d,
#     generate_fractal_noise_3d,
#     generate_perlin_noise_2d,
#     generate_perlin_noise_3d,
# )

import pointcept.datasets.transform as transform
from pointcept.utils.registry import Registry
from pointcept.datasets.transform import TRANSFORMS
from .sculpting import SculptingOcclude


# TRANSFORMS = Registry("transforms")
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


def get_perlin_noise_on_pts(
    cube_size_min, cube_size_max, points, feats, npoints=10, *args, **kwargs
):
    idxs = np.random.randint(0, len(points), size=npoints)
    cell_size = kwargs.get("cell_size")

    cubes = []
    cube_feats = []

    for idx in idxs:
        # aspect = np.random.randint(0, 3)
        _cube_size_min = np.ones(3) * cube_size_min
        _cube_size_max = np.ones(3) * cube_size_max
        # _cube_size_min[aspect] = cube_size_max / 5
        # _cube_size_max[aspect] = cube_size_max
        point = points[idx]
        f = feats[idx]

        cube_size = np.random.randint( _cube_size_min // cell_size, _cube_size_max // cell_size, (3,))
        # brain fart
        cube_size[1:]=cube_size[0]
        cube = get_perlin(
            noise_num_cells=(cube_size//2)*2,
            noise_cell_size=cell_size
        ).T + point.reshape((1, 3))

        feat = np.ones_like(cube) * f

        cubes.append(cube)
        cube_feats.append(feat)

    return np.vstack(cubes), np.vstack(cube_feats)


@TRANSFORMS.register_module()
class TrimmingOcclude(SculptingOcclude):
    def add_random_cubes(self, data_dict):
        xyz = data_dict["coord"]
        rgb = data_dict.get("color", self.get_random_colors(xyz.shape))
        normal = data_dict.get("normal", self.get_random_normals(xyz.shape))

        semantic_label = data_dict.get("segment", np.ones(len(xyz), dtype=int))

        if self.npoints is None:
            ncubes = int(self.npoint_frac * len(xyz))
        else:
            ncubes = self.npoints

        cubes, cube_feats = get_perlin_noise_on_pts(
            self.cube_size_min,
            self.cube_size_max,
            xyz,
            feats=rgb,
            npoints=ncubes,
            cell_size=self.cell_size,
            actual_cube=False,
            sphere=False,
            point_sampling=self.sampling,
            density_factor=self.density_factor,
        )

        xyz = np.vstack([xyz, cubes])

        # rand_colors = self.get_random_colors(cubes.shape)
        rgb = np.vstack([rgb, cube_feats])

        if normal is not None:
            rand_normals = self.get_random_normals(cubes.shape)
            normal = np.vstack([normal, rand_normals])

        # Randomly turn colors off
        # if np.random.rand() < self.kill_color_proba:
        #     rgb = rgb * 0.0 + np.random.rand() * 255

        dummy_cube = np.ones(len(cubes), dtype=np.int32)
        dummy_pc = np.ones_like(semantic_label, dtype=np.int32)

        semantic_label = np.hstack([dummy_pc, 0 * dummy_cube])
        instance_label = np.hstack([-1 * dummy_pc, -1 * dummy_cube])

        return (
            xyz.astype(np.float32),
            rgb.astype(np.float32),
            semantic_label.astype(np.int32),
            normal.astype(np.float32),
            instance_label.astype(np.int32),
        )

@TRANSFORMS.register_module()
class TrimmingMaskOcclude(object):
    def __init__(
        self,
        enable_feat_masking=True,
        mask_size=0.2,
        mask_ratio=0.5,
        cell_size=0.02,
        density_factor=0.1,
        npoint_frac=None,
        random_sizes=False,
        min_mask_size=0.1,
    ):
        self.enable_feat_masking = enable_feat_masking
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.cell_size = cell_size
        self.density_factor = density_factor
        self.npoint_frac = npoint_frac
        self.random_sizes = random_sizes
        self.min_mask_size = min_mask_size

    def hash(self, arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def get_sculpting_blocks_and_mask(
        self,
        coord,
    ):
        # Ratio of masked voxels
        # Size of each masked voxel
        if self.random_sizes:
            MASK_SIZE = self.min_mask_size + np.random.rand() * (self.mask_size - self.min_mask_size)
        else:
            MASK_SIZE = self.mask_size
        # Size of each point in the cube-grid for sculpting
        SCULPT_CELL_SIZE = self.cell_size
        SCULPT_CELL_DENSITY = self.density_factor

        # Offset to start at origin and have positive indexes
        min_coord = np.min(coord, axis=0)
        grid_coord = ((coord - min_coord) // MASK_SIZE).astype(np.int32)

        # get voxel ids(torch impl)
        unique_cells, clusters = torch.unique(
            torch.tensor(grid_coord), dim=0, return_inverse=True
        )
        unique_cells=unique_cells.numpy()
        clusters=clusters.numpy()

        # Pick cells for masking
        ncells = unique_cells.shape[0]

        if self.npoint_frac is None:
            ncubes = int(ncells * self.mask_ratio)
        else:
            ncubes = int(len(coord) * self.npoint_frac)

        ncubes = max(ncubes,1)

        picked_cells = np.random.randint(low=0, high=ncells, size=(ncubes,))

        # Voxel coordinates of picked cells
        p0s = unique_cells[picked_cells]
        p0s = (
            p0s * MASK_SIZE + min_coord  # cell coordinates  # min_coord per batch index
        )

        # Place cubes at each picked cell
        c = get_perlin(
            noise_num_cells=np.ones(3)*2*(int(MASK_SIZE // SCULPT_CELL_SIZE)//2),
            noise_cell_size=SCULPT_CELL_SIZE
        ).T
        # trick to do outer addition with broadcasting
        offsetted = p0s[None, ...] + c[:, None, :]
        offsetted = offsetted.reshape(-1, 3)

        mask = np.isin(clusters, picked_cells).astype(int)
        return offsetted, mask

    def get_random_colors(self, size, low=0, high=255):
        return np.random.randint(low, high, size).astype(np.float32)

    def get_random_normals(self, size):
        n = np.random.rand(*size).astype(np.float32) * 2 - 1
        n = n / np.linalg.norm(n, axis=-1)[:, np.newaxis]
        return n

    def __call__(self, data_dict):
        # Input PC data
        xyz = data_dict["coord"]
        rgb = data_dict.get("color", self.get_random_colors(xyz.shape))
        normal = data_dict.get("normal", self.get_random_normals(xyz.shape))

        # Will be passed through SonataLikeSculptor.before_train
        self.mask_ratio = data_dict.get("mask_ratio", self.mask_ratio)
        self.mask_size = data_dict.get("mask_size", self.mask_size)

        #self.mask_size = data_dict.get("mask_size", self.mask_size)

        cubes, mask = self.get_sculpting_blocks_and_mask(xyz)

        # mask will be 0 for original points, 1 for sculpted points, 2 for masked points
        mask = np.hstack((mask * 2, torch.full((len(cubes),), 1))).astype(np.int32)

        xyz = np.vstack([xyz, cubes])

        rgb = np.vstack([rgb, 0.0 * self.get_random_colors(cubes.shape)])

        if normal is not None:
            rand_normals = self.get_random_normals(cubes.shape)
            normal = np.vstack([normal, 0.0 * rand_normals])

        if self.enable_feat_masking:
            rgb[mask == 2] = [0.0, 0.0, 0.0]
            normal[mask == 2] = [0.0, 0.0, 0.0]

        data_dict["coord"] = xyz.astype(np.float32)
        data_dict["color"] = rgb.astype(np.float32)
        data_dict["normal"] = normal.astype(np.float32)
        data_dict["mask"] = mask.astype(np.int32)
        data_dict.pop("segment")
        data_dict.pop("instance")

        return data_dict
