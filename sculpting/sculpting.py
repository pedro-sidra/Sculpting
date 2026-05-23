import numpy as np
import torch
from .trimming import get_perlin_noise_on_pts
from .sculpting_ops import ( get_random_colored_cubes_on_pts, get_pointgrid,)
from .trimming import get_perlin

from pointcept.datasets.transform import TRANSFORMS

@TRANSFORMS.register_module()
class SculptingOcclude(object):
    def __init__(
        self,
        cube_size_min=0.1,
        cube_size_max=0.5,
        npoint_frac=0.005,
        npoints=None,
        cell_size=0.02,
        density_factor=0.1,
        kill_color_proba=0.5,
        sampling="random",
    ):
        self.cube_size_min = cube_size_min
        self.cube_size_max = cube_size_max
        self.npoint_frac = npoint_frac
        self.npoints = npoints
        self.cell_size = cell_size
        self.density_factor = density_factor
        self.kill_color_proba = kill_color_proba
        self.sampling = sampling

    def get_random_colors(self, size, low=0, high=255):
        return np.random.randint(low, high, size).astype(np.float32)

    def get_random_normals(self, size):
        n = np.random.rand(*size).astype(np.float32) * 2 - 1
        n = n / np.linalg.norm(n, axis=-1)[:, np.newaxis]
        return n

    def add_random_cubes(self, data_dict):

        xyz = data_dict["coord"]
        rgb = data_dict.get("color", self.get_random_colors(xyz.shape))
        normal = data_dict.get("normal", self.get_random_normals(xyz.shape))

        semantic_label = data_dict.get("segment", np.ones(len(xyz), dtype=int))

        if self.npoints is None:
            ncubes = max(int(self.npoint_frac * len(xyz)), 1)
        else:
            ncubes = self.npoints

        cubes, cube_feats = get_random_colored_cubes_on_pts(
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

    def __call__(self, data_dict):
        """
        for semseg models,
        data_dict.keys() = ['coord', 'color', 'normal', 'name', 'segment', 'instance']
        """

        (
            data_dict["coord"],
            data_dict["color"],
            data_dict["segment"],
            data_dict["normal"],
            data_dict["instance"],
        ) = self.add_random_cubes(data_dict)
        # from pointcept.utils import ForkedPdb; ForkedPdb().set_trace()
        return data_dict


def get_sculpting_block(num_cells, cell_size):
    return (
        get_pointgrid(int(num_cells)) * cell_size
    )

def get_trimming_block(num_cells, cell_size):
    return get_perlin( 
        noise_num_cells=np.ones(3)*2*(int(num_cells)//2),
        noise_cell_size=cell_size
    ).T

@TRANSFORMS.register_module()
class AdditiveMaskOcclude(object):
    def __init__(
        self,
        mode="Sculpting", # "Sculpting" or "Trimming"
        enable_feat_masking=True,
        mask_size=0.2,
        mask_ratio=0.5,
        cell_size=0.02,
        density_factor=0.1,
        npoint_frac=None,
        random_sizes=False,
        min_mask_size=0.1,
    ):
        self.mode=mode
        if mode.lower().startswith("t"): # Trimming
            self.get_mask=get_trimming_block
        else: # Sculpting
            self.get_mask=get_sculpting_block

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
        feat
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
        unique_cells, clusters, count = torch.unique(
            torch.tensor(grid_coord), dim=0, return_inverse=True, return_counts=True
        )

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
            p0s * MASK_SIZE + min_coord
        )
        first_point_idx = np.cumsum(np.insert(count, 0, 0)[0:-1])

        # Place cubes at each picked cell
        _cube_coords = []
        _cube_feats = []
        for i, cell in enumerate(picked_cells):
            cube_coords = self.get_mask( 
                num_cells=MASK_SIZE // SCULPT_CELL_SIZE,
                cell_size=SCULPT_CELL_SIZE
            )
            # trick to do outer addition with broadcasting
            cube_coords = p0s[i] + cube_coords
            point_feat=feat[first_point_idx[i]]
            cube_feats = point_feat * np.ones((len(cube_coords),point_feat.shape[0]))

            _cube_coords.append(cube_coords)
            _cube_feats.append(cube_feats)

        cube_coords = np.vstack(_cube_coords)
        cube_feats = np.vstack(_cube_feats)
        mask = np.isin(clusters, picked_cells).astype(int)
        return cube_coords, cube_feats, mask

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

        cube_coords, cube_feats, mask = self.get_sculpting_blocks_and_mask(xyz, rgb)

        # mask will be 0 for original points, 1 for sculpted points, 2 for masked points
        mask = np.hstack((mask * 2, torch.full((len(cube_coords),), 1))).astype(np.int32)

        xyz = np.vstack([xyz, cube_coords])

        rgb = np.vstack([rgb, cube_feats])

        if normal is not None:
            rand_normals = self.get_random_normals(cube_coords.shape)
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