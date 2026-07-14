import numpy as np
import torch
import numba as nb
from copy import deepcopy

import pointcept.datasets.transform as transform
from pointcept.datasets.transform import TRANSFORMS


@nb.njit
def jit_follow_k_2d(feats, out, mask, k, count, idx_sort, group_offsets):
    """JIT compiled kernel for fast 2D follow-K feature aggregation."""
    num_voxels = len(count)
    feat_dim = feats.shape[1]
    for i in range(num_voxels):
        start = group_offsets[i]
        end = start + count[i]
        
        found = False
        for j in range(start, end):
            idx = idx_sort[j]
            if mask[idx] == k:
                for d in range(feat_dim):
                    out[i, d] = feats[idx, d]
                found = True
                break
        
        # Fallback to mean if K is not found in this voxel
        if not found:
            for j in range(start, end):
                idx = idx_sort[j]
                for d in range(feat_dim):
                    out[i, d] += feats[idx, d]
            for d in range(feat_dim):
                out[i, d] /= count[i]
    return out


@nb.njit
def jit_follow_k_1d(feats, out, mask, k, count, idx_sort, group_offsets):
    """JIT compiled kernel for fast 1D follow-K feature aggregation."""
    num_voxels = len(count)
    for i in range(num_voxels):
        start = group_offsets[i]
        end = start + count[i]
        
        found = False
        for j in range(start, end):
            idx = idx_sort[j]
            if mask[idx] == k:
                out[i] = feats[idx]
                found = True
                break
        
        # Fallback to mean if K is not found in this voxel
        if not found:
            val = 0.0
            for j in range(start, end):
                val += feats[idx_sort[j]]
            out[i] = val / count[i]
    return out


@TRANSFORMS.register_module()
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


@TRANSFORMS.register_module()
class VoxelizeAgg(object):

    def __init__(
        self,
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        follow_ref="segment",
        how_to_agg_feats=dict(
            coord="mean",
            color="mean",
            normal="mean",
            segment="mode",
        ),
    ):
        self.grid_size = grid_size
        self.hash = (
            transform.GridSample.fnv_hash_vec
            if hash_type == "fnv"
            else transform.GridSample.ravel_hash_vec
        )
        assert mode in ["train", "test"]
        self.mode = mode
        self.follow_ref = follow_ref

        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord

        self.how_to_agg_feats = how_to_agg_feats
        self.agg_func_names = deepcopy(how_to_agg_feats)

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()

        # To voxel indexes
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord

        # Save the min coord in original values
        min_coord = min_coord * np.array(self.grid_size)

        # Hash of the grid coords -> to group the unique voxel coords
        key = self.hash(grid_coord)
        
        # --- ROBUSTNESS FIX: Ensure all aggregation variables match coord length ---
        for var_name in self.agg_func_names.keys():
            if var_name in data_dict and data_dict[var_name].shape[0] < key.shape[0]:
                pad_size = key.shape[0] - data_dict[var_name].shape[0]
                pad_shape = list(data_dict[var_name].shape)
                pad_shape[0] = pad_size
                pad_val = -1 if var_name in ["segment", "label"] else 0
                pad_arr = np.full(pad_shape, pad_val, dtype=data_dict[var_name].dtype)
                data_dict[var_name] = np.concatenate([data_dict[var_name], pad_arr], axis=0)

        # --- MULTI-K ENHANCEMENT: Base Sort & Specialized Sorts ---
        
        # 1. Establish the base grouping and sorting (used for standard mean/max/first)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        first_point_idx = idx_sort[np.cumsum(np.insert(count, 0, 0)[0:-1])]
        
        # 2. Extract original mask if we have any follow-* instructions
        original_mask = None
        has_follow = any(f.startswith("follow-") for f in self.agg_func_names.values())
        if has_follow:
            original_mask = data_dict[self.follow_ref].copy()

        # 3. Precompute a specialized lexsort mapping for EACH unique K requested
        special_sorts = {}
        if original_mask is not None:
            unique_ks = set()
            for f in self.agg_func_names.values():
                if f.startswith("follow-"):
                    k = 1 if f == "follow-mask" else int(f.split("-")[1])
                    unique_ks.add(k)
                    
            for k in unique_ks:
                is_label_k = (original_mask.reshape(-1) == k).astype(np.int32)
                # Pulls all points matching K to the front of their respective voxels
                idx_sort_k = np.lexsort((-is_label_k, key))
                first_idx_k = idx_sort_k[np.cumsum(np.insert(count, 0, 0)[0:-1])]
                special_sorts[k] = {"idx_sort": idx_sort_k, "first_idx": first_idx_k}

        # 4. Process Aggregations
        for var_name, agg_func in self.agg_func_names.items():
            if var_name not in data_dict:
                continue

            if agg_func == "first":
                data_dict[var_name] = data_dict[var_name][first_point_idx]
            elif agg_func == "rand_choice":
                idx_select = idx_sort[
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + np.random.randint(0, count.max(), count.size) % count
                ]
                data_dict[var_name] = data_dict[var_name][idx_select]
            elif agg_func == "mean":
                data_dict[var_name] = (
                    np.add.reduceat(
                        data_dict[var_name][idx_sort],
                        np.cumsum(np.insert(count, 0, 0)[0:-1]),
                    )
                    / count[:, np.newaxis]
                )
            elif agg_func == "max":
                data_dict[var_name] = np.maximum.reduceat(
                    data_dict[var_name][idx_sort],
                    np.cumsum(np.insert(count, 0, 0)[0:-1]),
                )
            elif agg_func == "min":
                data_dict[var_name] = np.minimum.reduceat(
                    data_dict[var_name][idx_sort],
                    np.cumsum(np.insert(count, 0, 0)[0:-1]),
                )
            elif agg_func == "mode":
                # Fallback to 'first' to prevent untracked tensors from keeping size N
                data_dict[var_name] = data_dict[var_name][first_point_idx]
                
            # --- FOLLOW-K: MULTI-K AGGREGATION ENHANCEMENT ---
            elif agg_func.startswith("follow-"):
                if original_mask is not None:
                    # Resolve K dynamically per variable
                    k = 1 if agg_func == "follow-mask" else int(agg_func.split("-")[1])
                            
                    if k in special_sorts:
                        idx_sort_k = special_sorts[k]["idx_sort"]
                        first_idx_k = special_sorts[k]["first_idx"]
                        
                        has_mask_k = (original_mask[first_idx_k] == k)
                        exact_vals = data_dict[var_name][first_idx_k]
                        
                        # Calculate mean using the K-specific sort array (perfectly valid for reduceat)
                        mean_vals = np.add.reduceat(
                            data_dict[var_name][idx_sort_k],
                            np.cumsum(np.insert(count, 0, 0)[0:-1]),
                        )
                        
                        if data_dict[var_name].ndim > 1:
                            mean_vals = mean_vals / count[:, np.newaxis]
                            mask_cond = has_mask_k.reshape(-1, 1)
                        else:
                            mean_vals = mean_vals / count
                            mask_cond = has_mask_k.reshape(-1)
                            
                        mean_vals = mean_vals.astype(exact_vals.dtype)
                        data_dict[var_name] = np.where(mask_cond, exact_vals, mean_vals)
                    else:
                        # Safety fallback if something goes horribly wrong
                        data_dict[var_name] = data_dict[var_name][first_point_idx]
                else:
                    # Fallback entirely to mean if there's no segment/mask array available
                    mean_vals = np.add.reduceat(
                        data_dict[var_name][idx_sort],
                        np.cumsum(np.insert(count, 0, 0)[0:-1]),
                    )
                    if data_dict[var_name].ndim > 1:
                        mean_vals = mean_vals / count[:, np.newaxis]
                    else:
                        mean_vals = mean_vals / count
                    data_dict[var_name] = mean_vals.astype(data_dict[var_name].dtype)

        if self.return_inverse:
            data_dict["inverse"] = np.zeros_like(inverse)
            data_dict["inverse"][idx_sort] = inverse
        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord[first_point_idx]
        if self.return_min_coord:
            data_dict["min_coord"] = min_coord.reshape([1, 3])
        return data_dict


@TRANSFORMS.register_module()
class VoxelizeAggJIT(object):
    """Numba-accelerated Voxelization, eliminating expensive lexsorts."""
    def __init__(
        self,
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        follow_ref="segment",
        how_to_agg_feats=dict(
            coord="mean",
            color="mean",
            normal="mean",
            segment="mode",
        ),
    ):
        self.grid_size = grid_size
        self.hash = (
            transform.GridSample.fnv_hash_vec
            if hash_type == "fnv"
            else transform.GridSample.ravel_hash_vec
        )
        assert mode in ["train", "test"]
        self.mode = mode
        self.follow_ref = follow_ref

        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord

        self.how_to_agg_feats = how_to_agg_feats
        self.agg_func_names = deepcopy(how_to_agg_feats)

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()

        # To voxel indexes
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord

        # Save the min coord in original values
        min_coord = min_coord * np.array(self.grid_size)

        # Hash of the grid coords -> to group the unique voxel coords
        key = self.hash(grid_coord)
        
        # --- ROBUSTNESS FIX: Ensure all aggregation variables match coord length ---
        for var_name in self.agg_func_names.keys():
            if var_name in data_dict and data_dict[var_name].shape[0] < key.shape[0]:
                pad_size = key.shape[0] - data_dict[var_name].shape[0]
                pad_shape = list(data_dict[var_name].shape)
                pad_shape[0] = pad_size
                pad_val = -1 if var_name in ["segment", "label"] else 0
                pad_arr = np.full(pad_shape, pad_val, dtype=data_dict[var_name].dtype)
                data_dict[var_name] = np.concatenate([data_dict[var_name], pad_arr], axis=0)

        # 1. Base grouping and sorting (only one argsort needed)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        
        # Fast offsets calculation for JIT compatibility
        group_offsets = np.cumsum(np.insert(count, 0, 0)[0:-1])
        first_point_idx = idx_sort[group_offsets]
        
        # 2. Extract original mask if we have any follow-* instructions
        original_mask = None
        has_follow = any(f.startswith("follow-") for f in self.agg_func_names.values())
        if has_follow:
            original_mask = data_dict[self.follow_ref].copy()

        # 3. Process Aggregations
        for var_name, agg_func in self.agg_func_names.items():
            if var_name not in data_dict:
                continue

            if agg_func == "first":
                data_dict[var_name] = data_dict[var_name][first_point_idx]
            elif agg_func == "rand_choice":
                idx_select = idx_sort[
                    group_offsets
                    + np.random.randint(0, count.max(), count.size) % count
                ]
                data_dict[var_name] = data_dict[var_name][idx_select]
            elif agg_func == "mean":
                data_dict[var_name] = (
                    np.add.reduceat(
                        data_dict[var_name][idx_sort],
                        group_offsets,
                    )
                    / count[:, np.newaxis]
                )
            elif agg_func == "max":
                data_dict[var_name] = np.maximum.reduceat(
                    data_dict[var_name][idx_sort],
                    group_offsets,
                )
            elif agg_func == "min":
                data_dict[var_name] = np.minimum.reduceat(
                    data_dict[var_name][idx_sort],
                    group_offsets,
                )
            elif agg_func == "mode":
                # Fallback to 'first' to prevent untracked tensors from keeping size N
                data_dict[var_name] = data_dict[var_name][first_point_idx]
                
            # --- JIT FOLLOW-K ---
            elif agg_func.startswith("follow-"):
                if original_mask is not None:
                    # Resolve K dynamically per variable
                    k = 1 if agg_func == "follow-mask" else int(agg_func.split("-")[1])
                    feats = data_dict[var_name]
                    
                    if feats.ndim > 1:
                        out = np.zeros((len(count), feats.shape[1]), dtype=feats.dtype)
                        data_dict[var_name] = jit_follow_k_2d(feats, out, original_mask, k, count, idx_sort, group_offsets)
                    else:
                        out = np.zeros(len(count), dtype=feats.dtype)
                        data_dict[var_name] = jit_follow_k_1d(feats, out, original_mask, k, count, idx_sort, group_offsets)
                else:
                    # Fallback entirely to mean if there's no segment/mask array available
                    mean_vals = np.add.reduceat(
                        data_dict[var_name][idx_sort],
                        group_offsets,
                    )
                    if data_dict[var_name].ndim > 1:
                        mean_vals = mean_vals / count[:, np.newaxis]
                    else:
                        mean_vals = mean_vals / count
                    data_dict[var_name] = mean_vals.astype(data_dict[var_name].dtype)

        if self.return_inverse:
            data_dict["inverse"] = np.zeros_like(inverse)
            data_dict["inverse"][idx_sort] = inverse
        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord[first_point_idx]
        if self.return_min_coord:
            data_dict["min_coord"] = min_coord.reshape([1, 3])
        return data_dict