import numpy as np
import torch
from copy import deepcopy

import pointcept.datasets.transform as transform
from pointcept.datasets.transform import TRANSFORMS


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
        
        # --- FOLLOW-K: SORTING ENHANCEMENT ---
        original_mask = None
        
        # Check if any aggregation function uses the follow-* pattern
        follow_mode = next((f for f in self.agg_func_names.values() if f.startswith("follow-")), None)
        
        if follow_mode is not None:
            mask_ref = data_dict.get(self.follow_ref,None)
            if mask_ref is not None:
                # Copy mask BEFORE loop so it isn't overwritten by size-V aggregations
                original_mask = mask_ref.copy()
                
                # Determine K from 'follow-{K}' or default to 1 for 'follow-mask'
                if follow_mode == "follow-mask":
                    k = 1
                else:
                    try:
                        k = int(follow_mode.split("-")[1])
                    except ValueError:
                        k = 1
                
                # Sort primarily by 'key' (grouping voxels) and secondarily by whether label == K.
                # A negative boolean puts True (-1) before False (0), naturally pulling label K to the front.
                is_label_k = (original_mask.reshape(-1) == k).astype(np.int32)
                idx_sort = np.lexsort((-is_label_k, key))
            else:
                idx_sort = np.argsort(key)
        else:
            idx_sort = np.argsort(key)
            
        key_sort = key[idx_sort]

        # unique values of the key
        # inverse: mapping from points to voxels (p2v_map)
        # count: points per voxel
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        # mapping from voxels to a single point (v2p_map)
        # Thanks to lexsort, if the voxel has a label==K point, it will automatically be at 'first_point_idx'!
        first_point_idx = idx_sort[np.cumsum(np.insert(count, 0, 0)[0:-1])]

        for var_name, agg_func in self.agg_func_names.items():
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
                
            # --- FOLLOW-K: AGGREGATION ENHANCEMENT ---
            elif agg_func.startswith("follow-"):
                if original_mask is not None:
                    # Resolve K dynamically per variable if they request different K's 
                    # (though sorting is bound to the first one found above)
                    if agg_func == "follow-mask":
                        k = 1
                    else:
                        try:
                            k = int(agg_func.split("-")[1])
                        except ValueError:
                            k = 1
                            
                    # Use original_mask (size N) to avoid IndexError if data_dict['segment'] shrunk
                    has_mask_k = (original_mask[first_point_idx] == k)
                    exact_vals = data_dict[var_name][first_point_idx]
                    
                    # Calculate mean as the fallback for voxels entirely without label K
                    mean_vals = np.add.reduceat(
                        data_dict[var_name][idx_sort],
                        np.cumsum(np.insert(count, 0, 0)[0:-1]),
                    )
                    
                    # Handle shapes seamlessly via reshape(-1, 1) and reshape(-1)
                    if data_dict[var_name].ndim > 1:
                        mean_vals = mean_vals / count[:, np.newaxis]
                        mask_cond = has_mask_k.reshape(-1, 1)
                    else:
                        mean_vals = mean_vals / count
                        mask_cond = has_mask_k.reshape(-1)
                        
                    mean_vals = mean_vals.astype(exact_vals.dtype)
                    data_dict[var_name] = np.where(mask_cond, exact_vals, mean_vals)
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
        data_dict['grid_size']=self.grid_size
        return data_dict