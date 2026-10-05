import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_am_data():
    """
    Generates a 2D point cloud with a base shape (circle) and applies
    Additive Masking according to the rules of the dissertation.
    """
    np.random.seed(42)
    
    # 1. Base Shape: A Ring (Label 0: Original)
    n_shape = 3000
    angles = np.random.uniform(0, 2 * np.pi, n_shape)
    radii = 5.0 + np.random.normal(0, 0.3, n_shape)
    x_shape = radii * np.cos(angles)
    y_shape = radii * np.sin(angles)
    points_shape = np.column_stack([x_shape, y_shape])
    
    # Initialize all base shape points as 'Original' (0)[cite: 1]
    labels_shape = np.zeros(n_shape, dtype=int) 
    
    # 2. Define Masking Region R and relabel
    # Let R be a square bounding box on the right side of the ring
    mask_cond = (points_shape[:, 0] >= 2.0) & (points_shape[:, 0] <= 7.0) & \
                (points_shape[:, 1] >= -2.5) & (points_shape[:, 1] <= 2.5)
    
    # Points inside R become 'Masked' (2)[cite: 1]
    labels_shape[mask_cond] = 2 
    
    # 3. Additive Masking: Sculpting a dense block (Label 1: Occlusion)
    n_mask = 5000
    x_mask = np.random.uniform(2.0, 7.0, n_mask)
    y_mask = np.random.uniform(-2.5, 2.5, n_mask)
    points_mask = np.column_stack([x_mask, y_mask])
    
    # Added points are labeled as 'Occlusion' (1)[cite: 1]
    labels_mask = np.ones(n_mask, dtype=int) 
    
    # 4. Combine datasets to simulate the corrupted input point cloud
    points = np.vstack([points_shape, points_mask])
    labels = np.concatenate([labels_shape, labels_mask])
    
    return points, labels

def voxelize_and_aggregate(points, labels, voxel_size):
    """
    Groups points into voxels and assigns labels based on max() priority.
    """
    # Map points to discrete voxel grid indices[cite: 1]
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Dictionary to group points by voxel index
    voxel_dict = {}
    for pt, lbl, v_idx in zip(points, labels, voxel_indices):
        v_tuple = tuple(v_idx)
        if v_tuple not in voxel_dict:
            voxel_dict[v_tuple] = {'pts': [], 'lbls': []}
        voxel_dict[v_tuple]['pts'].append(pt)
        voxel_dict[v_tuple]['lbls'].append(lbl)
        
    vox_centers = []
    vox_labels = []
    
    # Apply aggregation functions
    for v_tuple, data in voxel_dict.items():
        # Aggregation for coordinates: Mean of points inside voxel
        vox_centers.append(np.mean(data['pts'], axis=0))
        
        # Aggregation for labels: Max numerical value (Priority sampling)[cite: 1]
        # 0 (Original) < 1 (Occlusion) < 2 (Masked)[cite: 1]
        vox_labels.append(np.max(data['lbls']))
        
    return np.array(vox_centers), np.array(vox_labels)

def main():
    # Parameters
    voxel_size = 0.5
    
    # Generate and process data
    points, labels = generate_am_data()
    vox_points, vox_labels = voxelize_and_aggregate(points, labels, voxel_size)
    
    # Setup visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    
    # Define Colormap for the 3 labels
    # Label 0: Blue (Original), Label 1: Gray (Occlusion), Label 2: Orange (Masked)
    colors = ['#1f77b4', '#7f7f7f', '#ff7f0e']
    cmap = ListedColormap(colors) 
    
    # Plot 1: Raw Point Cloud
    ax1.scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap, s=2, alpha=0.6)
    ax1.set_title("Additive Masking - Point Cloud")
    ax1.set_aspect('equal')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_xticks([]) # Remove x tick labels
    ax1.set_yticks([]) # Remove y tick labels
    
    # Plot 2: Voxelized Point Cloud
    ax2.scatter(vox_points[:, 0], vox_points[:, 1], c=vox_labels, cmap=cmap, s=60, marker='s')
    ax2.set_title("Additive Masking - Voxels")
    ax2.set_aspect('equal')
    ax2.set_xlabel("X")
    ax2.set_xticks([]) # Remove x tick labels
    ax2.set_yticks([]) # Remove y tick labels
    
    # Color Legend Handles
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=10, label='Original (0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=10, label='Occlusion (1)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=10, label='Masked (2)')
    ]
    
    # Add legends to each individual subplot at the bottom left
    ax1.legend(handles=handles, loc='lower left', title="Color Legend")
    ax2.legend(handles=handles, loc='lower left', title="Color Legend")
    
    plt.tight_layout()
    plt.savefig("AM.png")

if __name__ == "__main__":
    main()