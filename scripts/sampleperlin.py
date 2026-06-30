#%%
import pandas as pd
from pypcd.pypcd import pandas_to_pypcd, encode_rgb_for_pcl
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from perlyn import generate_perlin_noise

def integral_from_minus_eps_to_eps(samples, epsilons):
    """Calculates the probability mass of the noise values around zero."""
    samples = np.asarray(samples).ravel()
    return np.array([np.mean(np.abs(samples) <= eps) for eps in epsilons])

def save_pcd(points_list, colors_list, noise_list, filename):
    """Helper function to compile and save Point Cloud Data."""
    if not points_list:
        print(f"No points to save for {filename}.")
        return
        
    combined_points = 0.02 * np.vstack(points_list)
    combined_colors = np.vstack(colors_list)
    combined_noise = np.concatenate(noise_list) # Concatenate the 1D noise arrays

    # Multiply by 255 before casting to uint8 so colors are scaled correctly for pypcd
    rgb = encode_rgb_for_pcl((combined_colors * 255).astype(np.uint8))

    # We add 'noise' here so pypcd registers it as a standard scalar field for CloudCompare
    pc_data = pd.DataFrame(dict(
        x=combined_points[:,0],
        y=combined_points[:,1],
        z=combined_points[:,2],
        rgb=rgb,
        noise=combined_noise 
    ))

    pandas_to_pypcd(pc_data).save_pcd(filename, compression="binary_compressed")
    print(f"Success! Saved to: '{filename}'")

# --- Configuration ---
target_shapes = [20, 15, 10, 5]
resolutions = [1, 2, 3, 5] # Varying resolutions for the first PCD
fixed_res = (2, 2, 2)
fixed_thresh = 0.04
seed = 42
num_mc_runs = 50

spacing = 5.0
grid_step = max(target_shapes) + spacing

# =====================================================================
# 1. Monte-Carlo Statistical Analysis
# =====================================================================
print("\n--- Running Monte-Carlo Statistical Analysis ---")
plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18})
plot_resolutions = [1, 2, 3, 5]
num_res = len(plot_resolutions)
fig, axes = plt.subplots(num_res, 2, figsize=(15, 6 * num_res))

# Ensure axes is 2D even if there's only 1 resolution
if num_res == 1:
    axes = np.array([axes])

good_thresholds = [] # Store heuristically calculated thresholds here

for row, res in enumerate(plot_resolutions):
    ax1 = axes[row, 0]
    ax2 = axes[row, 1]
    current_res = (res, res, res)
    print(f"\n-> Processing Monte-Carlo stats for Resolution: {current_res}")
    
    for i, size in enumerate(target_shapes):
        shape = (size, size, size)
        
        print(f"  -> Processing stats for shape: {shape}")
        
        all_counts = np.zeros(100)
        epsilons = np.linspace(0, 1.0, 300)
        all_integrals = np.zeros(300)
        
        for mc_seed in range(seed, seed + num_mc_runs):
            noise = generate_perlin_noise(shape=shape, resolution=current_res, seed=mc_seed)
            
            counts, bins = np.histogram(noise.ravel(), bins=100, range=(-1, 1), density=True)
            all_counts += counts
            all_integrals += integral_from_minus_eps_to_eps(noise, epsilons)
        
        # Average the distributions over the Monte-Carlo runs
        avg_counts = all_counts / num_mc_runs
        avg_integrals = all_integrals / num_mc_runs

        # Plot the averaged distributions
        bin_centers = (bins[:-1] + bins[1:]) / 2
        line, = ax1.plot(bin_centers, avg_counts, label=f"s={shape[0]}")
        line_color = line.get_color()
        ax2.plot(epsilons, avg_integrals, label=f"s={shape[0]}", color=line_color)

        # --- Heuristic Plotting ---
        # Only display thresholds for resolution 2 as requested
        if res == 2:
            baseline_shape = 20
            baseline_tau = 0.05
            
            # 1. Find the probability mass (CDF value) at the baseline threshold
            p0 = np.interp(baseline_tau, epsilons, avg_integrals)
            
            # 2. Scale target probability mass inversely with shape
            p_target = p0 * (baseline_shape / shape[0])
            
            # 3. Reverse CDF lookup: Find threshold yielding target probability
            # np.interp works here because avg_integrals (CDF) is monotonically increasing
            heuristic_thresh = float(np.interp(p_target, avg_integrals, epsilons))
            percent_captured = p_target
            
            # Append the calculated threshold so we can use it for the point cloud generation later
            good_thresholds.append(round(heuristic_thresh, 4))
            
            ax2.scatter(heuristic_thresh, percent_captured, color=line_color, zorder=5)
            ax2.annotate(rf"$\Gamma={heuristic_thresh:.3f}$", 
                         (heuristic_thresh, percent_captured), 
                         textcoords="offset points", 
                         xytext=(5, -10), 
                         fontsize=18, 
                         fontweight='bold',
                         color=line_color)
            
            # Linha horizontal até o eixo Y
            ax2.hlines(y=percent_captured, xmin=0, xmax=heuristic_thresh, color=line_color, linestyle='--', alpha=0.7)

    # Finalize subplots for this resolution
    ax1.set_xlabel(r"$N(x)$")
    ax1.set_ylabel(r"$p(N(x))$")
    ax1.set_title(f"PDF of N(x), G={res}")
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylabel(r"$P(|N(x)| \leq \epsilon)$")
    ax2.set_title(f"CDF of |N(x)|, G = {res}^3")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True)

plt.tight_layout()
plt.savefig("perlin_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================================================
# 2. Point Cloud Variant: Shape x Resolution
# =====================================================================
print("\n--- Generating Shape x Resolution Point Cloud ---")
pts_res = []
cols_res = []
noise_res = [] # Track scalar field values

for i, size in enumerate(target_shapes):
    for j, res in enumerate(resolutions):
        shape = (size, size, size)
        current_res = (res, res, res)
        
        noise = generate_perlin_noise(shape=shape, resolution=current_res, seed=seed)
        mask = np.abs(noise) < fixed_thresh
        
        if np.any(mask):
            X, Y, Z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
            
            # Center the block coordinates locally, then translate to grid position
            pts_x = X[mask] - (shape[0] - 1) / 2.0 + (i * grid_step)
            pts_y = Y[mask] - (shape[1] - 1) / 2.0 + (j * grid_step)
            pts_z = Z[mask] - (shape[2] - 1) / 2.0
            
            pts_res.append(np.vstack((pts_x, pts_y, pts_z)).T)
            
            # Store RGB based on the normalization
            norm_noise = (noise[mask] + 1.0) / 2.0
            cols_res.append(plt.cm.viridis(norm_noise)[:, :3])
            
            # Store the raw scalar noise values
            noise_res.append(noise[mask])

save_pcd(pts_res, cols_res, noise_res, "perlin_res.pcd")

# =====================================================================
# 3. Point Cloud Variant: Shape x Threshold
# =====================================================================
print("\n--- Generating Shape x Threshold Point Cloud ---")
pts_thresh = []
cols_thresh = []
noise_thresh = [] # Track scalar field values

# Build thresholds combining manual baselines and empirically derived limits
thresholds = [0.01] + good_thresholds + [0.2, 0.5]
print(f"Using mapped thresholds: {thresholds}")

for i, size in enumerate(target_shapes):
    shape = (size, size, size)
    
    # Generate noise once per shape for the threshold grid
    noise = generate_perlin_noise(shape=shape, resolution=fixed_res, seed=seed)
    X, Y, Z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    for j, thresh in enumerate(thresholds):
        mask = np.abs(noise) < thresh  
        
        if np.any(mask):
            # Center the block coordinates locally, then translate to grid position
            pts_x = X[mask] - (shape[0] - 1) / 2.0 + (i * grid_step)
            pts_y = Y[mask] - (shape[1] - 1) / 2.0 + (j * grid_step)
            pts_z = Z[mask] - (shape[2] - 1) / 2.0
            
            pts_thresh.append(np.vstack((pts_x, pts_y, pts_z)).T)
            
            # Store RGB based on the normalization
            norm_noise = (noise[mask] + 1.0) / 2.0
            cols_thresh.append(plt.cm.viridis(norm_noise)[:, :3])
            
            # Store the raw scalar noise values
            noise_thresh.append(noise[mask])

save_pcd(pts_thresh, cols_thresh, noise_thresh, "perlin_thresh.pcd")
#%%