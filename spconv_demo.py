import numpy as np
from scipy.ndimage import convolve
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_double_spiral_point_cloud(grid_size=32):
    """Generates a double 3D spiral curve to create more local interactions."""
    t = np.linspace(0, 4 * np.pi, 200)
    
    # Spiral 1
    x1 = np.int32(np.clip(grid_size / 2 + (grid_size / 3) * np.cos(t), 0, grid_size-1))
    y1 = np.int32(np.clip(grid_size / 2 + (grid_size / 3) * np.sin(t), 0, grid_size-1))
    z1 = np.int32(np.clip(np.linspace(4, grid_size - 4, 200), 0, grid_size-1))
    
    # Spiral 2
    x2 = np.int32(np.clip(grid_size / 2 + (grid_size / 3) * np.cos(t + np.pi), 0, grid_size-1))
    y2 = np.int32(np.clip(grid_size / 2 + (grid_size / 3) * np.sin(t + np.pi), 0, grid_size-1))
    z2 = np.int32(np.clip(np.linspace( grid_size - 4,4, 200), 0, grid_size-1))
    
    grid = np.zeros((grid_size, grid_size, grid_size))
    
    grid[x1, y1, z1] = 1.0 
    grid[x2, y2, z2] = 0.2
    
    return grid

def extract_point_cloud(grid, threshold=1e-5):
    """Extracts X, Y, Z coordinates and values for non-zero voxels."""
    coords = np.argwhere(grid > threshold)
    values = grid[coords[:, 0], coords[:, 1], coords[:, 2]]
    return coords, values

# ---------------------------------------------------------
# 1. Setup Data and Kernel
# ---------------------------------------------------------
GRID_SIZE = 32
input_grid = generate_double_spiral_point_cloud(GRID_SIZE)

# 3x3x3 blurring/averaging kernel
blur_kernel = np.ones((3, 3, 3)) / 5

# ---------------------------------------------------------
# 2. Dense 3D Convolution
# ---------------------------------------------------------
dense_convolved_grid = convolve(input_grid, blur_kernel, mode='constant', cval=0.0)

# ---------------------------------------------------------
# 3. Submanifold Sparse Convolution
# ---------------------------------------------------------
sparse_convolved_grid = dense_convolved_grid * (input_grid > 0)

# ---------------------------------------------------------
# 4. Extract Results
# ---------------------------------------------------------
orig_coords, orig_vals = extract_point_cloud(input_grid)
dense_coords, dense_vals = extract_point_cloud(dense_convolved_grid)
sparse_coords, sparse_vals = extract_point_cloud(sparse_convolved_grid)

# ---------------------------------------------------------
# 5. Plotly Interactive Visualization (2x2 Layout with Equation)
# ---------------------------------------------------------
fig = make_subplots(
    rows=2, cols=2,
    # Use 'xy' for the second column of the first row to display 2D text
    specs=[[{'type': 'scene'}, {'type': 'xy'}], 
           [{'type': 'scene'}, {'type': 'scene'}]],
    vertical_spacing=0.05,
    horizontal_spacing=0.05,
    subplot_titles=(
        f"Input PC<br>(Points: {len(orig_coords)})",
        "Convolution Kernel",
        f"Dense Convolution<br>(Points: {len(dense_coords)})",
        f"Sparse Convolution<br>(Points: {len(sparse_coords)})"
    )
)

def add_trace(fig, coords, values, row_idx, col_idx, trace_idx):
    if len(coords) > 0:
        scatter = go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=5, 
                color=values,
                colorscale='Viridis',
                cmin=0, cmax=1,
                line=dict(color='black', width=2),
                colorbar=dict(
                    title=dict(text="Feature Value", font=dict(size=14)), 
                    tickfont=dict(size=12),                               
                    x=1.05, y=0.5, len=0.5 
                ) if trace_idx == 4 else None 
            ),
            name=f'Plot {trace_idx}'
        )
        fig.add_trace(scatter, row=row_idx, col=col_idx)

# Top Left: Original Input
add_trace(fig, orig_coords, orig_vals, row_idx=1, col_idx=1, trace_idx=1)

# Top Right: Kernel Equation (using a 2D text scatter trace)
fig.add_trace(
    go.Scatter(
        x=[0.5], y=[0.5],
        text=["K = 1<sub>3x3x3</sub> / 5"],
        mode="text",
        textfont=dict(size=32, color="black"),
        hoverinfo="skip"
    ),
    row=1, col=2
)

# Add a white background shape behind the equation text
fig.add_shape(
    type="rect",
    x0=0, y0=0, x1=1, y1=1,
    fillcolor="white",
    line=dict(width=1, color="black"),
    layer="below",
    row=1, col=2
)

# Bottom Left: Dense Output
add_trace(fig, dense_coords, dense_vals, row_idx=2, col_idx=1, trace_idx=3)

# Bottom Right: Sparse Output
add_trace(fig, sparse_coords, sparse_vals, row_idx=2, col_idx=2, trace_idx=4)

# Base invisible axis options for the 3D point clouds
base_axis_opts = dict(
    range=[0, GRID_SIZE], 
    autorange=False, 
    showbackground=False, 
    showgrid=False,       
    zeroline=False,       
    showticklabels=False, 
    ticks='',             
    title=dict(text='')   
)

scene_base = dict(xaxis=base_axis_opts, yaxis=base_axis_opts, zaxis=base_axis_opts, aspectmode='cube')

fig.update_layout(
    title=dict(text="", font=dict(size=24), x=0.5, xanchor='center'), 
    font=dict(size=16),
    scene=scene_base,
    scene2=scene_base, # Maps to the bottom-left 3D plot
    scene3=scene_base, # Maps to the bottom-right 3D plot
    showlegend=False,
    width=1000, height=900,
    margin=dict(r=20, l=20, b=40, t=100) 
)

# Hide the 2D axes for the equation cell (row 1, col 2)
fig.update_xaxes(visible=False, range=[0, 1], row=1, col=2)
fig.update_yaxes(visible=False, range=[0, 1], row=1, col=2)

# Explicitly loop through and increase the font size of the subplot annotations (titles)
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=16)

# Save to HTML file
file_name = "convolution_comparison_2x2_equation_bg.html"
fig.write_html(file_name)
print(f"Successfully saved 2x2 Plotly figure to {file_name}")