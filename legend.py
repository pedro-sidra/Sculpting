import matplotlib.pyplot as plt
import matplotlib.patches as patches

# The 20 ScanNet classes and their corresponding RGB colors
scannet_classes = {
    "Wall": (255, 255, 255),
    "Floor": (166, 116, 4),
    "Cabinet": (85, 85, 0),
    "Bed": (0, 0, 255),
    "Chair": (255, 0, 0),
    "Sofa": (255, 0, 255),
    "Table": (255, 255, 0),
    "Door": (128, 0, 0),
    "Window": (14, 170, 255),
    "Bookshelf": (0, 0, 128),
    "Picture": (255, 170, 255),
    "Counter": (192, 192, 192),
    "Desk": (193, 230, 125),
    "Curtain": (191, 231, 205),
    "Refridgerator": (128, 128, 128),
    "Shower Curtain": (50, 255, 198),
    "Toilet": (255, 250, 250),
    "Sink": (0, 128, 128),
    "Bathtub": (69, 229, 0),
    "Otherfurniture": (0, 0, 0)
}

# Configure font to match standard typesetting
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12

# Create a figure with no axes
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

legend_patches = []
for label, rgb in scannet_classes.items():
    # Normalize RGB to [0.0, 1.0] for matplotlib
    norm_color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    
    # Create a patch for each class. 
    # The black edge ensures white/light colors are visible.
    patch = patches.Patch(
        facecolor=norm_color, 
        edgecolor='black', 
        linewidth=0.5, 
        label=label
    )
    legend_patches.append(patch)

# Add the legend to the axis
# ncol=4 organizes it into a neat horizontal grid; adjust as needed
legend = ax.legend(
    handles=legend_patches, 
    loc='center', 
    ncol=4, 
    frameon=False, 
    handlelength=1.5, 
    handleheight=1.2,
    columnspacing=2.0,
    labelspacing=0.8
)

# Save as a PDF vector graphic for lossless scaling
output_filename = 'scannet_legend.png'
plt.savefig(output_filename, format='png', bbox_inches='tight', dpi=300)
print(f"Legend successfully generated and saved to {output_filename}")

# Uncomment the line below if you want to preview it in a window
# plt.show()