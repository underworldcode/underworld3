# %% [markdown]
# # Example Output Management Template
#
# This template shows how to properly configure examples to use the output directory
# for generated files while keeping the repository clean.

# %%
import os
import pathlib

# %% [markdown]
# ## Output Directory Setup
#
# Configure all file outputs to go to the `output/` directory which is excluded
# from version control.

# %%
# Get the directory containing this example
example_dir = pathlib.Path(__file__).parent if '__file__' in globals() else pathlib.Path.cwd()

# Create output directory relative to this example
output_dir = example_dir / "../output" / "example_category"  # e.g., heat_transfer
output_dir = output_dir.resolve()
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Example directory: {example_dir}")
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## File Path Configuration
#
# Configure all generated files to use the output directory:

# %%
# Mesh files
mesh_file = output_dir / "simulation_mesh.msh"
refined_mesh = output_dir / "refined_mesh.h5"

# Data files  
temperature_data = output_dir / "temperature_field.h5"
velocity_data = output_dir / "velocity_field.vtk"
checkpoint_file = output_dir / "checkpoint_final.h5"

# Visualization files
streamline_plot = output_dir / "velocity_streamlines.png"
temperature_contours = output_dir / "temperature_contours.html"
animation_file = output_dir / "convection_evolution.mp4"

# Analysis results
statistics_file = output_dir / "simulation_statistics.txt"
convergence_data = output_dir / "convergence_history.csv"

print("Configured file paths:")
for name, path in [
    ("Mesh", mesh_file),
    ("Temperature", temperature_data), 
    ("Visualization", streamline_plot),
    ("Statistics", statistics_file)
]:
    print(f"  {name}: {path}")

# %% [markdown]
# ## Using Assets for Documentation
#
# For images that should be version controlled (diagrams, expected results):

# %%
# Reference images that are part of documentation
assets_dir = example_dir / "../assets" / "example_category"
concept_diagram = assets_dir / "concept_illustration.png"
expected_result = assets_dir / "expected_temperature_distribution.png"

# Display reference image (this would be version controlled)
# from IPython.display import Image, display
# display(Image(str(expected_result)))

# %% [markdown]
# ## Best Practices Summary
#
# 1. **Generated files** → `output/` directory (excluded from git)
# 2. **Documentation images** → `assets/` directory (version controlled)
# 3. **Use pathlib** for cross-platform path handling
# 4. **Create directories** as needed with `mkdir(parents=True, exist_ok=True)`
# 5. **Relative paths** from example location, not hardcoded absolute paths

# %%
print("✅ Output management template complete!")
print(f"   Generated files will go to: {output_dir}")
print(f"   Documentation assets from: {assets_dir}")
print("   All examples should follow this pattern.")