# Underworld3 Examples

This directory contains physics-organized examples that demonstrate Underworld3 capabilities from basic concepts to advanced research applications.

## Organization by Physics Domain

Examples are organized by physics domain with progressive complexity:

### 🌡️ Heat Transfer & Diffusion (`heat_transfer/`)
- **Basic**: Steady-state heat conduction, simple diffusion
- **Intermediate**: Time-dependent heating, temperature-dependent properties  
- **Advanced**: Coupled thermal processes, phase changes

### 🌊 Fluid Mechanics (`fluid_mechanics/`)
- **Basic**: Stokes flow, driven cavity problems
- **Intermediate**: Variable viscosity, non-Newtonian fluids
- **Advanced**: Navier-Stokes, complex rheology

### 🔥 Thermal Convection (`convection/`)
- **Basic**: Rayleigh-Benard convection, 2D systems
- **Intermediate**: 3D convection, variable properties
- **Advanced**: Mantle convection, planetary-scale systems

### 🏔️ Solid Mechanics (`solid_mechanics/`)
- **Basic**: Elastic deformation, simple loading
- **Intermediate**: Visco-elastic materials, stress analysis
- **Advanced**: Plate tectonics, fault systems

### 💧 Porous Flow (`porous_flow/`)
- **Basic**: Darcy flow, groundwater systems
- **Intermediate**: Multi-phase flow, permeability variations
- **Advanced**: Magma migration, hydrothermal systems

### 🌍 Free Surface (`free_surface/`)
- **Basic**: Surface deformation under loads
- **Intermediate**: Dynamic topography
- **Advanced**: Erosion processes, landscape evolution

### 🔬 Multi-Physics (`multi_physics/`)
- **Intermediate**: Coupled thermal-mechanical systems
- **Advanced**: Full thermo-mechanical convection, reactive transport

### 🔧 Meshing & Utilities (`utilities/`)
- **Basic**: Mesh generation, boundary conditions
- **Intermediate**: Adaptive refinement, parallel processing
- **Advanced**: Custom mesh generation, performance optimization

### 🚧 Work in Progress (`WIP/`)
- **Developer tools**: PETSc/DMPlex internals, performance benchmarking
- **Advanced techniques**: Specialized meshing, advanced I/O, research methods
- **Legacy examples**: Recovered from underworld3-documentation, under integration

## Navigation

Each physics domain contains:
- **`README.md`**: Overview and learning progression
- **`basic/`**: Introductory examples with detailed explanations
- **`intermediate/`**: Multi-physics or advanced parameter examples  
- **`advanced/`**: Research-level examples with complex physics

## Example Format

All examples follow the Python percent format (Jupytext compatible):
- `# %%` cell separators for Jupyter compatibility
- `# %% [markdown]` cells for educational content
- Consistent parameter organization at file top
- Clear section markers for easy navigation

## Getting Started

1. **New to computational geophysics?** Start with `heat_transfer/basic/`
2. **Familiar with finite elements?** Jump to your physics domain of interest
3. **Research applications?** Check `advanced/` examples in relevant domains

## Output Management

### Generated Files (`output/`)
Examples create temporary files that are **automatically excluded from version control**:
- **Mesh files**: `*.msh`, `*.h5`, `*.vtk`, `*.xdmf`
- **Visualizations**: `*.png`, `*.html`, `*.mp4`  
- **Data files**: `*.txt`, `*.csv`, `checkpoint_*`

Configure examples to save to the `output/` directory:
```python
output_dir = "../output"  # Relative to example location
mesh_file = f"{output_dir}/simulation.msh"
```

### Documentation Images (`assets/`)
**Curated images for documentation** are version controlled:
- Concept diagrams and illustrations
- Expected benchmark results
- Reference visualizations

### Cleanup
```bash
# Clean all generated output
rm -rf docs/examples/output/*

# Examples will regenerate files when run
```

## Example Dependencies

All examples require:
- `underworld3` (this package)
- `numpy`, `sympy` (mathematical operations)
- `matplotlib` (basic visualization, optional)

Advanced examples may require additional packages as documented.