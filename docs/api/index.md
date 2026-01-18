# Underworld3 API Reference

Underworld3 is a Python package for geodynamic modelling using PETSc finite elements.

## Core Modules

```{toctree}
:maxdepth: 2

meshing
discretisation
swarm
solvers
constitutive_models
function
scaling
```

## Supporting Modules

```{toctree}
:maxdepth: 2

coordinates
systems_ddt
maths
materials
model
utilities
visualisation
adaptivity
```

## Quick Links

### Mesh and Variables
- **{doc}`meshing`** - Create computational meshes (structured, unstructured, spherical)
- **{doc}`discretisation`** - Mesh variables and field data
- **{doc}`swarm`** - Particle swarms and Lagrangian tracking
- **{doc}`coordinates`** - Coordinate systems and transformations

### Solvers and Physics
- **{doc}`solvers`** - PDE solvers (Stokes, Poisson, advection-diffusion)
- **{doc}`constitutive_models`** - Material behaviour models (viscosity, diffusivity)
- **{doc}`systems_ddt`** - Time derivative discretisation
- **{doc}`materials`** - Multi-material systems

### Functions and Units
- **{doc}`function`** - Expressions, evaluation, and symbolic functions
- **{doc}`scaling`** - Units, quantities, and non-dimensionalisation
- **{doc}`maths`** - Mathematical operations and integrals

### Infrastructure
- **{doc}`model`** - Model management and configuration
- **{doc}`utilities`** - I/O, mesh import, and helper functions
- **{doc}`visualisation`** - Plotting and visualisation tools
- **{doc}`adaptivity`** - Adaptive mesh refinement (AMR)

## Indices

* {ref}`genindex`
* {ref}`modindex`
