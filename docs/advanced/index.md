---
title: "Advanced Usage"
---

# Advanced Usage

Master advanced Underworld3 techniques for research-grade simulations.

## Topics

### Parallel Computing
Write MPI-safe scripts and understand domain decomposition.

**[→ Parallel Computing Guide](parallel-computing.md)**

### Performance Optimization  
Profile, optimize, and scale your simulations.

**[→ Performance Guide](performance.md)**

### Multigrid Preconditioning (FMG vs GAMG)
Robust, anisotropy-tolerant solves on adapted meshes — build a mesh with
`refinement` and the solver uses geometric Full Multigrid automatically.

**[→ Multigrid Preconditioning](multigrid-preconditioning.md)**

### Complex Rheologies
Implement advanced material models and constitutive laws.

**[→ Complex Rheologies](complex-rheologies.md)**

### VEP with Transverse Isotropy for Fault Mechanics
Viscoelastic-plastic rheology with anisotropic weak planes and resolved
fault-plane yield for modelling fault zones.

**[→ VEP + Transverse Isotropy](vep-transverse-isotropy-faults.md)**

### Fault Zones That Cross
Finite-width fault zones fused into one region, and what the rheology of
the overlap is worth.

**[→ Crossing Fault Zones](crossing-fault-zones.md)**

### Gouge Zones
What a fault zone's width is for, and what collapsing it to a surface
throws away.

**[→ Gouge Zones](gouge-zones.md)**

### Custom Meshes
Create complex geometries with gmsh for research problems.

**[→ Custom Mesh Creation](custom-meshes.md)**

### Boundary Conditions on Curved Surfaces
Accurate free-slip and Neumann conditions on elliptical and non-planar boundaries.

**[→ Curved Boundary Conditions](curved-boundary-conditions.md)**

### Mesh Adaptation
Dynamic remeshing and adaptive refinement strategies.

**[→ Mesh Adaptation](mesh-adaptation.md)**

### Semi-Lagrangian Time Integration (SLCN / SL-BDF2)
How `AdvDiffusionSLCN` discretizes advection–diffusion in time: the BDF
time-derivative and Adams-Moulton/θ flux knobs, and how to pair them
(SLCN vs SL-BDF2).

**[→ Semi-Lagrangian Time Integration](semi-lagrangian-time-integration.md)**

### Porous Media Flow
Darcy flow, Richards equation, and variably-saturated groundwater modelling.

**[→ Porous Media Flow](porous-flow.md)**

### State Snapshots & Restore
A "stash for timesteps": snapshot the full model state, try a step,
restore exactly if you don't like it. For backtracking, adaptive Δt,
and predictor–corrector workflows.

**[→ State Snapshots & Restore](snapshot-restore.md)**

### Troubleshooting
Common issues, debugging strategies, and solutions.

**[→ Troubleshooting Guide](troubleshooting.md)**

## API Patterns

Understanding common design patterns helps you write better Underworld3 code.

**[→ API Patterns](api-patterns.md)**

## Prerequisites

This section assumes familiarity with:
- [Getting Started tutorials](../beginner/index.md)
- Python programming
- Basic parallel computing concepts

## Next Steps

Ready to contribute to Underworld3?

**[→ Developer Guide](../developer/index.md)**

```{toctree}
:maxdepth: 2
:hidden:

parallel-computing
performance
multigrid-preconditioning
solver-iteration-callbacks
complex-rheologies
vep-transverse-isotropy-faults
split-node-faults
fault-networks
crossing-fault-zones
gouge-zones
fault-mechanics-examples
custom-meshes
curved-boundary-conditions
mesh-adaptation
semi-lagrangian-time-integration
porous-flow
snapshot-restore
troubleshooting
api-patterns
SWARM-INTEGRATION-STATISTICS
```