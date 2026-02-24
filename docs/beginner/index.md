---
title: "Getting Started with Underworld3"
---

# Getting Started

Welcome to Underworld3! This section provides everything you need to start building geodynamic models.

## Learning Path

### 1. Installation
Get Underworld3 running on your system in about 5 minutes:

```bash
git clone https://github.com/underworldcode/underworld3
cd underworld3
./uw setup
```

**[→ Installation Guide](installation.md)**

### 2. Quick Start
Understand the basics and see your first model.

**[→ Quick Start Tutorial](quickstart.md)**

### 3. Script Parameters
Configure your models for notebooks and command-line execution.

**[→ Parameters Guide](parameters.md)**

### 4. Interactive Tutorials
Work through hands-on notebooks covering all core concepts.

**[→ Start Tutorials](tutorials/Notebook_Index.ipynb)**

## Tutorial Sequence

Our tutorial notebooks build progressively:

**Fundamentals**

1. [**Meshes**](tutorials/1-Meshes.ipynb) — Creating and visualising computational meshes
2. [**Variables**](tutorials/2-Variables.ipynb) — Defining fields on meshes and swarms
3. [**Symbolic Forms**](tutorials/3-Symbolic_Forms.ipynb) — Mathematical expressions with SymPy

**Solvers**

4. [**Poisson Solver**](tutorials/4-Solvers-i-Poisson.ipynb) — Solving diffusion problems
5. [**Poisson Validation**](tutorials/5-Solvers-i-Poisson-Validation.ipynb) — Validating against analytical solutions
6. [**Stokes Flow**](tutorials/6-Solvers-ii-Stokes.ipynb) — Incompressible fluid dynamics

**Time Dependence**

7. [**Timestepping**](tutorials/7-Timestepping-simple.ipynb) — Advection-diffusion with analytical comparison
8. [**Coupled Timestepping**](tutorials/8-Timestepping-coupled.ipynb) — Stokes + thermal convection loop
9. [**Unsteady Flow**](tutorials/9-Unsteady_Flow.ipynb) — Navier-Stokes pipe flow

**Materials and Particles**

10. [**Particle Swarms**](tutorials/10-Particle_Swarms.ipynb) — Lagrangian tracking and swarm variables
11. [**Multi-Material Models**](tutorials/11-Multi-Material_SolCx.ipynb) — SolCx benchmark with index swarms

**Units and Scaling**

12. [**Units System**](tutorials/12-Units_System.ipynb) — Physical units with Pint
13. [**Non-Dimensional Scaling**](tutorials/13-Scaling-problems-with-physical-units.ipynb) — Reference quantities and ND solves
14. [**Timestepping with Units**](tutorials/14-Timestepping-with-physical-units.ipynb) — Advection-diffusion with physical units
15. [**Thermal Convection**](tutorials/15-Thermal-convection-with-units.ipynb) — Rayleigh-Benard in an annulus

## What's Next?

Once comfortable with the basics, explore:

- **[Advanced Usage](../advanced/index.md)** - Parallel computing, performance, complex physics
- **[Developer Guide](../developer/index.md)** - Understand internals and contribute

## Getting Help

- Browse the [complete documentation](../index.md)
- Report issues on [GitHub](https://github.com/underworldcode/underworld3/issues)
- Join community discussions

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
parameters
tutorials/Notebook_Index
tutorials/1-Meshes
tutorials/2-Variables
tutorials/3-Symbolic_Forms
tutorials/4-Solvers-i-Poisson
tutorials/5-Solvers-i-Poisson-Validation
tutorials/6-Solvers-ii-Stokes
tutorials/7-Timestepping-simple
tutorials/8-Timestepping-coupled
tutorials/9-Unsteady_Flow
tutorials/10-Particle_Swarms
tutorials/11-Multi-Material_SolCx
tutorials/12-Units_System
tutorials/13-Scaling-problems-with-physical-units
tutorials/14-Timestepping-with-physical-units
tutorials/15-Thermal-convection-with-units
```