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

1. [**Meshes**](tutorials/1-Meshes.ipynb) - Creating and manipulating computational meshes
2. [**Variables**](tutorials/2-Variables.ipynb) - Defining fields on meshes and swarms  
3. [**Symbolic Forms**](tutorials/3-Symbolic_Forms.ipynb) - Mathematical expressions with SymPy
4. [**Poisson Solver**](tutorials/4-Solvers-i-Poisson.ipynb) - Solving diffusion problems
5. [**Stokes Flow**](tutorials/5-Solvers-ii-Stokes.ipynb) - Incompressible fluid dynamics
6. [**Timestepping**](tutorials/6a-Timestepping-simple.ipynb) - Evolution and time integration
7. [**Unsteady Flow**](tutorials/7-Unsteady_Flow.ipynb) - Time-dependent problems
8. [**Particle Swarms**](tutorials/8-Particle_Swarms.ipynb) - Lagrangian tracking

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
```