---
title: "Underworld3 Documentation"
---

# Welcome to Underworld3

**Computational Geodynamics Made Accessible**

Underworld3 is a Python library for computational geodynamics, providing finite element modeling for Earth science research.

```{image} media/SocialShareS.png
:alt: Underworld3 visualization examples
:width: 100%
:align: center
```

````{grid} 1 1 2 3
:gutter: 3

```{grid-item-card} Getting Started
:link: beginner/index
:link-type: doc

**For new users**

Learn Underworld3 fundamentals through interactive tutorials and hands-on examples.

- Install and configure Underworld3
- Create your first geodynamic model
- Work through progressive tutorials
- Understand meshes, variables, and solvers

+++
[Start Learning](beginner/index.md)
```

```{grid-item-card} Advanced Usage

**For researchers**

Master parallel computing, optimization, and complex physics for research-grade simulations.

- Write parallel-safe code using UW3 API
- Understand collective operations in PETSc
- Optimize simulation performance
- Implement complex rheologies

+++
*Documentation in progress*
```

```{grid-item-card} Developer Guide
:link: developer/index
:link-type: doc

**For contributors**

Understand the architecture, implementation details, and contribute to Underworld3 development.

- Architecture and design patterns
- Create new solvers and features
- Follow coding standards
- Contribute effectively

+++
[Developer Documentation](developer/index.md)
```

````

## Quick Navigation

### By Experience Level

````{grid} 1 1 3 3
:gutter: 2

```{grid-item}
**I'm new to Underworld3**

1. [Installation Guide](beginner/installation.md)
2. [Tutorial Notebooks](beginner/tutorials/Notebook_Index.ipynb)
3. [Next Steps](beginner/quickstart.md)
```

```{grid-item}
**I'm working with the code**

1. [Examples Repository](https://github.com/underworldcode/underworld3/tree/development/docs/examples)
2. [Benchmarks](https://github.com/underworld-community/UW3-benchmarks)
3. [API Documentation](api/index.md)
```

```{grid-item}
**I'm ready for research**

1. [Parallel Computing Guide](advanced/parallel-computing.md)
2. [Performance Tips](advanced/performance.md)
3. [Troubleshooting](advanced/troubleshooting.md)
```

````

### By Topic

````{grid} 2 2 4 4
:gutter: 2

```{grid-item}
**Basics**

- [Meshes](beginner/tutorials/1-Meshes.ipynb)
- [Variables](beginner/tutorials/2-Variables.ipynb)
- [Solvers](beginner/tutorials/4-Solvers-i-Poisson.ipynb)
```

```{grid-item}
**Physics**

- [Stokes Flow](beginner/tutorials/6-Solvers-ii-Stokes.ipynb)
- [Timestepping](beginner/tutorials/7-Timestepping-simple.ipynb)
- [Particle Swarms](beginner/tutorials/10-Particle_Swarms.ipynb)
```

```{grid-item}
**Advanced**

- [Complex Rheologies](advanced/complex-rheologies.md)
- [Mesh Adaptation](advanced/mesh-adaptation.md)
- [API Patterns](advanced/api-patterns.md)
```

```{grid-item}
**Reference**

- [API Overview](api/index.md)
- [Solvers API](api/solvers.md)
- [Meshing API](api/meshing.md)
```

````

## About Underworld3

Underworld3 is a Python library for computational geodynamics, built on:

- **PETSc** for scalable parallel finite element methods
- **SymPy** for mathematical self-description
- **Particle-in-cell methods** for Lagrangian tracking
- **Natural mathematical syntax** for intuitive model development

### Getting Help

- [Report issues on GitHub](https://github.com/underworldcode/underworld3/issues)
- [Browse API documentation](api/index.md)
- Join community discussions
- [Underworld Blog](https://www.underworldcode.org)

### Try It Now

Launch interactive tutorials directly in your browser:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/development?labpath=underworld3%2Fdocs%2Fbeginner%2Ftutorials%2FNotebook_Index.ipynb)

```{toctree}
:maxdepth: 2
:hidden:

beginner/index
advanced/index
developer/index
api/index
```
