---
title: "Underworld3 Developer Documentation"
subtitle: "Complete Guide to Architecture, Implementation, and Development"
---

# Welcome to Underworld3 Developer Documentation

This comprehensive documentation provides everything you need to understand, contribute to, and extend the Underworld3 computational geodynamics framework.

## What is Underworld3?

Underworld3 is a Python library for computational geodynamics and geophysical modeling, built on a foundation of:

- **PETSc parallel computing** for scalable finite element methods
- **SymPy symbolic mathematics** for mathematical self-description  
- **Particle-in-cell methods** for Lagrangian tracking
- **Natural mathematical syntax** for intuitive model development

## Getting Started

```{tip} Quick Navigation for Developers
- **New to UW3?** → Start with [Development Setup](development-setup.md)
- **Understanding the codebase?** → Read [Architecture Overview](UW3_Architecture_and_Documentation_Overview.md)
- **Working with math expressions?** → See [Mathematical Objects](UW3_Developers_MathematicalObjects.md)
- **Handling data arrays?** → Check [NDArray System](UW3_Developers_NDArrays.md)
- **Need coding standards?** → Reference [Style Guide](UW3_Style_and_Patterns_Guide.md)
```

## Documentation Structure

This documentation is organized into focused sections:

### Core Computational Systems
The foundation of Underworld3's numerical capabilities:

- **Meshing**: Geometric mesh generation and manipulation
- **Discretisation**: Finite element infrastructure and field management
- **Solvers**: PDE solvers and numerical methods
- **Constitutive Models**: Material physics and rheology

### Data Management Systems  
How Underworld3 handles data across parallel systems:

- **NDArray System**: Modern data access patterns (v0.99+)
- **Swarm System**: Particle tracking and Lagrangian methods
- **Variables**: Mesh and swarm field variables
- **Parallel Data**: MPI and distributed memory management

### Mathematical & Symbolic Systems
The mathematical foundation enabling natural syntax:

- **Mathematical Objects**: Natural math syntax implementation
- **Expressions & Functions**: Symbolic expression handling
- **Mathematics**: Vector calculus and tensor operations
- **Coordinate Systems**: Transformations and spatial handling

### Utility & Extension Systems
Supporting infrastructure and extensibility:

- **Visualization**: Plotting and rendering capabilities
- **Scaling**: Dimensional analysis and non-dimensionalization  
- **Cython Integration**: Performance-critical compiled extensions
- **JIT Compilation**: Just-in-time code generation

## Current Architecture Status

Based on our comprehensive analysis, here's the current state of Underworld3 subsystems. Areas requiring significant documentation work are marked with (§).

### Well-Documented Subsystems
- **Discretisation** (4,145 lines) - Finite element infrastructure
- **Solvers** (3,907 lines) - PDE solving capabilities  
- **Swarm** (4,484 lines) - Particle systems and tracking
- **NDArray** (946 lines) - Modern data access (v0.99+)
- **Constitutive Models** (1,967 lines) - Material physics

### Partially Documented Subsystems
- **Meshing** (4,437 lines) - Needs geometric parameter details (§)
- **Mathematics** (962 lines) - Inconsistent documentation quality (§)
- **Visualization** (788 lines) - Basic documentation present

### Documentation Gaps
- **Expressions/Functions** (1,192 lines) - Critical user-facing gap (§)
- **Scaling** (208 lines) - Minimal documentation (§)
- **Cython Integration** - No documentation for C bridge (§)

## Contributing to Documentation

We welcome contributions to improve this documentation! Each subsystem page includes:

- **Current implementation status**
- **API reference and examples**  
- **Developer guidelines**
- **Performance considerations**
- **Testing patterns**

### Documentation Priorities

Based on our analysis, these are the highest priority areas for contribution:

1. **Critical**: [Expressions & Functions](subsystems/expressions-functions.md) - Core user functionality
2. **Critical**: [Cython Integration](subsystems/cython-integration.md) - Bridge documentation missing
3. **High**: [Meshing](subsystems/meshing.md) - Geometric parameter guidance
4. **High**: [Mathematics](subsystems/mathematics.md) - Consistency improvements

## Development Workflow

Underworld3 follows established patterns for maintaining code quality:

```
Feature Branch → Development → Testing → Documentation → Code Review → Integration
```

### Key Principles

1. **Solver Stability**: Preserve performance of validated numerical methods
2. **Backward Compatibility**: Smooth migration paths for API changes  
3. **Mathematical Clarity**: Code should look like mathematical equations
4. **Parallel Safety**: All operations designed for distributed computing
5. **Progressive Enhancement**: New features coexist with legacy patterns

## Quick Reference

### Common Development Tasks

| Task | Documentation | Quick Link |
|------|---------------|------------|
| Set up development environment | [Development Setup](development-setup.md) | Environment configuration |
| Understand data access patterns | [NDArray System](UW3_Developers_NDArrays.md) | `var.array[...] = values` |
| Write mathematical expressions | [Mathematical Objects](UW3_Developers_MathematicalObjects.md) | `momentum = density * velocity` |
| Follow coding standards | [Style Guide](UW3_Style_and_Patterns_Guide.md) | Patterns and conventions |
| Add new functionality | [Contributing](contributing.md) | Development workflow |
| Debug parallel issues | [MPI Parallelism](advanced/mpi-parallelism.md) | Parallel debugging |
| Optimize performance | [Performance Guide](guidelines/performance-optimization.md) | Profiling and optimization |

### Build and Test Commands

```bash
# Build Underworld3 with pixi
pixi run underworld-build

# Run test suite
pixi run underworld-test

# Build documentation
pixi run docs-build
```

## Community and Support

- **Repository**: [github.com/underworldcode/underworld3](https://github.com/underworldcode/underworld3)
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join the community for questions and collaboration
- **Documentation**: Help improve these docs through pull requests

---

```{note} About This Documentation
This developer documentation covers Underworld3 version 0.99+. It includes both conceptual guides and practical implementation details for contributing to the Underworld3 ecosystem.
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

development-setup
contributing
UW3_Style_and_Patterns_Guide
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Architecture

UW3_Architecture_and_Documentation_Overview
UW3_Developers_MathematicalObjects
UW3_Developers_NDArrays
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Subsystems

subsystems/meshing
subsystems/discretisation
subsystems/solvers
subsystems/constitutive-models
subsystems/swarm-system
subsystems/data-access
subsystems/expressions-functions
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Advanced Topics

advanced/solver-development
advanced/mpi-parallelism
advanced/petsc-integration
```