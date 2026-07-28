---
title: "Underworld3 Developer Documentation"
subtitle: "Complete Guide to Architecture, Implementation, and Development"
---

# Developer Documentation

This comprehensive documentation provides everything you need to understand, contribute to, and extend the Underworld3 computational geodynamics framework.

## What is Underworld3?

Underworld3 is a Python library for computational geodynamics and geophysical modeling, built on a foundation of:

- **PETSc parallel computing** for scalable finite element methods
- **SymPy symbolic mathematics** for mathematical self-description  
- **Particle-in-cell methods** for Lagrangian tracking
- **Natural mathematical syntax** for intuitive model development

## Getting Started

```{tip} Quick Navigation for Developers
- **New to UW3?** → Start with [Development Setup](guides/development-setup.md)
- **Working with math expressions?** → See [Mathematical Objects](UW3_Developers_MathematicalObjects.md)
- **Handling data arrays?** → Check [NDArray System](UW3_Developers_NDArrays.md)
- **Need coding standards?** → Reference [Style Guide](UW3_Style_and_Patterns_Guide.md)
```

## Documentation Authority Map

One governing document per topic (this records the [UW3 Style Charter](UW3_STYLE_CHARTER.md)
§10 authority map — the Charter itself wins on any conflict). Other documents on the
same topic are reference or historical material subordinate to the governing document.

| Topic | Governing document |
|-------|--------------------|
| Coding style & API conventions | [UW3 Style Charter](UW3_STYLE_CHARTER.md) (detailed reference: [Style Guide](UW3_Style_and_Patterns_Guide.md)) |
| Data access | [subsystems/data-access.md](subsystems/data-access.md) (internals reference: [NDArray System](UW3_Developers_NDArrays.md)) |
| Local scattered-point interpolation | [subsystems/interpolation.md](subsystems/interpolation.md) |
| Rotated free-slip & wall-normal datum | [subsystems/rotated-freeslip.md](subsystems/rotated-freeslip.md) |
| Units | [design/UNITS_SIMPLIFIED_DESIGN_2025-11.md](design/UNITS_SIMPLIFIED_DESIGN_2025-11.md) |
| Testing tiers | [TESTING-RELIABILITY-SYSTEM.md](TESTING-RELIABILITY-SYSTEM.md) |
| Branching & releases | [guides/branching-strategy.md](guides/branching-strategy.md) |
| Docstring format | NumPy/Sphinx with RST `:math:` — Charter §6 and the [Style Guide docstring section](UW3_Style_and_Patterns_Guide.md) |
| Documentation file format | MyST Markdown (`.md`) for Sphinx — CLAUDE.md "Documentation Requests" section |

## Documentation Structure

This documentation is organized into focused sections:

- **Getting Started** — Development setup and contributing guidelines
- **Guides** — Practical how-to guides for scripts, notebooks, gmsh, code review
- **Architecture** — Mathematical objects and NDArray system internals
- **Design Documents** — Architecture decisions and design rationale
- **Subsystems** — Detailed documentation of each code module
- **Advanced Topics** — Solver development, MPI parallelism, PETSc integration

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
| Set up development environment | [Development Setup](guides/development-setup.md) | Environment configuration |
| Understand data access patterns | [NDArray System](UW3_Developers_NDArrays.md) | `var.array[...] = values` |
| Write mathematical expressions | [Mathematical Objects](UW3_Developers_MathematicalObjects.md) | `momentum = density * velocity` |
| Follow coding standards | [Style Guide](UW3_Style_and_Patterns_Guide.md) | Patterns and conventions |
| Add new functionality | [Contributing](guides/contributing.md) | Development workflow |
| Debug parallel issues | [MPI Parallelism](advanced/mpi-parallelism.md) | Parallel debugging |

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

guides/development-setup
guides/contributing
UW3_STYLE_CHARTER
UW3_Style_and_Patterns_Guide
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Guides

guides/HOW-TO-WRITE-UW3-SCRIPTS
guides/notebook-style-guide
guides/GMSH_INTEGRATION_GUIDE
guides/CODE-REVIEW-PROCESS
guides/style-gates
guides/SPELLING_CONVENTION
guides/version-management
guides/branching-strategy
guides/state-as-dataclass
guides/BINDER_CONTAINER_SETUP
guides/hpc-cluster-setup
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Design Documents

design/UNITS_SIMPLIFIED_DESIGN_2025-11
design/ND_UNITS_BOUNDARY_CONTRACT
design/WHY_UNITS_NOT_DIMENSIONALITY
design/SYMBOL_DISAMBIGUATION_2025-12
design/ADAPTIVE_MESHING_DESIGN
design/mesh-adaptation-formulation
design/LAYER2_SBR_ADAPT_ON_TOP
design/NVB_GRADED_ADAPT
design/MULTIGRID_MINIMAL_CONTROL_2026-07
design/solver-wall-clock-guard
design/ARCHITECTURE_ANALYSIS
design/MATHEMATICAL_MIXIN_DESIGN
design/COORDINATE_MIGRATION_GUIDE
design/GEOGRAPHIC_COORDINATE_SYSTEM_DESIGN
design/mesh-geometry-audit
design/SWARM_MODERNIZATION_DESIGN_2026-07
design/PROJECTED_NORMALS_API_DESIGN
design/TURBULENCE_MODEL_DESIGN
design/declined-coord-units-proposal
design/nonlinear-solver-homotopy-warmstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Architecture

UW3_Developers_MathematicalObjects
UW3_Developers_NDArrays
TEMPLATE_EXPRESSION_PATTERN
TESTING-RELIABILITY-SYSTEM
CHANGELOG
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Subsystems

subsystems/meshing
subsystems/mesh-shape-relaxation
subsystems/discretisation
subsystems/solvers
subsystems/boundary-stress-and-projection-postprocessing
subsystems/rotated-freeslip
subsystems/petsc-jacobian-layout
subsystems/constitutive-models
subsystems/constitutive-models-theory
subsystems/constitutive-models-anisotropy
subsystems/swarm-system
subsystems/data-access
subsystems/interpolation
subsystems/expressions-functions
subsystems/containers
subsystems/checkpointing-system
subsystems/model-orchestration
subsystems/jit-cache
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Advanced Topics

advanced/solver-development
advanced/mpi-parallelism
advanced/petsc-integration
```