---
title: "Discretisation Subsystem"
---

# Discretisation System Documentation

```{note} Well-Documented Subsystem
**Module**: `discretisation/` (4,145 total lines)  
**Priority**: 🟢 Low - already well documented  
**Current Status**: Good documentation ✅

This subsystem has comprehensive documentation with mathematical notation.
```

## Overview

The discretisation subsystem provides finite element discretization, field management, and PETSc integration.

### Current State
- **Files**: 
  - `discretisation_mesh.py`: 2,708 lines - Core Mesh class, PETSc integration
  - `discretisation_mesh_variables.py`: 1,437 lines - MeshVariable implementation
- **Complexity**: Very High - fundamental finite element infrastructure
- **Documentation Quality**: Good ✅

### Key Classes

| Class | Purpose | Lines | Documentation |
|-------|---------|-------|---------------|
| `Mesh` | Core mesh object with PETSc DM | 2,708 | Good |
| `MeshVariable` | Field variables on mesh | 1,437 | Good |
| `MeshVariable.array` | NDArray interface (v0.99+) | - | Good |

## Current Documentation

### Strengths
- ✅ Extensive mathematical docstrings
- ✅ PETSc integration documented  
- ✅ New NDArray interface documented
- ✅ Comprehensive class and method coverage

### Minor Improvements Needed
- ⚠️ Parallel patterns could be clearer
- Could benefit from more usage examples

## Recent Enhancements (v0.99)

- New `array` property with automatic PETSc synchronization
- Backward-compatible `data` property maintained  
- Comprehensive callback system for parallel safety

## Implementation Status

```{note} For Contributors
This subsystem already has good documentation. Potential improvements:
- Additional parallel usage patterns
- More complex integration examples
- Performance optimization guidance
- Advanced PETSc DM usage patterns
```

---

*This subsystem serves as a model for documentation quality in other areas.*