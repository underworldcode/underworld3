---
title: "Meshing Subsystem"
---

# Meshing System Documentation

```{important} High Priority Documentation Gap
**Module**: `meshing.py` (4,437 lines - largest module)  
**Priority**: 🟡 High  
**Current Status**: Partial documentation - needs geometric parameter details

This is a critical user-facing system that needs comprehensive documentation.
```

## Overview

The meshing subsystem handles computational mesh generation and manipulation for finite element analysis.

### Current State
- **Lines of Code**: 4,437 (largest single module)
- **Functions**: 17 mesh generation functions
- **External Dependencies**: gmsh, PETSc
- **Documentation Quality**: Partial ⚠️

### Key Components

```python
# Major mesh types that need documentation
- UnstructuredSimplexBox      # Cartesian box meshes
- SphericalShell              # Spherical annulus meshes  
- CubedSphere                 # Cubed sphere topology
- AnnulusInternalBoundary     # Annulus with internal boundaries
- QuadBox / HexBox            # Structured meshes
```

## Empty MPI Partitions

Use `mesh.isSimplex` for the mesh-wide cell family. PETSc's
`mesh.dm.isSimplex()` is a rank-local query and returns `False` on a rank
with no cells, even when the distributed mesh consists of triangles or
tetrahedra. UW3 infers the family collectively from populated ranks before
constructing coordinate finite elements or element metadata.

This is also an MPI correctness requirement: constructing simplex and
tensor-product coordinate elements on the same communicator consumes
different PETSc message tags. The first mesh may appear to construct
successfully, but a later HDF5 boundary-label load can deadlock in
`PetscSFSetUp_Basic`. This failure does not require a transport solver.

`tests/parallel/test_0781_empty_rank_mesh_sequence.py` covers triangles,
tetrahedra, quadrilaterals and hexahedra with deliberately empty partitions,
then verifies a second mesh's volume and boundary integrals using P2 data.

## Documentation Needs

### Critical Gaps
1. **Geometric parameter relationships and constraints**
2. **Boundary condition setup patterns**  
3. **Mesh refinement strategies**
4. **Parallel decomposition behavior**

### Current Status
- ✅ Function signatures documented
- ✅ Basic parameter descriptions  
- ❌ Geometric parameter details missing
- ❌ Usage examples sparse
- ❌ Mesh topology explanations needed

## Implementation Tasks

```{tip} For Contributors
This section needs:
- Complete geometric parameter documentation for each mesh type
- Boundary condition setup examples
- Mesh quality and refinement guidance  
- Parallel mesh distribution patterns
- Performance considerations for different mesh types
- Integration with discretisation system
```

## Related

- [Metric-driven mesh redistribution](mesh-metric-redistribution.md)
  — topology-preserving node redistribution toward a target
  size/density field (`smooth_mesh_interior`; the variational
  MMPDE mover). Restores the grading of a deformed
  adapted mesh or bunches nodes ~2× at a feature; contrast
  `mesh.adapt()` which remeshes.

---

*This document serves as a placeholder for comprehensive meshing system documentation.*
