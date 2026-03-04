# PETSc Merge Request: Ghost Facet Fix

## Submission details

- **Fork**: https://gitlab.com/lmoresi/petsc
- **Branch**: `fix/plexfem-ghost-facet-boundary-residual`
- **Target**: `petsc/petsc` → `release`
- **Create MR**: https://gitlab.com/lmoresi/petsc/-/merge_requests/new?merge_request%5Bsource_branch%5D=fix%2Fplexfem-ghost-facet-boundary-residual

## Title

DMPlex: filter ghost facets from boundary residual and integral assembly

## Description

### Problem

`DMPlexComputeBdResidual_Internal` and `DMPlexComputeBdIntegral` construct
`facetIS` as all facets at depth `dim-1` but do not exclude ghost facets
(point SF leaves). The volume residual code in `DMPlexComputeResidualByKey`
checks the `"ghost"` label before calling `DMPlexVecSetClosure` (line 5355),
but the boundary residual path has no equivalent check.

For **external** boundaries this is benign: ghost facets on the domain
boundary still have `support[0]` pointing to a valid local cell, and
`DMLocalToGlobal(ADD_VALUES)` correctly resolves ownership. But for
**internal** boundaries — facets in the mesh interior that carry a
`DMLabel` value and a `DM_BC_NATURAL` condition — facets at partition
junctions are present on multiple ranks. Each rank independently
integrates the boundary flux via `PetscFEIntegrateBdResidual`, and
`DMLocalToGlobal(ADD_VALUES)` sums all contributions, causing
double-counting.

The same issue affects `DMPlexComputeBdIntegral`: ghost boundary facets
are included in the per-face integral sum, which is then `MPI_Allreduce`d,
again double-counting shared facets.

### Use case: internal boundaries in geodynamics

[Underworld3](https://github.com/underworldcode/underworld3) is a
Python/PETSc geodynamics framework that uses DMPlex FEM throughout.
Geophysical models commonly require traction (natural) boundary conditions
on **internal surfaces** — for example, a fault plane or a material
interface embedded within the mesh. These are set up by labeling interior
facets with a `DMLabel` and registering `DM_BC_NATURAL` via
`PetscDSAddBoundary` / `PetscWeakFormAddBdResidual`.

This works correctly in serial but produces O(1) errors whenever the
labeled internal boundary is split across partition boundaries, because
shared facets are integrated on multiple ranks.

### Fix

After obtaining `facetIS` from `DMLabelGetStratumIS(depthLabel, dim-1, ...)`,
filter out SF leaves using `ISDifference`. This ensures each boundary facet
is processed by exactly one rank (the owner).

This follows the same pattern already used in `DMPlexComputeBdFaceGeomFVM`
(plexfem.c, around line 1087) which calls `PetscFindInt(face, nleaves, leaves, &loc)`
to skip ghost faces during FVM face flux computation.

### Testing

Tested with Underworld3 Stokes solver using `DM_BC_NATURAL` on an internal
boundary (annular mesh with labeled interior circle). Results match serial
to 10+ significant figures across all processor counts tested (1, 2, 4, 8).
Before the fix, any partition that splits the internal boundary produces
O(1) errors from double-counted boundary contributions.

The patch applies cleanly to all PETSc releases from v3.18.0 through
v3.24.5 — the affected code has been structurally unchanged across these
versions.

## Reference examples

- Stokes Green's function with internal boundary natural BCs:
  https://github.com/underworldcode/underworld3/blob/development/docs/examples/fluid_mechanics/advanced/Ex_Stokes_Annulus_Kernels.py
- Cartesian variant:
  https://github.com/underworldcode/underworld3/blob/development/docs/examples/fluid_mechanics/advanced/Ex_Stokes_Cartesian_Kernels.py
