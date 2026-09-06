# Integration-point variables

`uw.discretisation.IntegrationPointVariable` stores one value per quadrature
point per cell, on an element whose basis is the identity on the mesh's
integration rule. The assembler reads the stored values directly at the
integration points, with no interpolation. It is a peer of `MeshVariable`
and of swarm variables: a nodal field is *sampled*, a swarm carries
*particles*, and an integration-point variable carries values that are
*injected* into the weak form exactly where it is evaluated.

```python
import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=2)
eta_q = uw.discretisation.IntegrationPointVariable("eta_q", mesh)

eta_q.cell_data.shape        # (ncells, Nq, 1)
eta_q.coords                 # the physical integration points, same order
eta_q.cell_data[...] = 1.0
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_q.sym
```

## Why it exists

Two uses drove the design.

**Semi-Lagrangian history.** The SLCN scheme samples the previous solution at
the departure point of every node, builds a nodal history field, and the
assembler then interpolates that field to the quadrature points. Two
interpolations per step. If the departure points are traced from the
integration points instead, and the sampled values are stored here, the value
entering the weak form is the discrete solution evaluated exactly at the
departure point. Only the FE solution's own error remains.

**Material properties from particles.** A swarm property normally reaches the
constitutive law through a proxy mesh variable (nearest-neighbour or RBF to
the nodes, then the basis to the integration points), which smooths a
material interface over a cell. Reconstructing the property at each
integration point from the particles near it and writing it here keeps
sub-cell contrast, as the integration swarms of Underworld 1 and 2 did.

## The rule is a mesh property

Every field on a mesh is created on the rule fixed by `mesh.qdegree`, and
PETSc's `PetscDSSetUp` tabulates all fields of a discrete system on one rule.
So the element is built once per mesh, on `mesh.integration_rule`, whatever
the degrees of the fields it sits beside. A P1 temperature and a P2 velocity
on a `qdegree=2` triangle mesh are both integrated on the six-point rule, and
an integration-point variable on that mesh has six values per cell.

Point counts at `qdegree=2`: triangle 6, tetrahedron 14, quadrilateral 9,
hexahedron 27.

## What `evaluate` means

A delta field is defined only at its points. Between them it is *defined*
as piecewise constant on the nearest-integration-point partition of each
cell, and that is what `uw.function.evaluate` returns: locate the cell, take
the closest of its points. This is the one extension under which a query
agrees with what the assembler used at that point; an interpolant or a
projection would report a different viscosity from the one the solver saw.
Points no cell owns take the nearest integration point on the rank.

Evaluating the variable at its own `coords` returns its own `data` exactly.

If a smooth nodal picture is wanted (a plot, a diagnostic), project the
symbol onto a `MeshVariable` explicitly with `SNES_Projection`; the
projection of the field is an ordinary weak form and is exact for data that
the target space can represent.

## Guards

The field has no gradient (its tabulated derivative is identically zero), so
a derivative of its symbol in a weak form would be a silent zero. The JIT
refuses it at code generation:

```
RuntimeError: {h}_{,0}: derivative of an integration-point variable has no meaning ...
```

Off its own rule the field tabulates to zero, so a solver on a different
rule would drop the term it carries. A solver that attaches the mesh's
auxiliary vector checks its element against `mesh.integration_rule` and
raises if they differ. Boundary integrals evaluate the field on the face
rule and see zeros; that is correct for a history term and worth knowing for
anything else.

Scalar components only for now; use one variable per component.

## Implementation

- `src/underworld3/cython/uw_delta_space.h`: the `uwdelta` `PetscSpace`
  type, registered with `PetscSpaceRegister`. PETSc's own `PETSCSPACEPOINT`
  cannot be tabulated anywhere but its own points in its own order, which
  breaks `PetscFESetUp`, face tabulation and boundary integrals; the plugin
  type returns zeros off its rule and needs no PETSc patch, so stock conda
  PETSc works.
- `src/underworld3/cython/petsc_quadrature_fe.pyx`: `create_delta_fe`
  (the element, via `PetscFECreateFromSpaces` with a `PETSCDUALSPACESIMPLE`
  dual space of one point evaluation per rule point), `tabulate` (for
  tests) and `cell_quadrature_points` (the physical integration points from
  `DMPlexComputeCellGeometryFEM`, the assembler's own map).
- `_BaseIntegrationPointVariable` in `discretisation_mesh_variables.py`
  overrides the two discretisation hooks (`_create_petsc_fe`, `_basis_key`)
  and supplies the nearest-point evaluation; `IntegrationPointVariable` in
  `enhanced_variables.py` is the public wrapper.
- All dofs sit on the cell interior, so the local vector is cell-major,
  point-minor, and `cell_data` is a plain reshape.

Tests: `tests/test_0064_quadrature_point_fe.py` (the element),
`tests/test_0065_integration_point_variable.py` (the variable, the assembler
reading it, `evaluate`, the guards).

## Semi-Lagrangian history on the integration points

`uw.systems.ddt.IntegrationPointSemiLagrangian` is the SLCN history built on
this variable. Its slots `psi_star[k]` are integration-point variables, so
the value the weak form sees at each integration point is the solution from
`k+1` steps ago evaluated exactly at the departure point of that
integration point. A nodal history cannot be sampled from a delta field, so
the slot-to-slot chain of `SemiLagrangian` is replaced by nodal snapshots of
the solution and of the velocity at the last `order` times: slot `k` is
filled by tracing `k+1` RK2 segments back from every integration point,
segment `j` with the velocity at time `n-j` and that step's `dt`, and
evaluating the snapshot from time `n-k` at the foot. Every slot carries one
evaluation error rather than one per generation.

```python
DuDt = uw.systems.ddt.IntegrationPointSemiLagrangian(mesh, T, V_fn, degree=2, order=1)
adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=V_fn, DuDt=DuDt, order=1)
```

The diffusive flux history (`DFDt`) keeps its nodal projection, since it
carries derivatives. Scalar histories only; no ALE or old-frame trace-back,
no checkpoint state yet.

For a P2 field in a uniform velocity the slots reproduce the exact
departure-point values to round-off, for one and for two segments
(`tests/test_0066_integration_point_slcn.py`). On a rotating Gaussian at
Courant 1 (`cellSize=0.05`, `dt=0.1`, half a revolution) the L2 error drops
from 3.6e-3 (nodal SLCN) to 3.2e-3 and the peak is kept at 0.9998 instead
of 0.987. The trace-back samples six points per cell rather than the P2
nodes, so the update costs about twice the nodal one.
