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

Evaluating the variable at its own `coords` returns its own `data` to round-off
(the point selection is exact; the evaluator pipeline can add an ulp).

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
adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V_fn, DuDt=DuDt, order=1)
```

The diffusive flux history (`DFDt`) keeps its nodal projection, since it
carries derivatives. Scalar histories only; no ALE or old-frame trace-back,
no checkpoint state yet.

It is the transport manager of either advection-diffusion solver. In the
composed `uw.systems.AdvDiffusion` (#688) it runs at `order=2` (BDF2) and
at `order=1, theta=1`, where no spatial term sits on the old level; on a
rotating Gaussian the field matches the SLCN solver's to 3e-3. With the
Crank-Nicolson flux (`theta=0.5`) the composed solver differentiates the
old level, which a delta field cannot supply, and the JIT guard refuses
with a clear message; for that scheme use `AdvDiffusionSLCN`, whose
diffusive history is a separate nodal `DFDt`.

### The mid-point velocity is taken at the mid time

The RK2 trace, `x_mid = x - dt/2 v(x)`, `x_dep = x - dt v(x_mid)`, is second
order only if `v(x_mid)` is the velocity at `t^{n+1/2}`. Both schemes now
take it there: on the current interval by extrapolation from the two most
recent velocity fields, `1.5 v^n - 0.5 v^{n-1}` (the velocity at `n+1` is
not known when the history is built), and on the older segments of a
multi-step history by the average of the two known ends. With `v^n` alone
the foot is off by `b dt²/2` in a flow accelerating at rate `b`, first
order in an unsteady flow; the rotating Gaussian did not show it because
that velocity is steady. `tests/test_0066_integration_point_slcn.py`
checks both schemes against the exact foot in a uniformly accelerating
flow, for `V_fn` given as the variable, as `-v` and as `v/2`, with the
`v^n`-only foot as the control.

`V_fn` stays symbolic by design (`-v`, `v/2`, `c(t) v`, `v - v_mesh` must
all just work), and the previous velocity is **cached by evaluation**:
`V_fn` evaluated at the true nodes of a vector field of the highest degree
among the variables it contains. That captures everything the expression
depends on as it was at that time: the variables, constants that ramp,
swarm proxies, the mesh geometry. Substituting snapshots of the mesh
variables into the expression would read a ramping constant at its
current value. Evaluating at nodes *nudged* into their cells, the old
boundary-node workaround, left a `0.001 h |grad v|` bias that the
extrapolation fed into every trace and moved the Blankenbach 1a wall
Nusselt number by 0.9 %; the evaluator is exact at the true node
coordinates on simplex, quad and annulus meshes (2e-16), so no nudge is
used here. The test covers `V_fn` as the variable, `-v`, `v/2` and `c v`
with `c` changed between steps.

For a P2 field in a uniform velocity the slots reproduce the exact
departure-point values to round-off, for one and for two segments
(`tests/test_0066_integration_point_slcn.py`).

### The rule must oversample the history space

The solve fits the sampled departure-point values to the continuous space
by weighted least squares on the rule. With as many points per cell as the
element has local dofs (P2 on a triangle: 6 dofs, and 6 points at
`qdegree=2`) that fit is a per-cell interpolant through interior points,
which extrapolates, and at small Courant number a mode grows by about 1.1
per step: on the rotating Gaussian below at Courant 0.25 the run was flat
for 75 steps and then blew up. At twice the points (`qdegree=3`, 12 on a
triangle) the fit is contractive and the scheme is stable through a full
revolution. At 1.5x (PETSc's conical rule, 9 points at degree 4, selected
with `-<prefix>petscfe_default_quadrature_type conic`) it is also bounded
through a full revolution, with a 0.07 % rise in energy that saturates and
twice the L2 error of the 12-point rule. The constructor raises when the
rule has no more points than local dofs and warns below 2x. Raising the
rule costs every solver on the mesh its assembly time, which is the price
of this scheme; `qdegree` is the polynomial exactness of the rule, and the
extra exactness is incidental here, only the point count matters.

Why a fit at all: with the load vector formed from point samples and the
mass matrix exact on the same rule, the Galerkin step is algebraically the
weighted least-squares fit `min Σ_q w_q (T(x_q) - g_q)²`. The composed
field `T^n ∘ X_dep` is piecewise P2 on the *shifted* mesh, so on the actual
cells it carries interior kinks wherever a cell's feet straddle a source
edge, and it is not in the space. The fit contracts in the sampled norm,
not in L2; a grid-scale mode shifted by a fraction of a cell can have a
sampled norm above its true norm (an aliasing error of the rule), and that
ratio is the growth per step. The nodal scheme is stable for a different
reason: interpolation at the nodes is bounded by the source's nodal values.

Measured directly, by power iteration on the one-step operator (random
field renormalised every step, P2, `cellSize=0.1`, Courant 0.25):

| growth per step | pure advection | cell Péclet 100 |
|---|---|---|
| nodal SLCN | 0.9989 | — |
| integration-point, 6 points (1x) | 1.028 | 1.0095 |
| integration-point, 9 points (1.5x) | 1.005 | 0.9960 |
| integration-point, 12 points (2x) | 1.0003 | 0.9962 |

Under pure advection the integration-point map is never strictly
contractive; oversampling brings it toward neutral. That is the flip side
of not dissipating: the nodal scheme's 0.999 is its numerical diffusion.
With the physical diffusion a real problem carries (cell Péclet 100 here)
9 and 12 points are stable and 6 is not. The 1.5x case is therefore usable
with diffusion and slowly unstable without it (invisible over one
revolution, not over ten); 2x is neutral either way.

### Measured against nodal SLCN

Rotating Gaussian (solid-body rotation, width 0.1 at radius 0.5), P2, unit
square of side 2, `cellSize=0.05`, `qdegree=3`, half a revolution, pure
advection. L2 error against the exact rotated field and the peak value:

| Courant | nodal SLCN | integration-point | peak nodal / IP |
|---|---|---|---|
| 0.25 | 1.30e-2 | 6.3e-4 | 0.935 / 0.994 |
| 0.5 | 3.55e-3 | 9.0e-4 | 0.974 / 0.992 |
| 1 | 3.62e-3 | 3.24e-3 | 0.987 / 0.997 |
| 2 | 1.33e-2 | 1.33e-2 | 0.996 / 1.000 |

The gain is largest at small Courant number, where the nodal scheme
re-interpolates most often per unit of transport; at Courant 2 the RK2
trace-back error dominates and the two agree. Over a full revolution at
Courant 0.25 the error is 1.17e-3, twice the half-revolution value, so it
grows linearly.

What the scheme conserves (Courant 0.5, 63 steps, relative change):

| | integral of T | integral of T² |
|---|---|---|
| nodal SLCN | fluctuates within 3e-4 | -1.4 % (monotone) |
| integration-point | -3e-5 | -0.03 % |

Neither scheme is exactly conservative (a Galerkin projection of a
transported field conserves the integral only with exact integration), but
the second moment is where the nodal scheme's diffusion shows and the
integration-point scheme loses 45 times less of it.

The trace-back samples twelve points per cell rather than the P2 nodes, and
the snapshot evaluation at the moving feet misses the locator cache every
step, so the update costs about three times the nodal one.

## Swarm proxy at the integration points

A swarm variable normally reaches the weak form through a nodal proxy:
the particle field is reconstructed at the proxy's nodes from the nearest
particles and the assembler interpolates it to the integration points with
the basis. With `proxy_location="integration_points"` the proxy is an
integration-point variable, reconstructed from the nearest particles at
every integration point and read there directly. That is the
Ellipsis / Underworld PIC-LIP mapping: material properties are sampled
from the particles around each integration point, not smoothed to the
nodes and back.

```python
swarm = uw.swarm.Swarm(mesh)
M = uw.swarm.SwarmVariable("M", swarm, 1, proxy_location="integration_points")
swarm.populate(fill_param=3)
M.data[:, 0] = ...                                   # per particle
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0 * M.sym[0] + eta_1 * (1 - M.sym[0])
```

The reconstruction itself is unchanged (a linear-exact RBF over the nearest
particles, `rbf_interpolate`); only its target moved. A particle-carried
material step is reproduced at the integration points with less than half
the L2 error of the nodal proxy, and is exactly 0 or 1 one cell away from
the interface (`tests/test_0067_integration_point_proxy.py`).
`proxy_degree` and `proxy_continuous` are ignored for this proxy; the proxy
has no gradient, so a derivative of the swarm variable's symbol is refused.
Vector and tensor swarm variables get a multi-component proxy.

`Lagrangian_Swarm(..., proxy_location="integration_points")` applies the
same to the fully Lagrangian history: the slots carried on the particles
are reconstructed at the integration points and the weak form reads them
there, with no nodal history field. This is the Lagrangian option for large
particle swarms, where the particles carry the state and the mesh only
integrates it.

## Swarm proxy as a polynomial per cell

`proxy_location="cells"` is the third target. The proxy is a discontinuous
mesh variable of `proxy_degree`, and every cell holds the least-squares
polynomial through the particles that cell holds
(`utilities/cell_polynomial_projection.py`). The assembler reads it at the
integration points through the ordinary basis, so:

- a polynomial particle field up to `proxy_degree` is reproduced exactly;
- the value at the rule is a polynomial on the mesh cell, so the default
  rule integrates it exactly and the oversampling guard of the
  integration-point history does not apply;
- a material step on a cell edge is exactly 0 or 1 on either side, with no
  overshoot (the RBF reconstruction overshoots a step by up to 14%);
- the proxy has a gradient, so Crank-Nicolson and the Adams-Moulton flux
  of the history work;
- each rank fits its own cells from its own particles: no neighbour search
  across ranks, no halo particles.

A cell with fewer particles than the basis size plus two takes a linear
fit to the particles nearest its centroid, which is the RBF's
neighbourhood; linear, because a higher-degree polynomial extrapolated
from a distant neighbourhood is unbounded (a P2 extrapolation into the
emptied corner cells of a rotating box reached twice the field maximum).
That cell is consistent to first order but no longer a cell-local fit, and
a light swarm degrades the same way the RBF proxy does. A cell with no
particles keeps its previous proxy value: no particles is no information,
and that holds until the swarm is repopulated. The threshold and patch
size are `nmin` and `patch_nnn` on `CellPolynomialProjector.fit`.

```python
M = uw.swarm.SwarmVariable("M", swarm, 1, proxy_location="cells", proxy_degree=2)
```

### Why a least-squares fit and not a conservative transfer

The conservative particle-to-mesh transfer solves the rule mass matrix
against the particle moments, `M u = sum_p V_p phi(x_p) psi_p` (PETSc's
`DMSwarmProjectFields` on a Plex does exactly this, with unit weights). It
hands the mesh the particle sums exactly, but its nodal values carry the
error of the particle "quadrature", which scales with the field value over
the square root of the particle count, not with the field's variation over
the cell. Measured on `UnstructuredSimplexBox(cellSize=0.05)` with the
`populate()` lattice jittered by 30% of the particle spacing, L2 error at
the integration points (`~/+Simulations/integration_point_proxy`,
2026-09-08):

| field, fill (particles per cell) | RBF at the points | conservative P1 | conservative P2 | least squares P1 | least squares P2 |
|---|---|---|---|---|---|
| linear, 3 | 7e-16 | 3.5 | 5.8 | 2e-10 | 6e-10 |
| linear, 21 | 6e-16 | 0.89 | 2.6 | 9e-16 | 1e-15 |
| Gaussian (width 0.1), 3 | 1.1e-3 | 1.6e-1 | 2.7e-1 | 2.8e-3 | 4.4e-4 |
| Gaussian, 10 | 3.7e-4 | 6.8e-2 | 1.5e-1 | 1.7e-3 | 1.2e-4 |
| Gaussian, 21 | 1.2e-4 | 3.8e-2 | 1.3e-1 | 1.6e-3 | 5.5e-5 |
| Gaussian, 1 per cell on average (34% of cells empty) | 3.9e-3 | 2.5e-1 | 3.8e-1 | 8.9e-3 | 3.1e-3 |

Moment-matched particle weights (chosen so constants transfer exactly)
repair the conservative transfer's constant mode and cut the linear error
fifty-fold, but go negative on thin cells and still trail the fit by two
orders of magnitude on the Gaussian. Conservation and light sampling are in
tension: what makes a light swarm usable is polynomial reproduction with a
support that widens when the cell is thin, and the fit degree has to reach
the mesh degree to profit from particle density (the P1 fit is limited by
cell size and is worse than the RBF; the P2 fit beats the RBF three times
over at ten particles per cell and at every density tested). The cell-mean
of the fit matches the particle mean of the cell exactly; the integral of
the fitted field differs from the particle sum by the particle-quadrature
error, 1e-4 relative at ten particles per cell.

The refresh costs about the same as the RBF path: at ten particles per
cell on 944 cells, 8 ms (locate 6 ms, fit 2 ms) against 22 ms for the RBF
proxy with its kd-tree rebuilt.

### As the transport term of an advection-diffusion solve

`Lagrangian_Swarm(..., proxy_location="cells")` composed into
`uw.systems.AdvDiffusion` gives a particle-in-cell transport scheme: the
particles are advected (`swarm.advection`), each history slot is fitted
to the cells, the mesh solves the diffusion against that history, and the
particles re-read the solution (`particle_update="pic"`, the default;
`"flip"` adds the mesh increment instead and is kept for the MPM line of
work, it accumulates the projection increments). The history is sampled
at the particles the first time the swarm moves, through the swarm's
pre-advection hook; sampled at the first solve instead, it would see the
landed positions and lose a step.

Rotating Gaussian (sigma 0.1 at radius 0.4, one revolution, h = 0.1, P2,
C = 0.25, `~/+Simulations/integration_point_proxy/scripts/transport_gaussian.py`),
L2 error of the mesh field against the exact solution:

| Pe_h | nodal SLCN | integration-point SLCN | PIC, cells P2, 10 particles per cell | PIC, 21 per cell |
|---|---|---|---|---|
| infinite | 5.1e-2 (peak 0.69) | 9.1e-3 (peak 0.99) | 1.6e-2 (peak 0.86) | 5.2e-3 (peak 0.995) |
| 400 | 4.3e-2 | 6.8e-3 | 1.3e-2 | 4.1e-3 |
| 100 | 2.7e-2 | 3.4e-3 | 8.2e-3 | 2.4e-3 |
| ms per step | 510 | 1250 | 140 to 170 | 180 to 250 |

The particle scheme's error is the per-step re-projection (fit, then
Galerkin projection, then read-back) and falls with particle density; at
21 particles per cell it is below the integration-point history at a
fifth of the cost. At C = 2 all three schemes are limited by the midpoint
RK2 trajectory (half a radian per step, 4% phase lead), not by transport.
