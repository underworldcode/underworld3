# Rotated strong free-slip and the prescribed-normal datum

The rotated free-slip boundary condition imposes

$$\mathbf{u}\cdot\hat{\mathbf{n}} = \tilde u_n$$

strongly — as an exact per-node constraint, not a weak (Nitsche/penalty) term —
by rotating each boundary node's velocity components into a (normal,
tangential) frame and constraining the rotated normal component. `conds = 0`
(the default datum) is free-slip; a non-zero scalar `conds` prescribes the
wall-normal velocity, which is what the free-surface manager uses to keep the
surface a material boundary.

```python
stokes.add_rotated_freeslip_bc(0, "Upper", normal=nhat)          # free-slip
stokes.add_rotated_freeslip_bc(h_dot.sym[0], "Upper", normal=nhat)  # u·n̂ = field
```

`normal=None` uses the geometric facet normal (the default, and the one
consistent with what the assembler integrates — see below); a sympy `1×dim`
matrix in `mesh.X` supplies an analytic normal, exact for the TRUE surface
(`X/|X|` on a spherical cap, a constant on a planar face); a constant array is
also accepted. The datum must be a
*scalar* (a number, an expression of `mesh.X`, or a scalar field read); on an
enclosed boundary it must be discretely flux-free for incompressibility. A
corner or 3D-edge node shared between rotated boundaries has no single normal
and stays at the free-slip pinning (the datum is ignored there).

### The nodal normal is measure-weighted, not a bisector

A node that sits on more than one facet — a vertex in 2D, a vertex or an
edge-midpoint in 3D — gets one nodal normal, while the assembler integrates the
boundary term facet by facet. The two only agree when the node's normal is
parallel to the **measure-weighted** sum `Σ_f |f| n̂_f` (edge length in 2D, face
area in 3D), which is what the geometric path accumulates. Plain bisector
averaging `Σ_f n̂_f` — what UW3 did before issue #560 — is right only where the
facets are equal; on a **kinked** wall with unequal facets it leaves a residual
`sin(Δ/2)·(|f₁|−|f₂|)/6` in the node's free tangential row (Δ = kink angle), the
exact constant-pressure vector stops being a null vector of the constrained
operator, and the pressure gauge goes unpinned. **Axis-aligned** walls are
unchanged to the last bit — their facet normals have exactly 0/±1 components, so
`Σ_f |f| n̂_f` normalises to the same floats as `Σ_f n̂_f` whatever the weights.
A flat but *tilted* wall is not covered by that argument and can move by one ulp.

The sum runs over ALL facets meeting the node, so it must be completed **across
ranks**. Each boundary facet is labelled on exactly one rank, so a node on a
partition seam sees only some of its facets locally; the contributions are
summed through the DM's local↔global scatter before normalising, which is what
makes the normal partition-independent. Two things had to go with it: the
outward test now points away from the facet's own support cell (the mean of the
rank's coordinates is rank-local, and would let two facets of one node cancel),
and the node list comes from the local mesh's exterior facets rather than the
labelled subset, because a rank can own a node whose labelled facets are all on
neighbours.

### Which way is "outward" — and the sign of σ_nn on an inner boundary

The geometric normal points away from the facet's own support cell, which is the
**domain's** outward normal on any boundary. On a concave boundary — an annulus
or spherical-shell **inner** arc, the CMB — that points *toward* the centre of
curvature.

This changed at #560. The old rule pointed away from the mean of the mesh
coordinates, which on an inner arc is *into* the domain. So on a concave
boundary, through the geometric normal:

| quantity | before #560 | after |
|---|---:|---:|
| nodal radial component, annulus `Lower` | +1.000000 | **−1.000000** |
| `boundary_normal_traction("Lower")` | −5.233110e-02 | **+5.233110e-02** |

Magnitudes are identical to every digit; only the sign moves. `dynamic_topography_field`
is `h = −σ_nn/(Δρ g)` on top of that number, so it reverses there too, as does the
sign of a non-zero prescribed wall-normal datum (`u·n̂ = ũ_n`: positive now means
outflow *from the domain* on an inner arc, where before it meant inflow). Convex
boundaries — every box wall, an outer arc, a spherical cap — are unaffected: the two
rules agree there.

An **analytic** `normal=` is applied exactly as supplied and is *not* reoriented.
So `X/|X|` on an inner arc is inward-of-domain and gives σ_nn of the opposite sign
to the default. That is deliberate — the override means "use exactly this
direction", and silently flipping it would change the meaning of a user's datum —
but it means **you must pass `-X/|X|` on an inner boundary if you want the
domain-outward convention.** `test_1018_rotated_nodal_normal.py` pins both halves.

### Which normal to use

They answer different questions, and the trade is measurable. `|A z|/|A|_F` on
an annulus (`cellSize=0.15`), `z` = the attached constant pressure:

| boundary | np | geometric (default) | analytic `X/\|X\|` |
|---|---:|---:|---:|
| uniform arcs | 1 | **6.6e-20** | 1.1e-14 |
| uniform arcs | 4 | **6.7e-20** | 1.1e-14 |
| skewed (non-uniform facets) | 1 | **6.6e-20** | 3.1e-07 |
| skewed (non-uniform facets) | 4 | **6.7e-20** | 3.1e-07 |

The geometric normal is *consistent with the assembly*: it is the direction the
straight-facet boundary integral actually sees, so the constant pressure stays a
null vector to machine precision at every rank count. The analytic normal is
*consistent with the geometry*: it is tangent to the true surface, which the
faceted mesh only approximates — so the assembler and the constraint disagree by
an amount that grows with facet non-uniformity, and #560 does not remove it (the
analytic column is unchanged by this fix, and identical at every rank count).

Prefer the default. Reach for `normal=` when the constraint must follow the true
surface rather than the mesh — a coarse spherical shell where faceting, not the
gauge, is the dominant error — and be aware that the pressure gauge is then only
as good as the numbers above.

Why strong rather than Nitsche/penalty: the constraint holds to machine
precision (a penalty leaks ~1e-3, and the leak grows exactly where anisotropy
makes the boundary condition matter), it is correct on curved/tilted/deformed
boundaries, and the constraint **reaction is the boundary normal traction**
σ_nn (`solver.boundary_normal_traction`, `solver.dynamic_topography`) — the
quantity the free-surface machinery consumes. Reserve Nitsche for conditions
that must morph in time (Dirichlet→Neumann ramps).

**Implementation**: `src/underworld3/utilities/rotated_bc.py`; registration and
dispatch in `petsc_generic_snes_solvers.pyx` (`add_rotated_freeslip_bc`).

### The same rule holds for the other two free-slip paths

`add_constraint_bc` (the Lagrange-multiplier free-slip) and `add_nitsche_bc` do
not use `_boundary_velocity_nodes`. Their default constraint direction is
`mesh.boundary_normal(boundary)`, a P1 field assembled by
`Mesh._assemble_boundary_normal` — a second copy of the same accumulation, and
it was left rank-local when #560/#561 fixed the rotated one. That is **#564**:
on `Annulus(cellSize=0.12)` the worst nodal normal on the Upper arc was 3.0e-10
from the exact radial one in serial and **5.8e-02 (3.3°) at np=2, 3 and 4** —
one facet's normal instead of the average of two, so its size is set by the
facet's angular span and does not shrink with more ranks. End to end that moved
a constrained free-slip velocity by **3.4 %** between np=1 and np=2 (#495).

It is fixed the same way — the weighted contributions are summed through the
variable's own sub-DM local↔global scatter before normalising — and all three
accumulators now take their orientation and measure from one shared
`utilities/facet_normals.facet_measure_and_normal`, so they cannot drift apart
again. `boundary_flux._node_normals` is the third; its geometric branch is
unreachable today (its caller guards it with `if normal is not None`) and it
carries a `TODO(parallel)` rather than its own reduction.

Guard: `tests/parallel/test_1069_boundary_normal_parallel.py` (an analytic
oracle on the annulus, a global-facet-sum oracle in 2-D and 3-D, corner
preservation, a negative control, and the Nitsche end-to-end).

Two caveats that are *not* the normal, recorded so they are not re-derived:

* an **internal** boundary's facets have two support cells, so
  `Mesh._assemble_boundary_normal` skips them and
  `mesh.boundary_normal("Internal")` comes back **zero**. `rotated_bc` keeps
  PETSc's own face normal there instead. Neither is a supported configuration;
  do not read either as an endorsement.
* `add_nitsche_bc`'s default `local_h=True` scales the penalty by
  `mesh.cell_size()`, which is **partition-dependent in its own right** (it
  comes from a kd-tree query against this rank's centroids — see the
  `TODO(BUG)` on `Mesh._assemble_cell_size`). With `local_h=False` the Nitsche
  annulus agrees to 3.6e-10 at np=1…4; with it, to 6.6e-03. That is a separate
  defect from #564 and is not fixed by it.

## One solve path

There is a single driver, `solve_rotated_freeslip`: a manual outer
Newton/Picard loop that rotates the residual and tangent every iteration
(`F̂ = Q F`, `Ĵ = Q J Qᵀ`), imposes the constraint on the rotated normal rows,
and solves each increment with a self-contained fieldsplit-Schur KSP (geometric
FMG on the custom prolongation when the solver has a hierarchy, else GAMG; the
native 1/μ pressure mass as the Schur preconditioner). There is no separate
linear path and no up-front nonlinearity probe: a linear model converges after
its first increment and the loop self-terminates.

### Where the multigrid hierarchy comes from

Native geometric FMG cannot serve a rotated solve at all — the DM-coupled
hierarchy has no way to express the per-node rotation — so custom-P multigrid is
not an optimisation here, it is the only geometric option. The hierarchy is
resolved by `custom_mg.build_transfers`, which is the **same rule the standard
path uses**:

1. an explicit `set_custom_fmg` registration on the solver wins; otherwise
2. a **mesh-owned** coarse tail (`mesh._custom_mg_coarse_meshes`, what
   `mesh.adapt()` leaves on a refinement child) is picked up opportunistically,
   with the barycentric→RBF builder fallback, so a failed build degrades to GAMG
   rather than crashing the solve.

Step 2 used to be unreachable from here. The standard path's injection hook runs
*after* the rotated dispatch has already returned, and the rotated builder
consulted only `solver._custom_mg` — so an `adapt()` child under rotated
free-slip reported `velocity_pc == "GAMG"`, indistinguishable from having no
hierarchy at all, and lost its multigrid silently (#467). That is the
`adapt-on-top-faults` workflow's own configuration: a fault resolved by local
refinement on an adapt child, with rotated free-slip chosen over Nitsche because
it composes with transverse isotropy.

The option bundle for both the FMG and the GAMG velocity block comes from
`utilities/multigrid_options.py`, shared with the native and standard custom-P
routes, so the rotated block cannot be configured differently by accident. The
single deliberate difference is the coarse solve: `coarse="svd"`, because the
Galerkin-coarsened rotated block inherits the rigid-rotation null space where
`redundant`/LU hits a zero pivot. See
[the solvers subsystem doc](solvers.md) for the bundle itself.

### The velocity sub-KSP must converge, not just apply

The velocity block is preconditioned by multigrid, but the KSP *around* that
multigrid is FGMRES to `0.1 × tolerance` — never `preonly`. This is not a
tuning preference. `PCFieldSplit` forms the Schur complement
`S = A₁₁ − A₁₀ A₀₀⁻¹ A₀₁` and applies `A₀₀⁻¹` through this same velocity KSP,
so `preonly` replaces `A₀₀⁻¹` with a single multigrid cycle and hands the
pressure Krylov a *different* system `S̃ ≠ S`, preconditioned by a 1/μ mass
matrix built for `S`.

Measured on an annulus with a weak plane reaching the constrained boundary
(`η₁/η₀ = 1e-3`, transversely isotropic): under `preonly` the pressure residual
falls by 4.4e4 in about 16 iterations and then **stagnates at a floor ≈ 3.1e-7**,
spending the remaining 184 iterations of its cap for nothing — on every outer
iteration, 9 outer iterations in total. Under FGMRES it converges by 1.1e8
monotonically in 17 pressure iterations, and the outer solve takes 1. The
isotropic control moves the same way (5 → 1), so this is a property of the Schur
application rather than of the anisotropy.

Note the Schur application is bitwise reproducible under both settings, so the
floor is *not* "the operator changes between applications" — `S̃` is a fixed
linear operator, just the wrong one. The precise origin of the floor (most
plausibly a range/null-space inconsistency between `S̃` and the constant-pressure
null space attached to `S`) has not been isolated.

The native (non-rotated) Stokes path and the GAMG fallback have always wrapped
their multigrid this way; the rotated custom-FMG branch was the sole exception.
`max_it` matches both (200); `rtol` matches the GAMG fallback (0.1 × tol), where
the native path asks for 0.033 × tol.

Two things make this easy to miss, and both are now instrumented. The outer
iteration count does not reveal it — a full Schur factorisation with a good
pressure mass still reports ~1 outer iteration while the inner solve grinds
underneath. And a velocity FGMRES that exhausts its cap returns
`KSP_DIVERGED_ITS`, which `KSPCheckSolve` deliberately does not escalate, so it
would degrade silently where `preonly` simply could not fail; the rotated path
now warns on it.

`solver._rotated_freeslip_info` therefore reports `velocity_pc`, `schur_pre` and
`velocity_pc_type` (PETSc's own view of the sub-PC), plus `vel_its_last` and
`pres_its_last`. The last two are **last-application samples, not work**:
`KSPGetIterationNumber` reports only the most recent solve, and the velocity KSP
is applied once per Schur `MatMult`. They are named `_last` so they are not
confused with the summed counts `solver_health` uses as its work axis; wiring
the rotated path into `SolverInstrumentation.sub_reports()` is the proper fix and
has not been done. On the opt-in `solver._rotated_use_lu` path there are no
sub-KSPs at all and both fields record `None`.

The manual loop exists because the rotated operator `Q A Qᵀ` carries no DM
field information, so PETSc's DM-coupled fieldsplit cannot precondition it;
the increment is solved by an IS-built fieldsplit instead, driven from the
loop. The loop honours `zero_init_guess`, `picard`, and the solver's
`consistent_jacobian` tangent policy (frozen / Newton / continuation).

### How the datum passes through Newton

The constraint is affine, and the design keeps every **accepted iterate
feasible** (`u·n̂ = ũ_n` exactly, via an affine snap in the rotated frame), so
each Newton increment satisfies the *homogeneous* constraint `n̂·δ = 0` — the
datum never touches the tangent, the increment right-hand side, the FMG
prolongation, or the null-space handling.

The one deliberate exception is a **cold start with a non-zero datum**: the
first increment carries the datum jump through the affine lift
(`zeroRowsColumns(rows, diag, x̂, b̂)`) at the rest-state tangent, and is
accepted without a line search. Snapping the zero state onto the datum instead
manufactures an extreme boundary-strip strain state whose shear-thinning
tangent produces a first step orders of magnitude too large — no line-searched
step descends (measured). A warm start is assumed smooth and takes the exact
snap directly.

### Convergence semantics

The loop converges on `‖F̂‖ ≤ rtol·ref + atol` with
`ref = max(‖F̂(u₀)‖, ‖F̂(0)‖)`: the **rest-state residual is the intrinsic
forcing scale**, so a good warm start (whose own initial residual is small) is
not punished with an ever-stricter absolute target. A step-norm exit (tiny
Newton step) is *verified* against the same reference before it may report
convergence — a stiff tangent also produces tiny steps far from the solution,
and an unverified tiny step goes back through the line search (a slow crawl
continues; a genuine stall ends the loop with an explicit warning). An
unconverged exit always warns; the fields hold the last iterate.

### Null-space handling

Rigid-rotation candidates (one mode on a closed circle, three on a spherical
shell) are admitted to the increment null space and the post-solve gauge
removal only if they pass **two** tests: tangential to every rotated
constraint row (`Q·m ≈ 0` there), **and** a null vector of the assembled
operator (`‖J·m‖ ≈ 0` against the velocity diagonal scale). The second test is
what catches pinning the first cannot see — an essential condition on another
boundary leaves the mode tangential to the rotated wall while the eliminated
operator is *not* null on it, and admitting it projects an irreducible
component out of every increment (the residual then floors far above
tolerance instead of converging). The null space is therefore built after the
first Jacobian assembly.

### Parallel notes

- Every branch the loop takes is decided **collectively**. The datum-activity
  flag rides a PETSc `Vec` norm of the lift vector (the datum bookkeeping
  itself is rank-local — only owners of datum boundary nodes hold entries —
  and branching on it per-rank desynchronises the collective sequences: the
  np>1 deadlock class).
- All row surgery on vectors uses ownership-relative indices
  (`_zero_rows_local` / `_set_rows_local`); indexing a local slice with global
  rows is the np>1 crash class.
- Direct LU per increment (`solver._rotated_use_lu = True`) is a **serial**
  preconditioner-free diagnostic (the pressure-gauge pin is a naive per-rank
  scan, marked `TODO(BUG)`); use it to separate "the operator/constraint is
  wrong" from "the preconditioner is struggling".

## Result and reaction

The solve fills the solver's fields and stores a result dict
(`solver._rotated_freeslip_info`) with the rotation, the constrained rows, the
per-increment KSP iteration counts, the convergence verdict, and the
**reaction** — the converged Cartesian residual `F(u)`, which for a linear
residual equals `A·u − b` exactly. `boundary_normal_traction` projects the
nodal reaction onto the boundary normal (corner-correct) and de-smears it with
the shared boundary-mass machinery in `utilities/boundary_flux.py`;
`dynamic_topography_field` writes `h = −(σ_nn − σ̄_nn)/(Δρ g)` onto a surface
field for the free-surface integrator.

### Which de-smearing mass (3-D)

`mass="auto"` uses the **P1-projected** recovery on a 3-D P2 trace. The full
("consistent") P2 **triangle** mass has vertex rows summing to exactly zero, so
it is singular on constants along those rows and `M⁻¹` amplifies any
perturbation of the nodal load at *vertices* by O(1) — independently of
resolution. Measured on the Zhong l=2 shell (#633): consistent-mass vertex
values are 7.6% low with no grad-div penalty, 28% low at `penalty=10` and 79%
low at 100, and the error is flat across a 3.2× refinement. The 2-D P2 **line**
mass has positive vertex row sums and is unaffected — which is why this appears
only in 3-D, and not because of dimension or curvature as such.

This is a trade rather than a strict improvement. Consistent-mass *midpoint*
values are superconvergent (0.1–1.5% at every penalty) and the P1 projection
gives that up, because under it the midpoints are the P1 interpolant of the
vertices. `"p1"` is the default because its error converges under refinement
(worst node 0.170 → 0.049 over cellSize 0.30 → 0.16) where the consistent
vertex error does not. **If you sample only midpoints, pass
`mass="consistent"`.** See also #404 (the vertex-integral checkerboard) and
#637 (only P1/P2 triangular traces are supported in 3-D at all).

## Tests

`tests/test_1018_rotated_freeslip.py` (serial: essential-equivalence, FMG,
tangent policies, datum linear + nonlinear),
`tests/parallel/test_1066_rotated_datum_parallel.py` (np≥2: partition
independence of the linear datum and the nonlinear Newton datum path).
