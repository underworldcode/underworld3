# Constrained free-slip via a recoverable Lagrange multiplier (dynamic topography)

**Status**: shipped as `uw.systems.Stokes_Constrained` (serial). The constraint
is enforced by a multiplier carried **inside** the saddle point (one coupled
solve); the converged boundary multiplier is the normal traction = dynamic
topography. An earlier augmented-Lagrangian **outer-loop** variant was removed in
favour of this in-saddle formulation (it is straightforward to reproduce in
Python if needed). Validated against the exact SolCx analytic solution
(`tests/test_1062_constrained_solcx.py`).

## Motivation

Free-slip / no-normal-flow on curved (annulus, spherical) boundaries is
currently enforced with **penalty-like** methods — a penalty natural BC
(`add_natural_bc(penalty · Γ·v · Γ, ...)`) or Nitsche. These are fragile: the
penalty magnitude must be tuned against the Rayleigh number and viscosity. Too
weak and a coherent radial throughflow appears (an under-scaled `1e4` natural BC
is ~100× too weak at Ra=1e6); too strong and the system ill-conditions and the
Stokes solve diverges in line search.

This feature enforces `u·n = g` on a curved boundary with a **true Lagrange
multiplier** `λ` instead of a penalty. Because the converged multiplier *is* the
normal traction holding the boundary, it is simultaneously a direct estimate of
**dynamic surface topography**, `h = λ / (Δρ g)`. The equilibrium `λ` is also the
target end-state toward which a free surface can be integrated over a time
interval (connecting to the ETD free-surface work on
`feature/exp-integrator-freesurface`).

## Formulation

Stokes with a surface constraint `u·n = g` on Γ, multiplier `λ`:

```
[ A    Bᵀ   Cᵀ ] [u]   [f]
[ B    0    0  ] [p] = [0]      A = viscous,  B = div,  C = ∫_Γ (n·v) ψ
[ C    0    0  ] [λ]   [g]      (C couples only the boundary trace of u)
```

`C` is **co-dimension-1**: it touches only velocity DOFs on Γ. A monolithic
third FE field would therefore either waste interior DOFs or need a boundary
trace space PETSc/DMPlex does not provide on the same DM. We instead solve the
boundary Schur complement `S_λ = C K⁻¹ Cᵀ` by an **outer loop**, leaving the
validated 2×2 Stokes assembly untouched.

### Augmented Lagrangian (ALG2) — the production algorithm

Each Stokes solve carries a natural-BC traction with **both** the multiplier and
a penalty augmentation, reusing the existing penalty machinery:

$$\mathbf{t} = \bigl[\lambda + r\,(\mathbf{u}\cdot\mathbf{n} - g)\bigr]\,\mathbf{n}
\quad\text{on } \Gamma,$$

and the multiplier is updated with the same `r`:

$$\lambda \leftarrow \lambda + r\,(\mathbf{u}\cdot\mathbf{n} - g).$$

The penalty term `r(u·n)n` preconditions *every* boundary mode uniformly, so the
outer loop converges in a handful of iterations; the multiplier removes the
penalty's accuracy bias, so a **moderate, well-conditioned** `r` gives both fast
convergence *and* an exact constraint — unlike a pure penalty, which must be made
large and fragile.

## Phase-0 spike findings (what shaped the design)

Spikes (`/tmp/s3_*.py`) on a 2D annulus (no-slip inner boundary to remove the
rigid-rotation null space, multiplier free-slip on the outer boundary):

- **Plain Uzawa works but is slow.** A damped-Richardson update
  `λ ← λ + ρ(u·n)` converges and matches the penalty solution, but a single
  scalar `ρ` cannot kill both the fast and slow boundary-Schur modes — the
  residual contracts the dominant mode in ~5 iterations then crawls.
- **`ρ ∝ μ`, NOT `ρ ∝ μ/h`.** The optimal Richardson step is `ρ ≈ C·μ` with `C`
  a geometry constant, **independent of mesh resolution** (`ρ=8μ` converged in 5
  iterations at cellSize 0.1/0.05/0.025). The naive `μ/h` scaling over-steps on
  refinement and stalls.
- **CG is the wrong accelerator.** CG on `S_λ` diverged: each matvec is an
  *inexact* iterative Stokes solve (plus pressure-null-space noise), and the
  nodal Euclidean inner product is not the one in which `S_λ` is SPD. Krylov
  acceleration needs an exact symmetric operator; this is neither.
- **Augmented Lagrangian is the right accelerator** (per L. Moresi). It converges
  in **2 iterations** where plain Uzawa took 21, reusing the existing penalty BC.
  This is the implemented algorithm.

## Implementation

`SNES_Stokes_Constrained(SNES_Stokes)` in `src/underworld3/systems/solvers.py`,
exported as `uw.systems.Stokes_Constrained`. **Purely additive** — the validated
2×2 saddle-point assembly and fieldsplit configuration are untouched, honouring
"solver stability is paramount".

```python
stokes = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = mu
stokes.bodyforce = buoyancy * unit_r
stokes.add_dirichlet_bc((0.0, 0.0), "Lower")            # no-slip inner

lam = stokes.add_constraint_bc("Upper", g=0.0)          # free-slip outer (Gamma_P1)
stokes.solve()                                          # no constraint tuning needed

topo = stokes.topography("Upper", buoyancy_scale=delta_rho_g)   # h = lambda/(drho g)
```

The outer loop is hidden behind ``solve()``: the tolerance is **relative** to the
velocity scale (``RMS(u.n-g) < rtol·RMS|u|``, default ``rtol=1e-3``) so it is
problem-independent, and the augmentation defaults to ``1e3·μ(x)`` (local-
viscosity-weighted). On SolCx this gives ~2e-4 accuracy in 3 outer iterations at
viscosity contrast 1 → 10⁶ with no user tuning.

Key design points:

- **Multiplier representation.** `λ` is an ordinary full-mesh scalar field at the
  *velocity degree* (P2). Only its trace on Γ enters the weak form. Matching the
  velocity degree means the multiplier reaches every velocity normal-trace DOF
  (including P2 mid-edge), so there is no penalty floor on the constraint — a P1
  multiplier plateaus at ~`‖t‖/r` on the mid-edge component.
- **Clean topography field.** The dual update is restricted to boundary nodes via
  a P1 boundary marker (`petsc_dm_find_labeled_points_local`) sampled at the
  multiplier's nodes — boundary mid-edge nodes interpolate to 1, interior to
  < 0.5. Interior `λ` therefore stays **exactly zero**, so `λ` is a directly
  usable topography field (not interior garbage).
- **Coupling registered once.** `add_natural_bc([λ + r(u·n − g)]·n, boundary)`
  is set up a single time; only `λ`'s boundary data changes between solves, so
  nothing recompiles.
- **`add_constraint_bc(boundary, g=0, normal=None, augmentation=None)`** —
  `normal` defaults to the smooth projected normals `mesh.Gamma_P1`;
  `augmentation` defaults to a viscosity-scaled `r = 10³·μ`.

## Validation

`tests/test_1061_constrained_freeslip_annulus.py` (level_2 / tier_b), buoyancy-
driven annulus, no-slip inner + multiplier free-slip outer, vs a `1e6` penalty
reference. All pass (~14 s):

| Check | Result |
|---|---|
| `RMS(u·n)` on outer boundary (no penalty coefficient) | 6.5e-5 |
| `relL2(v_multiplier vs v_penalty)` | 3.1e-3 |
| Outer iterations (augmented Lagrangian) | 2 |
| Interior `λ` (clean field) | exactly 0 |
| boundary `corr(λ, −n·σ·n)` (dynamic-topography stress) | 0.9999 |

The consistent-boundary-flux identity `λ = −n·σ·n|_Γ` is the independent
cross-check: the multiplier's boundary trace equals the recovered normal Cauchy
stress (negative sign = the reaction traction holding the boundary), confirming
`λ` is the dynamic topography signal.

## The augmentation parameter `r`: true-work trade-off

`r` is a *speed* knob, not an *accuracy* knob — this is the key advantage over a
pure penalty. Sweep on the annulus (constraint tol 1e-4, wall time for one cold
solve; `tot_lin` = total outer Schur-KSP linear iterations across the loop):

cellSize 0.1 (~2940 dof):

| `r` | outer its | tot_lin | wall (s) | relL2 vs penalty |
|---:|---:|---:|---:|---:|
| 10 | 26 | 26 | 3.77 | 3.1e-3 |
| 100 | 4 | 4 | 0.61 | 3.2e-3 |
| 300 | 3 | 3 | 0.47 | 3.1e-3 |
| 1,000 | 3 | 3 | 0.53 | 3.1e-3 |
| 3,000 | 2 | 2 | **0.38** | 3.1e-3 |
| 10,000 | 2 | 2 | 0.51 | 2.9e-3 |
| 100,000 | 2 | 5 | 1.83 | 1.6e-3 |

cellSize 0.05 (~10852 dof) shows the same shape (min wall ≈ 1.25 s at r=1e3;
6.98 s at r=10; 7.07 s at r=1e5).

- **Outer iterations fall with `r`** (`26 → 4 → 3 → 2`) — bigger penalty, faster
  dual convergence (`contraction ≈ ‖S_λ‖/(r+‖S_λ‖)`).
- **But the inner solve stiffens at large `r`.** Linear iterations per outer
  solve stay at 1.0 up to `r=10⁴`, then rise (2.5 at `r=10⁵`), and wall time
  balloons (the velocity sub-block conditioning degrades — visible in wall time
  even before the outer KSP count moves).
- **True work is U-shaped**: both extremes are 3–8× slower than the optimum. The
  efficient basin is `r ∈ [300, 10⁴]` (>1.5 decades) at both resolutions; the
  default `r = 10³·μ` sits inside it.
- **Accuracy is `r`-independent** (relL2 ≈ 3.1e-3, flat across four decades of
  `r`). So `r` is tuned for *speed* with a benign failure mode — too small just
  costs iterations, too large just costs inner work; **the answer is never
  wrong**. Contrast a pure penalty, where the magnitude must be tuned against
  forcing strength and viscosity to get *accuracy* (too small ⇒ wrong), which is
  the fragility this method removes.

## Option trade-offs and what is deferred

| Option | Verdict |
|---|---|
| (A) full-domain 3rd FE field + ε-screening | 3-way nested fieldsplit; ε re-introduces tuning. Rejected as primary. |
| (B1) boundary-stratum-only PetscFE field | No DMPlex support on the same DM. |
| (B2) co-dim-1 submesh + MATNEST | The honest monolithic form; deferred. |
| (C) reuse pressure / `_constraints` | `p` enforces `∇·u=0` interior, not `u·n=0` on Γ — not redundant. The CBF identity is a *validation* tool, not an implementation. |
| **(D) augmented-Lagrangian outer loop** | **Implemented.** Non-invasive, 2 iterations, exact, recoverable topography. |

Deferred to follow-up PRs (Phase-0 spike S2 gathers the fieldsplit-feasibility
evidence): the monolithic 3-field / co-dim-1 representation; 3D spherical shells;
**parallel** (the boundary mask and the nodal update are serial); the
**both-boundaries-free-slip** annulus case (admits a rigid-rotation velocity null
space needing explicit removal); and live free-surface equilibrium integration
(pass `λ` as the target normal-stress end-state).

## Monolithic `P'=[p,h]` fieldsplit — feasibility spike

A future direction (and a general "inject arbitrary constraints into the saddle
point" capability): rather than a third field forcing a nested 3-way Schur, group
pressure and the multiplier into a composite `P' = [p, h]` and keep a **2-way
`u | P'`** split.

**Spike result (confirmed):** a 3-field DM `(u, p, h)` on a real mesh, with the `p`
and `h` index sets grouped (`pc_fieldsplit_1_fields 1,2`, or an explicit
concatenated IS), produces exactly a 2-block `u | [p,h]` Schur fieldsplit
(block sizes 84 | 84 = u | (p+h) on a coarse test). The nested-Schur /
KSP-reconfiguration concern is therefore moot — the split structure is identical to
the current `u | p` solver. (`/tmp/spike_pph.py`.)

**Remaining work (bounded, ~2 weeks of Cython, behind a subclass):**
- Register `h` as field 2 (one `dm.setField`).
- `h`-equation residual: boundary part `∫_Γ ψ(n·u − g)` (the existing Nitsche path
  already registers field-1 boundary residuals — same pattern) plus a small interior
  screening `ε∫_Ω h ψ` to de-singularise the interior `h` block.
- Three new Jacobian blocks: `uh`, `hu` (boundary integrals — the `UW_PetscDSSetBdJacobian`
  machinery already does arbitrary field pairs for `up`/`pu`) and `hh` (interior mass);
  `ph`/`hp` are zero.
- Group `[p,h]` in the fieldsplit and extend the Schur PC (`p` keeps its `1/μ` mass PC;
  `h` gets the screening diagonal).
- The 3-IS field decomposition / nullspace path (currently assumes 2 fields).

**Caveats:** the interior screening reintroduces a small `ε` (benign — it does not bias
the boundary multiplier); the work touches the validated `uu/up/pu/pp` assembly, so it
must live behind a subclass and be regression-tested. The one genuine PETSc limitation
remains a *co-dimension-1* `h` (boundary-only DOFs) — avoided here by the full-domain +
screening representation.

**Recommendation:** the architecture is de-risked, but since the outer loop converges in
2–3 iterations with no user-facing tuning, monolithic is scheduled work (a general
saddle-point-constraint capability), not an urgent replacement.

## Files

- `src/underworld3/systems/solvers.py` — `SNES_Stokes_Constrained`, `_ConstraintBC`.
- `src/underworld3/systems/__init__.py` — `Stokes_Constrained` export.
- `tests/test_1061_constrained_freeslip_annulus.py` — validation.

Pre-existing (unrelated) failures noted: `tests/test_1060_nitsche_freeslip.py`
has two failing assertions on `development` independent of this work.
