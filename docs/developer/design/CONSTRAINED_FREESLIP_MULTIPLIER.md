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

`C` is **co-dimension-1**: it touches only velocity DOFs on Γ. The **shipped**
solver carries `λ` as a third field **inside** the saddle point and solves the
whole 3×3 system in one coupled solve (the `[p, λ]` rows are grouped into a
single Schur factor — see the "Monolithic `P'=[p,λ]` fieldsplit" section). The
multiplier carries the *exact* constraint; there is **no outer loop**.

### Augmented-Lagrangian stabilisation `r`

The u-row carries `λ` plus an augmented-Lagrangian penalty:

$$\mathbf{t} = \bigl[\lambda + r\,(\mathbf{u}\cdot\mathbf{n} - g)\bigr]\,\mathbf{n}
\quad\text{on } \Gamma,$$

which adds a `uu` boundary stiffness `r(n⊗n)` that conditions the `[p, λ]` Schur
complement **without biasing the multiplier** (the λ-row stays the exact
constraint). It is *not* an outer multiplier update — `λ` is solved
monolithically. Because accuracy is independent of `r`, `r` is a cost-only knob:
larger values reduce the iteration count up to a broad plateau, well below the
roundoff limit. Default `r = augmentation_base · μ(x)` with
`augmentation_base = 1e4` (viscosity-weighted, mesh-independent).

> **Historical note.** An earlier *outer-loop* (Uzawa / ALG2) variant updated
> `λ ← λ + r(u·n − g)` between Stokes solves. It was superseded by — and removed
> in favour of — the in-saddle formulation above. The Phase-0 spike findings
> below motivated that exploration and are kept for context.

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

`solve()` does **one coupled solve** — no outer iteration or constraint tuning.
The augmentation defaults to `1e4·μ(x)` (local-viscosity-weighted); accuracy is
independent of it (the λ-row carries the exact constraint), so no per-problem
tuning is needed.

Key design points:

- **Multiplier representation.** `λ` is a full-mesh scalar field at the *velocity
  degree* (P2). Only its trace on Γ enters the weak form. Matching the velocity
  degree means the multiplier reaches every velocity normal-trace DOF (including
  P2 mid-edge), so there is no constraint floor.
- **Boundary-only reduction → clean topography.** The interior (off-boundary) λ
  DOFs are constrained directly in the PetscSection, so the solved `[p, λ]` block
  carries only the boundary trace (~√ndof DOFs, ~1.1× Dirichlet rather than ~3×).
  Interior `λ` is absent, so the boundary `λ` is a directly usable topography
  field. The reduction is lossless (machine-precision constraint) and default-on.
- **Coupling registered once.** The boundary residual/Jacobian
  (`λ·n`, the AL stiffness `r(n⊗n)`, and the `uλ`/`λu` couplings) are registered
  a single time; nothing recompiles between solves.
- **`add_constraint_bc(boundary, g=0, normal=None, augmentation=None)`** —
  `normal` defaults to the smooth projected normals `mesh.Gamma_P1`;
  `augmentation` defaults to a viscosity-scaled `r = 10⁴·μ`.

## Validation

Two regression tests cover the shipped solver:

- `tests/test_1061_constrained_freeslip.py` — box (vs an exact Dirichlet
  free-slip reference) and buoyancy-driven annulus (vs a `1e6` penalty
  reference): constraint enforced (`RMS(u·n)` small with no penalty coefficient),
  velocity matches the reference, and `corr(λ, −n·σ·n) ≈ 0.9999` (topography).
- `tests/test_1062_constrained_solcx.py` — free-slip via four in-saddle
  multipliers on the **SolCx** benchmark (1e6 viscosity jump) compared to the
  **exact analytic** solution: velocity `rel ≈ 8.7e-6` (== the Dirichlet
  baseline), constraint `RMS(u·n) ≈ 1.6e-10`.

The consistent-boundary-flux identity `λ = −n·σ·n|_Γ` is the independent
cross-check: the multiplier's boundary trace equals the recovered normal Cauchy
stress (negative sign = the reaction traction holding the boundary), confirming
`λ` is the dynamic topography signal.

## The augmentation parameter `r`: true-work trade-off

`r` is a *speed* knob, not an *accuracy* knob — this is the key advantage over a
pure penalty, and it carries over to the in-saddle solver (accuracy is
`r`-independent; `r` only sets the iteration count). The sweep table below is from
the **historical outer-loop** variant (its "outer iterations" have no analogue in
the one-shot coupled solve), but the shape and the conclusion stand. For the
in-saddle solver with a scalable (FMG) inner solve the iteration count falls
monotonically with `r` to a saturation floor, with no high-`r` penalty until
roundoff; the default `r = 10⁴·μ` sits comfortably in that regime.

Historical outer-loop sweep on the annulus (constraint tol 1e-4, wall time for one
cold solve; `tot_lin` = total outer Schur-KSP linear iterations across the loop):

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
| **(D) monolithic in-saddle multiplier (grouped `[p,λ]` Schur)** | **Implemented** (the shipped solver). One coupled solve, exact, recoverable topography, boundary-only reduction. |
| (E) augmented-Lagrangian outer loop | Earlier exploration; **removed** in favour of (D). Easy to reproduce in Python. |

Deferred to follow-up PRs: a true co-dim-1 / MATNEST `λ` representation; 3D
spherical shells; **parallel** (the boundary handling is serial); the
**both-boundaries-free-slip** annulus case (rigid-rotation velocity null space
needing explicit removal); and live free-surface equilibrium integration (pass
`λ` as the target normal-stress end-state).

## Monolithic `P'=[p,λ]` fieldsplit — the shipped design

This is the implemented approach (and a step toward a general "inject arbitrary
constraints into the saddle point" capability): rather than a third field forcing
a nested 3-way Schur, group pressure and the multiplier into a composite
`P' = [p, λ]` and keep a **2-way `u | P'`** split.

**Spike result (confirmed):** a 3-field DM `(u, p, h)` on a real mesh, with the `p`
and `h` index sets grouped (`pc_fieldsplit_1_fields 1,2`, or an explicit
concatenated IS), produces exactly a 2-block `u | [p,h]` Schur fieldsplit
(block sizes 84 | 84 = u | (p+h) on a coarse test). The nested-Schur /
KSP-reconfiguration concern is therefore moot — the split structure is identical to
the current `u | p` solver. (`/tmp/spike_pph.py`.)

**What was implemented (behind the `SNES_Stokes_Constrained` subclass):**
- `λ` registered as field 2 (`dm.setField`).
- `λ`-equation residual: boundary part `∫_Γ ψ(n·u − g)` plus a small interior
  screening `ε∫_Ω λ ψ` to de-singularise the interior block — which is then
  **constrained away** by the boundary-only reduction (the interior λ DOFs are
  pinned in the PetscSection, so only the boundary trace is solved).
- Boundary Jacobian blocks `uλ`, `λu` and the AL stiffness `uu += r(n⊗n)`
  (`ph`/`hp` are zero); registered via the `UW_PetscDSSetBdJacobian` machinery.
- `[p,λ]` grouped in the fieldsplit by field index (keeps the velocity DM
  hierarchy for geometric MG/FMG); the gauge nullspace handled as a combined
  `(p, λ)` mode on enclosed problems.

**Caveats:** the work touches the validated `uu/up/pu/pp` assembly, so it lives
behind the subclass and is regression-tested (`tier`-graded). Serial only for
now. A true *co-dimension-1* `λ` (boundary-only DOFs end-to-end) remains the
honest long-term form; the full-domain field + boundary-only reduction is the
pragmatic path that ships today.

## Files

- `src/underworld3/systems/solvers.py` — `SNES_Stokes_Constrained`, `_BlockConstraintBC`.
- `src/underworld3/cython/petsc_generic_snes_solvers.pyx` — multiplier-field
  registration, boundary residual/Jacobian coupling, fieldsplit grouping,
  nullspace, and the section-based interior-multiplier reduction.
- `src/underworld3/systems/__init__.py` — `Stokes_Constrained` export.
- `tests/test_1061_constrained_freeslip.py` — box + annulus validation.
- `tests/test_1062_constrained_solcx.py` — SolCx analytic validation.
