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

lam = stokes.add_constraint_bc(0.0, "Upper")            # free-slip outer (Gamma_P1)
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
- **`add_constraint_bc(conds, boundary, normal=None, augmentation=None)`** (value-first, Style Charter API conventions) —
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

## Conditioning, the augmentation's true role, and rigid-body null spaces

This section records what a focused study of strong **boundary viscosity
contrast** (annulus lateral `μ_hi ≥ 10³`, SolCx's `1e6` jump) revealed about
*why* the constraint block is sometimes hard to solve — and corrects an
over-simplification in the "augmentation is purely a speed knob" framing above.

### The constraint Schur preconditioner (`selfp`, the shipped default)

The grouped `[p, λ]` block is preconditioned through its Schur complement. With
the base-Stokes default `pc_fieldsplit_schur_precondition = a11`, that
preconditioner is the assembled `a11` block — a viscosity-scaled **mass** on
pressure (`1/μ`) and the small screening on `λ`. The pressure part is the right
scaling (`S_p ≈ μ⁻¹ M_p`); the **`λ` part is not** — the true constraint Schur is

$$S_\lambda = C\,A^{-1}C^{\mathsf T},$$

a boundary operator that scales like `1/μ` but is *not* a simple `λ`-mass. Under
strong contrast the bare `a11` mass is a poor approximation of `S_λ` and the
solve walls (or needs a very large augmentation to compensate).

`SNES_Stokes_Constrained` therefore defaults to
**`pc_fieldsplit_schur_precondition = selfp`** instead. `selfp` forms

$$S \approx A_{11} - A_{10}\,\operatorname{diag}(A_{00})^{-1}A_{01}$$

from the *actual* operator blocks; its `λλ` corner is
`−C diag(A)⁻¹ Cᵀ`, i.e. the true constraint Schur, **automatically and at no
extra assembly**. On a smooth lateral viscosity ramp the outer Krylov count is
flat (≈2–6 iterations) across `μ_hi = 1 … 10⁶`, where the bare `a11` mass climbs
to ~25 and diverges. Override with
`solver.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"` if needed.

**Why `selfp` and the `r ∝ μ` augmentation are mutually consistent.** The bare
constraint Schur is `S₀ = C A⁻¹ Cᵀ ~ 1/μ`. The augmentation adds `r·N` to `A`
(with `N ~ CᵀC`, the boundary `n⊗n`), so by Woodbury the *augmented* Schur is
`S_r = C(A + rN)⁻¹Cᵀ ≈ S₀(I + r S₀)⁻¹` — equal to `S₀ ~ 1/μ` for small `r` and
tending to `1/r` for large `r`, with the **crossover at `r·S₀ ~ 1`, i.e. `r ~ μ`**.
So the default `r = augmentation_base · μ(x)` is the AL-natural (crossover)
scaling. And `selfp` builds its Schur from `diag(A + rN)`, whose boundary diagonal
is `μ + r = μ(1 + augmentation_base)` — so `selfp`'s `λλ` block `~ 1/r`
**automatically tracks the augmented Schur, because `diag(A)` already contains the
penalty.** This is *why* they compose so well: the approximate inverse "sees" the
augmentation. A **constant** `r` (instead of `∝ μ`) was tested and is strictly
worse — it is negligible on the stiff side (`r ≪ μ`, no regularisation) *and*
breaks the uniform `diag(A)` scaling that `selfp` relies on, giving garbage
velocity at SolCx 1e6 (`velerr 1–12` across `r = 10²–10⁴`, vs `1.3e-4` for
`r ∝ μ`). The `μ`-weighting is load-bearing; keep it.

### The augmentation has two roles — conditioning *and* null-space regularisation

The earlier sections describe `r` as a *speed knob* that conditions the `[p, λ]`
Schur complement. That is correct, but incomplete. `r(n⊗n)` does **two**
independent things:

1. **Conditions `S_λ` from the velocity (A) side.** Stiffening the boundary
   velocity is one way to make the constraint Schur well-behaved. `selfp`
   conditions the *same* operator from the Schur side. They are **substitutes**:
   on a viscosity-contrast sweep, increasing `r` *or* switching to `selfp` both
   collapse the iteration count. With `selfp` the augmentation can drop to zero on
   a **single-constraint** problem and the solve still converges.

2. **Regularises the velocity block's rigid-body null space** — a *structural*
   role that `selfp` alone cannot fill (see below). This is why an all-free-slip
   enclosed box still needs `r > 0` even with `selfp`.

So `r` is not "just an accelerator". On problems with a velocity anchor (a no-slip
patch) it is optional with `selfp`; on problems with **no** velocity anchor it is
doing genuine null-space regularisation.

### Rigid-body null spaces: a general principle (not specific to constraints)

The velocity operator `A = ∫ 2μ ε(u):ε(v)` has the **zero-strain rigid-body
motions in its kernel for *any* mesh** — **2 translations + 1 rotation in 2D, 3 +
3 in 3D** — removed *only* by Dirichlet velocity DOFs. In a monolithic direct
solve this is hidden, but a **Schur-factored** solve inverts `A` as an *inner*
block that sees only `A` (not the constraint rows or the pressure), so the
singularity surfaces there.

```{important}
Whenever **nothing pins velocity anywhere** — all free-slip, all-natural, the
free-slip spherical shell — the inner `A`-solve is singular along the rigid-body
modes. The constraints kill those modes in the *full* system but never in the
*inner* `A`-solve, so it can drift before the outer coupling acts. In the
**block-constrained** solver the working remedy is the **augmentation**
`r(n⊗n)`: it stiffens those modes out of `A` directly, so the inner solve is
well-posed. This is a *third*, structural role of the augmentation (beyond
conditioning the Schur complement), and it is why an all-free-slip enclosed
problem needs `r > 0` even with `selfp`.
```

```{warning}
**Do not** try to remove the inner-`A` singularity by attaching the rigid-body
modes to the *coupled* null space (`petsc_velocity_nullspace_basis`) on a
**constrained** problem — neither the rotation nor the translations work cleanly:

- The **rotation** carries a *real* part of the answer: a closed `u·n=0`
  incompressible flow generically has nonzero circulation `∫(x u_y − y u_x) dV`,
  so it is not orthogonal to the rotation mode. Projecting it out corrupts the
  velocity (measured: SolCx velerr `3e-1`; with all three modes, `≈1.0`).
- The **translations** *are* orthogonal to the solution in **L2**
  (`∫u_i dV = 0` for incompressible `u·n=0` on a closed boundary), so in principle
  they are harmless to suppress. But PETSc projects against the **Euclidean nodal**
  inner product, and the unweighted nodal sum `Σ u_i` is not zero even when the
  integral is — so projecting the constant mode still injects error (SolCx velerr
  `1e-4 → 1.4e-2`). Honouring the L2 orthogonality needs **mass-weighted** null
  vectors.

Attaching the modes to the **velocity sub-block** instead (the inner solve) was
prototyped and is *less wrong* than the coupled route — at `aug=10⁴` it degrades
SolCx velocity to `4e-3` rather than destroying it (`≈1.0`) — but it still does
not match the augmentation (`1.3e-4`) and does not enable small/zero `aug`. The
reason is that `selfp` builds its Schur preconditioner from `diag(A)`, which is
**not** rank-deficient even when `A` is, so the preconditioner is blind to the
sub-block null space. A clean version would need the null space attached natively
(`DMSetNullSpaceConstructor`, before setup) *and* a Schur PC that respects it —
an unproven, non-trivial enhancement. **In practice the augmentation is the
working remedy for unanchored free-slip** (it stiffens the modes out of `A`
directly, with no pseudo-inverse or inner-product subtleties). Note
`petsc_velocity_nullspace_basis` remains correct for a genuine *coupled* null
mode, e.g. the rigid **rotation** of a free-slip spherical shell solved *without*
a constraint multiplier.
```

The buoyancy-driven annulus escapes the singularity entirely because its
**no-slip inner boundary** already pins all rigid-body modes in `A`; there `aug`
is fully optional (aug=0 and aug=10⁴ velocities agree to `3e-4`) and small `aug`
gives clean velocity, pressure, *and* topography at once.

One further consequence:

- The bare velocity rigid-body modes are the **GAMG near-null space** of the
  velocity block. Set as a *near*-null space on that sub-block
  (`MatSetNearNullSpace`, distinct from the coupled-operator null-space projection
  warned against above) they improve the velocity multigrid coarse spaces —
  standard elasticity/Stokes-multigrid practice — independent of the singularity
  question. This is the same sub-block plumbing the clean inner-`A` fix would use.

### Solver type at extreme contrast

A fixed-viscosity Stokes solve is **linear**. The default `newtonls` performs an
inexact-Newton defect correction; when the Schur approximation is stiff (extreme
contrast) it can take many "Newton" steps or stall, even though each linear solve
is cheap. Using `snes_type = "ksponly"` solves the linear system directly and is
markedly more robust (and faster) for constrained free-slip at strong contrast.

### Topography: the mixed / penalty / augmented triad

The boundary constraint mirrors the **incompressibility** triad, with `λ` playing
the role of `p` (`λ : u·n=g  ::  p : ∇·u=0`):

| formulation | analogue | topography signal |
|---|---|---|
| **mixed** (`r = 0`, multiplier) | Taylor–Hood `p` (a real DOF) | `λ` *is* `−σ_nn`, directly |
| **penalty** (no `λ`, large `r`) | `p = −λ_pen ∇·u` | `−r(u·n)` (recovered from the residual) |
| **augmented** (`λ + r`) | Uzawa / ALG2 | `−σ_nn = λ + r(u·n)` — a **mix** |

The δu equation gives the boundary traction `−(λ + r(u·n − g))`, so the recovered
topography splits between the multiplier `λ` and the penalty term `r(u·n)`. The
constraint residual `u·n ≈` (FE/KSP tolerance) is roughly `r`-independent, so the
penalty share `r(u·n)` grows **linearly with `r`** (measured: ≈`10⁻⁴`, `10⁻²`,
`1.0` of the signal at `r/μ = 1, 10², 10⁴`). At small `r`, `λ` *is* the clean
topography (`corr(λ, −σ_nn) = 0.99995`); at large `r` it inflates and the
correlation degrades. Crucially the penalty term is pointwise **noise** (`r ×`
the noisy residual), so adding it back does **not** recover a cleaner signal —
the fix is to use a **smaller `r`**.

This used to be a trade-off (large `r` for conditioning vs small `r` for clean
topography). **`selfp` breaks it**: it conditions the constraint Schur from the
operator side, so a small (or, with a velocity anchor, zero) augmentation gives
both a well-conditioned solve *and* a clean multiplier. For topography work, keep
a *modest* `r` for constraint *tightness* (`u·n → 0`), not for conditioning.

### Is there a workable-AL / clean-topography overlap? (the unanchored case)

When the velocity is **unanchored** (all free-slip, no Dirichlet) the augmentation
is *required* for rigid-body regularisation, so it has a **floor**; topography
contamination gives it a **ceiling**. The two move oppositely with the boundary
viscosity contrast `μ`:

- the **regularisation floor** *rises* with `μ` (a stiffer inner-`A` needs more
  `r(n⊗n)` to lift the rigid modes) — `aug ≲ 1` at `μ=10²`, `aug ~ 10³` at `μ=10⁶`;
- the **topography ceiling** *falls* like `1/μ` (the penalty noise is
  `r·(u·n) = aug·μ·ε`).

Measured on the SolCx 4-wall (unanchored), velocity-vs-analytic and
`corr(λ, −σ_nn)` together:

| regime | overlap |
|---|---|
| **anchored** (no-slip core/base, any `μ`) | floor = 0; `aug→0` gives clean `u`, `p`, *and* topography — overlap is everything |
| **unanchored, moderate `μ` (≲10⁴–10⁵)** | **wide** — the whole `aug ∈ [1, 10⁴]` range is accurate *and* `corr(λ,CBF)=1.0000` (`μ=10²`) |
| **unanchored, extreme `μ` (~10⁶)** | **none** — regularisation needs `aug≥10³`, where topography has collapsed (`corr 0.7–0.9`); the seemingly-clean low-`aug` corr is spurious (garbage `u` ↔ garbage CBF) |

So the conflict appears only in the corner of *fully-unanchored* geometry **and**
`~10⁶` contrast. There, lean on the block solver's exact constraint enforcement
(`RMS(u·n) ~ 10⁻⁹`) and recover topography from the consistent-boundary-flux
stress `−n·σ·n` (no `r`-amplification), not from `λ` directly.

### The same story for the interior incompressibility penalty

The boundary constraint `u·n=g` (multiplier `λ`) is the surface analogue of the
interior constraint `∇·u=0` (multiplier `p`). UW3's Stokes carries an **optional
augmented-Lagrangian grad-div penalty** for incompressibility, `λ ∫ μ (∇·u)(∇·v)`,
on by setting `solver.penalty` (default `0`). Everything above transfers:

- **Scaling.** The penalty is multiplied by the local viscosity `μ`
  (`constitutive_model.K`) — exactly as the boundary AL is `∝ μ` — so the ratio
  penalty/`μ` stays uniform. The `penalty` parameter is therefore a
  *dimensionless* `O(1)` base. A bare constant (the previous behaviour) over-
  stiffens low-`μ` regions into velocity locking under contrast (measured: a
  constant `100` gave SolCx velocity error `0.1`, while the `μ`-scaled `O(1)`
  penalty gives `~10⁻⁵`).
- **Pressure correction.** Because the penalty sits in the *operator*, the
  recovered `p` is the multiplier, not the mechanical pressure:
  `p_mech = p − penalty·μ·(∇·u)`. At convergence they agree; pointwise they differ
  by ≈ a couple of percent of `|p|` at `penalty = O(1)`. For a pressure-dependent
  constitutive law use `p_mech`; for visualisation, raw `p` is adequate.
- **Usually unnecessary.** The `1/μ` Schur preconditioner already conditions the
  pressure block (outer KSP is unchanged with or without the penalty), so the
  default `penalty = 0` — where `p` is the clean physical pressure and needs no
  correction — is the right starting point.

## Files

- `src/underworld3/systems/solvers.py` — `SNES_Stokes_Constrained`, `_BlockConstraintBC`.
- `src/underworld3/cython/petsc_generic_snes_solvers.pyx` — multiplier-field
  registration, boundary residual/Jacobian coupling, fieldsplit grouping,
  nullspace, and the section-based interior-multiplier reduction.
- `src/underworld3/systems/__init__.py` — `Stokes_Constrained` export.
- `tests/test_1061_constrained_freeslip.py` — box + annulus validation.
- `tests/test_1062_constrained_solcx.py` — SolCx analytic validation.
