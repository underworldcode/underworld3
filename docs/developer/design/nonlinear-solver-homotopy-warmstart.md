# Nonlinear solver: automatic warm-start and single-parameter yield homotopy

**Status: PROPOSED.** Design only — implementation is deferred to a dedicated,
carefully-benchmarked session because it changes core `solve()` behaviour.
(Design captured 2026-07, from the Spiegelman viscoplastic hardening work.)

## Why this exists

Hard viscoplastic (Drucker–Prager) Stokes problems do not converge from a cold
Newton start on the sharp yield surface. The working recipe is well understood —
a viscous warm start, then a **δ-continuation** (solve at a smooth yield law, then
warm-start progressively sharper solves), with the consistent-Newton tangent and a
multigrid smoother that tolerates the non-symmetric Newton operator.

The problem is that *setting that recipe up correctly is a minefield*, and the
minefield is the real defect. Across a single working session the recipe was
mis-configured many times — default line search, a pressure-nullspace-singular LU,
`preconditioner="fmg"` giving a 1-iteration outer KSP, the default Chebyshev
smoother diverging on the non-symmetric operator, cold-start `ε̇=0 → τ_y/0 → NaN`.
Each produced a *different* failure a few steps in. **If the correct setup is that
fiddly for an expert, it is an API regression, not a user error.** The goal of this
design is to make the correct path the default path: the user writes their rheology
and calls `solve()`.

## What was established (evidence)

These results are on the Spiegelman notch at the genuinely hard regime
(`η_bg = 1e26 Pa·s`, `V = 10 mm/yr`, C = 1e8 Pa, φ = 30°), coarse mesh, and are
reproducible via `convergence.py` in the hard-case study.

- **Multi-solve δ-continuation converges.** Constant δ per solve, solved to
  tolerance, warm-started into the next (smaller) δ, consistent-Newton tangent,
  `bt` line search, non-symmetry-safe smoother: settles at δ≈2e-3 with the plastic
  band active. This is the trusted path.
- **In-SNES δ-ramp does not — proven, not masked.** Ramping δ *inside one SNES
  solve* via a `SNESSetUpdate` callback (with the *same* proven config) diverges
  with `DIVERGED_LINEAR_SOLVE` after 2 iterations and grinds for ~2 hours doing it.
  The mechanism is fundamental: the continuation only sharpens δ from a **converged**
  iterate, where the consistent-Newton Jacobian is well-conditioned; the in-SNES
  ramp sharpens δ **mid-solve**, at a far-from-solution iterate, where the
  consistent-Newton Jacobian on a sharpening yield surface is ill-conditioned and
  the linear solve fails. **Do not re-attempt hiding δ inside a single Newton
  solve.** Hide the *continuation*, not the ramp.
- **The δ-march is cheap.** With the power-mean smoother, once the first δ converges
  the warm start already satisfies every sharper δ in ≈0 Newton iterations, so an
  automatic residual-guided descent costs almost nothing.
- **The power-mean smoother is faithful.** At the settled δ it undershoots the true
  `Min` yield by ≤0.2 %, and 0 % in genuinely yielded cells (see
  {doc}`jacobian-consistent-tangent`). Earlier "it undershoots badly" readings were
  an artefact of a diagnostic that evaluated a sharper power-mean, not the exact
  `Min`.
- **The recursion prerequisite is fixed** (landed on the same branch as this design:
  the DP model's `yield_stress_min` guard used the wrong sentinel and wrapped every
  yield in `Max(<−∞ Parameter>, …)`, which recursed under the consistent tangent).

The through-line: every failure was a **solver-configuration** error, upstream of
the Jacobian and unrelated to the recursion fix. The `get_snes_diagnostics()` field
`linear_iterations` is the tell — ≈1 linear iteration per Newton step means the
linear solve is doing no real work.

## The design

Three layers, from most general to most specific. Layer 1 stands alone and helps
every nonlinear solver; layers 2–3 build on it.

### Layer 1 — automatic warm-start (all nonlinear solvers)

A single **Picard (frozen-coefficient) step is a general cold-start warm-up**. It is
defect-correction iteration 1: contractive, moves a cold guess into the Newton
basin. UW3 already exposes it (`solve(picard=N)`) and already has the Picard tangent
(`consistent_jacobian=False`) and a Picard→Newton α-blend.

Rules:

- **Cold** (no usable solution): take **one** Picard step, then Newton.
- **Warm** (a usable solution is present): straight Newton, **no** Picard step — a
  Picard step from a good iterate takes a linear-convergence step off the good
  quadratic starting point (unnecessary, mildly harmful).
- **Linear problem**: free either way — the frozen operator *is* the real operator,
  so the single Picard step *is* the exact solve and Newton then reports converged in
  0 iterations. UW3 already probes the assembled Jacobian for solution-dependence, so
  a linear problem can also simply skip the warm-up. Overhead ≈ one convergence check.

**Cold-vs-warm is auto-detected**, and detection is safe by construction: guessing
*cold* when actually warm costs one Picard step that instantly re-converges; the only
harmful direction (warming off stale, unrelated field data) is what the opt-out flag
is for.

- **`solver.has_solution`** — a **public** property, set `True` only after a solve
  that **converged** (`reason > 0`), `False` initially, after a **diverged** solve,
  and after a **structural rebuild** (mesh change / adaptivity / mesh-mover — the
  existing `is_setup=False` invalidation is the natural hook). It is kept through
  *coefficient* changes (new viscosity, new δ, new BCs) so continuation and
  time-stepping warm-start correctly. It doubles as a user-facing status flag.
- Secondary signal for a hand-set initial guess that has not been solved yet:
  `‖v‖ ≈ 0 ⇒ cold`.
- **Emergent auto-recovery**: a diverged solve leaves `has_solution=False`, so the
  *next* `solve()` auto-cold-starts (Picard warm-up) rather than warming off garbage.

**API change**: today `zero_init_guess` defaults to `True` (always cold). Make it a
tri-state whose **default auto-detects**; `zero_init_guess=True` means "force fresh /
discard any solution", `False` means "insist on warming". The onus flips from "flag
when you have a solution" to "flag when you want to throw one away".

```{note}
This changes a core default and is "benchmark before flipping" territory. It is
mostly a bug-fix — today `solve()` in a loop silently re-zeros each iteration — but
must be validated. Explicit warm (`zero_init_guess=False`) is already the common
deliberate choice in the codebase; explicit cold is rare, so the flip aligns with
existing practice. See the open verification points.
```

### Layer 2 — advertised single-parameter homotopy

A constitutive model **advertises** a single-parameter homotopy, exactly as it
advertises an approximate-Jacobian flux:

- `cm.supports_yield_homotopy` → `True` on `ViscoPlasticFlowModel` / VEP, `False` on
  plain viscous. `solve(homotopy=True)` on an unsupported model raises a clear error.
- `cm._yield_homotopy_control()` → puts the model in its smooth mode
  (`yield_mode="softmin"`, `yield_smoother="powermean"`), returns the δ `constants[]`
  atom and the recommended tangent (Newton for non-elastic DP, Picard for elastic
  VEP). The model owns "what the homotopy means for me", so the mechanism generalises
  to other models and applications.

`solve(homotopy=True, homotopy_options=...)` then runs the **continuation** (never
the in-SNES ramp):

1. Set δ = δ₀ **large**. At large δ the power-mean is the harmonic mean, which is
   bounded by η_bg even as `ε̇ → 0`, so the cold Picard warm-up (Layer 1) is
   well-conditioned — **this is what removes the need for a separate viscous
   pre-solve, and why the cold `ε̇=0` NaN does not occur.**
2. Newton to tolerance at δ₀ (the tangent from `_yield_homotopy_control`).
3. **Residual-guided descent**: march δ down while solves keep converging (accelerate
   when a step is easy, ease off when it is hard), warm-started each time, and settle
   at the smallest feasible δ. This is the default because the descent is cheap and
   strictly better than "assume homotopy is not needed → stall → retreat".
4. Report via `get_snes_diagnostics()`.

`homotopy_options` (all defaulted) exposes only the march knobs — `delta0`, `down`,
`dmin`, `entry_maxit`, `step_maxit`, and the settle criteria. Fine-tuning is optional.

**Scope: single parameter only.** A one-parameter homotopy is easy to generalise;
multi-parameter is not, and is unnecessary here. The rate-strengthening ξ term is a
*non-homotopic* regularisation, not a homotopy — it belongs in a user-domain loop
*around* `solve()`, not inside it.

### Layer 3 — smoother as a consistent-Newton consequence

Turning on the consistent tangent adds the `∂η/∂(grad v)` term, which makes the
velocity-block operator **non-symmetric**. The default multigrid smoothers
(Chebyshev / Richardson) assume a symmetric/SPD operator: Chebyshev can **diverge**
on a non-symmetric operator, Richardson stalls. The fix is one line — a
non-symmetry-safe smoother, `mg_levels_ksp_type = gmres` (with `mg_levels_pc_type =
sor`).

This is a consequence of `consistent_jacobian=True`, **not** of homotopy — anyone
using the consistent tangent with FMG hits it.

**LANDED, and the benchmark changed the shape of the fix.** The smoother is now
`gmres` + `sor` unconditionally in the FMG bundle, with `mg_levels_ksp_norm_type =
none` (a fixed-cost V-cycle: exactly `max_it` smoother iterations, no residual norm,
no early exit). Two findings drove that:

- **The gmres margin grows with multigrid depth.** Measured on the Spiegelman notch
  (Drucker–Prager, `η_bg=1e26`, `V=10`, `δ=0.01`) over a *nested* hierarchy built by
  uniformly refining an ultra-coarse 492-cell base, per-V-cycle contraction ρ at
  `ξ=0.01` and the same four smoother iterations: 3 levels — 0.722 richardson vs
  0.686 gmres (5 %); 4 levels — 0.746 vs **0.560** (25 %). A deeper cycle applies the
  smoother on more coarse operators, each non-symmetric under the consistent tangent,
  so smoother quality compounds. A measurement on a 2-level hierarchy shows almost no
  effect and is **misleading** — a two-level cycle is a coarse-grid correction, not a
  V-cycle.
- **Therefore no gate.** Since shallow hierarchies are not worth having (maintainer
  ruling), the default is set for the deep case unconditionally — no level-count gate
  and no `consistent_jacobian` gate. Four smoother iterations, not more: per unit work
  gmres/4 (ρ^(1/4) = 0.87) beats gmres/8 (0.91).

**What this does not fix.** At the hard small-`ξ` corner the inner solve fails under
every smoother tried — richardson outright, and gmres with **ρ > 1** (the V-cycle
diverges) at 4 levels, where the 3-level run still returned a finite ρ≈0.9. The
coarsest grid cannot represent the weak-zone/notch viscosity contrast, so geometric
interpolation cannot carry it: this is *operator conditioning*, and the lever for it is
the δ/ξ continuation (Layer 2), not the smoother. Do not expect a smoother to rescue
small ξ.

Measured with `fmg_contraction_probe.py` / `smoother_depth_sweep.py` in the Spiegelman
hard-case study (ρ_MG = per-V-cycle contraction of the velocity block).

## The user-facing result

Once all three land:

```python
stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
cm.Parameters.shear_viscosity_0 = ...
cm.Parameters.yield_stress = ...          # a plain pressure-dependent yield
stokes.solve(homotopy=True)               # warm-start, δ-descent, smoother — all automatic
if stokes.has_solution:
    ...
```

The only knobs a user ever touches are the two deliberate opt-outs: force-fresh
(discard an existing solution) and fine-tune-the-march.

## Implementation stages (for the dedicated session)

1. **Layer 1a — `has_solution` property + Picard-on-cold warm-up** — **LANDED**
   (`petsc_generic_snes_solvers.pyx`; test `test_0201`). Behind the existing
   `zero_init_guess` semantics (no default flip). Additive, low-risk. As built:
   `has_solution` lives on `SolverBaseClass`, is recorded by
   `_record_convergence_status()` at the end of every `solve()` (the rotated
   free-slip path records from its info dict, since it runs its own KSP loop),
   and resets in the `is_setup = False` setter. The cold Picard warm-up is scoped
   to a cold Stokes solve **under the consistent-Newton tangent**
   (`consistent_jacobian is True`) — the opt-in nonlinear regime that needs it —
   so the default (frozen) tangent path is bit-identical and the common linear
   solve pays nothing. Broadening the warm-up to all nonlinear solvers (a cached
   nonlinearity probe rather than the tangent proxy) is a natural Layer-1b
   extension. Recipe + config-trap-list captured in the `nonlinear-solver` skill.
2. **Layer 1b — flip `zero_init_guess` to auto-detect.** The benchmarked default
   change. Gate the free-surface-chain and mover cases first (below).
3. **Layer 3 — non-symmetry-safe smoother default** — **LANDED** (`gmres`+`sor`+
   `norm_type=none` in the FMG bundle; test `test_1014`). Independent of homotopy.
   Benchmarked as described in the Layer 3 section: applied unconditionally rather
   than gated on the consistent tangent, because the gain scales with multigrid depth
   and shallow hierarchies are not worth special-casing.
4. **Layer 2 — `supports_yield_homotopy` / `_yield_homotopy_control` model hook and
   the `solve(homotopy=True)` continuation** — **LANDED** (test `test_1057`). As built:
   the control is a `YieldHomotopyControl` carrying a **model-owned δ setter** (the
   isotropic models update the `constants[]` atom, TI-VEP rebuilds — and going through
   the model's own `yield_softness` property is what stops a later
   `_get_yield_softness()` resetting δ to a stale value), the recommended tangent, and
   the δ atom. The march is residual-guided as designed, and additionally *reverts and
   retries a failed δ more gently* before settling. A VEP model's `timestep` is
   forwarded to the inner solves. **Cold-start finding:** a cold power-mean solve died
   with `DIVERGED_FNORM_NAN`, and the first diagnosis (a `0/0` needing a strain-rate
   floor) was **wrong**. At `ε̇=0` the plastic viscosity is `+inf`, which `Min` and the
   sqrt soft-min carry correctly to the viscous branch — both cold-start fine. Only the
   power-mean broke, because it formed its harmonic mean as
   `η_ve·η_pl/(η_ve+η_pl)` = `inf/inf`. Rewriting that as the identical `η_ve/(1+f)`
   fixes it with no floor and no numerical change. The bug was never homotopy-specific
   or cold-start-specific: `η_pl` is infinite at *every rigid (unyielded) point*, so it
   could poison a converged solve wherever a plug forms. The continuation logic existed as
   a reference implementation (`underworld3.systems.yield_continuation`, the extracted
   form of `convergence.py::run_continuation`); it is folded in, driven by the model hook
   and Layer 1's warm-start.

## Open verification points / risks

- **Free-surface chain of solves** (on a feature branch, not yet inspected): a chain
  of *deliberately independent* solves is the one pattern that might rely on the
  implicit cold default. Confirm it either wants explicit force-fresh or is correct
  warm, before flipping the default.
- **Adaptivity / mesh-mover reset**: confirm `has_solution` resets on the same
  structural-invalidation event that already nukes the initial guess after a remesh.
- **Empirical basin test**: confirm a *single* Picard step at δ=δ₀ actually lands in
  the Newton basin on the hard case — i.e. that it genuinely replaces the explicit
  viscous pre-solve. Likely, given large-δ boundedness, but test it.
- **Benchmark the two core-default changes** (Layer 1b, Layer 3) against the existing
  solver regression suite — "Solver Stability is Paramount".

## References

- Reference implementation of the continuation: `convergence.py::run_continuation`
  in the Spiegelman hard-case study; extracted form in
  `underworld3.systems.yield_continuation`.
- Consistent tangent + δ soft-min families: {doc}`jacobian-consistent-tangent`.
- Solver strategy catalogue (consult and contribute): `solver-strategies-catalogue`.
- Diagnostics: `SNES_*.get_snes_diagnostics()` / `check_snes_convergence()` /
  `solve_with_diagnostics()` — `linear_iterations` is the diagnostic tell.
- The recipe and its trap list are also captured in the `plasticity-solvers` /
  nonlinear-solver skill.
