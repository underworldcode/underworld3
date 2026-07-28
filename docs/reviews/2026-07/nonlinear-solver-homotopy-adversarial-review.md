# Adversarial review — nonlinear solver warm-start + yield homotopy (4 layers)

Branch `feature/nonlinear-warmstart-homotopy` vs `development`
(16 commits, 2402 insertions; `level_1 and tier_a` suite is green at 470 passed).

Method: four independent adversarial reviewers, each given one dimension and
instructed to break the change rather than praise it, plus the author's own pass.
Every finding below was re-verified against the source before being reported;
findings the reviewers raised that did not survive verification are not listed.

**Verdict: NOT ready to flag ready.** Three critical and seven major defects, most of
them in the new code rather than pre-existing. Two of the four layers (1a, 3) look
sound; the homotopy driver (Layer 2) and the default flip (Layer 1b) both need work.

---

## CRITICAL

### C1 — `entry_maxit` / `step_maxit` are inert; a failing δ-step burns 50 iterations
`systems/yield_continuation.py:151` sets `solver.petsc_options["snes_max_it"]`, but
`SNES_Stokes_SaddlePt.solve` hardcodes `snes_max_it = 50`
(`petsc_generic_snes_solvers.pyx:8474`) and pushes it with `setValue` +
`setFromOptions` immediately before the real solve (`:8543`) — the file's own comment
at `:8468` documents that this clobbers any user-set value.

Consequences: both documented `homotopy_options` budget keys are silent no-ops; the
docstring promise that "a tight budget lets a too-hard step abort cheaply" is false
(a failing δ costs 50 Newton iterations, ×3 with the default retries); and the
residual-guided step logic at `:175` compares the *real* `nit` (cap 50) against a
*fictional* `budget` (10/30), so the "ease off" branch mis-fires and drives `step`
toward its 0.95 clamp. The mechanism that does work is
`snes.setTolerances(max_it=…)` after `setFromOptions` (as `_snes_solve_with_retries`
already does at `:1495`).

### C2 — every retry after a failed δ is COLD, defeating the revert
`yield_continuation.py:181` reverts `u`/`p` to the last converged state so the retry
can warm-start from it. But the failed inner solve already ran
`_record_convergence_status()`, setting `has_solution = False`, so the retry's
`solver.solve(zero_init_guess=not solver.has_solution, …)` (`:155`) resolves to
**cold** — a from-zero solve at a δ *sharper* than the one that just failed. That is
exactly the regime this module's own docstring says does not converge. The revert is
wasted, the retry is near-guaranteed to fail, and the march settles earlier than it
should.

### C3 — `homotopy=True` on a VEP model advances the stress history once per δ-step
`ViscoElasticPlasticFlowModel` advertises the homotopy
(`constitutive_models.py:2289`), and `SNES_Stokes.solve` forwards `timestep` into
every inner solve (`systems/solvers.py:1478`). Each inner solve therefore re-enters
the `has_stress_history` branch and runs the full time-integration tail —
`DFDt.update_pre_solve(timestep)`, the stress projection, the `psi_star` shift loop,
and `DFDt.update_post_solve(timestep)` (`solvers.py:1525-1585`). An N-step march
advances the elastic stress history by **N·dt for one requested dt**, and the
driver's revert restores only `u`/`p`, not the history, so a failed step leaves the
extra shift in place. `test_1057` only exercises the non-elastic model, so nothing
catches it.

---

## MAJOR

### M1 — `_force_setup=True` now warm-starts, inverting its own contract
All four solve bodies resolve `zero_init_guess` *before* the invalidation it depends
on (`pyx` resolve/invalidate pairs 3444/3450, 4494/4497, 5217/5220, 8394/8402):

```python
stokes.solve()                    # converged -> has_solution True
stokes.solve(_force_setup=True)   # resolves WARM off the stale flag, THEN rebuilds
```

The `is_setup` setter's own comment names "explicit `_force_setup`" as a structural
invalidation that must drop the warm-start claim. Before the flip this path was cold.
Fix: resolve after the `_force_setup` block.

### M2 — a march that settles via the failure exit leaves `has_solution=False` on a good solution
On the retries-exhausted `break`, the fields are reverted to the settled solution and
δ is reset to `settled`, but nothing restores `has_solution` (left `False` by the
failed solve). The report says `converged: True` with a real `settled_delta`, yet the
next `solve()` auto-cold-starts and discards the continuation result — precisely the
advertised usage (`solve(homotopy=True)` once, then a time loop). `solve_report` has
the same problem: it describes the *failed* δ, not the settled one.

### M3 — the march always cold-starts its entry solve
`yield_continuation.py:148` runs `solver.is_setup = False` on the first iteration,
whose setter clears `has_solution`, so `not solver.has_solution` on the next line is
unconditionally `True`. A time loop calling `solve(homotopy=True)` per step discards
the previous step's converged state every time, contradicting the module's own
comment ("Cold only if there is genuinely nothing to warm-start from").

### M4 — no `try/finally`: an exception mid-march leaves solver and model corrupted
`yield_continuation.py:145-203` has no exception handling. If an inner solve raises
(PETSc error, JIT failure, the VEP `timestep is None` ValueError), then `snes_max_it`
is left at the march value, `consistent_jacobian` at `control.tangent`, the model in
`softmin`/`powermean` at the *failed* δ, and `u`/`p` hold the diverged iterate with no
revert.

### M5 — unrestored side effects on the user's solver and model
`solver.consistent_jacobian = control.tangent` (`:128`) is never restored: a user who
set `consistent_jacobian = "continuation"` (a supported value) silently gets `True`
or `False` forever after. `_yield_homotopy_control()` likewise leaves
`yield_mode="softmin"` and `yield_smoother="powermean"` permanently — a model the user
configured as `yield_mode="min"` (the `ViscoPlasticFlowModel` **default**) comes back
as a power-mean soft-min, so every later plain `solve()` runs different physics than
asked for, unwarned.

### M6 — the power-mean cannot reach exact `Min`, but the docs say it does
Sharpness is `s = 1/(δ + 0.001)`, which **saturates at 1000**:

| δ | 1e-2 | 1e-3 | 1e-4 | 1e-6 | 0 |
|---|---|---|---|---|---|
| s | 91 | 500 | 909 | 999 | **1000** |

`constitutive_models.py:989` and `:1031` state "Both approach exact `Min` as δ → 0".
False for the power-mean family: δ=0 gives a finite power-mean (~0.2 % below true
`Min`), not the sharp surface. Since the entire homotopy premise is "march δ→0 to
reach the sharp yield surface", the contract needs restating: with `powermean` you
converge to a slightly smoothed yield law. Corollary (minor): the march happily
descends below δ≈1e-4 where nothing changes — a demo run reached δ=2.4e-10 in 9 steps,
and `settled_delta` then reads far sharper than the law actually is.

### M7 — `TransverseIsotropicVEPFlowModel` advertises the homotopy but ignores half the control
It returns `supports_yield_homotopy = True` (`constitutive_models.py:3800`) and
inherits `_yield_homotopy_control()`, but never calls `_combine_yield`: its yield law
is inlined (`:3485`, `:3578`, `:3986`, `:4049`) reading `self._yield_softness` as a raw
float and hardcoding the **sqrt** family. So `yield_smoother = "powermean"` is a silent
no-op there — and the power-mean's large-δ harmonic limit is the stated justification
for the march's well-posed cold entry. Its `yield_softness` setter also never syncs
`_yield_softness_expr`, so the `control.delta` atom handed back is meaningless for
this model.

---

## MINOR

- **m1** — `uw.pprint(0, …)` (`yield_continuation.py:168,187,194`) uses the retired
  `pprint_old(ranks, …)` shape; the current signature is `pprint(*args, proc=0, …)`,
  so the literal `0` is printed. Confirmed empirically: every march line reads
  `0 [yield-continuation] δ=…`.
- **m2** — diverged **linear** rotated free-slip records success. `pyx:8456` does
  `.get("converged", True)`, but `solve_rotated_freeslip()` returns only
  `ksp_reason`/`ksp_its` (`rotated_bc.py:379`), so the default fires even when
  `_warn_if_ksp_diverged` just warned. Provably inconsistent with the sibling line:
  `_capture_rotated_report` derives `converged = reason > 0` from the same dict, so
  `solve_report.converged is False` while `has_solution is True`.
- **m3** — the retry log prints the wrong δ as having failed (`:194` recomputes
  `d/step`, which is the last *good* δ).
- **m4** — `failures` is a whole-march counter, never reset on success, so `retries`
  is not "per failed δ" as documented.
- **m5** — `delta0` is unvalidated (`down` is). `delta0 <= 0` reaches
  `s = 1/(δ+0.001)` and divides by zero at δ = −0.001.
- **m6** — no cap on δ-steps: `step` is clamped to ≤0.95, so a hostile march can take
  ~135 solves from `delta0=1.0` to `dmin=1e-3` with no `max_steps` escape (compounded
  by C1's 50-iteration steps).
- **m7** — `solve(homotopy=True)` silently drops `picard`, `divergence_retries`,
  `evalf`, `order`, `_force_setup`, `debug`, and even `zero_init_guess`.
- **m8** — two dispatch points with different behaviour: `pyx:8399` forwards no
  `solve_kwargs`, so a `SNES_Stokes_SaddlePt` subclass that doesn't override `solve`
  would give a VEP model inner solves with no `timestep`.
- **m9** — Charter §6: `YieldHomotopyControl` and `SolveReport` are named in public
  docstrings but not exported from `systems/__init__` (deep-import-only).
- **m10** — Charter §4: three undocumented exception swallows in
  `_capture_solve_report` (`pyx:1239-1250`).
- **m11** — stale docstrings after the flip: `SNES_Darcy.solve` still says
  "If True (default)"; `SNES_TransientDarcy.solve` says "(default True)".
- **m12** — duplicate test number: `test_1055_solve_report.py` and
  `test_1055_yield_smoother.py` are both new on this branch; `test_1057` then skips
  1056.
- **m13** — latent, pre-existing: the `yield_stress_min != 0` guard fixed on the DP
  model survives at `constitutive_models.py:2030` (VEP) and `:3534` (TI-VEP). Same
  class of bug; per Charter §2 it wants a `# TODO(BUG):` rather than a silent fix.

---

## Test quality (Charter §8) — the review's sharpest hit on the author

Three of the new tests would pass with the feature deleted:

- `test_0201::test_zero_init_guess_is_tristate_and_auto_detects` asserts the **private**
  `_resolve_zero_init_guess` back to itself; it never drives the public `solve()` path,
  so it is blind to M1 (the `_force_setup` ordering bug) — the exact defect it should
  have caught.
- `test_0201::test_repeated_default_solve_agrees_to_solver_tolerance` passes unchanged
  if the flip is reverted (two cold solves of a linear Poisson also agree), and never
  asserts the second solve was actually warm.
- `test_0201::test_cold_warmstart_under_consistent_newton_converges` claims to exercise
  the automatic Picard branch but uses a **linear** `ViscousFlowModel`, which converges
  cold regardless; it passes with the warm-up line deleted.
- `test_1057::test_solve_homotopy_marches_and_reports` asserts only `steps >= 1` and
  `settled_delta <= delta0`, both satisfied by an implementation that does one solve
  at δ₀ and stops — it never asserts the march descended.

No parallel coverage was added for any of the four layers (Charter §11). Layer 3 is
the notable gap: `test_default_fmg_bundle_is_parallel_safe` is a *serial* test
asserting option strings, and never exercises the new `gmres`/`norm_type=none`
smoother at np>1 where it interacts with the redundant-LU coarse solve.

---

## Constitutive maths — added by the fourth reviewer (most severe of the four)

### C4 (CRITICAL) — the power-mean returns NaN wherever the yield stress goes negative
`constitutive_models.py:1005-1017`. With a pressure-dependent Drucker–Prager
`τ_y = C + sinφ·p`, tension drives `τ_y < 0` ⇒ `η_pl < 0` ⇒ `f < 0`, and `a = 1+f`,
`b = 1+1/f` are then negative bases raised to the non-integer power `-s` ⇒ **NaN**
(reproduced against the real UW3 expression: `η = nan`, all Jacobian entries NaN,
where `yield_mode="min"` returns a finite floored `0.001`). The old hard-`Min` path
degraded gracefully; the new smooth path does not.

This is not a corner case for this feature: `_yield_homotopy_control()` flips the
model into `powermean` **unconditionally and without checking that the yield stress is
bounded below**, so `solve(homotopy=True)` NaNs on the first residual for any
pressure-dependent DP model that lacks a lower bound — i.e. the exact target problem.
(The Spiegelman driver wraps `sympy.Max(C + sinφ·p, 0)` by hand, which is why the
hard-case study never hit it.)

### C5 (CRITICAL) — `_apply_floor(value, 0)` has a rounding scale of exactly zero
`constitutive_models.py:1088`, reached from `:1293`. The floor is rounded by
`ε = δ·floor`; with `floor = 0` that is `smooth_max(τ_y, 0, 0) = ½(τ_y + |τ_y|)` —
the exact kink the smooth floor exists to remove, with a `0/0` derivative. Over the
whole region where raw `τ_y ≤ 0` the floored value is exactly `0.0`, so `η_pl = 0`,
`η = 0`, and every Jacobian entry is NaN (reproduced). For a zero floor the new
"smooth" path is **strictly worse than the `sympy.Max` it replaced**, which is clean
there. `_apply_floor`'s own docstring admits it "needs a non-zero `floor` for a length
scale" and nothing enforces it — and the same commit's sentinel change
(`!= 0` → `!= -sympy.oo`) is what made `yield_stress_min = 0` reachable.

### M8 (MAJOR) — δ doubles as the floor-rounding scale, so early march solves run a *different model*
`constitutive_models.py:1088`. `yield_continuation` starts at `delta0=1.0` (and the
`yield_smoother` setter forces δ→1.0 when switching to powermean), giving `ε = 1.0·floor`
and measured `smooth_max(F)/F = 1.500` — the viscosity and yield-stress floors sit up to
**50 % above** the values the user requested. It converges away as δ→dmin, but it means
the early continuation steps solve a perturbed problem, it is undocumented, and there is
no independent knob for the floor rounding.

### M9 (MAJOR) — the cold-start claim is over-stated: the Newton *tangent* at ε̇=0 is NaN
`constitutive_models.py:1124-1128`. The comment I added ("no strain-rate floor is
needed … the soft-min carries it correctly") is true of the **residual** only: measured
at v=0, `η = 1.0000003` (finite — the inf/inf fix works) but **all five Jacobian entries
are NaN**. Partly intrinsic (`dε_II/dE = E/(2ε_II)` → 0/0), partly added by the
power-mean. It only works today because `solve()` interposes `picard = 1`, and that
guard is `consistent_jacobian is True` — so `consistent_jacobian="continuation"`, or
`zero_init_guess=False` on an all-zero field, assembles a NaN Jacobian.

**This settles the open question I flagged to the maintainer**: a strain-rate floor
*is* legitimately needed — not for the residual (where removing it was right) but for
the **tangent**.

### Corroborated: δ=0 ⇒ exact `Min` is false for the power-mean (see M6)
Measured deviation from exact `Min`: 5.2e-4 at δ=0, 1.2e-3 at δ=1e-3. Note the `+0.001`
floor **equals the default `dmin=1e-3`**, so the default march ends at `s = 500` — half
the achievable sharpness. The sqrt family is genuinely exact at δ=0; this is
power-mean-only. Six docstrings state it wrongly (`:653`, `:959`, `:995`, `:1039`
["`s = 1/δ`", but the code is `1/(δ+0.001)`], `:1211`, `:1353`).

### Verified CLEAN by the same reviewer (against PETSc source)
- **The harmonic-mean rewrite is numerically sound.** `η_ve·η_pl/(η_ve+η_pl)` vs
  `η_ve/(1+f)` agree to ≤1 ulp at (1e26,1e21) both orders, (1e26,1e26), (1e-20,1e26);
  the product form only overflows above η≈1e154; the identity is exact. The new form is
  strictly better — the only one that survives `η_pl = ∞`. My "no numerical change"
  claim holds.
- **The FMG smoother bundle is correct.** `gmres` registers `KSP_NORM_NONE` for both
  PC sides (`gmres.c:894`); `mg_levels_ksp_converged_maxits` is **required, not
  redundant**, and is present — `KSPConvergedDefault` (`iterativ.c:1529`) tests it
  *before* the `KSP_NORM_NONE` early return, so without it `KSPSolve_GMRES` would
  hard-set `KSP_DIVERGED_ITS` and PCMG would flag PC failure. Restart 30 > max_it 4, so
  the cycle never restarts. The GAMG delete list does include
  `mg_levels_ksp_norm_type`. The fgmres claim is verified at both configuring sites.
  Nit: the comment's "same four smoother iterations" understates cost — GMRES adds ~5
  Krylov vectors and Gram–Schmidt per level over Richardson's 2.

### Reviewer claim NOT sustained
The reviewer states the power-mean NaN fix "ships no test that would catch a
regression". Not correct: `test_1057::test_cold_viscoplastic_solve_survives_zero_strain_rate`
parametrises a cold (ε̇=0) solve over all three yield modes and does fail without the
fix. The reviewer inspected `test_1055_yield_smoother.py` only. Their broader point
stands, though — **no** test covers the negative-`τ_y` path (C4) or the zero-floor path
(C5).

---

## What held up

- **Parallel correctness of the new control flow** — traced clean. `getConvergedReason`
  is collective and rank-identical, so `_resolve_zero_init_guess` cannot split ranks;
  the march's predicates are all rank-uniform; the revert is inside
  `synchronised_array_update` on every rank.
- **Backward compatibility of the tri-state flip** — no caller in `src/`, `tests/`,
  `docs/` or the skills passes `zero_init_guess` positionally or truth-tests it before
  resolution.
- **`_record_convergence_status()` coverage** — reached on every normal exit of all
  four solve bodies and the rotated early return; every wrapper funnels through them.
- **Recursion/reentrancy of `solve(homotopy=True)`** — `solve_kwargs` never carries
  `homotopy`, so the inner solves cannot re-enter the march.
- **Return-value contract** — the dict survives `timing`, `memprobe` and
  `SNES_Stokes_Constrained.solve(*args, **kwargs)`.
- **March arithmetic** — `step ∈ [0.05, 0.95]` strictly, `d` strictly positive and
  decreasing, `d <= dmin` reachable, `delta0 <= dmin` exits correctly. The problem is
  cost (m6), not termination.
- **Layers 1a and 3 as such** — `has_solution`'s lifecycle and the FMG smoother bundle
  drew no correctness findings beyond the ones listed.
