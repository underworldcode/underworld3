# Jacobian unwrap-before-differentiate bug — handoff brief

> **Status:** root cause found and confirmed (2026-06-17, during the gradient-plasticity
> spike on `feature/gradient-plasticity`). Fix **not yet started**. This brief is written
> so a fresh, carefully-supervised session can implement and validate it on a clean branch.
>
> **Branch policy:** do this on a new `bugfix/jacobian-unwrap-to-constants` worktree off
> `origin/development`. **Do not** build on the `feature/gradient-plasticity` spike branch
> — that branch carries unrelated experimental block-coupling scaffolding.

---

## 1. Summary

The SNES Jacobian assembly differentiates the residual flux `F1` (and `F0`) **while the
constitutive viscosity is still a wrapped `UWexpression` atom**. `sympy.derive_by_array`
therefore treats the effective viscosity as a constant, so the term `∂η/∂(grad v)` is
**silently dropped from every Jacobian**. The consequence:

> **UW3 viscoplastic Stokes has been running an accidental defect-correction / Picard
> tangent, not full Newton — not by design, but because the unwrap happens *after* the
> derivative instead of *before* it.**

This is the real source of the long-standing "~20 Picard iterations is *intrinsic* to UW3
viscoplasticity" folklore. It is **not** intrinsic.

Why it stayed hidden: constant-viscosity problems are unaffected (η has no `grad v`
dependence), so the entire constant-viscosity test suite passes bit-identically. The only
symptom is slow convergence on *nonlinear* rheology — which had a ready, innocent
explanation.

---

## 2. Root-cause diagnosis (confirmed)

In `src/underworld3/cython/petsc_generic_snes_solvers.pyx`, the Jacobian blocks are formed
like this (saddle-point Stokes shown; the SNES_Scalar / SNES_Vector variants are identical
in structure):

```python
# Don't unwrap here — let getext()'s two-phase unwrap handle it.
F1 = sympy.Array(self.F1.sym)
...
G2 = sympy.derive_by_array(F1, self.u.sym)        # <-- F1 still wrapped here
G3 = sympy.derive_by_array(F1, self.Unknowns.L)   # <-- ditto
```

The `# Don't unwrap here …` comment flags the exact ordering inversion. `getext()`'s unwrap
runs **after** these derivatives, so the derivative is taken with the effective viscosity
still an opaque atom.

Confirming evidence (`ViscoPlasticFlowModel`, η₀=1, τ_y=0.5):

```python
F1 = sympy.Array(s.F1.sym)
derive_by_array(F1, L00)[0]   #  ->  [ λ·{η_eff,p} + 2·{η_eff,p}, 0 ]   (η_eff FROZEN)
```

The atom `{η_eff,p}` **is** `cm.viscosity` (a `UWexpression`), and unwrapping it directly
works:

```python
uw.function.unwrap(cm.viscosity, return_self=False)
#  ->  Min(1.0, 0.25/sqrt({U}_{0,0}**2/2 + {U}_{0,1}**2/4 + ...))   (grad v exposed)
```

The gap is that `unwrap` is **not reaching the atom when it is embedded inside the sympy
`Matrix`/`Array`** that the assembly differentiates. If we recursively expand the embedded
`UWexpression` atoms **first**, then differentiate, the full consistent tangent appears —
including the `Heaviside` derivative of the `Min` (the yield switch):

```python
F1x = deep_unwrap(F1)            # subs every UWexpression atom -> unwrap(atom), to fixpoint
derive_by_array(F1x, L00)[0]
#  ->  2·Min(1.0, 0.25/sqrt(...))  − 0.25·U00²·Heaviside(1.0 − 0.25/sqrt(...))·...   ✓
```

So the consistent tangent was never unavailable — it was being differentiated on the wrong
side of the unwrap.

---

## 3. The fix

A new **unwrap mode/option** (per the design owner): expand every `UWexpression` atom down
to — **but not including** — `_is_constant` atoms, leaving those as symbols. Apply it to
`F0` and `F1` **before** `derive_by_array` when forming `G0–G3`.

```python
# pseudocode for the assembly change
F0u = unwrap(F0, to_constants=True)   # expand all non-constant UWexpressions; keep _is_constant atoms
F1u = unwrap(F1, to_constants=True)
G0 = sympy.derive_by_array(F0u, self.u.sym)
G1 = sympy.derive_by_array(F0u, self.Unknowns.L)
G2 = sympy.derive_by_array(F1u, self.u.sym)
G3 = sympy.derive_by_array(F1u, self.Unknowns.L)
# (G0/G1 likewise for F0u; cross blocks up/pu the same way)
```

### Why the `_is_constant` guard is the crux (do not skip it)

A **blanket** unwrap is wrong: it bakes the constant parameters (η₀, τ_y, …) into numeric
literals — observed directly (`Min(1.0, 0.25/…)` instead of `Min({η}, {τ_y}/…)`). That
breaks the **`constants[]` runtime-parameter mechanism** (changing η₀/τ_y without a JIT
recompile). The whole reason the unwrap was deferred to `getext()` was to preserve constants.

So the option must:
- recurse through `UWexpression` atoms and substitute each with its unwrapped content,
- **stop at any atom whose `_is_constant` is true**, leaving it as a symbol,
- be idempotent / run to a fixpoint (the viscosity nests `ε̇_II` inside `Min` inside `η_eff`).

This is the same expansion `getext()` ultimately performs, just made available **before**
differentiation, with constants protected.

### Where to apply it

`src/underworld3/cython/petsc_generic_snes_solvers.pyx`, every site that forms `G0–G3` via
`derive_by_array` on `self.F1.sym` / `self.F0.sym`. Anchor on the comment string
`# Don't unwrap here — let getext()'s two-phase unwrap handle it` (appears ~3×):
- `SNES_Scalar._setup_pointwise_functions`
- `SNES_Vector._setup_pointwise_functions`
- `SNES_Stokes_SaddlePt._setup_pointwise_functions` (also its `up`/`pu` cross blocks)

The unwrap option itself lives in `src/underworld3/function/expressions.py`
(`unwrap(fn, depth=None, keep_constants=True, return_self=True)`, ~line 431, and
`unwrap_expression`, ~line 181). The current `keep_constants` flag does **not** give the
required behaviour for *embedded* atoms (it left them frozen in tests) — verify and extend.

---

## 4. Validation matrix (this is a Jacobian change — every solver is affected)

**Must be bit-identical (Jacobian unchanged where η is constant):**
- `tests/test_1010_stokesCart.py`
- `tests/test_1015_analytic_solcx.py` (SolCx — piecewise-constant viscosity)
- `tests/test_0610_constitutive_tensor_regression.py`
- `tests/test_snes_vector_asymmetric_jacobian.py` (the G-block layout guard)
- the pressure-nullspace / shell-nullspace tests (1013/1014/1056)
- **Add a residual-probe check**: assert the assembled Jacobian is bit-identical
  before/after the change for a constant-viscosity Stokes solve.

**Should improve (fewer iterations / Newton convergence; answers unchanged):**
- `tests/run_vp_shear_box.py` (viscoplastic yield) — expect iteration count to drop
- `tests/test_1052_VEP_stability_regression.py`, the `vep_*` drivers
- compare SNES iteration counts before/after on a genuinely nonlinear viscoplastic case

**Must still work (the reason the guard exists):**
- `constants[]` runtime-parameter update: change η₀ / τ_y on a built solver and re-solve
  **without** a JIT recompile; confirm it still updates (this is what the `_is_constant`
  guard protects).
- `set_jacobian_F1_source`: confirm it still functions. Note it is revealed to be a
  **workaround for this bug** (manually supplying a differentiable F1) — once the unwrap
  is fixed it should be largely unnecessary for the viscosity case; the softmin-at-yield-kink
  use may remain legitimate.

**Full suite:** `pytest -m "tier_a or tier_b"` before proposing the PR.

---

## 5. Downstream payoff (context, not in scope for this fix)

- Once landed, the gradient-plasticity block coupling (spike branch) gets its `ē→u`
  cross-block (`us_G2 = ∂σ/∂ē`) **for free** — the deep-unwrap differentiates it with the
  correct Mandel scaling, removing the hand-derived term we could not get right by hand.
- A consistent operator is also a prerequisite to evaluating a **Vanka** preconditioner
  fairly (today it preconditions a defect-corrected approximation).
- These are reasons the fix matters, not tasks for this branch.

---

## 6. Cautions

- Solver stability is paramount (repo rule: no solver changes without extensive
  benchmarking). Keep the change minimal and the bit-identical constant-viscosity checks
  front and centre.
- Watch JIT compile time / expression size: deep-unwrapping before differentiation can
  enlarge the symbolic expressions (more terms to compile). Check that compile times and
  the `Min`/`Heaviside`/`Piecewise` JIT paths behave (cf. the historical "`sympy.simplify`
  time-bomb" notes — do **not** add `simplify`).
- `Min`/`Max` differentiate to `Heaviside`; ensure the JIT (`utilities/_jitextension.py`)
  emits these correctly (it already handles them for boundary/Nitsche terms).

---

## 7. References

- Project memory: `project_jacobian_unwrap_constants_bug.md`
- Reproducing snippets: this brief, §2 (run against any `ViscoPlasticFlowModel` Stokes).
- Spike that surfaced it: `feature/gradient-plasticity` (block smoothing field,
  `add_smoothing_field`, `_smoothing_full_tangent`) — for context only.
