# Consistent Jacobian tangent for nonlinear (viscoplastic) solves

**Status**: Implemented — PR #258 (2026-07-02): opt-in `solver.consistent_jacobian`, default off (Picard tangent unchanged).

## The bug

The SNES Jacobian assembly in `src/underworld3/cython/petsc_generic_snes_solvers.pyx`
formed the `G0`–`G3` blocks by calling `sympy.derive_by_array` / `sympy.diff` on the
residual flux `F1` **while the effective viscosity was still a wrapped `UWexpression`
atom**. `derive_by_array` therefore treated the viscosity as a constant, so the term
`∂η/∂(grad v)` was **silently dropped from every Jacobian**.

Consequence: UW3 viscoplastic Stokes was running an accidental **Picard /
defect-correction** tangent, not full Newton — not by design, but because the unwrap
happened *after* the derivative instead of *before* it. Constant-viscosity problems were
unaffected (η has no `grad v` dependence), which is why the entire constant-viscosity
suite passed bit-identically and the bug stayed hidden behind the "≈20 Picard iterations
is intrinsic" folklore.

## The fix (this PR — opt-in, default-off)

A new unwrap mode and a gate, so the default path is **bit-identical** to the historical
(Picard) behaviour and the consistent tangent is **opt-in**.

- **`symbolic_keep_constants` unwrap mode** (`function/expressions.py`): expands every
  `UWexpression` atom down to — but **not including** — truly-constant atoms (η₀, τ_y, …),
  which stay as symbols so the `constants[]` runtime-update mechanism survives. The
  "stop at constant" predicate is the *same* `_is_truly_constant` that `getext()`'s
  `_extract_constants` uses, so the kept-symbolic set and the `constants[]` set provably
  cannot drift (drift-guard test).

- **`solver.consistent_jacobian`** (`SolverBaseClass`), one of:
  - `False` (**default**) — differentiate the flux *as wrapped*: frozen viscosity, the
    historical Picard tangent. **Bit-identical** to prior behaviour.
  - `True` — unwrap before differentiating: full consistent Newton tangent
    (`Min → Heaviside` yield switch appears).
  - `"continuation"` — Picard → Newton. Blend `J(α) = J_picard + α·(J_newton − J_picard)`
    with `α` a `constants[]` parameter ramped 0 → 1 at solve time (`_continuation_solve`):
    Picard locates the basin, Newton sharpens inside it. Because `α` lives in `constants[]`,
    switching costs **no JIT recompile**, and `α=0` is bit-identical to Picard.

- **Model-owned smooth-tangent hook** `Constitutive_Model.flux_jacobian` (default `None`):
  a model may supply a smooth surrogate flux for the *Jacobian only* while the residual
  keeps the exact law. Consumed via `_newton_flux`, which is **guarded** so the default
  (Picard) path never even evaluates it.

The residual is never routed through the new path, so a converged solution always satisfies
the exact constitutive law regardless of the tangent used.

### Non-regression evidence
- Constant-viscosity Jacobian is **symbolically bit-identical** (0/N blocks change).
- `test_1010`, SolCx `test_1015`, `test_0610`, the asymmetric-Jacobian guard, units (64) all
  pass; `level_1 tier_a` 225/225.
- Crash-isolated (`--forked`) `level_2 tier_a` failure set is **identical to pristine
  `development`** — the pre-existing reds (`test_1012` gmsh crash + 3 `test_1052` VEP) are
  not introduced here. (Creating the continuation `α` lazily was required to avoid a global
  symbol-counter perturbation that flipped two flaky VEP variable-dt tests.)

## Why a smooth Jacobian on a sharp residual is *not* the fix (tangent hierarchy)

For a hard-`Min` yield residual there is a strict ordering of tangents:

1. **Picard** (η frozen, `J = η·∂E/∂L`) — not the true Jacobian, but a *contractive*
   defect-correction operator whose fixed point **is** the `Min` solution. Slow (linear)
   but globally stable.
2. **Consistent Newton** (exact `Heaviside` tangent) — fast near the solution; the kink
   discontinuity breaks the line search far from it.
3. **Inconsistent smooth Jacobian** (harmonic tangent + `Min` residual) — the consistent
   tangent of a *different* (harmonic) problem; it points at the wrong solution and has
   neither Picard's contractivity nor Newton's consistency. **Worst of the three** — it
   diverges *more* than Picard on hard-yield VEP (measured 8/15 vs Picard 3/15 on the VEP
   loading test).
4. **Full smooth** (smooth residual *and* Jacobian) — consistent and smooth, converges, but
   solves a *smoothed* problem (different physics).

So "exact residual + smooth Jacobian" is option (3) — appealing but mathematically a dead
end. This is why the model-owned harmonic `flux_jacobian` override is **not** in this PR.
The valuable, well-posed part — the consistent tangent and the Picard→Newton continuation —
helps the *non-elastic* viscoplastic Stokes it was built for (the shear box converges via
continuation).

## Successor work (separate PR): δ-homotopy + yield-law unification

The robust route for hard-yield convergence is **problem-space** continuation, not tangent
tricks: ramp the *residual* smooth → sharp.

- Use the softmin softness **δ** as the homotopy parameter. softmin at **δ=0 is identically
  `Min`** (`g(f) = max(1, f)` exactly), and is C∞ for δ>0. So ramping δ → 0 gives a
  differentiable path whose endpoint has **zero physics change** (exact yield law) — unlike a
  yield-stress staircase, which moves the target.
- Empirically: solving smooth-first then warm-starting the sharp solve converges the hard
  VEP loading test **0/15** (vs cold 3/15).
- Production form mirrors the `α`-continuation: make δ a `constants[]` parameter and ramp it
  within one solve (per-iteration via the `add_update_callback` / `SNESSetUpdate` hook), so
  there is one compiled problem, one BDF-history update per step, and no recompile.
- Unify the yield-mode "zoo": collapse `softmin`/`smooth`/`min` into one δ-parameterised law
  (δ=0 default = exact `Min`; δ>0 = a *controlled* smooth-min), keep `harmonic` as a distinct
  physical model, drop the redundant `smooth` formula. "Smooth-min" then means *incomplete
  blending* (final δ>0), not an ad-hoc surrogate with simulation-dependent deviation from τ_y.

That work depends on this PR's `constants[]`-ramp machinery, so it lands after.
