---
name: plasticity-solvers
description: How to get hard-Min viscoplastic / visco-elastic-plastic (VEP) Stokes solves to CONVERGE in Underworld3 — the consistent Newton tangent (solver.consistent_jacobian), the δ-soft-min yield law, and the yield homotopy that pairs them. Reach for THIS first when a Drucker-Prager / yield-stress Stokes solve stalls, diverges (DIVERGED_LINEAR_SOLVE / line-search fail), or grinds through ~20+ nonlinear iterations. Tells you which tangent to use per model, how to confirm you are actually running Newton, and the measured failure modes.
---

# plasticity-solvers

The workable recipe for **nonlinear convergence of yielding (viscoplastic / VEP)
Stokes** in Underworld3. Hard-`Min` yield laws have a non-differentiable kink that
breaks naive solvers; this encodes the combination that converges.

**One call does it:**

```python
stokes.constitutive_model = cm           # ViscoPlastic / ViscoElasticPlastic / TI-VEP
cm.Parameters.yield_stress = tau_y       # finite -> plasticity active
cm.enable_yield_homotopy()               # <- the strategy: δ-ramp + the right tangent
stokes.solve(zero_init_guess=False)
```

`enable_yield_homotopy()` is the recommended default for any hard-Min solve. It picks
the Jacobian tangent per model (below) and ramps the soft-min δ→0 within one solve.

---

## The two ingredients

Yielding viscoplasticity is `η_eff = Min(η_visc, η_yield)`,
`η_yield = τ_y/(2·ε̇_II)`. The `Min` kink is what makes it hard.

1. **Consistent Newton tangent** — `solver.consistent_jacobian`:
   - `False` (default) — frozen-viscosity **Picard** tangent: contractive, globally
     stable, **linear** (slow). Bit-identical to long-standing UW3.
   - `True` — full **Newton** tangent (the assembled `dF1/dL` carries `∂η/∂(grad v)`):
     **quadratic** near the solution; the kink can break it far from it.
   - `"continuation"` — Picard→Newton α-blend (α a `constants[]` atom ramped 0→1).

2. **δ-soft-min yield law + homotopy** — `g = 1 + ½(f-1+√((f-1)²+δ²)) − offset`,
   `η_eff = η_ve/g`, `f = η_ve/η_pl`:
   - **δ = 0 (default) ≡ exact `Min`** to machine precision.
   - δ and the onset offset are `constants[]` atoms → δ is **runtime-rampable with no
     JIT recompile**. `enable_yield_homotopy()` ramps δ from `delta_start`→0 within one
     solve (residual-paced absolute schedule, via the `SNESSetUpdate` hook). The smooth
     (δ>0) problem warm-starts the sharper one; δ ends at 0 so the **converged answer is
     on the exact yield surface**.

This is **problem-space** continuation (ramp the residual smooth→sharp). A smooth
Jacobian on a sharp `Min` residual is the consistent tangent of a *different* (harmonic)
problem and diverges worse than Picard — don't do that.

---

## Confirm you are actually running Newton

A consistent-Newton solve should converge **quadratically** — the residual roughly
squares each iteration and reaches ~1e-12 in 3–6 nonlinear steps. A **linear** tail
(residual dropping by a roughly constant factor over ~15–25 steps) means you are on the
Picard tangent — check `solver.consistent_jacobian is True` and that the viscosity is a
field of the unknowns, not a constant.

Direct symbolic check that the Newton term is in the Jacobian (`dF1/dL` differs between
the frozen and unwrapped flux by exactly the `∂η/∂(grad v)` term):

```python
import sympy
from underworld3.function.expressions import unwrap_expression
F1 = sympy.Array(stokes.F1.sym)            # residual flux (η wrapped)
L  = sympy.Array(stokes.Unknowns.L)        # velocity-gradient symbols
G_picard = sympy.derive_by_array(F1, L)
F1_unwrapped = sympy.Array(
    [unwrap_expression(e, mode="symbolic_keep_constants") for e in F1], F1.shape)
G_newton = sympy.derive_by_array(F1_unwrapped, L)
assert sympy.simplify(sympy.Array(G_newton) - sympy.Array(G_picard)) != \
       sympy.Array([0]*len(list(sympy.flatten(F1)))*len(list(sympy.flatten(L))))
# nonzero difference  ==  the Newton form is present
```

---

## Which tangent for which model (measured)

`enable_yield_homotopy(consistent_tangent="auto")` is **model-aware**:

| Model | `"auto"` picks | Why |
|-------|----------------|-----|
| `ViscoPlasticFlowModel` (non-elastic) | **Newton** + δ-ramp | δ-ramp keeps the residual smooth while Newton finds the basin; both sharpen to exact Min together. From a viscous guess the consistent tangent is robust (≈3–4 quadratic iters). |
| `ViscoElasticPlasticFlowModel` (VEP) | **Picard** + δ-ramp | The consistent yield tangent over the elastic stress-history block makes the Jacobian **indefinite → `DIVERGED_LINEAR_SOLVE`**. Picard is contractive; with the δ-ramp it converges to the exact yield surface. |
| `TransverseIsotropicVEPFlowModel` (TI-VEP) | **Picard** + δ-ramp | same as VEP (elastic). |

Override with `consistent_tangent=True` / `False` / `"continuation"`.

> Measured: VEP loading-through-yield — Picard+δ-ramp converges (σ locks at τ_y),
> Newton+δ-ramp diverges every step (`DIVERGED_LINEAR_SOLVE`).

---

## Failure modes → fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `DIVERGED_LINEAR_SOLVE`, 0 iters, VEP | consistent Newton tangent on the elastic block → indefinite | use Picard (`consistent_tangent=False`, or `"auto"` on a VEP model) |
| Line-search failure at first step | `bt` line search trips on the kink as δ steps down | `enable_yield_homotopy` sets `snes_linesearch_type="basic"` (full step); keep it |
| Converges but σ sits **below** τ_y | solving a *smoothed* problem (fixed δ>0, or a smooth surrogate) | use the δ-ramp (δ→0) so the endpoint is exact Min |
| Linear (~20-iter) convergence | running Picard when you wanted Newton | `consistent_jacobian=True` on a non-elastic model (see "Confirm" above) |

---

## Gotchas

- **`./uw build` → `amr-dev` env.** Verify `uw.__file__` is the worktree site-packages.
- **Run VEP tests UNFORKED** — `pytest --forked` SIGABRTs here (fork of multithreaded PETSc).
- `enable_yield_homotopy` raises the `snes_max_it` floor to 100 and needs the model
  **attached to a solver first** (`solver.constitutive_model = model`).
- `harmonic` yield mode is a **distinct physical model** (parallel blend), not an
  approximation to Min — the homotopy does not apply to it.
- If you project η, use a **low-order** field (P0/P1) — higher order overshoots and η
  is not guaranteed positive.

---

## Reference

- `ViscousFlowModel._combine_yield`, `enable_yield_homotopy` / `_yield_homotopy_step`
  in `constitutive_models.py`; `solver.consistent_jacobian` / `_jacobian_source`.
- Design: `docs/developer/design/jacobian-consistent-tangent.md`.
- Tests: `tests/test_1053_yield_homotopy.py`.
- Benchmark: `docs/examples/WIP/Benchmark/Ex_VP_Spiegelman_Benchmark.py`.

<sub>Footnote: before this work UW3 differentiated the flux with the viscosity still
wrapped, so `∂η/∂(grad v)` was dropped and viscoplastic solves silently ran the Picard
tangent — the origin of the "~20 iterations is intrinsic" folklore.</sub>
