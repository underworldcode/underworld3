---
name: plasticity-solvers
description: How to get hard-Min viscoplastic / visco-elastic-plastic (VEP) Stokes solves to CONVERGE in Underworld3 — Newton with the automatic Picard entry (solver.consistent_jacobian), which tangent per model, grid sequencing for the hard cases, and the δ-soft-min substrate (yield_mode / yield_smoother / yield_anchor) as a modelling choice. Reach for THIS first when a Drucker-Prager / yield-stress Stokes solve stalls, diverges (DIVERGED_LINEAR_SOLVE / line-search fail), or grinds through ~20+ nonlinear iterations. Tells you which tangent to use per model, how to confirm you are actually running Newton, and the measured failure modes. For the solver-config trap list and multigrid, see `nonlinear-solver`.
---

# plasticity-solvers

The workable recipe for **nonlinear convergence of yielding (viscoplastic / VEP)
Stokes** in Underworld3. Hard-`Min` yield laws have a non-differentiable kink that
breaks naive solvers; this encodes what the yield campaigns measured actually works —
and records what was retired.

**The default call is now just:**

```python
stokes.constitutive_model = cm           # ViscoPlastic / ViscoElasticPlastic / TI-VEP
cm.Parameters.yield_stress = tau_y       # finite -> plasticity active
stokes.consistent_jacobian = True        # Newton tangent (non-elastic DP; see table)
stokes.solve()                           # cold start takes ONE Picard step automatically
```

---

## The doctrine (measured, 2026-07 campaigns)

Yielding viscoplasticity is `η_eff = Min(η_visc, η_yield)`,
`η_yield = τ_y/(2·ε̇_II)`. The `Min` kink is what makes it hard.

1. **Picard is an ENTRY requirement, not an accelerator.** On a cold start under
   the consistent tangent, `solve()` takes one automatic Picard (frozen-tangent)
   step and then runs Newton (fires only when `picard==0`,
   `consistent_jacobian is True`, and the start is cold — from a warm iterate,
   0 Picard really is 0). Do NOT front-load Picard where Newton works: measured,
   opening with 5 / 25 Picard steps cost 12 / 30 total iterations against pure
   Newton's 7.

2. **Newton-first; spend Picard only to rescue.** When Newton fails
   (`DIVERGED_LINE_SEARCH` / `DIVERGED_LINEAR_SOLVE`) **or stalls admissibly**
   (steps accepted, residual flat — no FAIL reason ever fires), revert to the best
   iterate and buy a Picard block: `solve(picard=N)`, or
   `consistent_jacobian="continuation"` (staged Picard→Newton α-blend; α is a
   `constants[]` atom, no recompile). Rescue on *stagnation*, not only on a
   failure reason — the failure-only trigger measured byte-identical to doing
   nothing at the cliff.

3. **Grid sequencing is the validated warm start for hard problems.** Solve
   coarse (it finds the localisation structure cheaply), transfer the state up
   (the linear-exact local RBF, #430), warm-start the fine solve. Measured on the
   notch: 2–3× deeper residual, more localised, fewer iterations than any cold
   fine strategy. No packaged API yet — hand-roll the cascade with
   `uw.function.evaluate` per level; PETSc's `-snes_grid_sequence` does NOT work
   on UW3 meshes. See `docs/developer/design/multilevel-nonlinear-stokes-strategy.md`.

4. **`solver.has_solution` / tri-state `zero_init_guess`** make warm-started
   campaigns safe: `None` (default) auto-detects; a diverged solve or a remesh
   clears the flag so the next solve cold-starts (with its Picard entry) instead
   of warming off a corrupted iterate.

---

## The retired doctrine — do not resurrect it

An earlier line of work paired the δ-soft-min with a **yield homotopy** and shipped
a model-level enable method for an in-SNES δ-ramp. That API **has been removed from
the source**, and the doctrine it taught rested on a unit-scaling error in the
campaign that motivated it.
Re-measured on the correctly-scaled problem (13 points across two parameter axes):

- the δ-march **never succeeded where a direct hard-Min solve failed**, and where
  both work the direct solve is 4–5× faster with better residuals;
- homotopy rescues **Picard**, not Newton — under the consistent tangent it adds
  nothing;
- ramping δ **inside** a single SNES solve is separately proven dead (diverges
  `DIVERGED_LINEAR_SOLVE` within ~2 iterations even on the proven config).

The ruling that closed the campaign: **regularise the PROBLEM (give the shear band
a physical length scale), not the solver.** Where a hard-Min solve will not
converge, sharpening δ is not the missing lever — a viscous seed, the Picard
rescue, and grid sequencing are.

---

## Which tangent for which model (measured)

`solver.consistent_jacobian` takes `False` | `True` | `"continuation"`:

| Model | Use | Why |
|-------|-----|-----|
| `ViscoPlasticFlowModel` (non-elastic) | **`True`** (Newton) | Quadratic near the solution; the automatic Picard entry handles the cold start. |
| `ViscoElasticPlasticFlowModel` (VEP) | **`False`** (Picard) | The consistent yield tangent over the elastic stress-history block makes the Jacobian **indefinite → `DIVERGED_LINEAR_SOLVE`**. Picard is contractive. |
| `TransverseIsotropicVEPFlowModel` (TI-VEP) | **`False`** (Picard) | Same as VEP (elastic). |
| Any, far from the solution | **`"continuation"`** | Staged Picard→Newton; Picard locates the basin, Newton finishes. Beat pure Newton at every notch point measured — but its stage switch is a residual LEVEL and one-way, so it can overspend Picard on easy problems. |

> Measured: VEP loading-through-yield — Picard converges (σ locks at τ_y),
> Newton diverges every step (`DIVERGED_LINEAR_SOLVE`).

---

## Confirm you are actually running Newton

A consistent-Newton solve on a smooth-enough problem converges **quadratically** —
the residual roughly squares each iteration and reaches ~1e-12 in 3–6 nonlinear
steps. A **linear** tail (a roughly constant reduction factor over ~15–25 steps)
means you are on the Picard tangent — check `solver.consistent_jacobian is True`
and that the viscosity is a function of the unknowns, not a constant. (On genuinely
hard localising problems the quadratic phase may never be reached — that is the
problem, not the tangent; see the doctrine above.)

Direct symbolic check that the Newton term is present (`dF1/dL` differs between the
frozen and unwrapped flux by exactly the `∂η/∂(grad v)` term):

```python
import sympy
from underworld3.function.expressions import unwrap_expression
F1 = sympy.Array(stokes.F1.sym)
L  = sympy.Array(stokes.Unknowns.L)
G_picard = sympy.derive_by_array(F1, L)
F1_unwrapped = sympy.Array(
    [unwrap_expression(e, mode="symbolic_keep_constants") for e in F1], F1.shape)
G_newton = sympy.derive_by_array(F1_unwrapped, L)
# a nonzero difference == the Newton form is present
```

---

## δ smoothing — a modelling choice, not a convergence strategy (#475 substrate)

If you want a *rounded* yield law at all (as physics or as a formulation choice),
the substrate is three model properties; δ is a `constants[]` atom, so changing it
never recompiles:

- **`yield_mode`**: `"min"` (default — exact hard `Min`), `"softmin"` (the
  δ-parameterised family below), `"harmonic"` (a **distinct physical model**, a
  parallel blend — not an approximation to `Min`).
- **`yield_smoother`**: `"sqrt"` or `"powermean"`. **δ is NOT the same parameter
  in the two families**: the power mean's sharpness is `s = 1/(δ + 0.001)`, so
  δ ≤ 1 and δ = 1 IS the harmonic mean; the sqrt family's δ is a percentage stress
  deviation, generous entry O(10), and δ = 0 is exactly `Min`. The power mean at
  δ = 0 lands within 0.07 % of `Min` — an order of magnitude inside a 1e-8 solver
  tolerance.
- **`yield_anchor`**: which point is pinned to the exact law — the SIDE of `Min`
  belongs to the anchor, not the family. `"onset"` (default, historical) is exact
  on the unyielded branch but sits BELOW `Min` at and above yield — a *weaker*
  problem than the sharp one. `"yield"` pins τ/τ_y = 1 exactly and sits on-or-above
  `Min` everywhere; the cost is stiffer unyielded material (bounded ×2 sqrt,
  ×2^δ powermean, both → 1 as δ → 0).

**If you march δ toward the sharp law, the only sound discipline is multi-solve:**
hold δ constant for a full solve to tolerance, warm-start the next smaller δ,
sharpen only between converged solves. Never ramp δ inside one SNES solve. The
packaged march is `stokes.solve(homotopy=True)` /
`underworld3.systems.yield_continuation` — usable, with two open caveats (#473):
its documented cold-start guarantee does NOT hold on a multi-material
(`Piecewise`) yield stress, so give it a viscous pre-solve anyway; and its
adaptive step control is effectively one-shot (one early decision pins the step
for the whole march). Do not expect it to cross a cliff the direct solve cannot —
measured, it never has.

---

## Floors

- **`shear_viscosity_min`** (default `-oo` = off) is applied through
  `uw.maths.smooth_max`, but the default rounding scale is zero under
  `yield_mode="min"` and `δ·|floor|` under the smooth modes — so it **vanishes as
  δ → 0**, leaving an exact `Max` corner that kills the consistent tangent
  (`nl=0, DIVERGED_LINEAR_SOLVE`). Set **`viscosity_min_rounding`** (a few per
  cent of the floor) and the cutoff is differentiable at any δ, including 0.
- A viscosity floor bounds the viscosity contrast and therefore how localised the
  solution can be — relaxing it toward zero is a solution-SELECTION continuation,
  independent of δ. Use it deliberately.
- **Do not add a strain-rate floor for the cold start.** At `ε̇=0`, `η_pl=+inf`
  is carried correctly to the viscous branch by `Min` and by both smooth families;
  only a hand-rolled product-over-sum harmonic blend breaks (`inf/inf`) — write it
  as `η_ve/(1+f)`.

---

## Failure modes → fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `DIVERGED_LINEAR_SOLVE`, 0 iters, VEP | consistent Newton over the elastic block → indefinite | Picard (`consistent_jacobian=False`) |
| `DIVERGED_LINEAR_SOLVE` at nl=0 with a viscosity floor set | δ→0 leaves the floor's `Max` corner exact | set `viscosity_min_rounding` |
| Newton stalls with no divergence reason | admissible uselessness — steps accepted, residual flat | revert to best iterate, Picard block (`picard=N` / `"continuation"`); consider grid sequencing |
| Converges but σ sits **below** τ_y | a fixed δ>0 soft-min under the default `"onset"` anchor is a WEAKER law | that is the modelling choice you made — use `yield_anchor="yield"`, or δ→0 / `yield_mode="min"` for the exact surface |
| Linear (~20-iter) convergence | Picard tangent when you wanted Newton | `consistent_jacobian=True` on a non-elastic model (see "Confirm" above) |

---

## Gotchas

- **`./uw build` → `amr-dev` env.** Verify `uw.__file__` is the worktree site-packages.
- **Run VEP tests UNFORKED** — `pytest --forked` SIGABRTs here (fork of multithreaded PETSc).
- `harmonic` yield mode is a **distinct physical model**, not an approximation to Min.
- If you project η, use a **low-order** field (P0/P1) — higher order overshoots and η
  is not guaranteed positive.

---

## Reference

- Yield law: `ViscousFlowModel._combine_yield`, `yield_anchor`, `yield_smoother`,
  `viscosity_min_rounding` in `constitutive_models.py`.
- Tangent: `solver.consistent_jacobian` / `_jacobian_source` in
  `petsc_generic_snes_solvers.pyx`; design
  `docs/developer/design/jacobian-consistent-tangent.md`.
- Warm start / continuation: `docs/developer/design/nonlinear-solver-homotopy-warmstart.md`;
  grid sequencing: `docs/developer/design/multilevel-nonlinear-stokes-strategy.md`.
- Tests: `test_0201_solver_has_solution_warmstart.py`, `test_1055_yield_smoother.py`,
  `test_1057_yield_homotopy_solve.py`, `test_1059_yield_anchor.py`.
- Solver-config traps, smoother, FMG/multigrid: the `nonlinear-solver` skill.

<sub>Footnote: before this work UW3 differentiated the flux with the viscosity still
wrapped, so `∂η/∂(grad v)` was dropped and viscoplastic solves silently ran the Picard
tangent — the origin of the "~20 iterations is intrinsic" folklore.</sub>

## SNESFAS — do not reach for it

Nonlinear multigrid (SNESFAS) looks tempting for hard viscoplastic solves but is
**not a viable option** at present (maintainer ruling 2026-07-17): there are no
good preconditioners for the nonlinear hierarchy, and it abandons the robust
linear-solver path (consistent tangent / continuation + fieldsplit + MG) that
this skill is built around. It stays options-only for experiments; treat it as a
future investigation. See `docs/developer/design/solver-strategies-catalogue.md`
and `MULTIGRID_MINIMAL_CONTROL_2026-07.md` (ruling 6).
