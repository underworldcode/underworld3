# Monge–Ampère mesh redistribution: Newton/cofactor linearisation

> **Status**: exploration (Phase 0), `feature/winslow-mesh-smoother`,
> 2026-05-17. Companion to
> `docs/developer/subsystems/mesh-metric-redistribution.md` (the
> shipped BFO-Picard + direct-solver work) and the project memory
> `project-ma-efficiency-direct-solver`.

## Motivation

The shipped MA path (`_winslow_elliptic`) is a damped
**Benamou–Froese–Oberman Picard** iteration: each iteration solves a
*constant-coefficient* Poisson `Δφ = √((φxx−φyy)²+4φxy²+4g)−2` with
the recovered Hessian of the previous iterate, ~20–25 iterations,
under-relaxation `ω=0.4`. The constant operator is what made the
factor-once-reuse direct-solver speedup (~10×) possible — but that is
a **serial** expedient (sparse direct factorisation does not scale to
large-3D parallel per-timestep use; this build has only MUMPS + GAMG,
no hypre/SuperLU_DIST).

A **Newton / quasi-Newton** linearisation is the textbook approach for
smooth MA / mesh redistribution / OT. Linearising
`R(φ)=det(I+D²φ)−g`:

$$ \operatorname{cof}(I+D^2\varphi_k) : D^2\,\delta\varphi
   \;=\; g-\det(I+D^2\varphi_k), \qquad
   \varphi_{k+1}=\varphi_k+\lambda\,\delta\varphi $$

Using the Jacobi (Piola) identity `∂_i cof(M)_{ij}=0`, the weak form
is the **symmetric variable-coefficient elliptic** problem

$$ a(\delta\varphi,v)=\int (C_k\nabla\delta\varphi)\cdot\nabla v,
   \qquad C_k=\operatorname{cof}(I+D^2\varphi_k), $$

with `C_k` SPD iff `φ_k` is convex (Brenier branch). In 2D
`C_k = [[1+φyy, −φxy],[−φxy, 1+φxx]] = det(M_k)·M_k⁻ᵀ`. Only **first
derivatives of the unknown** appear (in the flux `F1=C_k∇δφ`); all
2nd-derivative content is in the *coefficient* `C_k`, read from the
existing recovered-Hessian field (`_hessian_recovery_class`,
first-derivatives-only — UW3-legal).

It slots into the existing `uw.systems.Poisson` (`SNES_Scalar`):
`F1 = constitutive_model.flux = c·∇u`, so a `DiffusionModel` subclass
with `_c = C_k` *is* the Newton operator; `f = det(I+H_k)−g` is the
source; `constant_nullspace` handles the pure-Neumann singularity
exactly as the BFO path does. Single-field scalar SNES — **not** the
rejected fully-coupled (φ,H) SNES.

### What it can and cannot change

- **Cannot** change the fixed-node grading ceiling (≈1.5–1.8× for an
  8–20× target). Same equation, same recovered Hessian ⇒ same fixed
  point. The OT ~10× needs *more nodes* (settled — see
  `project-ma-recovered-hessian-picard-inadequate`). Newton is **not a
  grading lever**; `ma_cost_grading.py` (1.02/1.43/1.71/1.54) is the
  regression guard.
- **Can** change convergence: few Newton iterations vs ~20–25 Picard
  ⇒ insensitive to per-iteration setup cost (the GAMG-resetup failure
  mode), and the per-step operator is SPD variable-coefficient
  elliptic ⇒ **AMG-friendly** ⇒ the right structure for the parallel
  rework.

## Phase 0 — residual-contraction quantification

**Goal**: confirm Newton contracts the MA residual
`r_k = det(I+H_k) − g` in a handful of iterations vs the BFO-Picard's
~20–25, on the canonical res-16 Annulus, *before* any source changes.
Both schemes share the φ field, the recovered-Hessian solver, the `c`
normalisation, `g`, the constant nullspace and pinned BCs — the **only
difference is the inner potential update**. Geometry is held fixed
(no node move) to isolate solver contraction.

Script: `scripts/ma_newton_phase0.py` (no `src/` changes; uses
`smoothing._hessian_recovery_class`, `_use_direct_solver`,
`_auto_pinned_labels`).

### Results

**Run 1 (AMP=8, RES=16) — a methodological finding.** Measuring
contraction of `r_k = det(I+H_k) − g` (H recovered) was the *wrong
yardstick*: it has a large **irreducible floor** that *neither*
scheme reduces — BFO plateaus at `‖r‖≈0.29` (from 0.46), Newton at
`≈0.34`. That floor is precisely the recovered-Hessian
under-estimation of `det(D²φ)` that the project memory identifies as
the root cause of the ≈1.5–1.8× single-solve cap (the FE-MA fixed
point is *self-consistently under-deformed*; `det(I+H_rec)−g` is O(0.3)
even at the exact FE solution). Strong confirmation that **Newton on
the cofactor operator cannot beat the grading ceiling** (same
recovered Hessian, same floor) — exactly as predicted; it is not a
grading lever.

Consequence for the experiment: a residual-decrease line search on
`‖det(I+H_rec)−g‖` is meaningless here (it rejected almost every
Newton step, collapsing `λ→0.008` and freezing the iteration — *not*
a fair Newton test). The correct Phase-0 question is the
*efficiency* one: **does Newton reach the same fixed-node transport
map in far fewer iterations than BFO's ~20–25?** The valid metric is
the **transport-map increment** `Δ_k = ‖∇φ_k − ∇φ_{k-1}‖∞` (→0 as
the map converges) and the realised `max|∇φ|` / honest grading
(must match BFO — the regression guard). Run 2 uses that, with
fixed damping (no residual-rejection; keep only a `det(I+H)>0`
convexity backtrack).

**Run 2 (AMP=8, RES=16) — transport-map contraction.** Metric:
`d_k=max|∇φ_k|`, increment `Δ_k=max|∇φ_k−∇φ_{k-1}|`; final honest
`d/n` after one signed-area-backtracked move. Three Newton
convexity-safeguard variants, all vs the shipped BFO-Picard.

| scheme | converges? | iters (Δ<1e-3·d₀) | final d/n | note |
|---|---|---|---|---|
| **BFO-Picard** (`+√` branch, ω=0.4) | yes | **16** | **1.713** | shipped; reference |
| Newton, residual line-search | no | — (frozen) | — | λ→0.008; the `det(I+H_rec)−g` floor (Run 1) makes the search objective meaningless |
| Newton, `det>0` backtrack only | no | — (stalls) | 1.58 | recovered-H noise breaks convexity under a finite step ⇒ λ→0.002, under-deforms |
| Newton, **PD-projected H** (eps=0.05) | no | — (creeps) | 1.49 | no λ collapse, but `Δ_k` plateaus ≈2e-3 (never contracts), overshoots `max|∇φ|` past BFO, map inverts cells (move scale→0.5) |

### Verdict — Newton/cofactor is NOT the efficiency/parallel path

Decisive negative result, consistent with and extending the settled
memory:

1. **It cannot beat the grading cap** (predicted): same recovered
   Hessian ⇒ same `det(I+H_rec)−g` floor (Run 1). Not a grading lever.
2. **It is *less robust* than BFO at the same recovered-Hessian
   quality** (new): all three convexity safeguards from the standard
   remedy list fail to reach BFO's fixed point — the iteration either
   freezes, stalls under-deformed (1.58), or creeps past it into a
   cell-inverting state (1.49). BFO reaches d/n 1.713 in 16 iters.
3. **Root cause**: BFO's `Δφ=√((φxx−φyy)²+4φxy²+4g)−2` is not "just a
   linearisation" — it is a *closed-form convex-branch solve* that
   expresses the new Laplacian via `g` and only the **deviatoric**
   part of the recovered Hessian, side-stepping the noisy/
   under-estimated full `det`. The cofactor-Newton operator feeds the
   full noisy recovered Hessian into *both* the variable coefficient
   `C_k` *and* the residual `det(I+H_k)−g`; at this recovery quality
   that is fragile (non-convex repulsion) or, once convexity is
   forced by projection, no longer the true MA equation (drifts
   instead of contracting). UW3 forbids 2nd derivatives of mesh-var
   functions, so a genuinely sharp `D²φ` (which Newton needs) is not
   available — the original footgun. Newton would only pay off with a
   fundamentally better Hessian / a wide-stencil MA discretisation:
   research effort, **no expected grading gain (settled) and now a
   demonstrated robustness loss**. Do not pursue.

### Implication for the parallel requirement

The validated efficiency lever stays the **factor/setup-once-reuse on
the constant BFO Laplacian** (shipped, ~10× serial via MUMPS). For
**parallel**, port that exact pattern to GAMG (the only AMG in this
build): build the GAMG hierarchy **once per `_winslow_elliptic` call**
(the operator is constant across the ~25 BFO iters) via
`snes_lag_jacobian=-2` / `KSPSetReusePreconditioner` with the constant
near-nullspace already wired, and warm-start the Krylov from the
previous Picard φ. Parallel-scalable, keeps BFO's robust convex-branch
structure, preserves grading. This — not Newton — is the parallel
work item. Script: `scripts/ma_newton_phase0.py`; data
`/tmp/metric_mesh/ma_newton_phase0.npz`.
