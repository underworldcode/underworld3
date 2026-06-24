# Yield homotopy convergence study — plan and findings

Status: **proof-of-concept established; academic investigation planned.** This note is the
version-controlled planning record. The runnable harness lives outside the repo at
`~/+Simulations/spiegelman_hardcase/` (drivers/, logs/, figures/, meshes/, README.md) —
durable and browsable; see its README for the live run log.

## Idea

Hard-`Min` Drucker–Prager viscoplastic Stokes is hard to converge because (a) the yield
kink is non-smooth and (b) the DP yield `σ_y = C + sinφ·P` depends on the **solution
pressure**, so the consistent Newton tangent is **non-symmetric** (an `∂η/∂P` coupling of
the deviatoric stress to pressure with no mirror in continuity — this is exactly why
Spiegelman et al. 2016 Fig 5 shows von Mises converging but DP failing).

The **δ-soft-min yield law** (`ViscousFlowModel._combine_yield`) regularises the
*constitutive law*, smoothing the **residual and the consistent Jacobian together** (kept
mutually consistent — not the dead-end smooth-Jacobian/sharp-residual tangent). A
**homotopy** that drives δ→0 then (i) automatically approaches the hard-Min solution and
removes the free δ parameter, and (ii) **opens the problem to iterative (FMG) solvers** —
the smoothed operator is MG-tractable where the δ=0 kink operator is not. Reaching exactly
δ=0 is not required: settling at the smallest feasible δ is a useful, well-defined answer.

## Proven so far (direct solver, diagnostic only — NOT the deliverable)

On the genuine hard regime — Spiegelman notch, size-3 (9645 cells), φ=30, **strong layer
η_bg=3e25 Pa·s, V=10 mm/yr** (Fig 5's "stronger layer / larger compression velocity"):

| strategy | result |
|----------|--------|
| Newton (consistent, δ=0) | FAIL (−3, kink) |
| Picard | FAIL (−5, stalls) |
| fixed homotopy (δ_start too small) | FAIL (over-pushes δ→0) |
| **automatic δ-continuation** | **CONVERGES, settles δ=0.5**, no parameter chosen |

The δ-continuation fully converges at δ=64, warm-starts halved δ, marches down, settles
where the next δ fails: path `64→32→16→8→4→2→1→0.5✓ → 0.25 FAIL`. **Warm-starting
compounds**: cold solves only reach δ≈2 (sweep threshold), warm reaches 0.5.

FMG (the deliverable) reproduced the core point — its δ=64 step converged cleanly under
FMG where FMG-Newton on δ=0 failed — but errored (PETSc 63, ARG_OUTOFRANGE) partway down
the march under tight ksp caps; **fixing this is task 1**.

## What δ means physically (regime-independent)

In the dimensionless overstress ratio `f = η_ve/η_pl = 2η_ve·ε̇_II/τ_y`, the normalised
stress is `τ/τ_y = f/g(f,δ)`. So δ has a **regime-independent** meaning — a fixed
*percentage deviation* from the yield value, independent of the actual η/τ_y — which is
how to set it consistently across problems (figure: `figures/homotopy_yield_curves.png`):

| δ | 0.5 | 1 | 2 | 8 | 64 |
|---|-----|---|---|---|----|
| current sqrt-soft-min: max τ/τ_y | +1% | +3% | +9% | +28% | +62% |

**The current sqrt-soft-min *overshoots* yield** in the transition (carries up to 1–60%
over τ_y) before asymptoting to τ_y. The **power-mean / harmonic family** (below)
approaches yield strictly **from below** (τ/τ_y ≤ 1 always) — arguably the more physical
regularisation.

## Investigation plan (next session)

1. **Fix the FMG error 63**; confirm the FMG δ-continuation settles cleanly at (3e25,V10),
   then across a regime grid. FMG (geometric MG) is *the deliverable* — it scales; direct
   is reference only. AL `penalty` conditions the Schur (helps outer Krylov); document its
   tuning. Watch the AL-augmented velocity block under MG smoothers.
2. **Two smooth-min families — evaluate both, power-mean first.** Add the power-mean
   `η_eff = (η_ve^(−s) + η_pl^(−s))^(−1/s)`, **s = 1/δ** (s=1 = harmonic mean, parameter-
   free; s→∞ = exact Min) to `_combine_yield` behind a `yield_smoother=` selector. It is
   simpler (no offset) and **undershoots** yield (no over-yield). Compare convergence
   robustness *and* physical fidelity head-to-head with the sqrt-soft-min.
3. **Sequencing study.** δ=64/32/16 are indistinguishable (smoothing saturates at large
   δ); the action is all near δ→0. March in **s = 1/δ** (the power-mean exponent — the
   natural measure) or in the physical overshoot/undershoot fraction, rather than
   geometric-in-δ. Compare against the convergence-adaptive `run_continuation`.
4. **Probe→refine continuation (two difficulty metrics).** Instead of guessing a high
   δ_start and marching down, auto-discover the entry:
   - **Probe (cold, hard→easy):** from the viscous guess, start sharp (large s / small δ)
     and relax smoothing until the FIRST cold solve converges → **δ_entry / s_entry**, the
     "smoothest starting solve required" — an intrinsic problem-difficulty scalar.
   - **Refine (warm, easy→hard):** from that solution, warm-start (v,p) back toward
     hard-Min, settling at **δ_settle / s_settle**.
   The **gap** δ_entry→δ_settle measures how much the warm-starting buys (at (3e25,V10):
   cold threshold δ≈2, warm rides to 0.5). Map (δ_entry, δ_settle) across the Fig-5 plane.
   Small extension of `run_continuation` (add a cold probe-up loop before the warm march).
5. **Regime map.** Sweep the Fig-5 plane (η_bg × V): where each {family × sequence ×
   solver} converges. Reproduce the vM-converges / DP-fails / homotopy-rescues figure.
6. **Metrics.** Total Newton iterations, δ_entry/δ_settle, closeness to hard-Min, yield
   overshoot/undershoot, robustness across regimes.
7. **Write-up** + figures (convergence histories, yield-law curves, regime map).

## Pointers

- Harness: `~/+Simulations/spiegelman_hardcase/` (README.md is the live index).
- Driver knobs: `SOLVER=fmg|direct`, `ETA_BG`, `V_MMYR`, `DELTA_START`, `PENALTY`,
  `SMOOTH_KSP`, `SMOOTH_IT`, `KSP_MAXIT`, `SWEEP_DELTA`.
- Code: `_combine_yield`, `enable_yield_homotopy`/`_yield_homotopy_step`,
  `solver.consistent_jacobian` (PR #258). Tests: `tests/test_1053_yield_homotopy.py`.
- Solver design: `jacobian-consistent-tangent.md`. Skill: `plasticity-solvers`.
