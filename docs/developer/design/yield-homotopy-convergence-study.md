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

## Results — 2026-06-25 session

### Task 1 — FMG error 63 fixed; FMG δ-continuation settles cleanly
Root cause was a **SNES monitor leak**, not anything numerical: the Stokes `solve()` path
re-runs `snes.setFromOptions()` 2–3× per call (to switch SNES type between the nrichardson
pre-solve and the Newton/Picard solve). Each call re-registers the option-configured
`snes_monitor` on the *persistent* SNES, so repeated solves (the δ-continuation) accumulate
monitors until PETSc throws `ARG_OUTOFRANGE` "Too many monitors set" (`MAXSNESMONITORS`=5).
Fixed in `petsc_generic_snes_solvers.pyx` with `snes.monitorCancel()` before each per-solve
`setFromOptions()`. (Latent bug for *any* repeated Stokes solve with a monitor set — e.g.
time-stepping.) The harness now also captures the residual history **in memory** via the
SNESSetUpdate hook, since the `:file` monitor truncates on each re-registration.

**FMG δ-continuation at (3e25, V10) now settles cleanly** — marches `64→…→0.002`,
converging at *every* step (reason=2), settling at δ≈0.002 (≈ hard-Min). This is *stronger*
than the direct diagnostic (which settled δ=0.5): the warm-start chain reaches hard-Min.
The regime is still **cold-hard** (standard cold Newton/Picard fail there); the continuation's
warm-starting is what makes the kink reachable — see the probe→refine δ_entry below.

### Task 2 — power-mean smooth-min added; **now the preferred family** (L.M.)
`ViscousFlowModel.yield_smoother = "sqrt" (default) | "powermean"`. The power-mean p-norm
`η_eff = (η_ve^(−s) + η_pl^(−s))^(−1/s)`, `s = 1/δ` (s=1 harmonic, s→∞ Min), is evaluated in
an overflow-safe **harmonic-normalised factored form** `N·(a^(−s)+b^(−s))^(−1/s)` with
`a=1+f`, `b=1+1/f`, `N=η_ve η_pl/(η_ve+η_pl)`: both bases ≥1 and at least one ≈1, so it
never over/underflows on geodynamic ranges (η~1e21–1e26) where the naive `η^(−s)` form
overflows above s≈40.

**L.M. preference (2026-06-25):** the power-mean is what the soft-min was always meant to
be — *arbitrarily close to hard-Min, one free parameter, and at low-δ / high-s close enough
to the yield surface that the rounded corner is immaterial*. Crucially it **undershoots**
(τ ≤ τ_y always — never over-yields), unlike the sqrt soft-min which overshoots τ_y by
≤~60% mid-transition. So the study now **leads with power-mean**; sqrt is the documented
contrast. Figures: `powermean_yield_curves.png` (undershoots) vs `homotopy_yield_curves.png`
(overshoots).

Head-to-head δ-continuation at (3e25, V10), FMG: both families settle δ≈0.002 (hard-Min);
power-mean does it in fewer total Newton iters (2 vs 9). Because power-mean is smooth *and*
undershoots, a **single fixed small-δ / high-s solve** may converge cold without any
continuation — the probe phase measures that δ_entry (see below).

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

## Phase 1 results (2026-06 session) — COMPLETE

**Solver/infra fixes:** FMG "error 63" was a **SNES monitor leak** (the Stokes
`solve()` re-runs `setFromOptions()` per call → re-registers the option monitor on
the persistent SNES → PETSc `MAXSNESMONITORS` overflow). Fixed with
`snes.monitorCancel()` before each per-solve `setFromOptions()` (latent bug for any
repeated Stokes solve with a monitor). Power-mean `yield_smoother` family added
(overflow-safe harmonic-normalised form; undershoots τ_y).

**The two families both reach Min; sqrt is the chosen homotrope.** sqrt reaches
exact Min at δ=0 (validated, ‖F(x,δ=0)‖ unchanged ×1.0 — the smooth solution *is*
hard-Min because plasticity pins the stress). Power-mean reaches Min only
asymptotically (s→∞) and, crucially, **is a bad homotopy from cold**: its
smoothing *weakens* the load-bearing layer, so a smooth start collapses into a
spurious degenerate basin (η→0, fully-yielded), and a sharp start hits the kink —
no safe cold entry. sqrt's smoothing *strengthens* (overshoot) → safe from any
generous start. So **sqrt is THE homotrope** (it mirrors the physical sequenced
yield-stress reduction, single parameter δ, and steps onto the exact discontinuous
Min at the end); power-mean is paper-only contrast (the above/below blend figures
`figures/{homotopy,blend}_yield_curves.png`).

**Regime map (cold-Newton-δ0 vs single smooth-δ solve vs δ-sequencer), ordered by
driving stress η_bg·V** (size-3, φ=30, FMG; `~/+Simulations/spiegelman_hardcase/
logs/regime_summary.txt`):

| η_bg·V | regime | verdict | mechanism |
|--------|--------|---------|-----------|
| ~2.5e24 | 1e24,2.5 | **COLD-OK** | sub-yield (no plasticity) |
| ~1e26 | 1e25,10 | **SMOOTH-COLD-OK** | cold δ=0 (kink) fails; ONE smooth δ=0.5 solve = hard-Min (×1.0) — no march |
| ~2.5e26 | 1e25,25 | **SEQUENCER-NEEDED** | single δ=0.5 below cold-basin threshold; δ:64→0 march reaches hard-Min |
| ~3e26 | 3e25,10 | **SEQUENCER-NEEDED** | the genuine homotopy win |
| ≳1e27 | 1e26,10/25 | **beyond the homotopy** | see below |

**What the sequencer wins:** the mid-band (η_bg·V ~ 2–3e26) — cold δ=0 dies at the
kink, a single smooth solve is below its cold-basin threshold, but the sequenced
march reaches exact hard-Min. A single smooth-δ cold solve already suffices at the
onset band (~1e26): the smooth (kink-free) Jacobian converges cleanly and its
solution is within tolerance of hard-Min (stress-pinned). The sequencer removes the
need to *guess* a workable δ (it always starts safe at δ=64 and marches).

**Why the extreme corner (≳1e27) is hard — NOT the kink** (key diagnostic finding):
at (1e26,·) the non-dim τ_y is tiny ⇒ f = stress/τ_y ~ 1e4 *everywhere* ⇒ material
deep in the yielded branch, nowhere near the corner. So (1) the δ-homotopy's
corner-smoothing **doesn't engage**; (2) the ~4-order viscosity contrast makes the
Schur ill-conditioned and the **consistent Newton tangent's non-symmetric A breaks
the linear solve** (`DIVERGED_LINEAR_SOLVE` even with an exact MUMPS velocity block;
the viscous pre-solve at the same contrast is machine-zero clean, and **Picard's
symmetric A solves fine but slowly**); (3) **AL penalty fixes the linear divergence
for the direct solver** (failure then moves to the nonlinear line search) **but is
hostile to FMG** (the γ·grad-div augmentation wrecks the MG smoother — FMG+AL ran
47 min on one δ=64 step without converging). So the extreme corner is an
extreme-contrast linear-algebra + tangent-fragility problem, *outside* the
corner-homotopy's mechanism, and the natural Schur fix (AL) doesn't transfer to the
scalable FMG deliverable.

## Phase 2 (proposed, L.M.) — a SECOND homotopy axis: rate-dependent τ_y (damage)

The δ-homotopy regularises the **Min corner** (f≈1). The extreme corner showed the
*other* difficulty axis — the **deep-yielded branch** (f≫1) at extreme contrast —
which corner-smoothing cannot touch. A **strain-rate dependence of the yield
stress** (physically: damage accumulation *rate* ∝ ε̇) regularises *that* axis:

- Write `τ_y = τ_y(ε̇)` so `η_pl = τ_y(ε̇)/(2ε̇)` becomes a **power-law-like viscous
  rheology** — smooth, coercive, well-conditioned — instead of the singular
  `σ_y/(2ε̇)`. **SIGN MATTERS:** the solver-helpful, uniqueness-restoring direction
  is rate-***strengthening*** (`∂τ_y/∂ε̇ > 0`, an overstress term — classic
  viscoplastic/Perzyna regularisation): it adds a **positive-definite** tangent
  contribution → less non-normal operator → cures the consistent-tangent
  linear-solve fragility found above, **stays MG-friendly** (unlike AL), and
  regularises localisation. Power-law Stokes is a standard well-posed problem.
  NOTE: literal **damage is rate-*weakening*** (`∂τ_y/∂ε̇ < 0`) — the
  *destabilising* direction that *promotes* localisation and non-uniqueness, so on
  its own it makes the linear solve harder. The reconciliation is a viscous/rate
  regularisation *of* the damage evolution (delay-damage): net softening, but an
  instantaneous rate-strengthening overstress provides the regularisation. The
  homotopy dials the strengthening coefficient → 0. (Design choice to settle:
  homotope the regularisation (strengthening, clean) vs the damage feedback itself
  (softening, selects a band but fights the solver) — they behave oppositely.)
- **Solution selection:** rate-independent perfect plasticity is genuinely
  *non-unique* (slip-line / shear-band multiplicity). A rate-dependence **selects a
  unique pattern** ("snaps" the solution). The homotopy then dials the
  rate-dependence → 0 while staying in that basin ⇒ recovers a **physically
  damage-rate-selected** solution in the rate-independent limit, not just *a*
  solution.
- **Fits the existing machinery:** the rate-dependence coefficient ξ is a rampable
  `constants[]` atom (like δ), ramped → 0 via the same `SNESSetUpdate` hook — no
  recompile. δ for the corner, ξ for the rate/damage regularisation, used together
  or separately. Open question: interaction of the two axes, and whether ξ alone
  rescues the (1e26,·) corner under FMG.

## Pointers

- Harness: `~/+Simulations/spiegelman_hardcase/` (README.md is the live index).
- Driver knobs (`drivers/convergence.py`): `SOLVER=fmg|direct`, `ETA_BG`, `V_MMYR`,
  `SMOOTHER=sqrt|powermean`, `PENALTY`, `KSP_MAXIT`, `LS_MAXIT` (bt backtrack cap),
  `SWEEP_DELTA` (+`SWEEP_CONSISTENT`/`SWEEP_PICARD`), `STRATEGY` (label substring
  filter), `REGIME=1` (cold-δ0 / cold-smooth-`NEARMIN_DELTA` / sequencer + 4-way
  CLASSIFY; `REGIME_NOCOLD`, `REGIME_COLD0_MAXIT`, `REGIME_COLD_MAXIT`),
  `CONT_MAXIT`/`CONT_MAXIT0` (sequencer per-step early-abort budget), `PROBE_REFINE`.
  Regime sweep + blend figures: `drivers/{regime_map,plot_blend_curves,plot_blend_path}.py`.
- Code: `_combine_yield`, `enable_yield_homotopy`/`_yield_homotopy_step`,
  `solver.consistent_jacobian` (PR #258). Tests: `tests/test_1053_yield_homotopy.py`.
- Solver design: `jacobian-consistent-tangent.md`. Skill: `plasticity-solvers`.
