---
title: "SNES convergence: set snes_atol to the problem scale"
---

# SNES `snes_atol` — guess-independent convergence

**Status:** design proposal, pending sign-off + benchmarking.
**Scope:** UW3 SNES solver wrapper — internal to the `solve()`
path in `cython/petsc_generic_snes_solvers.pyx` (which already
branches on `zero_init_guess`). No user-facing API. Affects *every*
UW3 SNES solve.
**Origin:** the adaptive-mesh / Stokes warm-start divergence
investigation (2026-05). This note is the root-cause writeup +
proposed fix; the mesh-mover work was unrelated — it merely exposed
this.

## Summary

UW3's `tolerance` setter configures `snes_rtol` but **never sets
`snes_atol`**, leaving it at PETSc's default (`~1e-50`). PETSc's
default convergence test then has only one viable criterion: a
**relative tolerance referenced to the residual at the initial
guess**. A warm-started solve whose initial residual is already
small (re-solving a near-solved state — exactly what you *want* to be
cheap) is handed an unreachably tight target and fails
(`DIVERGED_LINE_SEARCH`), while the *same problem* cold-started
converges. The fix is to also set `snes_atol` to the problem's
natural residual scale, so convergence is judged **absolutely
(guess-independent)** — including the desirable "re-solve the
solution ⇒ zero iterations" behaviour.

## Evidence (PETSc 3.25 source)

`SNESConvergedDefault` (`src/snes/interface/snesut.c`):

```c
if (!it) {                              /* iteration 0 = initial guess */
    snes->ttol   = fnorm * snes->rtol;  /* rtol target ∝ ‖F(x0)‖ */
    snes->rnorm0 = fnorm;
}
...
} else if (fnorm < snes->abstol && (it || !snes->forceiteration)) {
    *reason = SNES_CONVERGED_FNORM_ABS;  /* absolute — guess-independent */
} ...
if (it && !*reason) {
    if (fnorm <= snes->ttol) *reason = SNES_CONVERGED_FNORM_RELATIVE;
    else if (snorm < snes->stol * xnorm)
        *reason = SNES_CONVERGED_SNORM_RELATIVE;   /* it>=1 only */
}
```

Key facts, verified in-tree:

1. `rtol` is **defined** relative to the initial-guess residual
   (`ttol = rtol·‖F(x0)‖`, set once at `it==0`). There is **no
   option** to reference it to the problem/RHS scale. PETSc has not
   changed this.
2. The **absolute** path (`fnorm < snes_atol`) is gated by
   `(it || !snes->forceiteration)`, so it is evaluated **even at
   `it==0`**. With `snes_atol` set to the problem scale and
   `snes_force_iteration` off (UW3's default), re-solving an
   already-solved state converges at iteration 0 with **zero Newton
   steps** — the intended behaviour.
3. The step-norm path (`snorm < stol·xnorm`) is gated by `it && ...`
   — it cannot deliver zero-iteration convergence and is pre-empted
   when the line search aborts at the 0→1 transition.

UW3 (`petsc_generic_snes_solvers.pyx`, `tolerance` setter):

```python
self.petsc_options["snes_rtol"] = self._tolerance       # set
self.petsc_options["ksp_rtol"]  = self._tolerance * 1e-1 # set
self.petsc_options["ksp_atol"]  = self._tolerance * 1e-6 # set
#  snes_atol : NEVER set  → PETSc default ~1e-50 → absolute path dead
```

So convergence is decided **solely** by `fnorm ≤ rtol·‖F(x0)‖`.

## Failure mechanism

For a warm-started solve where the carried-forward guess is close to
the solution, `‖F(x0)‖` is small ⇒ `ttol = rtol·‖F(x0)‖` is a tiny
absolute number, often below what the (relative-tolerance) inner KSP
delivers for the Newton correction. The line search cannot achieve
sufficient decrease toward an unreachable target ⇒
`DIVERGED_LINE_SEARCH`. Cold-start (`x0 = 0`) gives
`‖F(x0)‖ ≈ ‖RHS‖` (large) ⇒ a sane `ttol` ⇒ converges. This is
guess-relative, not problem-relative — and it means *improving the
guess makes convergence harder*, the opposite of what a solver should
do.

Observed across the adaptive-convection runs: warm Stokes diverged
repeatedly through violent transients (every step, until the field
calmed), each instance recovering cleanly from a cold restart;
`ksponly`/`basic` line-search "worked" only by bypassing the test;
improving the warm guess (V,P remap) did **not** help — all exactly
as the mechanism predicts.

## Proposed fix

**`snes_atol` is internal to the solver and never user-facing.**
There is no new API, no `tolerance_abs` knob — exposing it would
repeat the mistake this whole investigation argued against (robust
defaults, not fragile expert knobs). The solver derives and applies
it automatically, **per solve, conditioned on `zero_init_guess`**:

```
if not zero_init_guess:                 # WARM start
    F0    = ‖F(x=0)‖   for the CURRENT operator/RHS   # problem scale
    saved = snes_atol
    snes_atol = snes_rtol * F0          # temporary, guess-independent
    <SNES solve>                        # → SNES_CONVERGED_FNORM_ABS
    snes_atol = saved                   # restore
else:                                   # COLD start
    <SNES solve>                        # rtol·‖F(x0=0)‖ already = scale
```

* **Warm solve:** the guess-relative `ttol = rtol·‖F(x_warm)‖` is
  unreachable; the solver instead **computes the problem-scale
  target residual and temporarily sets `snes_atol` to it for that
  solve only**, then restores. Convergence then takes the absolute,
  guess-independent path (`SNES_CONVERGED_FNORM_ABS`, evaluated even
  at `it==0`), so re-solving an already-solved state converges in
  **zero Newton iterations** — the intended behaviour.
* **Cold solve:** untouched. `‖F(x0=0)‖` *is* the problem scale, so
  the existing `rtol` path already targets the right residual; the
  cold solve is also the natural place to (re)source the scale.

**Scale currency (design decision).** The target must be the
**current** problem scale, *recomputed each warm solve* — one extra
function evaluation at `x=0`, negligible against the solve — **not**
a frozen startup `‖F₀‖`. The RHS scale (e.g. `‖buoyancy‖`) varies
substantially through a transient; a frozen scale would be stale
exactly where warm-start divergence bites. `F(0)` remains a valid
scale for nonlinear rheology, so this is a convergence-*criterion*
fix independent of linearity. (The `--stokes-snes-atol-auto`
confirmation harness uses a *frozen* startup scale — a valid proof
of the mechanism, but a simplification; production recomputes.)

## Impact & risk

This changes the convergence criterion for **every UW3 SNES solve**
(Stokes, scalar Poisson, projections, advection–diffusion; the
mesh-mover's `ksponly` sub-solves are unaffected — no Newton test).
Per the repository rule *solver stability is paramount — no changes
without benchmarking*:

* **Cold-started** solves: behaviour ≈ unchanged
  (`atol` not applied; `rtol·‖F0_cold‖` already the accuracy floor).
* **Warm-started** solves: spurious divergence *fixed*;
  "re-solve the solution ⇒ 0 iterations" now works; accuracy is the
  same `rtol·‖F(0)‖` a working cold solve targets — no under-solving.
* The `snes_atol` mutation is **scoped to one solve and restored**,
  so it cannot leak across solvers/steps or interact with a user's
  own `petsc_options`.
* Benchmark the standard suite (Stokes/Poisson convergence-order,
  the `tier_a` set) before merge — it must show unchanged accuracy
  and order, only removed spurious warm divergences.

Recommended landing: internal to the `solve()` path in
`petsc_generic_snes_solvers.pyx` (which already branches on
`zero_init_guess`); no API surface; benchmark suite green; one line
in the solver guide noting the automatic behaviour.

## Validation

* Root cause verified against PETSc 3.25 source (above) and the UW3
  `tolerance` setter.
* Confirmation experiment (`scripts/adaptive_saturation.py
  --model a16r15a --stokes-snes-atol-auto`, equidist R=1.5, warm,
  V,P-remap on, default `newtonls`+`bt`, **no cold-recover**;
  `‖F0‖=24.75 ⇒ snes_atol=2.47e-4`): **full settled run** —
  warm `STOKES DIVERGED` **24 → 31** (i.e. *no net benefit*, if
  anything slightly worse, vs the identical run without the
  absolute criterion). *(An earlier step-70 partial read showed a
  spurious 24→9 — corrected here: it was a mid-trajectory snapshot
  before the later transient windows, not the result.)*

  This is *consistent with* the mechanism, and clarifies the scope:

  * The absolute path (`SNES_CONVERGED_FNORM_ABS` at `it==0`) only
    fires when `‖F(x_warm)‖ < snes_atol`. In a **violent-transient
    -dominated** run the warm-guess residual is almost always
    ≫ `atol` (the field changed substantially per step), so the
    absolute path essentially never triggers; SNES proceeds to the
    line search, which aborts on the inexact inner Newton step
    *before any convergence test is consulted*. The
    near-converged-guess class this fix targets is **nearly absent**
    in this benchmark, so `snes_atol` provides no net benefit here
    and merely perturbs which steps fail (net +7).
  * Where it *does* help — and the reason it should still land — is
    the regime it is *for*: steady-state continuation, restarts,
    lightly-evolving problems, any re-solve of a near-solved state.
    There `‖F(x_warm)‖ < atol` genuinely holds and SNES converges
    in **zero Newton iterations** instead of failing on an
    unreachable guess-relative `ttol`. That is a real, general UW3
    gap (PETSc-source-verified), independent of this benchmark.
    This experiment does **not** exhibit that regime, so it neither
    confirms nor refutes the fix's value there — it only shows the
    fix does not help violent transients (as the mechanism
    predicts).

**Conclusion:** `snes_atol` is a correct, general improvement for
the near-converged-guess regime (justified by the PETSc-source
diagnosis, *not* demonstrated by this transient-dominated run —
which shows no benefit, as expected). It is **not** the cure for
warm-start through a violent transient. That cure is a separate,
*demonstrated* result: an accurate inner Newton solve (`a16r15d`,
MUMPS-LU inner solve, warm, default `bt`, no recover/atol →
**24 → 0** warm `STOKES DIVERGED`) — the inner KSP must deliver an
acceptable step on the graded / stiff-Robin operator, generalised
as a tight inner tolerance / strong PC / direct where affordable
(not "always direct"). Cold-restart-on-divergence is the
operational safety net. The pieces are independent and
complementary; this note covers only the `snes_atol` piece — see
the inner-solve result for the transient cure.
