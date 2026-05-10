# Integrator zoo for free-surface kinematic update — supplementary characterisation

> Supplementary material to *A semi-Lagrangian kinematic update and an
> amplitude-invariant relaxation CFL for free-surface viscous flow*.
> The main paper recommends RK4-SL with relaxation-CFL safety factor
> $c=1$ as the production scheme. This supplement maps the operating
> envelope of the alternatives so readers can match a scheme to their
> own constraints (problem stiffness, time-budget, A-stability needs,
> implementation complexity). No single winner is declared — different
> use cases will choose differently from the table.

## Two orthogonal design choices

The schemes characterised here combine two orthogonal choices that
can be made independently. The cross-product gives the rows of the
characterisation matrix below.

### Choice A: time integrator

How the surface ODE $\dot h = u_n^{\rm eff}$ is advanced in time.

- **FE / RK2 / RK4** — explicit one-step schemes. RK4 has $4$ stages
  per step; RK2 has $2$; FE has $1$.
- **AB2** — explicit multistep (history-locked). One Stokes solve per
  step + one cached past rate.
- **BDF2** — implicit multistep. Each step is an implicit equation in
  $h^{n+1}$, solved by damped Picard iteration; each Picard iteration
  is one Stokes solve. 5–12 iterations per step at $c \in [0.5, 4]$.
- **curvS** — saturated 1st-order ETD with curvature-derived $\gamma$.
  One Stokes solve per step + analytic prefactor.

### Choice B: mesh handling at intermediate stages (suffix `_sl` vs `_load_sl`)

For multi-stage / multi-iteration time integrators (RK2, RK4, BDF2),
each "intermediate solve" can be done in two ways:

- **`_sl` (deform-each-stage):** restore mesh to start-of-step;
  deform to predicted intermediate state via the Poisson diffuser;
  solve Stokes at that mesh; repeat for next stage. The full
  kinematic dance — every intermediate state is a real mesh.
- **`_load_sl` (surface-load mesh-freezing):** keep mesh at
  start-of-step throughout the intermediate stages; apply the
  predicted $\Delta h$ as a Neumann surface traction $-\rho g \,\Delta h\, \hat r$
  on the upper boundary. Stokes solves at the un-deformed mesh, with
  the traction approximating the deformation effect on the velocity
  field (linearised free-surface response). Mesh deforms exactly
  **once** per step using the assembled increment, via the same
  Poisson diffuser path used by `_sl`.

  The two halves of this trick are coupled — *not deforming* the
  mesh forces the *load compensation*, otherwise all RK stages see
  the same un-deformed mesh and the multi-stage scheme collapses
  to FE-SL. The compensation has a structural validity condition
  ($\Delta h \ll h$ per step); it is **not a free lunch but a
  regime-restricted optimisation**. See the relaxation-benchmark
  results below for the failure threshold.

The load trick saves: (i) the diffuser Poisson solve at every
intermediate stage; (ii) the mesh `_deform_mesh` + restore operations;
(iii) when swarms are added, every intermediate-stage swarm migration.
On this benchmark, $\sim 60\%$ wall-time savings for RK4 and $\sim 65\%$
for BDF2 (see table). The trade-off: trajectory is the linearised-load
approximation rather than exact mesh-deformation Stokes response.
Trajectory differences from the `_sl` versions are $\sim 10^{-4}$ in
$h_p^{\rm final}$ — within the discretisation noise floor.

For **single-stage** schemes (FE-SL, AB2-SL, curvS) there are no
intermediate stages — they have only one Stokes solve per step at
the actual current mesh — and so they have no `_load` variant.

### Choice (C): kinematic update (the Phase III result, fixed)

All schemes named `*_sl` use the semi-Lagrangian-horizontal
discretisation of the kinematic boundary condition (main paper).
Without this, the radial-only kinematic update has structural
$\sim 2\%$ volume drift independent of resolution. The pre-Phase-III
schemes `fe`, `fe_dtf*` are listed below as the baseline.

## Setting

All schemes share the Phase III structure:

- Continent isostasy benchmark (rock annulus $0.5 \le r \le 1$, free
  surface at $r_o = 1$, no-slip at $r_i = 0.5$, body force
  $-(1 - \beta B)\hat r$ with block buoyancy $\beta = 0.2$ on a 0.4-rad
  half-sector inside $r \ge 0.7$).
- Mesh: structured polar V2/P1 (Q2-Q1 Taylor–Hood), 20 cells radially.
- Run to $T = 540$ (forced equilibrium plateau at $h_{\rm pole} \approx 0.038$).
- Δt selected by $\min(\text{bulk-CFL}\cdot \text{dtf},\, c/\gamma_{\rm used})$
  with $\gamma_{\rm used}$ the monotone L²-history estimator from the
  main paper.

## Diagnostics

For each (scheme, $c$) combination we record:

- **Cost** — reported on two axes:
  - *Total Stokes solves* — number of `stokes.solve()` calls. Counts
    each Picard iteration of an implicit step, each RK stage, and
    the single rate evaluation of explicit single-stage schemes.
  - *Wall time* — total seconds on a single 2.5 GHz core.
  These two axes do not perfectly agree because not every Stokes
  solve costs the same: Picard iterations 2..N within a BDF2 step
  inherit a tight warm-start from the previous iteration (mesh
  barely changes), so the SNES+KSP loop converges in fewer internal
  iterations than a cold-start solve. On this benchmark the empirical
  per-solve cost is 3.7 s for BDF2-SL and AB2-SL, 4.4 s for
  RK2-SL and RK4-SL — the implicit-scheme advantage is ~15%. Wall
  time is the more honest cost metric; total solves is a proxy that
  is convenient because it can be compared across hardware.
- **Volume drift** — $|\Delta A / A_0|$ at $t = 540$, computed on the
  curved-element representation of the deformed mesh.
- **Pole jitter** — RMS of $h_p(t_n) - \tilde h_p(t_n)$ where
  $\tilde h_p$ is the linear interpolant through neighbouring samples.
  Detects step-to-step period-2 oscillation that volume drift alone
  can miss. Reported relative to the equilibrium $h_p \approx 0.038$.

## Characterisation table

"Total solves" counts every call to `stokes.solve()`: stages-per-step
for explicit schemes, summed Picard iteration count for BDF2.

| scheme | $c$ or dtf | $\langle\Delta t\rangle$ | $n_{\rm steps}$ | total solves | $h_p^{\rm final}$ | $|\Delta A / A_0|$ | jitter / $h_{\rm eq}$ | wall (s) | regime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FE-SL  | dtf=0.05 | 1.55  | 348 | 348  | +0.0383 | 0.024%   | 0.04%  | 1106 | 1st-order ref |
| RK2-SL | 0.25     | 2.78  | 195 | 390  | +0.0383 | 0.020%   | 0.14%  | 1525 | 2nd-order, deeply safe |
| RK2-SL | 0.5      | 5.54  | 98  | 196  | +0.0383 | 0.016%   | 0.47%  | 836  | 2nd-order, safe |
| RK2-SL | 1.0      | 11.0  | 50  | 100  | +0.0382 | 0.001%   | 1.06%  | 438  | 2nd-order, at margin |
| RK2-SL | 2.0      | 18.3  | 30  | 60   | +0.0262 | 0.79%    | 2.05%  | 251  | linear-stability bound exceeded |
| RK4-SL | 0.25     | 4.10  | 132 | 528  | +0.0383 | 0.023%   | 0.24%  | 2026 | 4th-order, deeply safe |
| RK4-SL | 0.5      | 8.14  | 67  | 268  | +0.0383 | 0.018%   | 0.78%  | 1113 | 4th-order, safe |
| RK4-SL | 1.0      | 16.05 | 34  | 136  | +0.0382 | **0.002%** | 1.95% | 596 | **production reference** |
| RK4-SL | 2.0      | 25.06 | 22  | 88   | +0.0197 | **2.20%**| 15.6%  | 393  | linear-stability bound exceeded |
| AB2-SL | 0.25     | 7.19  | 76  | 76   | +0.0385 | 0.091%   | 7.3%   | 280  | **ruled out** — early-step trajectory artefacts |
| AB2-SL | 0.5      | 8.81  | 62  | 62   | +0.0390 | 0.21%    | 12.2%  | 247  | **ruled out** — borderline period-2 |
| AB2-SL | 1.0      | 11.7  | 47  | 47   | +0.0339 | 1.72%    | 52.1%  | 193  | **ruled out** — full period-2 instability |
| BDF2-SL| 0.5      | 10.48 | 52  | 286  | +0.0383 | 0.029%   | 1.41%  | 1084 | A-stable, low truncation |
| BDF2-SL| 1.0      | 20.57 | 27  | 169  | +0.0385 | 0.061%   | 2.29%  | 644  | A-stable, mid Δt |
| BDF2-SL| 2.0      | 39.67 | 14  | 146  | +0.0388 | 0.123%   | 3.76%  | 544  | A-stable, large Δt |
| BDF2-SL| 4.0      | 57.93 | 10  | 120  | +0.0390 | 0.177%   | 6.33%  | 458  | A-stable, Picard near-cap |
| BDF2-load-SL | 0.5  | 8.26  | 66  | 351  | +0.0385 | 0.068%   | 1.05%  | 397  | implicit + load |
| BDF2-load-SL | 1.0  | 16.28 | 34  | 207  | +0.0386 | 0.103%   | 1.73%  | 228  | implicit + load (Pareto-equiv to RK4-load) |
| BDF2-load-SL | 2.0  | 31.65 | 18  | 177  | +0.0389 | 0.174%   | 2.84%  | 165  | implicit + load |
| BDF2-load-SL | 4.0  | 56.21 | 10  | 120  | +0.0393 | 0.246%   | 6.25%  | 118  | implicit + load (cheapest BDF) |
| RK2-load-SL | 0.5 | 4.35 | 125 | 250  | +0.0383 | 0.026%   | 0.33%  | 552  | linearised surface load |
| RK2-load-SL | 1.0 | 8.63 | 63  | 126  | +0.0383 | 0.022%   | 0.82%  | 283  | **load Pareto-favoured** |
| RK2-load-SL | 2.0 | 16.88| 33  | 66   | +0.0328 | 0.0012%  | 1.84%  | 153  | linear-stab cliff (gentle) |
| RK4-load-SL | 0.5 | 7.03 | 77  | 308  | +0.0383 | 0.028%   | 0.62%  | 423  | linearised surface load |
| RK4-load-SL | 1.0 |13.87 | 39  | 156  | +0.0383 | 0.023%   | 1.54%  | **240** | **load Pareto-favoured** |
| RK4-load-SL | 2.0 |23.53 | 23  | 92   | +0.0268 | 0.014%   | 6.39%  | 144  | linear-stab cliff (gentle) |
| curvS  | dtf=1.0  | 30.7  | 18  | 18   | +0.0393 | 0.06%    | 4.30%  | 59   | saturation prefactor, 1 solve/step |
| curvS  | dtf=0.5  | 15.8  | 35  | 35   | +0.0378 | 0.18%    | 1.88%  | 146  | saturation prefactor |
| curvS  | dtf=0.25 | 8.04  | 68  | 68   | +0.0373 | 0.26%    | 0.68%  | 265  | saturation prefactor, smaller Δt |

### Reference runs (high-fidelity, no self-comparison)

To avoid using a same-scheme-family reference to evaluate any scheme,
we run two *independent* reference cases at higher spatial resolution:

| reference | scheme | res | $\langle\Delta t\rangle$ | $n_{\rm steps}$ | total solves | $h_p^{\rm final}$ | $|\Delta A / A_0|$ | wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| res=20 RK4 self-ref | RK4-SL c=0.1 | 20 | 1.65 | 328 | 1312 | +0.0383 | 0.026% | 4944 |
| res=20 FE cross-ref | FE-SL dtf=0.05 | 20 | 1.55 | 348 | 348 | +0.0383 | 0.024% | 1106 |
| **res=40 FE cross-ref** | FE-SL dtf=0.05 | 40 | 0.74 | 733 | 733 | +0.0385 | 0.028% | 4827 |
| res=40 RK4 self-ref | RK4-SL c=0.1 | 40 | 0.65 | 826 | 3304 | +0.0385 | 0.030% | 23974 |
| **res=80 FE no-overlap ref** | FE-SL dtf=0.5 | 80 | 3.63 | 150 | 150 | +0.0384 | 0.027% | ~2820 |

Five reference runs converge on the same answer:

- res=20, RK4-SL c=0.1: $h_p = 0.0383$, $|\Delta A| = 0.026\%$
- res=20, FE-SL dtf=0.05: $h_p = 0.0383$, $|\Delta A| = 0.024\%$
- **res=40, FE-SL dtf=0.05**: $h_p = 0.0385$, $|\Delta A| = 0.028\%$
- **res=40, RK4-SL c=0.1**: $h_p = 0.0385$, $|\Delta A| = 0.030\%$
- **res=80, FE-SL dtf=0.5**: $h_p = 0.0384$, $|\Delta A| = 0.027\%$

Across **two scheme families** (RK4 vs FE) and **three resolutions**
(res=20, 40, 80), $h_p^{\rm final}$ varies by at most $2 \times 10^{-4}$
and $|\Delta A|$ stays within $[0.024\%, 0.030\%]$. The trajectory
shape is set by the SL kinematic update — not by the time integrator,
not by the spatial resolution above res=20, not by Δt below the
relax-CFL margin. The $|\Delta A| \approx 0.025\%$ floor is the
discretisation+solver-noise limit on this V2/P1 polar mesh family.

The res=40 references push the noise floor lower by quadrupling the
DOFs. The res=80 reference (FE-SL only) provides a no-resolution-overlap
comparator: routine production runs at res=20 and the secondary res=40
references both share an *integer-multiple* DOF relation with each
other, but res=80 has 16× the DOFs of res=20 and quadruples res=40 —
its discretisation noise floor is independent of either. They are the
authoritative comparison targets for the production-c sweep results
above. **All time-sweep schemes are evaluated against these references,
not against any RK4 result at the same resolution as themselves**,
to avoid the self-comparison pitfall.

### Caveat on the RK4 c=1 / RK2 c=1 numbers

The headline production numbers are:

- RK4-SL c=1: $|\Delta A| = 0.002\%$
- RK2-SL c=1: $|\Delta A| = 0.0015\%$

Both are *below* the discretisation noise floor of 0.025% (set by
the references above). This is **luck** — accidental sign cancellation
in the cumulative drift across the run, not a property of the
scheme. The schemes' true accuracy on this benchmark is no better
than 0.025%; the production runs happen to land favourably in the
noise. Readers re-running this benchmark at different resolution,
solver tolerance, or with slightly different $\beta$ should expect
$|\Delta A| \in [0.005\%, 0.05\%]$ range, not a reproduction of the
0.002% number specifically.

## Per-scheme commentary

### FE-SL — first-order baseline

- **Cost class:** 1 Stokes solve per step. Cheapest per-step among
  consistent schemes.
- **Stability:** $\gamma \Delta t < 2$. Easy to violate when the
  bulk CFL gives large $\Delta t$.
- **Accuracy:** 1st order, but for relaxation-dominated problems the
  per-step error stays modest. With $\text{dtf} = 0.05$ on this
  benchmark, $|\Delta A| = 0.024\%$ — comparable to RK4-SL with the
  large-$\Delta t$ relaxation CFL.
- **Use it when:** time stepping is constrained externally to
  $\Delta t \ll 1/\gamma$ (e.g. by another physics module), and you
  want minimum implementation overhead.

### RK2-SL, RK4-SL — explicit higher-order with relaxation CFL

- **Cost class:** 2 / 4 Stokes solves per step.
- **Stability:** $\gamma \Delta t < 2$ / 2.78. With the relaxation-CFL
  monotone $\gamma$ history, $c$ is the safety factor on this bound.
- **Accuracy:** clean trajectory at $c \le 1$. RK4-SL $c=1$ is the
  paper's production scheme.
- **Use it when:** Stokes solves are cheap relative to other costs,
  and you want predictable behaviour with a clear stability margin.

### AB2-SL — explicit multistep — **ruled out for this problem class**

- **Cost class:** 1 Stokes solve per step + 1 surface-history term.
- **Stability:** linear bound $\gamma \Delta t < 1$, but in practice
  the relevant constraint near a *forced* equilibrium is tighter.
  At $c = 0.5$ ($\gamma \Delta t = 0.5$, well inside the linear bound)
  the scheme oscillates with period 2 around the equilibrium plateau:
  $h_p$ alternates between $\sim h_{\rm eq} \pm 0.005$ each step.
  At $c = 1$ this becomes catastrophic
  (jitter = 52% of $h_{\rm eq}$, trajectory loses fidelity).
- **Trajectory pathology even when stable:** even at $c = 0.25$
  (deep in the safe regime), the first 5 steps show large
  oscillations — overshoot to $h_p = 0.043$ at step 2, undershoot to
  0.029 at step 3, then a 24%-overshoot spike to $0.051$ at step 4
  before settling. The mechanism: each AB2 step caches `u_n_eff` whose
  baked-in $\Delta t$ depends on the previous step's trace-back distance;
  when $\Delta t$ changes between steps, the predictor extrapolates from
  a stale rate. The trajectory does converge to the correct equilibrium
  shape after $t \approx 100$, but the early-time history is wrong.
- **Why this rules it out:** for problems where the *trajectory*
  matters (e.g. coupled to other physics, or where the early-time
  loading history is part of the answer), AB2-SL gives the right
  asymptotic answer for the wrong reasons. The end state is correct;
  the path is not.
- **Variable-step robustness clamp:** an internal $\omega = \Delta t^n / \Delta t^{n-1}$
  cap at 1.5 forces a fall-back to FE-SL when $\Delta t$ grows too
  fast. Without it, the predictor coefficient $1 + \omega/2$ blows up
  during the early-step Δt-tuning regime. The clamp prevents
  divergence but does not prevent the trajectory artefacts.
- **Retained in the supplementary** as cautionary characterisation.
  Adams–Bashforth-style multistep methods are sometimes attractive
  for their per-step economy; we report this so practitioners
  considering AB2 in this setting can see the failure mode in advance.

### BDF2-SL — implicit, A-stable

- **Cost class:** 5–12 Picard iterations $\times$ Stokes solve per
  step (averages: 5.5 at $c=0.5$, 6.3 at $c=1$, 10.4 at $c=2$, 12.0 at $c=4$).
  Each iteration restores the start-of-step mesh, deforms by the
  predicted increment, and re-solves Stokes. Implementation: damped
  Picard with $\alpha = 1 / (1 + \max(\gamma_{\rm used} \Delta t, 1))$
  keeps the iteration well-behaved through bootstrap and into the
  large-$\gamma\Delta t$ regime where the linear Picard map would diverge.
- **Stability:** A-stable. No $\Delta t$ cap from linear stability —
  the binding constraint is accuracy. Trajectory remains bit-equal
  to RK4-SL $c=1$ at the halfway snapshot regardless of $c$ for
  $c \le 4$ tested here.
- **Accuracy:** 2nd-order, with clean monotonic Δt² truncation
  scaling visible across $c = 0.5..4$ — see `fig_convergence_time.png`.
  Volume drift grows from $0.029\%$ at $c=0.5$ to $0.177\%$ at $c=4$.
  Compare RK2-SL/RK4-SL on the same plot, where the $c = 2$ point is
  the catastrophic-failure jump (>1% drift). BDF2's A-stability buys
  *graceful degradation* under Δt overshoot, not improved per-step
  accuracy.
- **Trajectory shape:** slight overshoot of $\sim 1\%$ above
  $h_{\rm eq}$ at large $c$ (e.g. $h_p^{\rm final} = 0.0390$ at $c=4$
  vs $h_{\rm eq} \approx 0.0383$). This is the symmetric counterpart
  of the RK undershoot at the first step (RK4-SL c=1 reaches
  $h_p^{\rm final} = 0.0382$). For applications sensitive to early-time
  trajectory shape, the BDF overshoot is qualitatively the same trade-off
  as the RK undershoot; both are bounded and well-behaved.
- **Iteration cost grows with $c$:** the $\alpha$ damping formula
  is Newton-equivalent only when the linearisation matches the
  local Jacobian. At $\gamma \Delta t \sim 1$ contraction is $\sim 0.3$
  (6 iterations to tol $10^{-4}$); at $\gamma \Delta t \sim 4$
  contraction degrades to $\sim 0.7$ (12 iterations, sometimes
  hitting the iteration cap). The big-$\Delta t$ savings are
  partially absorbed by Picard cost growth — but only partially:
  total solves drop from 286 ($c=0.5$) to 120 ($c=4$).
- **Use it when:** A-stability is non-negotiable (extreme stiffness,
  e.g. $\gamma$ varying by orders of magnitude across the domain),
  or when you want graceful failure rather than the explicit-scheme
  cliff at $c = 2$.

#### BDF2-load-SL (the load variant of BDF2)

Each Picard iteration uses the predicted $h^{n+1,(k)}$ as a
surface load instead of deforming the mesh; mesh deforms once per
step at the end with the converged $h^{n+1}$.

- **Wall time vs BDF2-SL:** **~3× speedup** (per-Picard cost drops from
  3.8 s to 1.1 s — even cheaper per solve than RK4-load, because the
  start-of-step mesh is *identical* across all Picard iterations,
  giving SNES warm-starts that are essentially perfect refinements).
- **Trade-off:** the load approximation introduces extra drift.
  At $c = 1$, BDF2-load gives $|\Delta A| = 0.103\%$ vs BDF2-SL's
  $0.061\%$ — about $1.7\times$ more drift for the same step count.
  At $c = 4$ the gap widens to $0.25\%$ vs $0.18\%$.
- **vs RK4-load:** at comparable wall time (228 s vs 240 s), BDF2-load
  c=1 has $\sim 5\times$ more volume drift (0.103% vs 0.023%). The
  load approximation effectively makes everything a 2nd-order scheme
  in time (the linearised free-surface response is the bottleneck);
  RK4-load is more efficient at extracting that 2nd-order accuracy
  from a fixed solve budget. **BDF2-load doesn't dominate RK4-load**
  on this benchmark.
- **Where BDF2-load wins:** at the cheapest end of the cost spectrum.
  BDF2-load c=4 reaches t=540 in 118 s wall — the cheapest stable
  scheme in the entire zoo. If $|\Delta A| \approx 0.25\%$ is
  acceptable, this is the choice.

### RK2-load-SL, RK4-load-SL — RK with linearised surface-load estimator

- **Cost class:** same Stokes-solve count per step as RK*-SL, BUT
  every estimator-stage solve happens at the **same start-of-step
  mesh** — only the surface load (a Neumann BC) changes between
  stages. No diffuser Poisson solve, no `mesh._deform_mesh` per
  stage, no swarm migration (when swarms are added). Final mesh
  deformation goes through the standard diffuser path **once** per
  step.
- **Per-solve cost:** ~3× lower than RK*-SL on this benchmark
  (1.5–1.8 s vs 4.4 s) because the iterate-to-iterate mesh
  perturbation is *zero* — KSP/SNES warm-starts are nearly perfect.
- **Stability margin:** identical to RK*-SL — the load
  approximation doesn't change the linear stability analysis of
  the underlying RK time integrator. $c=2$ exits the L-stability
  region for both RK2 and RK4, same as their SL counterparts.
- **Failure mode at $c \ge 2$:** **gentler than RK*-SL.** The mesh
  only deforms once per step using the RK-weighted assembled
  increment, so per-stage instability averages out before mesh
  corruption. RK4-SL c=2 catastrophically loses 2.2% volume; RK4-load
  c=2 only drifts 0.014% but still has wrong $h_p^{\rm final} = 0.027$.
  Volume drift is no longer the binding stability indicator —
  pole jitter and trajectory shape are. Note the symmetric
  observation: at the production-c, load schemes' volume drift
  is *worse* (noise-floor-limited at ~0.025%, no lucky cancellation),
  but their failure mode is *milder*.
- **Trajectory at $c \le 1$:** matches RK*-SL to $\sim 10^{-4}$ in
  $h_p^{\rm final}$. The linearised surface load is faithful to the
  exact mesh-deformation Stokes response when $\Delta h$ is small
  per step — exactly the regime the relaxation-CFL keeps us in.
- **Pareto position at $c = 1$:** **clear win on wall time vs accuracy**.
  RK4-load c=1 reaches t=540 in 240 s wall with $|\Delta A| = 0.023\%$
  (right at the noise floor); RK4-SL c=1 takes 596 s for $|\Delta A| = 0.002\%$
  (lucky sub-noise-floor). When sub-noise-floor accuracy is not a
  meaningful target — which is true for any practitioner whose
  benchmark resolution is $\le$ res=20 — RK4-load c=1 is preferable.
- **Use it when:** mesh deformation cost is significant relative to
  Stokes solve cost (especially when swarms are involved),
  trajectory accuracy at the noise floor is acceptable, and the
  graceful degradation at the stability margin is desirable.

### curvS — saturated kinematic-ETD with curvature-derived $\gamma$

- **Cost class:** 1 Stokes solve per step.
- **Stability:** L-stable via the saturation prefactor
  $(1-\alpha)/\gamma$.
- **Accuracy:** systematic shape bias from the analytic Cathles
  $\gamma$. At small $\Delta t$ the bias *grows* (cumulative effect)
  rather than shrinking — see the main paper's Discussion section
  on saturation-scheme convergence pathology.
- **Use it when:** a pre-existing FSSA pipeline is already in place
  and a more accurate replacement is not yet available. For new
  implementations, RK4-SL or BDF2-SL are preferred.

## Relation to FSSA-stabilised methods (and beyond)

The schemes in this zoo deliberately avoid the saturation prefactor
that defines the FSSA family (Crameri & Tackley 2012 and successors).
The classical FSSA recipe is:

1. Pick a fixed $\gamma$ (often the Cathles dispersion estimate
   $\rho g / 2\eta k$, or a curvature-derived fit).
2. Form the saturated kinematic-ETD prefactor
   $(1 - e^{-\gamma\Delta t})/\gamma$.
3. Apply this prefactor to a single explicit normal-velocity sample
   per step. The prefactor saturates to $1/\gamma$ as $\Delta t \to \infty$,
   giving "pseudo-L-stable" behaviour at large $\Delta t$.

This is robust against blow-up but introduces a **systematic shape
bias** when $\gamma$ is mis-estimated, and produces the Δt-dependent
convergence pathology shown in our `curvS` rows: $|\Delta A|$
*grows* as $\Delta t$ shrinks, the opposite of what truncation-driven
convergence would predict (see `fig_convergence_time.png`).

Sticky-air variants (Lu et al. 2025, Quinquis et al., earlier
Kaus formulations) avoid the prefactor by introducing a thin
low-viscosity layer above the rock so the upper boundary is
*not* the geometric free surface, then track an internal interface
either with markers (Eulerian) or via ALE deformation. These
methods sidestep the FSSA saturation question but introduce a
different family of numerical issues: marker noise, tracer
re-seeding, sharp interface viscosity contrasts, and (for ALE
variants) mesh-quality decay near the deforming interface.

This zoo represents a third path:

- **True free surface.** No sticky-air layer; the geometric upper
  boundary $r = r_o$ deforms physically, with natural-stress
  boundary conditions.
- **No saturation prefactor.** All schemes here either respect a
  linear-stability bound through the relaxation CFL ($c \le 1$ for
  FE, RK2, RK4) or are A-/L-stable with truncation-only Δt
  bound (BDF2). $\gamma$ is *measured* from the current surface
  state, not assumed from constitutive constants.
- **Honest stability cliffs.** Schemes that violate their stability
  bound (RK4-SL c=2) actually fail — visible in volume drift
  *and* pole jitter — rather than being silently kept stable by
  an over-saturating prefactor.
- **Surface-load variants generalise the FSSA "surface traction"
  trick** (the $-\rho g \Delta h \hat r$ Neumann BC on the upper
  boundary) but only in the **estimator** stages of multi-stage
  schemes. The final mesh deformation goes through the same
  Poisson-diffuser path as the un-stabilised `_sl` schemes, so the
  end-of-step state is geometrically correct. We do not stabilise
  the integrator; we just avoid mesh deformation between
  intermediate solves.

For practitioners coming from FSSA: think of this as
"the FSSA surface-load idea, applied per-stage in an RK or
Picard loop, *without* the saturation prefactor". The trade-off
is honest: cheaper-per-step than non-stabilised RK-SL,
slightly less accurate (because the linearised free-surface
response in the estimator isn't exact), and limited to the
RK / BDF stability margin (no FSSA-style protection at large
$\Delta t$). For applications where a recoverable failure
mode and an honest noise floor are preferred to silent
stabilisation, the load variants here are competitive on cost
with FSSA.

### Three ways to handle the surface evolution

Reading the zoo through the FSSA lens reveals a 3-way taxonomy:

| approach | how it handles deformation between solves | validity regime |
|---|---|---|
| **exact mesh `_sl`** | actually moves the mesh, samples $u_n$ at the deformed geometry | unconditionally faithful (mesh quality permitting) |
| **linearised load `_load_sl`** | freezes mesh, applies linear-in-$\Delta h$ Neumann traction $-\rho g \Delta h \hat r$ | $\Delta h \ll h$ per step only |
| **FSSA / curvS / ETD-with-curv-γ** | freezes mesh, applies analytic exponential prefactor $(1-e^{-\gamma\Delta t})/\gamma$ | requires accurate $\gamma$; biased if $\gamma$ is wrong |

The load short-cut and the FSSA prefactor address the same goal —
avoid moving the mesh between solves — by different approximations.
FSSA uses the *analytic exponential*, which is correct for a single
decaying mode but biased when $\gamma$ is mis-estimated. The load
short-cut uses the *linearisation* of the same response, which is
strictly worse for an exponentially-decaying process: linearising an
exponential at $\gamma\Delta t \sim 1$ is exactly the failure mode
ETD was invented to fix. The load variants therefore inherit FSSA's
"freeze-the-mesh" architectural saving but lose ETD's analytic
treatment of the exponential, and they break in the regime where
ETD literature already shows linearisation fails.

**The genuine "beyond FSSA" schemes here are the exact-mesh `_sl`
family.** They neither linearise nor pre-factor — they sample $u_n$
at the actual deformed geometry and rely on the relaxation CFL to
keep $\Delta t$ within the integrator's linear-stability bound. The
relaxation benchmark verifies that RK4-SL c=1 recovers Cathles
$\gamma_k$ to within 3%, with no constitutive constants in the
algorithm. The load variants are a separate point in the design
space, useful when $\Delta h \ll h$ holds (forced equilibria near
steady state, e.g. continent isostasy at the plateau) but not a
general replacement.

## Selection guide

For a relaxation-dominated free-surface problem with the
relaxation-CFL machinery available:

- **Default — accuracy is paramount and Stokes is affordable:** RK4-SL $c=1$
  (136 solves, $|\Delta A| = 0.002\%$) or RK2-SL $c=1$ (100 solves,
  $|\Delta A| = 0.001\%$). On this benchmark the two are
  Pareto-equivalent at the relax-CFL margin: RK4's higher per-step
  truncation order is below the per-step measurement-noise floor,
  so RK2 captures the same trajectory quality more cheaply. RK4
  retains an advantage on problems with finer temporal structure
  (sharp loading, rapid transients) where higher per-step truncation
  order matters. The cliff at $c \ge 2$ argues for not pushing the
  safety factor above 1 on either scheme.
  *Note: the sub-noise-floor $|\Delta A|$ is luck (sign cancellation),
  not a property of the scheme — see "Caveat" below.*

- **Wall-time-sensitive *AND* problem stays near forced equilibrium
  (Δh per step ≪ h):** RK4-load-SL $c=1$ (156 solves, 240 s wall,
  $|\Delta A| = 0.023\%$) or RK2-load-SL $c=1$ (126 solves, 283 s wall,
  $|\Delta A| = 0.022\%$). Linearised-surface-load variants drop wall
  time ~60% by skipping per-stage diffuser Poisson and mesh deform.
  Trajectory matches RK-SL to $10^{-4}$ in $h_p^{\rm final}$; volume
  drift sits at the discretisation noise floor.
  **Important caveat:** the linearisation is faithful only when
  $\Delta h \ll h$ per step. For *pure-relaxation* regimes where
  $\Delta h \approx h$ at $\gamma\Delta t \sim 1$, the load variants
  over-damp by 3–4× and should be used at $c \le 0.25$ if at all.
  This is exactly the regime where ETD literature already showed
  linearisation fails; the load variants inherit that limitation.
  The exact-mesh `_sl` schemes have no such regime dependence and
  remain the recommended default for problems where the surface
  evolution amplitude is not a priori bounded.
- **Stiffness varies wildly, or you need graceful Δt-overshoot
  behaviour:** BDF2-SL. Graceful Δt² accuracy degradation, no
  catastrophic-failure cliff. Cost ranges from $2 \times$ RK4-SL
  ($c=0.5$, 286 solves, 0.029%) to slightly cheaper than RK4-SL
  ($c=4$, 120 solves, 0.18%). At $c=1$, BDF2-SL is comparable in
  cost to RK4-SL ($c=0.5$) with comparable trajectory accuracy.
- **Adams–Bashforth-style cheap multistep:** **not recommended for
  this problem class.** Even at deep safety factors AB2-SL produces
  early-step trajectory artefacts; correct equilibrium for the wrong reasons.
- **Existing FSSA / curvS pipeline:** keep using it until trajectory
  fidelity becomes binding; the Cathles-$\gamma$ bias may be
  acceptable depending on the use case. The convergence pathology
  (smaller $\Delta t$ → larger drift) is shown in `fig_convergence_time.png`.

## Figures

- `fig_zoo_comparison.png` — side-by-side $h_p(t)$ trajectories and
  final $\delta r(\theta)$ profiles for the production-c picks
  (FE-SL ref, RK2-SL c=1, RK4-SL c=1, RK2-load c=1, RK4-load c=1,
  AB2-SL c=0.25, BDF2-SL c=1, BDF2-load c=1). Family by colour,
  mesh-handling by dash, integrator order by linewidth.
- `fig_zoo_residuals.png` — final-profile residuals
  $\delta r_{\rm scheme}(\theta) - \delta r_{\rm ref}(\theta)$ vs the
  res=80 reference, plus mode-amplitude residual
  $|a_m|_{\rm scheme} - |a_m|_{\rm ref}$. Exposes long-wavelength
  errors invisible in the overlaid profile plot — curvS shows the
  largest spike at the block edges, BDF2 c=4 has elevated low-mode
  errors.
- `fig_convergence_time.png` — continent-benchmark final
  $|\Delta A / A_0|$ vs $\langle\Delta t\rangle$ per (scheme, $c$).
  BDF2 traces a clean slope-1 line in the truncation-dominated arm;
  RK2 and RK4 are V-shaped because they're noise-floor-dominated at
  small Δt and exhibit a stability cliff at $c = 2$.
- `fig_zoo_pareto.png` — two-panel Pareto: total Stokes solves vs
  $|\Delta A|$ (left) and wall time vs $|\Delta A|$ (right). The
  load variants cluster left of their `_sl` counterparts at the
  same accuracy band, confirming the wall-time savings.
- **`fig_relax_gamma.png`** — Cathles relaxation benchmark: per-mode
  $\gamma_{\rm fit} / \gamma_{\rm Cathles}$ for each scheme. Faithful
  schemes cluster around 1 at modes 6, 10; mode 2 deviates to ~0.4
  uniformly (annulus geometry departing from half-space dispersion).
  RK4-load c=1 stands out at $\sim 4 \times$ over-damping —
  the regime-dependent load-approximation failure.

## Cathles relaxation-benchmark results (Case A)

A 15-run sweep on the relaxation benchmark (modes $k \in \{2, 6, 10\}$
× schemes {FE-SL dtf=0.05, RK4-SL c=1, RK4-load-SL c=1, AB2-SL c=0.25,
curvS dtf=1}) tests per-mode decay-rate recovery against analytic
Cathles dispersion.

| mode | FE-SL ref | RK4-SL c=1 | RK4-load c=1 | AB2-SL c=0.25 | curvS dtf=1 |
|---:|:---:|:---:|:---:|:---:|:---:|
| 2  | 0.44 | 0.41 | **1.19** | 0.40 | 0.11 |
| 6  | 1.07 | 0.98 | **3.82** | 0.97 | 0.95 |
| 10 | 1.09 | 0.97 | **3.84** | 0.96 | 0.90 |

(Numbers are $\gamma_{\rm fit} / \gamma_{\rm Cathles}$. Cathles
$\gamma_k = 1 / (2k)$ in our nondim. See `fig_relax_gamma.png`.)

### Findings

1. **Mode 2 deviation is geometric, not numerical.** All faithful
   schemes (FE-SL, RK4-SL, AB2-SL) cluster at ratio ≈ 0.41 — meaning
   the *true* annulus relaxation rate for $k=2$ is $\sim 0.10$, not
   the half-space estimate of $0.25$. The annulus thickness
   $r_o - r_i = 0.5$ is comparable to the wavelength
   $2\pi/k = \pi$ at $k=2$, so the half-space assumption fails.
   For $k \ge 6$ the wavelength fits inside the annulus and the
   half-space dispersion is recovered to within 5%.

2. **RK4-SL c=1 and AB2-SL c=0.25 recover the (annulus) γ_k cleanly
   without any prefactor.** This is the headline "beyond FSSA"
   verification: the relaxation CFL with $\gamma_{\rm used}$
   measured from the surface state alone gives the right decay rate,
   no hardcoded constants required.

3. **curvS systematically under-damps**, especially at low $k$.
   The curvature-derived $\gamma$ is biased — $0.11$ at $k=2$,
   $0.90$ at $k=10$. The Phase III convergence-pathology finding
   (curvS gets *worse* with smaller Δt) is verified per-mode.

4. **RK4-load over-damps catastrophically in pure relaxation.**
   At mode 10, step 1 gives $A = 0.0253$ (15% under analytic), and
   step 2 collapses to $A = 0.000486$ — a 50× decay over $\Delta t = 20.6$,
   implying $\gamma_{\rm eff} \approx 0.19$, almost **4× the Cathles
   rate**. The load surface traction
   $-\rho g \, \Delta h\, \hat r$ at large $\gamma\Delta t$ overshoots
   when $\Delta h$ is comparable to $h$ itself.

### Regime-dependent failure of the load approximation

This is the failure mode the continent benchmark hid:

| benchmark | $h$ scale | $u_n$ scale | $\Delta h_{\rm load} / h$ | load works? |
|---|---:|---:|---:|---|
| Continent (Case B, this paper) | $h_{\rm eq} \approx 0.04$ small | small (residual) | $\ll 1$ | **yes** |
| Cathles relaxation (Case A) | $A_0 \approx 0.05$ initially, decaying | large (full relaxation) | $\sim 1$ at $\gamma\Delta t = 1$ | **no — over-damps** |

The load approximation is effectively a linearised FSSA-style
stabilisation. It inherits FSSA's saturation bias when
$\gamma\Delta t$ approaches 1 *and* the displacement per step
approaches the displacement amplitude. Forced-equilibrium problems
where $h$ stays small avoid this regime; pure-relaxation problems
where $h$ starts at full IC amplitude expose it immediately.

**Recommendation:** for the load variants, use $c \le 0.25$
(γΔt ≤ 0.25, where the linearisation is still faithful) on
relaxation-dominated problems, or use the exact-mesh `_sl` variant.
The 60% wall-time savings of `_load_sl` at $c=1$ on continent
isostasy do not transfer to the pure-relaxation regime.

**Composition caveat — load-shortcut is single-physics-cohort.**
The load variants keep the mesh at the start-of-step configuration
during all intermediate stages. If the cohort includes other
state variables that are *integrity-coupled* to the mesh — most
importantly, swarms whose particles must remain inside the deforming
domain because they carry material properties evaluated at Stokes
integration points — then holding the mesh frozen while those
variables advance at per-stage rates breaks the geometric consistency
required for the Stokes assembly. Two compounding failure modes:

1. *Particles drift outside the frozen-mesh cell layout* during
   per-stage advection. Migration would relabel their cell
   assignment, but the next stage's Stokes solve is on the same
   frozen mesh — geometrically inconsistent with the new particle
   positions.
2. *Property projection becomes mesh-vs-swarm inconsistent.* The
   Stokes assembly projects particle properties to integration
   points; with swarm advanced but mesh frozen, the projection
   uses fresh particle positions on stale integration-point
   locations. Silent correctness bug, distinct from the
   linearisation-vs-exponential issue above.

**The load-shortcut variants are intended for free-surface-only
cohorts (mesh + surface profile, no state-coupled swarms).** Once
swarms or per-integration-point fields join the cohort, fall back
to the exact-mesh `_sl` schemes which advance every cohort member
to its true intermediate-stage configuration before the next stage's
rate evaluation.

### Solution-state-driven Δt limiter for the load variants

The validity condition $\Delta h_{\rm pred} \ll h$ can be enforced
adaptively from the current solution state alone, with no knowledge
of $\gamma$ or the physics regime. The limiter, applied at the start
of each step before the load is computed:

```
ratio = Δt × max|u_n_now| / max|h_now|
if ratio > 0.25:
    Δt ← 0.25 × max|h_now| / max|u_n_now|
```

This uses only `u_n` (free from the previous step's terminal Stokes
solve) and `h` (the current surface profile). On RK4-load-SL mode 10
$c=1$ — the previously-catastrophic case — the limiter reduces
$\Delta t$ from 10.3 → 5.16 at step 1 and keeps it at ~5 throughout,
yielding $\gamma_{\rm fit} / \gamma_{\rm Cathles} = 1.13$ in 11 steps —
the same result as $c=0.25$ chosen statically. The limiter
*formalises* the regime-restriction: a problem whose dynamics keep
$\Delta h \ll h$ never triggers it (Δt grows freely up to the
relax-CFL or bulk-CFL bound); a problem with $\Delta h \approx h$
triggers it on every step (Δt is held at the load-validity bound).

Honest cost consequence: the limiter saves the load scheme from
silent over-damping but does not recover its cost advantage. On
relaxation mode 10, RK4-load-SL with the limiter uses 44 Stokes
solves to reach the decay target, vs RK4-SL c=1's 16 solves at the
same trajectory accuracy. **The load short-cut wins on cost only
when its validity condition is naturally satisfied** (forced-
equilibrium-near-steady-state regime); the limiter makes that
regime-dependence self-evident from the solution state.

**Verification at mode 10**, varying $c$ on RK4-load-SL:

| scheme | $c$ | $\gamma\Delta t$ | $n_{\rm steps}$ | $\gamma_{\rm fit}/\gamma_{\rm Cathles}$ |
|---|---:|---:|---:|---:|
| RK4-load-SL | 0.25 | 0.25 | 11 | 1.13 (12% over) |
| RK4-load-SL | 0.5  | 0.50 | 5  | 1.35 (35% over) |
| RK4-load-SL | 1.0  | 1.00 | 2  | **3.84 (3.8× over)** |
| RK4-SL      | 1.0  | 1.00 | 4  | 0.97 (3% under, reference) |

The faithful threshold for the load approximation is $\gamma\Delta t \le 0.25$.
RK4-SL at $c=1$ stays accurate (3% under Cathles) because it samples
at the actual deformed mesh — not affected by the load linearisation.

For BDF2 we also checked the implicit family in pure relaxation:

| scheme | $c$ | $n_{\rm steps}$ | $\gamma_{\rm fit}/\gamma_{\rm Cathles}$ | comment |
|---|---:|---:|---:|---|
| BDF2-SL      | 1.0 | 4 | 1.41 | 2nd-order truncation at $\gamma\Delta t=1$ |
| BDF2-load-SL | 1.0 | 4 | n/a | A goes negative, sign-flip — fails |

BDF2-SL's 41% over-damp is the expected 2nd-order BDF2 truncation
at $\gamma\Delta t = 1$ (linear analysis: $|\xi| = 0.447$ vs true
$e^{-1} = 0.368$, 21% error per step). It would converge to the
correct rate at smaller $c$. The BDF2-load failure at $c=1$ is the
load linearisation regime-dependence again — sign-flip means the
estimator predicts $\Delta h$ overshooting through zero, exactly the
failure mode RK4-load c=1 exhibits at modes 6 and 10.

This is also the place where the parallel-runner setup paid off:
running the same scheme set on Case A and Case B exposed an
implicit physics dependency in the load approximation that a
single-benchmark study would have missed.

## Pending: additional benchmarks for publication

Cases A (Cathles relaxing topography) and B (continent isostasy)
together exercise both the pure-relaxation regime and the forced-
equilibrium regime, and have already exposed orthogonal failure
modes (curvS bias at low k from Case A; load approximation
over-damping in pure relaxation also from Case A; the RK stability
cliff at $c=2$ from Case B). For full FSSA-literature comparison,
two further benchmarks would help:

### Larger-amplitude isostatic loading

- Same block geometry as continent isostasy but with $\beta = 0.5$
  or domain-scaled to give $h_p^{\rm eq} / r_o \sim 0.1$ (currently
  $\sim 0.04$).
- **What it adds:** tests the SL traceback when $u_t \Delta t / r_o$ is
  no longer $\ll 1$; tests that mesh quality holds without ALE
  remeshing.

### Crameri–Tackley sloshing (recommended, classic FSSA test)

- Box geometry, dense block sinks through ambient. Free surface
  deforms in response.
- **What it adds:** *explicitly* compares against the published
  FSSA literature on its home-turf benchmark. We expect the
  saturation-free schemes here to perform comparably without the
  prefactor — the headline result.

### Subduction-style with free surface (stretch goal)

- Slab pull deforms the upper boundary; the free surface is the
  observable for surface-process coupling.
- **What it adds:** demonstrates the load variants are robust under
  a sustained driving force (not just relaxation); also exercises
  the regime that motivated FSSA in the first place.

### Cost expectation (rough)

| benchmark | runs | est. wall (8-way parallel) | status |
|---|---:|---:|---|
| Cathles, 3 modes × 5 schemes | 15 | 15 min | **done** |
| Larger-β isostasy | ~7 schemes | 1–2 h | pending |
| Crameri sloshing (Cartesian) | needs box-mesh adaptation | 4–8 h dev + 4 h sweep | pending |
| Subduction stretch | needs slab setup | 1+ day dev + sweep | pending |

The larger-β benchmark reuses the existing annulus infrastructure;
the box-geometry benchmarks (sloshing, subduction) require porting
the SL traceback and surface-load variables to Cartesian boundaries,
which is straightforward but not free.

## Reproducibility

### Runners (two parallel callers of the same scheme bodies)

- `docs/developer/design/_phase_i_fs_continent_fs_snapshots.py` — Case B
  (continent isostasy, forced equilibrium).
- `docs/developer/design/_phase_i_fs_relax_sl_zoo.py` — Case A
  (Cathles relaxing topography).

Both implement the same scheme set: `fe_sl`, `rk2_sl`, `rk4_sl`,
`ab2_sl`, `bdf2_sl`, `rk2_load_sl`, `rk4_load_sl`, `bdf2_load_sl`,
plus the legacy `fe`, `rk2`, `rk4`, `rk2_full`, `curvS`, `midpoint`.
The shared kernels (SL traceback, surface-load infrastructure,
monotone $\gamma$ history, Picard damping) appear identically in
both files — useful for the eventual extraction of a deformable-
surface abstraction.

### Sweep driver and diagnostics

- `~/+Simulations/FreeSurface/sweep_driver.py` — job matrix for Case B.
- `~/+Simulations/FreeSurface/spectral_diagnostics.py` — pole-jitter
  + low-k excess metrics.
- `~/+Simulations/FreeSurface/analyse_integrator_zoo.py` — Pareto +
  characterisation table for Case B.
- `~/+Simulations/FreeSurface/analyse_relax_zoo.py` — per-(mode, scheme)
  $\gamma$-recovery analysis for Case A.
- `~/+Simulations/FreeSurface/plot_zoo_comparison.py`,
  `plot_zoo_residuals.py` — Case B figures.

### Output

- `~/+Simulations/FreeSurface/zoo_summary.csv` — one row per run
  (Case B sweep).
- `~/+Simulations/FreeSurface/zoo_relax_summary.csv` — one row per
  (mode, scheme) for the relaxation sweep.
- Per-run subdirectories under
  `~/+Simulations/FreeSurface/{time_sweep_res20, reference_highres,
   space_sweep_rk4sl_c1, relax_modesweep, reference}/<label>/output/`
  contain the per-step npz files and `work_log.csv`.
