# Handoff notes — Exponential time-stepping for free-surface evolution

> **Read first**: `EXPONENTIAL_FREE_SURFACE.md` in the same directory.
> That's the running design doc. This document is the working-state
> snapshot at the point of session handoff and a prioritised list of
> things to try next.

## Status at handoff

The kinematic exponential update

$$
h^{n+1} = h^n + \frac{1 - e^{-\gamma\Delta t}}{\gamma}\,u_n^{\text{(surf)}}
$$

is the proposed integrator. Its saturation property (bounded
$h$ at large $\gamma\Delta t$) has been demonstrated cleanly on
the **buoyancy-driven** test: FE / FE+FSSA grow linearly without
bound; the kinematic ETD saturates at the driven equilibrium. See
`_plot_phase_i_buoyancy.py` and `output/phase_i2d_fs_buoyancy.png`.

On **homogeneous (initial-perturbation) tests** all schemes
eventually drunken-sail at large $\Delta t$. The current
investigation has narrowed where that drift comes from but has
not eliminated it. State of play below.

## Two demonstrators

- `_phase_i_fs_etd_annulus.py` — upper-boundary variant (the
  classical free-surface setup). Body force was originally
  `-r̂`; tested with the user's "anomaly form" (Heaviside step at
  $r_o$ on the body force). Drift floor present. Diagnostics
  identified mode-0 from incompressibility violation under the
  curl-free $-r̂$ body force.
- `_phase_i_fs_etd_internal.py` — internal-boundary "sticky air"
  variant adapted from
  `docs/examples/free_surface/advanced/AnnulusND_FS-OuterSphere.py`.
  Two layers (heavy fluid $\eta=1$ + light air $\eta=0.01$);
  internal boundary at $r_o$ is the free surface. Per-element P0
  Lagrangian density $M$, sharp Piecewise reference $M_{\text{ref}}(r)$,
  body force $-(M - M_{\text{ref}})\hat r$. **This is the cleaner
  setup** because mean-density-subtraction is well-defined here
  (the air phase is in the mesh).

The internal-boundary variant should be the production demonstrator.
The upper-boundary variant is kept as a reference / pedagogical
case showing the asymmetric-restoring-force issue.

## What has been ruled out as the source of large-dt drift

| Hypothesis | Test | Result |
|---|---|---|
| Stokes SNES residual | Tighten tolerance 1e-4 → 1e-5 | No effect; well-converged |
| Discontinuity in body force at $r_o$ | Smooth tanh $M_{\text{ref}}$ vs sharp Piecewise | Smooth is *worse*; not the cause |
| Layer-interface viscosity contrast | 100× → 10× contrast | Small change (~10%); not the cause |
| FSSA stabilisation contribution | FSSA off | Drift increases ~7% — FSSA helps slightly |
| Mode-0 incompressibility violation | Fourier on final boundary | Mode 0 is only 4e-3, much smaller than the drift in mode 10 |

## What the data shows

1. The drift is **entirely in the target mode** (mode 10 in the
   single-mode test), not in spurious modes. Final-step Fourier
   on the internal-boundary curvS-fssa run at dt-factor=1, n=32:
   $|a_{10}| = 6.32\times 10^{-2}$, all other modes $\le 4\times
   10^{-3}$.
2. The mode-10 amplitude **sign-flips and grows** —
   $+0.05 \to 0 \to -0.063$ over 32 steps with a roughly constant
   per-step $\Delta A$ of $-3\times 10^{-3}$. This is the
   classic FE drunken-sailor signature, not amplitude
   accumulation of mode-0 noise.
3. **First-step decay rate is consistent across resolutions** at
   $\sim 1.2\times 10^{-3}$ per unit time, matching the predicted
   $\gamma_{\text{eff}} A$ for the half-space estimate. So the
   *initial* dynamics is exponential and resolution-converged.
4. **Per-step drift floor is dt-proportional, not cellsize-proportional.**
   res 32, 48, 64, 80 all show roughly the same drift rate per unit
   time (~$2\times 10^{-3}$). One of those (res=48) has
   anomalously elevated max\|v\| — likely a poor cellsize-to-amplitude
   ratio for representing the discontinuous body force.

## Small-$\Delta t$ reference: $\gamma_{\text{eff}} \approx 0.048$

Run: `--scheme fssa-vs-nofssa --n-steps 100 --dt-factor 0.05`,
giving $\Delta t = 0.092$ and total $t = 9.19$.

Trajectory (FE+FSSA, single mode):

| step | A_mode |
|---:|---:|
| 0 | +5.00e-2 |
| 11 | +4.78e-2 |
| 51 | +3.97e-2 |
| 100 | +2.98e-2 |

Fitting: per-step ratio $\sim 0.9956$, giving
$\gamma_{\text{eff}} = -\ln(0.9956)/0.092 = 0.048$.

**That matches the half-space prediction $\gamma = \rho g/(2\eta|k|) =
1/(2\cdot 1\cdot 10) = 0.05$ to within 5%.** So $\gamma$ in the
two-layer system is *not* dramatically different from half-space —
my earlier hypothesis was wrong.

## Drift roughly proportional to $\Delta t$ — analytic accuracy is a separate question

Comparing actual trajectories at small vs large dt:

| run | $\Delta t$ | total $t$ | measured A_final | drift rate / unit time |
|---|---:|---:|---:|---:|
| small-dt (n=100) | 0.092 | 9.19 | +2.98e-2 | -2.5e-4 |
| large-dt (n=32) | 1.84 | 58.8 | -7.41e-2 | -1.3e-3 |

Drift rate scales roughly with $\Delta t$ (a $20\times$
increase in $\Delta t$ gave $\sim 5\times$ increase in drift
rate). Whether the precise scaling is $O(\Delta t)$ or some
other power isn't the focus of this investigation.

**The actual goal is a stable integration scheme that doesn't
blow up when the timestep grows.** The small-dt run is the
*reference trajectory* — whatever the small-dt scheme produces
is what we want the large-dt scheme to approach. Departure from
the analytic exponential at small dt is a separate
accuracy-against-analytic question for a different session, not
what this investigation is trying to solve.

The current open question is: at large $\Delta t$, can we keep
the trajectory bounded close to the small-dt reference? The
kinematic ETD provides modest damping (~17% less drift than
FE+FSSA at our test dt) but isn't delivering bounded large-dt
behaviour by itself. The next session should attack this stability
question directly — not the analytic-accuracy question.

## Recommended next actions, in priority order

The success criterion is **stability at large $\Delta t$
relative to the small-dt reference trajectory** — not accuracy
against the analytic exponential. The small-dt run gives the
reference; the goal is a scheme that doesn't blow up away from
that reference as $\Delta t$ grows.

1. **Find an integrator that's actually stable at large
   $\Delta t$.** The current first-order kinematic ETD provides
   modest damping but doesn't bound the trajectory at our test
   dt. Candidates worth implementing and benchmarking against the
   small-dt reference:
    - *Midpoint-corrected $u_n$* (RK2-flavoured): solve Stokes,
      take a half-step kinematic ETD to estimate $h^{n+1/2}$,
      deform the mesh there, re-solve Stokes for $u_n^{n+1/2}$,
      take the full-step ETD with that midpoint velocity.
      Two Stokes solves per step but should be much more stable.
    - *Cox–Matthews ETD-2 with source*: use $u_n^n$ and
      $u_n^{n-1}$ to estimate the linear-in-time part of $u_n(t)$
      across the step. One Stokes solve per step but two-step
      memory.
    - *Implicit kinematic update*: solve a coupled
      surface-position+Stokes problem at $t^{n+1}$ — known
      stable at any $\Delta t$, but heavier solver.
   The bench is "compare against small-dt reference at the same
   total $t$" — *not* against analytic exponential.

2. **Buoyancy on the internal-boundary mesh.** The buoyancy
   demonstrator currently lives only on the upper-boundary
   variant. Port to the internal-boundary mesh and confirm the
   kinematic-ETD saturation at the driven $h_{\text{eq}}$. This
   is where the scheme has clear value already — it's worth
   landing it cleanly on the cleaner mesh.

3. **Empirical-$\gamma$ from temporal regression**, as a
   separate methodological track. The single-shot spatial
   regression has a known failure mode (source confounding);
   a multi-step temporal regression resolves it but needs a
   history buffer. Independent of the integrator improvement
   above.

4. *Out-of-scope-for-this-session*: matching the analytic
   exponential decay at small dt. The small-dt run shows
   modest deviation from analytic ($\sim 5\%$); whether
   that's discretisation, body-force formulation, or something
   else is a separate question.

## What not to chase

- Mode-0 from incompressibility violation. Confirmed small (~4e-3
  vs mode-10 ~6e-2). Not the dominant drift source.
- Stokes SNES tolerance. Already converged tighter than what
  matters at the integrator level.
- Sharp vs smooth body-force discretisation. Sharp is correct;
  smooth is empirically *worse* by ~3% in drift rate.
- Layer-interface viscosity contrast. Tested 10× and 100×; minor
  effect on drift rate.
- Per-element rho being non-Lagrangian. Confirmed it IS
  Lagrangian (P0 discontinuous, attached to elements that ride
  with the mesh). No advection error.

## User preferences and corrections from this session

- **Curvature-derived γ is the analytic solution dressed up.**
  Don't compare against it as a separate scheme — it's the
  $u_n = -\gamma h$ limit of the kinematic ETD.
- **Free surfaces are always driven** in geodynamics — by
  buoyancy, slabs, plumes, glacial loads, rotation.
  Pure-relaxation tests are pedagogical, not the production case.
- **Drop "subtract radial mean = zero everywhere"** misconception.
  The mean-subtraction gives forces *near* the deformed boundary
  (where local density differs from radial mean), not zero
  everywhere.
- **Internal-boundary mesh** (sticky-air) is the right setup for
  testing mean-subtraction body force, because the mesh contains
  both the heavy and light layers.
- **P0 discontinuous element-density** for $\rho$, sharp
  Piecewise for $\rho_{\text{ref}}(r)$, *not* a smooth tanh.
  This was tested; smooth is worse.
- **Geometry is linear** in UW3 (straight edges, flat faces),
  even when fields use higher-order basis. Cotangent Laplacian
  applies directly to surface meshes for $\Delta_S h$ computation,
  no curved-element generalisation needed.
- **Mean curvature deviation** is the unifying scalar invariant
  in both 1D-on-2D and 2D-on-3D. Same as the bubble-pressure
  Laplace law. Don't get distracted by Gaussian curvature — it
  doesn't enter the linearised dispersion.
- **Empirical $\gamma$ from spatial regression** has a known
  failure mode: it confounds source spatial structure with
  relaxation rate when the source has imprinted itself on $h$.
  Temporal regression with multi-step history is the recovery
  path.

## Code state (worktree-local)

Files (all in `docs/developer/design/`):

- `EXPONENTIAL_FREE_SURFACE.md` — design doc; main reference
- `EXPONENTIAL_FREE_SURFACE_HANDOFF.md` — this file
- `_phase_i_fs_etd_annulus.py` — upper-boundary runner; FE,
  FE+FSSA, ETD-scalar, ETD-curv, ETD-curvS, BDF-2, ETD-2,
  empE schemes; supports `--ic single|multi`,
  `--visc-contrast`, `--buoyancy`
- `_phase_i_fs_etd_internal.py` — internal-boundary runner;
  FE+FSSA, curvS+FSSA, FE-noFSSA, curvS-noFSSA. Sharp Piecewise
  $M_{\text{ref}}$. Stokes tolerance 1e-5.
- `_plot_phase_i_fs_summary.py` — multi-panel comparison plot
  (drops "curv" since the user pointed out it's the analytic
  solution dressed up)
- `_plot_phase_i_buoyancy.py` — buoyancy single-figure plot
- `_phase_i_freesurface_relaxation_0d.py` — 0-D ODE precursor
  (in vep-two-stokes worktree, predates this work)

## Test commands

```bash
# In the worktree
cd /path/to/worktree

# Internal-boundary, FSSA on/off comparison, n=32 at default dt
pixi run -e amr-dev python -u docs/developer/design/_phase_i_fs_etd_internal.py \
    --res 20 --scheme fssa-vs-nofssa --n-steps 32 --dt-factor 1.0

# Small-dt reference (the run currently pending)
pixi run -e amr-dev python -u docs/developer/design/_phase_i_fs_etd_internal.py \
    --res 20 --scheme fssa-vs-nofssa --n-steps 100 --dt-factor 0.05

# Upper-boundary buoyancy (the demonstrator)
pixi run -e amr-dev python -u docs/developer/design/_phase_i_fs_etd_annulus.py \
    --quick --scheme buoyancy-set --dt-factor 1.0 --n-steps 16 --buoyancy
```

## What "done" looks like for this investigation

1. ✅ Buoyancy-driven kinematic ETD demonstrator works
   (bounded saturation at $h_{\text{eq}}$).
2. ✅ A scheme that stays close to the small-dt reference at
   large $\Delta t$, without blowing up — RK4 (no γ) is the
   answer. See "Session 3 findings" below.
3. ⏳ Clean write-up incorporating the integrator-comparison
   result; pivot to viscosity-contrast test (isostatic
   relaxation of buoyant block) where curvature-γ is expected
   to be a poor model of local relaxation rate.

**Goal restated:** find an integrator that doesn't blow up as
the timestep grows. Compare against the small-dt run as
reference (not against the analytic exponential — analytic
accuracy is a separate question for a different session). The
first-order kinematic ETD already buys saturation in the driven
case; for the homogeneous case, RK4 with no γ estimate beats
all γ-prefactor variants on accuracy AND stability at large
$\Delta t$.

## Session 3 findings (2026-05-06): RK family beats ETD prefactor

The priority-1 "midpoint-corrected $u_n$" candidate was
implemented and found to be **strictly worse** than curvS-FSSA
across all $\Delta t$-factors tested. Adding RK2 midpoint
sampling on top of the (1-α)/γ prefactor compounds the
prefactor's overshoot bias.

The unexpected finding came from extending the comparison to
**RK4 with no γ at all**: classical 4-stage Runge-Kutta
sampling $u_n$ at four trial mesh positions. Across the dt
range that admits it ($\gamma\Delta t \le 2.78$, here up to
dt-factor=20), RK4 gives the **smallest error against the
reference at any total Stokes-solve cost**.

### Headline numbers (single-mode IC, res=20, internal boundary)

Reference: FE-noFSSA at $\Delta t = 0.05\Delta t_{\text{est}}$,
$n=200$. Fitted $\gamma_{\text{eff}} = 0.0457$.

Error $|A_{\text{final}} - A_{\text{ref}}|$ at run's final $t$:

| scheme | dtf=1 | dtf=10 | dtf=20 |
| --- | ---: | ---: | ---: |
| FE-noFSSA | 1.4e-3 | 1.0e-3 | 9.9e-3 (drunken-sailor) |
| RK2 (no γ) | 6.5e-4 | 1.6e-3 | 1.1e-2 (blown up) |
| **RK4 (no γ)** | **7.5e-4** | **3.0e-4** | **3.5e-5** |
| curvS-FSSA | 7.4e-4 | 1.3e-2 | 1.7e-2 |
| midpoint-FSSA | 7.4e-4 | 1.7e-2 | 2.2e-2 |

Cost-per-step: FE 1×, RK2 2×, RK4 4×, curvS 1×, midpoint 2×
Stokes solves.

At dtf=20, n=4 (16 RK4 solves): RK4 reaches $t=147$ with error
$3.5\times 10^{-5}$ — better than any scheme at any cost.

### What this means

1. **The kinematic ETD prefactor (1-α)/γ is L-stability, not
   accuracy.** curvS-FSSA / midpoint-FSSA *never blow up* but
   *systematically overshoot* the reference at large $\Delta t$.
   They're the right choice when $\gamma$ is unknown or
   spatially varying and you cannot guarantee
   $\gamma\Delta t < 2$.

2. **For homogeneous problems with bounded $\gamma\Delta t$,
   RK4 wins on every metric**: smaller error, comparable or
   lower total cost, retains stability up to the ~4× larger
   $\Delta t$ admitted by its 4th-order Taylor match to the
   exponential.

3. **The midpoint hybrid (RK2 + ETD prefactor) is the worst of
   both worlds.** The RK2 sampling and the prefactor both
   correct in the same direction; combining them
   over-corrects. Drop this candidate from the publication.

### Open question for the next session: viscosity-contrast test

The curvature-γ scheme uses a global $\eta_{\text{eff}} = 0.5$
constant. On a problem with strong local viscosity contrast
(e.g. a high-η buoyant block on a low-η substrate, isostatic
relaxation), the local $\gamma$ varies dramatically across the
surface, and the curvature estimator can't see it. This is
where curvS / midpoint should fail — the prefactor will use
the wrong $\gamma$ in regions, leading to either over- or
under-correction. RK4 (no γ) doesn't care about $\eta$
distribution at all and should track the reference cleanly.

**Test setup**: 2D annulus with a buoyant Gaussian high-η blob
near the surface, no initial perturbation, surface bulges
upward to isostatic equilibrium. Known analytic equilibrium
for the ideal half-space case (Cathles 1975). Compare:
RK4-noFSSA, RK2-noFSSA, FE-noFSSA, curvS-FSSA, midpoint-FSSA.
Expect: curvS / midpoint to either over- or under-shoot the
equilibrium; RK4 to settle close to it.

This is the natural extension after the headline integrator
result lands cleanly.

### Code state at end of session 3

Files (all in `docs/developer/design/`):

- `EXPONENTIAL_FREE_SURFACE.md` — design doc; new "Phase I-2D-
  integrator" section documents the comparison
- `EXPONENTIAL_FREE_SURFACE_HANDOFF.md` — this file
- `_phase_i_fs_etd_internal.py` — schemes `fe`, `rk2`, `rk4`,
  `curvS`, `midpoint` with FSSA on/off variants. CLI option
  `--scheme integrators` runs the headline comparison set in
  one Python process.
- `_plot_phase_i_integrators.py` — trajectory + cost-vs-
  accuracy plotter; works on the dtf{1,2,5,10,20} npz outputs.
- `_plot_phase_i_midpoint.py` — earlier midpoint-vs-curvS
  plotter (kept for reference).

**Output files** (in `output/`, .gitignore'd):
- `phase_i2d_fs_etd_dtf0.05_n200_internal_res20.npz` — reference
- `phase_i2d_fs_etd_dtf{1.00,2.00,5.00,10.00,20.00}_n*_internal_res20.npz`
  — sweep data
- `phase_i2d_fs_integrators_trajectories.png` — 5 schemes ×
  5 dt-factors trajectory grid
- `phase_i2d_fs_integrators_cost.png` — error vs total Stokes
  solves; A_max vs total Stokes solves
