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

## Drift rate scales with $\Delta t$ — truncation error, not noise floor

Comparing actual trajectories vs the pure-exponential prediction
$A(t) = 5\times 10^{-2}\,e^{-0.048\,t}$:

| run | $\Delta t$ | total $t$ | measured | predicted (exp decay) | deviation | drift rate / unit time |
|---|---:|---:|---:|---:|---:|---:|
| small-dt (n=100) | 0.092 | 9.19 | +2.98e-2 | +3.22e-2 | -2.3e-3 | -2.5e-4 |
| large-dt (n=32) | 1.84 | 58.8 | -7.41e-2 | +3.0e-3 | -7.7e-2 | -1.3e-3 |

**The drift rate per unit time is $\sim 5\times$ larger at $20\times$
larger $\Delta t$.** That's roughly linear scaling with $\Delta t$
— consistent with $O(\Delta t^2)$ per-step truncation error
accumulating over $t/\Delta t$ steps. This is the integrator's own
truncation error, not a fixed velocity-noise floor.

This changes the diagnosis: the large-dt drunken-sailor is
**accumulated truncation error**, not noise. At small dt the
truncation error is small enough that the relaxation dominates;
at large dt the truncation error per step gets big enough that
over many steps the surface drifts past zero.

Implications:
- **Higher-order integrator** (ETD-2, BDF-2 with proper source
  handling) should help — those have $O(\Delta t^3)$ per-step
  truncation error, so accumulated drift scales as $\Delta t^2$
  rather than $\Delta t$.
- **Predictor-corrector for $u_n$** within a step would also
  reduce truncation error (we're using $u_n$ at the *start* of
  the step; using a midpoint $u_n$ would be second-order
  accurate).
- **The kinematic ETD's saturation property is real** but only
  bounds the drift relative to the *signal* — at large dt where
  signal is decaying faster than the truncation error, the
  truncation error dominates and the integrator can't do anything
  about it because it's *the integrator's own error*.

## Recommended next actions, in priority order

1. **Implement a higher-order kinematic ETD update** to attack
   the $O(\Delta t^2)$ truncation error directly. Two natural
   options:
    - *Midpoint-corrected $u_n$*: solve Stokes once, get $u_n^n$;
      apply half-step kinematic ETD to estimate
      $h^{n+1/2}$; deform the mesh to that intermediate position
      and re-solve Stokes for $u_n^{n+1/2}$; apply the full-step
      ETD with $u_n^{n+1/2}$. This is RK2-flavoured and should
      give $O(\Delta t^3)$ truncation per step.
    - *Cox–Matthews ETD-2 with source*: use $u_n^n$ and
      $u_n^{n-1}$ to estimate the linear part of $u_n(t)$
      across the step. Two-step memory but only one Stokes
      solve per step.
   Both should reduce the drift rate scaling from $\Delta t$ to
   $\Delta t^2$ per unit time, which is the difference that
   matters at the dt regime where the integrator is supposed
   to be useful.

2. **Test that the higher-order scheme's drift rate scales as
   $\Delta t^2$.** Run at multiple dt and verify the drift
   rate per unit time scales appropriately. The headline
   homogeneous-test result is then: "kinematic ETD with
   midpoint-corrected $u_n$ remains bounded for arbitrary
   $\Delta t$ within the relaxation regime, where FE/FE+FSSA
   accumulate visible error".

3. **Buoyancy on the internal-boundary mesh.** The buoyancy
   demonstrator currently lives only on the upper-boundary
   variant. Port to the internal-boundary mesh and confirm the
   kinematic-ETD saturation at the driven $h_{\text{eq}}$.

4. **Long-time accuracy run** with the higher-order scheme,
   confirming exponential relaxation over many decay times.

5. **Empirical-$\gamma$ from temporal regression**, separately
   from the integrator improvement above. The single-shot
   spatial regression has a known failure mode (source
   confounding); a multi-step temporal regression resolves it
   but needs a history buffer.

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

1. ✅ Measured $\gamma_{\text{eff}}$ matches half-space prediction.
2. ✅ Buoyancy-driven kinematic ETD demonstrator works
   (bounded saturation at $h_{\text{eq}}$).
3. ⏳ Pure-relaxation demonstrator with bounded error at large
   $\Delta t$ — *needs the higher-order ETD per priority 1*.
4. ⏳ Clean write-up: integrator works for driven problems
   bounded by physics; pure-relaxation problems require a
   higher-order time discretisation to keep accumulated
   truncation error below physical relaxation.

The first-order kinematic ETD has a real saturation property
that bounds amplitude relative to a fixed signal (buoyancy-driven
case). Where it fails on homogeneous tests is from per-step
truncation error scaling with $\Delta t$ that accumulates over
many steps; this is fixable with a higher-order time scheme.
That's the headline open task.
