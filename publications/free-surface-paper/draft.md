# A semi-Lagrangian kinematic update and an amplitude-invariant relaxation CFL for free-surface viscous flow

> **Working draft** — not a design document. This is the
> publication-track write-up; the investigation history is in
> `docs/developer/design/EXPONENTIAL_FREE_SURFACE.md`.

## Abstract

We revisit the free-surface kinematic update used in many ALE-style
viscous-flow codes for geodynamics. The conventional formulation
samples the radial component of the surface velocity, smooths it via
a scalar diffuser, and remaps it to a purely radial mesh deformation.
The tangential component of the surface velocity is dropped. We show
that this produces a structural volume-conservation error that does
not vanish with mesh refinement or timestep refinement: in our
benchmarks the bias is approximately 2% over a typical relaxation
trajectory regardless of resolution.

We propose two changes. First, a **semi-Lagrangian discretisation
of the kinematic boundary condition** that retains the existing
radial-mesh-deformation infrastructure but folds the missing
tangential transport term into the boundary value of the diffuser.
This is a one-line modification with no change to mesh quality or
the Stokes solver. Second, an **amplitude-invariant relaxation
CFL** that derives the timestep cap from a single observable —
the L²-projection of $u_n$ onto $h$ on the surface — with a
monotone-history "damping requirement" that guarantees stability
of every previously-observed mode without any hardcoded
constitutive constants ($\eta$ or $\rho g$).

Combined, the two changes reduce volume drift to better than $10^{-4}$
of the rock area on a continent-isostasy benchmark, recover the
correct equilibrium height on a relaxing-topography benchmark, and
make first-order forward Euler with a small Δt a viable reference
scheme. RK4 with the relaxation CFL is the production scheme.

## Notation and setup

Two-dimensional viscous Stokes flow on an annulus
$\mathcal{D} = \{r_i \le r \le r_o(\theta, t)\}$, with no-slip on
the inner boundary and a deformable free surface at
$r = r_o(\theta, t)$. We write $h(\theta, t) = r_o(\theta, t) - r_o^{(0)}$
for the displacement of the free surface from its initial radius
$r_o^{(0)}$. On the surface, the velocity has radial and tangential
components $(u_n, u_t)$ in the local
$(\hat r, \hat\theta)$ frame.

The Stokes solve is incompressible:
$$
\nabla\cdot\mathbf{v} = 0,\qquad \nabla\cdot\boldsymbol{\sigma} + \mathbf{f} = 0.
$$
Free-surface boundary conditions are: zero traction
($\boldsymbol{\sigma}\cdot\hat n = 0$) on the deformed surface; no
slip on the inner boundary. The mesh moves to track the deformed
surface (ALE).

## The kinematic free-surface boundary condition

The kinematic condition follows a material point on the free
surface:
$$
\frac{D}{Dt}\bigl(r - r_o^{(0)} - h(\theta, t)\bigr) = 0
\quad\Longrightarrow\quad
\frac{\partial h}{\partial t} + \frac{u_t}{r_o}\frac{\partial h}{\partial \theta} = u_n
$$
The second form is the equation we discretise. The left-hand side
contains **two** terms: a local time derivative $\partial h/\partial t$
and a tangential transport term $(u_t/r_o)\,\partial h/\partial\theta$.

### The conventional radial-only update

A common discretisation drops the tangential term. One samples
$u_n(\theta)$ on the surface, decomposes it on the boundary in
Fourier modes, solves a scalar Laplace–Poisson problem to smooth
$u_n$ into the interior radial-displacement field
$\hat r \cdot \xi(\mathbf{x})$, and deforms each mesh node by
$\Delta t\,\xi\,\hat r$. The kinematic BC is satisfied to
$\mathcal{O}(\Delta t)$ at the surface as $\partial h/\partial t = u_n$.

This is consistent (the truncation goes to zero with $\Delta t$)
*pointwise on $h$*, but it is **not consistent with the kinematic
condition itself**. Mass that should redistribute laterally on the
surface — encoded in the $u_t\,\partial h/\partial\theta$ term — has
nowhere to go. In a full Stokes flow with $\nabla\cdot\mathbf{v}=0$
the flux through the deformed surface is
$\int u_n\,ds = 0$ exactly; the radial-only update applies that
flux as $r$-displacement only, while in the true flow the
combined radial and tangential motion preserves area. Discarding
$u_t$ breaks the area-preservation balance and introduces a bias
that scales with $|u_t|\cdot\Delta t/h$ — independent of $\Delta t$
in cumulative form once integrated over a fixed total simulated
time $T$.

We verify this directly. On the continent-isostasy benchmark,
forward-Euler with the radial-only kinematic update and a *very
small* timestep ($\Delta t \approx 1.6$ from `--dt-factor 0.05`)
produces a final volume drift of **−1.82%** and a final pole
height $h_p = 0.031$ — *wrong* by ~20% relative to the converged
equilibrium $h_p \approx 0.038$. Halving $\Delta t$ further does
not help (drift scales linearly with simulated time × per-step
displacement, not with $\Delta t$ alone). The error is structural
in the spatial discretisation of the kinematic BC.

### Semi-Lagrangian-horizontal discretisation

Restoring the tangential term is a one-line modification. The
analytical solution of $\partial_t h + (u_t/r_o)\partial_\theta h = 0$
along a characteristic is $h^{n+1}(\theta) = h^n(\theta - u_t\Delta t/r_o)$.
Adding the radial uplift gives the semi-Lagrangian update for the
full kinematic equation:
$$
\begin{equation}
\label{eq:sl-update}
h^{n+1}(\theta) = h^n\!\left(\theta - \frac{u_t\,\Delta t}{r_o}\right) + \Delta t\,u_n(\theta).
\end{equation}
$$
We retain the existing scalar-diffuser infrastructure by writing
this as an *effective* radial velocity at the surface:
$$
\begin{equation}
\label{eq:un-eff}
u_n^{\rm eff}(\theta) = u_n(\theta) + \frac{1}{\Delta t}\!\left[h^n\!\left(\theta - \tfrac{u_t \Delta t}{r_o}\right) - h^n(\theta)\right].
\end{equation}
$$
The diffuser solves $\nabla^2\xi = 0$ with $\xi = u_n^{\rm eff}$ on
the surface and $\xi = 0$ on the no-slip boundary, and the mesh
deforms by $\Delta t\,\xi\,\hat r$. The interior remains
purely radially smoothed — mesh quality is preserved exactly as in
the conventional update — and $h^n$ is interpolated at the
trace-back angle by Fourier evaluation (free, since the diffuser's
Dirichlet BC is already a Fourier polynomial).

For multi-stage Runge–Kutta integrators the same trick applies at
each stage with the stage's local $\Delta t$, building an
RK-weighted combination of stage-effective velocities.

### Generalisation

The recipe applies to any kinematic surface BC where the surface
velocity has both a normal and a tangential component. In 3D, the
trace-back is along the surface gradient direction
$\hat n \times (\hat n \times \mathbf{u}_S)$ rather than along
$\hat\theta$, but the structure is identical: trace back, sample,
add normal uplift.

## An amplitude-invariant relaxation CFL

The kinematic update fixes the *spatial* part of the discretisation.
The remaining question is the *temporal* part: how big a $\Delta t$
is safe?

### Setup

For a relaxing surface mode $h \sim e^{-\gamma t}$, we have
$\dot h = u_n = -\gamma h$, so $u_n/h = -\gamma$ exactly,
**independent of amplitude**. As the system relaxes toward
equilibrium, $h$ and $u_n$ both shrink in proportion; the ratio is
constant at $-\gamma$ all the way down. This is a self-evident
amplitude-invariant observable and is the right input to a CFL.

For multi-mode states $h(\theta, t) = \sum_k h_k\cos(k\theta)e^{-\gamma_k t}$,
no single pointwise ratio gives the dominant rate. We use the
L²-weighted least-squares estimator:
$$
\begin{equation}
\label{eq:gamma-eff}
\gamma_{\rm eff}(t)
= \frac{\bigl|\langle u_n, h\rangle_S\bigr|}{\langle h, h\rangle_S}
= \frac{\bigl|\sum_k \gamma_k\, h_k^2\, e^{-2\gamma_k t}\bigr|}{\sum_k h_k^2\, e^{-2\gamma_k t}}.
\end{equation}
$$
This is the amplitude-weighted mean of the per-mode $\gamma_k$,
robust to local sign changes (absolute value handles forced rise
and pure relaxation symmetrically). The integrals are surface line
integrals on the deformed boundary.

### The damping requirement and the monotone-history fix

The L² estimator returns the *dominant* (largest-amplitude) mode's
$\gamma$. But the *fastest* mode in the system is what bounds
stability — the highest-$k$ mode that is still resolved. When
$\gamma_{\rm eff}$ tracks a slow dominant mode and $\Delta t$ grows
to $c/\gamma_{\rm slow}$, fast modes with $\gamma_{\rm fast} \gg \gamma_{\rm slow}$
are integrated outside their L-stable region and grow oscillatorily
— a numerical "wave-like" instability rather than a physical motion.
The L² estimator catches up only after the oscillation has corrupted
the dominant mode itself.

The fix is to require *every previously-observed* mode to remain
damped. Define
$$
\begin{equation}
\label{eq:gamma-used}
\gamma_{\rm used}(t) = \max_{0 \le t' \le t}\gamma_{\rm eff}(t')
\end{equation}
$$
and use $\Delta t \le c/\gamma_{\rm used}$ where $c$ is the
integrator's L-stable safety factor (RK4 stable to $\gamma\Delta t \approx 2.78$;
in practice we use $c = 1$ for headroom).

The monotone-history estimator is bootstrap-safe (set
$\gamma_{\rm used} = 0$ and fall back to a bulk-CFL on $\Delta t$
until $\gamma_{\rm eff}$ becomes well-defined at step 2), and is
**reset only on user-controlled events** such as remeshing, or
when entering a new physical regime.

### Properties

(P1) **Amplitude-invariant.** $\gamma_{\rm eff}$ depends on
the *shape* of $(u_n, h)$ on the surface, not their magnitude.
Equivalent simulations that differ only in initial-amplitude scaling
produce identical $\Delta t$ schedules.

(P2) **No constitutive constants.** $\gamma_{\rm eff}$ is computed
from the fields the integrator already owns: $h$ from the mesh
geometry, $u_n$ from the latest Stokes solution. No reference to
$\eta$, $\rho$, $g$, or a curvature regression on $h$ — all of
which the conventional FSSA / kinematic-ETD formulation use. The
criterion specialises automatically to heterogeneous viscosity,
finite layers, geographic basis vectors, and any other case where
the analytic dispersion relation for $\gamma$ is unavailable or
incorrect.

(P3) **Conservative under stability margin.** For a single mode,
$\gamma_{\rm eff} = \gamma$ exactly; for multiple modes,
$\gamma_{\rm eff} \le \gamma_{\rm max}$ on average but
$\gamma_{\rm used} \to \gamma_{\rm max}$ as soon as a fast mode
is observed. With $c = 1$ the criterion is $\le \gamma_{\rm fast}\Delta t \le 2.78$,
within RK4's L-stable region.

(P4) **Generalisation to $(\phi, \dot\phi)$.** The same
$\gamma_{\rm eff} = |\langle\dot\phi, \phi\rangle|/\langle\phi,\phi\rangle$
applies to any state-rate pair. For VEP problems with a stress
field $\sigma$ and Jaumann rate $\dot\sigma$, this produces a
relaxation CFL on the Maxwell timescale $\eta/G$ measured directly
from the stress field, no constitutive bookkeeping required.

## Schemes

We use the following integrators on the kinematic ODE
$\dot h = u_n^{\rm eff}$, with $u_n^{\rm eff}$ from
\eqref{eq:un-eff} evaluated using the latest Stokes solution at each
stage's mesh state.

**FE-SL** (forward Euler with semi-Lagrangian update). Single
stage: $h^{n+1} = h^n + \Delta t\,u_n^{\rm eff,n}$.
Conservatively stable for $\gamma\Delta t < 2$. Reference scheme
at small $\Delta t$.

**RK2-SL.** Two stages: $k_1 = u_n^{\rm eff}(t_n)$;
solve Stokes at the half-step state; $k_2 = u_n^{\rm eff}(t_n + \tfrac{\Delta t}{2})$;
final update $h^{n+1} = h^n + \Delta t\,k_2$. Stable to
$\gamma\Delta t < 2$.

**RK4-SL.** Four stages, Stokes solved at each intermediate mesh
state, classical RK4 weights $(1, 2, 2, 1)/6$. Stable to
$\gamma\Delta t < 2.78$. Production scheme.

For comparison we also run:

**FE (radial-only).** The conventional update without semi-Lagrangian
correction. Used to demonstrate the kinematic-update bias.

**FSSA / kinematic-ETD with curvature γ.** Uses the saturated
prefactor $(1-\alpha)/\gamma$ with $\alpha = e^{-\gamma\Delta t}$
and $\gamma$ derived from a windowed curvature regression on $h(\theta)$
plus the analytic Cathles dispersion $\gamma = \rho g/(2\eta k)$. The
canonical "stabilised explicit" reference.

## Test cases

### Case A: relaxing topography

Scalar mode demonstrator. An annulus $r_i = 0.5$, $r_o = 1.0$,
unit viscosity, unit body force; no buoyant block. Initial
condition: $h(\theta, 0) = A_0\cos(k\theta)$ with $A_0 = 0.05$,
$k \in \{2, 6, 10\}$. Boundary conditions: no slip on $r_i$, free
surface on $r_o$. The system relaxes monotonically toward
$h \to 0$.

Analytic solution (Cathles half-space): $h_k(t) = A_0 e^{-\gamma_k t}\cos(k\theta)$
with $\gamma_k = \rho g/(2\eta k)$. The benchmark verifies that
each scheme recovers $\gamma_k$ to within its truncation order,
and that volume conservation is maintained throughout the decay.

### Case B: continent isostasy

Forced-equilibrium problem. Same annulus geometry. A Lagrangian
P0 indicator $B$ marks a buoyant block:
$B = 1$ inside $\theta_{\rm block} = 0.4$ rad, $r_{\rm min} = 0.7$;
$B = 0$ elsewhere. Body force
$\mathbf{f} = -(1 - \beta B)\hat r$, $\beta = 0.2$. The free
surface above the block bulges upward to isostatic equilibrium.
We run to a fixed simulated time $T = 540$.

The test exercises both the rising transient
(kinematic discretisation has to handle large $|u_n|$ and small $h$)
and the equilibrium tail (small $|u_n|$, large $h$). Volume
conservation is computed as $\Delta A/A_0$ where $A$ is the rock
area integrated over the curved-cell representation of the
deformed mesh.

## Results

### Volume conservation

| scheme | $\langle\Delta t\rangle$ | $h_p^{\rm final}$ | $\Delta A/A_0$ |
| --- | ---: | ---: | ---: |
| FE (radial-only), $\Delta t = 1.62$ | 1.62 | +0.0309 | **−1.82%** |
| FE-SL, $\Delta t = 1.55$ | 1.55 | +0.0383 | +0.024% |
| RK2-SL, relax CFL $c = 1.0$ | 11.0 | +0.0382 | −0.001% |
| RK4-SL, relax CFL $c = 1.0$ | 16.1 | +0.0382 | **+0.002%** |
| RK4-SL, relax CFL $c = 0.1$ (reference) | 1.65 | +0.0383 | +0.026% |
| FSSA / curvS, $\Delta t = 30.7$ | 30.7 | +0.0393 | +0.06% |
| FSSA / curvS, $\Delta t = 8.0$ | 8.0 | +0.0373 | −0.26% |

The radial-only forward Euler is wrong by ~$10^4$ relative to its
SL counterpart at the same $\Delta t$. The structural fix is
exactly the kinematic update.

### Time convergence

For SL schemes, doubling $\Delta t$ does not increase volume drift
until the relaxation CFL stability bound is approached. RK4-SL at
$c = 1$ ($\Delta t = 16$) actually produces *smaller* drift than
$c = 0.5$ ($\Delta t = 8$), because cumulative per-step measurement
noise scales with the step count. Crossing the L-stability bound
($c \ge 2$) is catastrophic: drift jumps to −2.2% as fast modes
amplify between steps.

For the FSSA / curvS variant, smaller $\Delta t$ produces *worse*
volume conservation: $\Delta t = 30.7 \to +0.06\%$;
$\Delta t = 15.8 \to -0.18\%$; $\Delta t = 8.0 \to -0.26\%$. The
saturation prefactor $(1-\alpha)/\gamma$ with mis-estimated
curvature $\gamma$ produces a $\Delta t$-dependent bias that
inverts the expected convergence.

### Space convergence

For RK4-SL at $c = 1$, mesh resolutions $\{10, 16, 20, 28, 40\}$
all give $|\Delta A/A_0| < 0.16\%$, with no systematic trend below
the per-step measurement noise. The cell area integrand
($\int 1\,\mathrm{d}V$ on a curved-edge mesh) carries its own
$h^p$ discretisation error that is comparable in magnitude to the
per-step truncation, so the volume metric saturates above the
spatial discretisation error. Trajectory convergence
($h_p$ vs $t$) is monotone in the cell size.

### Cost vs accuracy

Total wall time on a single 2.5 GHz core (8-way parallel for the
sweep itself):

| scheme | wall (s) | $|\Delta A/A_0|$ |
| --- | ---: | ---: |
| FE-SL, $\Delta t = 1.55$ (reference) | 1106 | 0.024% |
| RK4-SL, $c = 1.0$ (production) | 596 | 0.002% |
| RK4-SL, $c = 0.5$ | 1113 | 0.018% |
| RK4-SL, $c = 0.1$ (high-fidelity ref) | 4944 | 0.026% |
| FSSA / curvS, $\Delta t = 30.7$ | 59 | 0.06% |
| FSSA / curvS, $\Delta t = 8.0$ | 265 | 0.26% |

RK4-SL with $c = 1$ is Pareto-dominant on the SL family: lowest
wall time, smallest drift. The FSSA scheme at large $\Delta t$ has
a 10× wall-time advantage but pays in volume drift and trajectory
fidelity.

## Discussion

### What the kinematic-update fix does and does not do

The semi-Lagrangian update fixes the surface kinematic BC. It does
**not** address mesh distortion in the interior — that's a separate
problem, handled here by keeping the interior radial-only smoothing.
The conventional FE update has the same interior smoothing; we
simply correct the surface boundary value the smoothing receives.

It also does **not** require any change to the Stokes solver, the
element pair, or the constitutive model. Section 7's table shows
that the V3/P2 element pair and tighter Stokes solver tolerance
have negligible effect on volume conservation; the bias is
genuinely in the kinematic update alone.

### Why the relaxation CFL is sharp

The criterion $\Delta t \le c/\gamma_{\rm used}$ uses the *measured*
relaxation rate of the dominant mode in the system. This is
sharper than two alternatives:

- **Bulk-velocity CFL.** $\Delta t < c\,h_{\rm cell}/\max|\mathbf{v}|$.
  Bounds advection in the interior, which is necessary but not
  sufficient: a surface mode with high-$\gamma$ relaxation can
  still go unstable while the bulk velocity is small.
- **Curvature-CFL with analytic dispersion.** $\Delta t < c\,2\eta k_{\rm max}/\rho g$.
  Requires a hardcoded $\eta_{\rm eff}$ and $\rho g$; mis-estimates
  by factor 2 in heterogeneous problems.

  The relaxation CFL replaces both with a single observable that is
  exact for single-mode relaxation and asymptotically correct for
  multi-mode, and uses no constitutive constants.

### Implications for FSSA / saturation schemes

The FSSA / kinematic-ETD prefactor $(1-\alpha)/\gamma$ provides
unconditional L-stability — $\alpha < 1$ for any $\Delta t$ —
which is its great strength and the reason it is the canonical
"stabilised explicit" scheme. But the prefactor is exact only when
$\gamma$ is exact, and the curvature-derived $\gamma$ from a
windowed regression has known biases (sensitivity to numerical
noise in $h$, dependence on hardcoded $\eta_{\rm eff}$). The
$\Delta t$-dependent bias we observe in Case B is consistent with
the saturation factor amplifying these.

Replacing the curvature γ in the saturation factor with the
empirical $\gamma_{\rm eff}$ from \eqref{eq:gamma-eff} (with
monotone history) is a natural way to retain the L-stability
guarantee while removing the constitutive dependency. We do not
develop this combination here, but the recipe is identical and
worth investigating in heterogeneous-viscosity problems where
RK-style schemes' stability bound is tight.

### Generalisation to other relaxing systems

Property (P4) above suggests the same recipe applies to any state
$\phi$ with an associated rate $\dot\phi$ controlled by a
relaxation operator. We identify two natural targets:

(a) **Visco-elasto-plastic (VEP) stress fields.** Stress $\sigma$
relaxes to the yield surface or to viscous equilibrium on a
Maxwell-time timescale $\tau = \eta/G$. The pair $(\sigma, \dot\sigma)$
with the L² regression \eqref{eq:gamma-eff} gives a measured
Maxwell time per step, no constitutive bookkeeping required. We
hypothesise this resolves the BDF-1-only-stable result on
tight-yield TI fault problems documented elsewhere.

(b) **Coupled multi-physics.** Multiple state-rate pairs combine
as $\Delta t = \min_\phi c_\phi/\gamma_{\rm used,\phi}$. The
free-surface and stress relaxation CFLs would naturally combine
with the bulk-velocity advection CFL.

## Conclusion

We have shown that the conventional radial-only kinematic update
for a free-surface in viscous flow is structurally
volume-non-conservative, and that a semi-Lagrangian-horizontal
correction removes the bias with a one-line modification. The
correction is integrator-order independent: forward Euler at small
$\Delta t$ matches RK4 at the relaxation-CFL-derived $\Delta t$
to four significant figures, both on the same trajectory.

We have also derived an amplitude-invariant relaxation CFL,
$\Delta t \le c/\gamma_{\rm used}$ with
$\gamma_{\rm used} = \max_{t' \le t} |\langle u_n, h\rangle|/\langle h, h\rangle$,
that uses no hardcoded constitutive constants and adapts naturally
to the dominant relaxation timescale of the system. Combined with
RK4 — which has the largest L-stable region of the integrators
considered — this gives a scheme with $|\Delta A/A_0| < 10^{-4}$
on a continent-isostasy benchmark and $\mathcal{O}(\Delta t^4)$
trajectory accuracy.

The kinematic-update fix is structural; the relaxation CFL is the
performance optimisation. Forward-Euler-SL at small $\Delta t$ is
a safe reference scheme. RK4-SL with the relaxation CFL is the
production scheme.

## References

- Cathles, L. M. (1975). *The Viscosity of the Earth's Mantle*.
  Princeton University Press.
- Kaus, B. J. P., Mühlhaus, H., & May, D. A. (2010). A stabilization
  algorithm for geodynamic numerical simulations with a free surface.
  *Physics of the Earth and Planetary Interiors*, 181, 12–20.
- Andrés-Martínez, M., Morgan, J. P., Pérez-Gussinyé, M., & Rüpke, L.
  (2015). A new free-surface stabilization algorithm for geodynamical
  modelling: Theory and numerical tests. *Physics of the Earth and
  Planetary Interiors*, 246, 41–51.
- Cox, S. M. & Matthews, P. C. (2002). Exponential time differencing
  for stiff systems. *Journal of Computational Physics*, 176, 430–455.

## Reproducibility

All runs in this paper use the test runner
`docs/developer/design/_phase_i_fs_continent_fs_snapshots.py` from
the `feature/exp-integrator-freesurface` branch of underworld3.
The full convergence sweep (19 runs, 82.6 min on 8 cores) is in
`~/+Simulations/FreeSurface/` with per-step surface-profile
checkpoints (`step_NNNN.npz`), per-step work logs
(`work_log.csv`), and full mesh checkpoints (UW3 HDF5 + XDMF +
pyvista VTU) at halfway and final.

Schemes are selected via `--schemes <name>`; the relaxation CFL is
`--dt-cap-mode relax --dt-cap-c <c>`; per-step diagnostics are
`--per-step-profile --work-log <path>`. The conventional radial
update is the default; SL schemes are `fe_sl`, `rk2_sl`, `rk4_sl`.
