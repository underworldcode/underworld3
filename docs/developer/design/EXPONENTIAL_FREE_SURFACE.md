# Exponential time integration for free-surface evolution

> **Status**: investigation in progress.
> **Worktree**: `feature/exp-integrator-investigation` (vep-loading-unloading).
> **Companion**: `EXPONENTIAL_VE_INTEGRATOR.md` (the VE/VEP work that
> spawned this idea).

## What ETD really buys

Free-surface evolution in geodynamics is *always* a **relaxation
problem**: the surface moves toward an equilibrium configuration in
which deformation is balanced by gravitational restoring forces.
Decay isn't usually toward a flat state — internal buoyancy, slabs,
plumes, glacial loads, rotation rate changes all set a non-trivial
equilibrium $h_{\mathrm{eq}}$. But the *structure* is universal:
$\dot h \to 0$ as $h \to h_{\mathrm{eq}}$, and the rate of approach
is set by a viscous-gravitational timescale $\tau$ that depends on
local geometry, viscosity, and density jump.

The hard part of integrating this with a forward-Euler kinematic
update $h^{n+1} = h^n + \Delta t\,u_n$ is twofold:

1. **Drunken-sailor instability** when $\Delta t > 2\tau$. The
   surface overshoots equilibrium by more than the perturbation;
   the next step overshoots in the opposite direction with larger
   amplitude.
2. **Wasted work near equilibrium.** As $h \to h_{\mathrm{eq}}$,
   $u_n \to 0$ and the FE displacement $\Delta t\,u_n$ becomes
   small. We'd *like* to take large $\Delta t$ here and converge
   quickly; instead we're constrained by the $2\tau$ stability
   limit at the local rate, even when nothing physically interesting
   is happening.

The kinematic exponential update

$$
\boxed{\;
h^{n+1} \;=\; h^n \;+\; \frac{1 - e^{-\gamma\Delta t}}{\gamma}\,u_n
\;}
$$

solves both. Two limits:

- $\gamma\Delta t \to 0$: $(1-\alpha)/\gamma \to \Delta t$ —
  recovers FE.
- $\gamma\Delta t \to \infty$: $(1-\alpha)/\gamma \to 1/\gamma$ —
  surface increment is bounded by $u_n/\gamma$ regardless of how
  big $\Delta t$ is. The step *saturates* at $h^{n+1} = h^n + u_n/\gamma$,
  i.e. it jumps directly to the local linearised equilibrium
  estimate.

This is what makes the integrator useful: it lets $\Delta t$ grow as
the system approaches equilibrium, without going unstable.

The remaining question is **how to estimate $\gamma$** in the
general case.

## The role of FSSA

The standard stabilization in geodynamics codes is the **Free-Surface
Stabilization Algorithm (FSSA)** of Kaus, Mühlhaus & May (2010): a
Robin-type natural BC $\tfrac{1}{2}\rho g\,\Delta t\,(\mathbf{u}\cdot\hat n)\,\hat n$
added to the Stokes weak form on the free surface. FSSA is *additive*
to ETD — it modifies the Stokes BVP so the velocity returned at each
step is more consistent with the upcoming surface motion. ETD is then
the integrator that uses that velocity. The pair is the natural
deployment, and the experiments below show FSSA + ETD performing as
well as or better than either alone.

## Physics setup

For an isoviscous half-space below a free surface with density jump
$\Delta\rho$ to vacuum, gravity $g$:

$$
\gamma(k) = \frac{\rho g}{2\eta |k|}
\qquad\Rightarrow\qquad
\tau(k) = \frac{2\eta |k|}{\rho g}
$$

Short wavelengths relax slowly ($\gamma \to 0$ as $|k| \to \infty$), long
wavelengths also relax slowly for a *finite* layer ($\gamma \sim k^2 H^2$ for
$kH \ll 1$). The peak relaxation rate occurs at $kH \sim O(1)$, which
is where drunken-sailor bites hardest.

**Local $\tau$ from curvature.** For $h(s) = A\cos(k_s\,s)$ with arclength
$s$:

$$
\frac{\partial^2 h}{\partial s^2} = -k_s^2 h
\quad\Rightarrow\quad
k_s^2 = -\frac{\partial_s^2 h}{h}
$$

So curvature-over-amplitude is a local proxy for $k^2$. It's noisy at
zero crossings of $h$, and second differences amplify mesh noise, but
it doesn't require a Fourier basis or knowledge of the dominant
wavelength.

In an annulus with arclength $s = r\theta$, $\partial^2/\partial s^2 = (1/r^2)\,\partial^2/\partial\theta^2$, so
the discrete estimator on the upper boundary is:

$$
k_s^2(\theta) \approx \frac{1}{r_o^2}
\frac{|\partial^2_\theta h|}{|h| + \varepsilon}
$$

with $\varepsilon$ regularizing zero crossings.

## Phase I-2D: scalar-mode demonstrator

**Setup.** Annulus, $r_i = 0.5$, $r_o = 1$, viscous Stokes ($\eta = 1$),
body force $-\hat r$, perturbation $h(\theta, 0) = 0.05\sin(10\theta)$ on the upper
surface. Time-stepping with the AnnulusND_FS workflow (Stokes + Poisson
diffuser + mesh deformation), tracking the mode-10 projection of the
surface displacement $A(t)$ as a scalar diagnostic.

**Schemes.** All four combinations of:
- FSSA on/off — Robin BC $\tfrac{1}{2}\rho g\,\Delta t\,(\mathbf{u}\cdot\hat n)\,\hat n$ on Upper
- Update FE/ETD — FE: $x \leftarrow x + \Delta t\,v_{\text{diffused}}$. ETD: project boundary
  $\delta r$ onto $\sin(10\theta)$, update mode amplitude with
  $A^{n+1} = e^{-\Delta t\,\gamma_{\text{local}}}A^n$ using $\gamma_{\text{local}} = -\dot A/A$,
  reconstruct, diffuse into interior.

**Result at $\Delta t = \texttt{estimate\_dt()}$ ($\Delta t/\tau \approx 0.25$):**

| scheme | $A_{\text{final}}$ after $t\approx 45$ | $A_{\max}$ final | behaviour |
| --- | ---: | ---: | --- |
| FE only | -5.21e-2 | 5.48e-2 | sign-flip + grew past $A_0$ |
| FE + FSSA | -4.01e-2 | 4.23e-2 | sign-flip, only mild damping |
| ETD only | -1.28e-2 | 1.27e-2 | clean decay to ~0, fallback overshoot |
| ETD + FSSA | -1.13e-2 | 1.12e-2 | same as ETD only |

$A_0 = 5.0\times 10^{-2}$. Plot: `output/phase_i2d_fs_etd_dtf1.0_n8.png`.

**Reading.**

1. *FSSA alone does not prevent drunken-sailor at this $\Delta t$.* The FE
   advection step is still too large; FSSA only fixes the Stokes BVP,
   not the time-step.
2. *ETD on the surface mode update kills drunken-sailor* even without
   FSSA. The constant-magnitude FE bite is replaced by a multiplicative
   $e^{-\Delta t/\tau}$, which respects sign and decays through zero.
3. *ETD + FSSA matches ETD only*. They're not redundant in principle
   (FSSA fixes Stokes, ETD fixes the integrator), but in this regime
   the ETD update alone gets within a few percent of the with-FSSA
   case.
4. *Caveat: the demonstrator cheats* by using a single dominant mode
   with a clean scalar $\tau$. Multi-mode initial conditions and lateral
   viscosity contrast are the real test.

**Code:**
- `docs/developer/design/_phase_i_fs_etd_annulus.py` — runner
- `docs/developer/design/_plot_phase_i_fs_etd.py` — plotter
- `docs/developer/design/_phase_i_freesurface_relaxation_0d.py`
  (in vep-two-stokes worktree) — 0-D ODE precursor

## Phase I-2D-curv: curvature-derived $\tau$

The scalar-mode estimator only works because the original test had a
single dominant mode. The publication-track form needs a local $\tau$
that doesn't presume modal structure.

**The estimator (final form):**

For $h(\theta)$ sampled on upper-boundary nodes, ordered by $\theta$, compute
$\partial^2 h/\partial\theta^2$ by central periodic FD. Then per-node, in a sliding window
of $\pm W$ neighbours ($W = 4$ works well at the resolution tested),
estimate $k^2$ by *regression*:

$$
k_s^2(\theta_i) \approx -\frac{\sum_{j \in W} h''(\theta_j)\,h(\theta_j)}
                              {\sum_{j \in W} h(\theta_j)^2}
\quad / \quad r_o^2
$$

This is robust at zero-crossings of $h$ because both numerator and
denominator vanish in step (for a smooth signal with definite
wavelength), and the ratio remains well-defined. The naive pointwise
form $|\partial^2 h/h|$ is fragile near zeros and gave bad results on the
viscosity-contrast test (eps regularization picked up wrong values).
The windowed regression is the publication-track form.

Local rate: $\gamma(\theta) = \rho g / (2\,\eta(\theta)\,|k_s(\theta)|)$ (half-space dispersion).
Per-node ETD: $h^{n+1}(\theta) = e^{-\Delta t\,\gamma(\theta)}\,h^n(\theta)$. The increment is
Fourier-decomposed ($a_m$, $b_m$ up to $n_{\text{modes}} \approx N_{\text{boundary}}/3$) and
re-inserted as a sympy expression on the diffuser BC, propagating
the displacement smoothly into the interior mesh.

**Single-mode IC test ($\Delta t/\tau \approx 0.25$, mode 10):**

| scheme | $A_{\max}$ final | $A/A_0$ ratio per step | comment |
| --- | ---: | ---: | --- |
| FE | 5.48e-2 | linear | drunken sailor, sign-flip |
| FE + FSSA | 4.23e-2 | linear | drunken sailor, mild damping |
| ETD scalar | 1.27e-2 | exp + fallback | clean, then zero-crossing artifact |
| **ETD curv** | **5.02e-3** | **0.749 (constant)** | **clean exponential, no fallback** |

Inferred $\tau$ from curvature ETD: 19.4. Predicted half-space $\tau(k=10) = 20.0$.
Agreement within 3%. The curvature estimator recovers the modal $\tau$.

**Multi-mode IC ($0.025\sin(10\theta) + 0.0125\sin(25\theta)$):**

| scheme | $A_{\text{mode}}$ final | $A_{\max}$ final | comment |
| --- | ---: | ---: | --- |
| FE | -1.52e-1 | 1.67e-1 | drunken sailor, blow-up |
| FE + FSSA | -9.65e-2 | 1.09e-1 | drunken sailor with FSSA |
| ETD scalar | -8.83e-2 | 9.85e-2 | mode-10 estimator broken by mode-25 |
| ETD scalar + FSSA | -6.08e-2 | 7.12e-2 | same |
| **ETD curv** | **+5.93e-5** | **8.19e-5** | **clean monotonic decay** |

This is the headline result. The scalar-mode ETD **fails** when the IC
isn't a single mode — its $\tau$ estimator ($-A/\dot A$ on the dominant
mode projection) gets the wrong rate when other modes contribute. The
curvature-windowed estimator handles it cleanly because each location
sees the locally-dominant wavelength and uses its own $\tau$.

**Lateral viscosity contrast ($\Delta t/\tau_{\text{local}} \approx 10$ in weak zone):**

$\eta$ drops 20× across a Gaussian window centred at $\theta=0$ (half-width 0.4
rad). At dt-factor=10 ($\Delta t = 3.3$ after estimate_dt scaling):

| scheme | $A_{\max}$ final | blow_up |
| --- | ---: | --- |
| FE | 6.24e-1 | YES |
| FE + FSSA | 3.89e-1 | no (but $>7\times$ initial) |
| ETD scalar | 2.22e-16 | no (collapsed via fallback) |
| ETD scalar + FSSA | 2.89e-2 | no |
| **ETD curv** | **1.26e-2** | **no** |

Curvature ETD remains bounded in a regime where FE blows up and
FE+FSSA is not stabilizing the FE step. The factor of ~30 in stable
$\Delta t$ is the practical win.

**Plots:** `output/phase_i2d_fs_etd_*.png` (one per IC × visc setting).

## Reading

1. **ETD on the boundary advection is the right level of abstraction.**
   The drunken-sailor instability has two sources: (i) the Stokes
   solve giving an inconsistent velocity (FSSA fixes this), and (ii)
   the FE step on the boundary being too large for the local
   relaxation timescale. FSSA addresses only the first; for the
   second you need either smaller Δt or an exponential update.

2. **The curvature-windowed estimator works.** A pointwise
   $|\partial^2 h/h|$ ratio is too fragile at zero-crossings, but a windowed
   regression gives a robust local $k$ that tracks the dominant
   wavelength even under multi-mode and lateral-$\eta$-contrast
   conditions.

3. **ETD-curv is a drop-in for the FE step in the FSSA workflow.**
   No changes to Stokes, just the surface-advection and mesh-deform
   sub-step. The diffuser-based interior propagation is preserved.
   FSSA on/off makes essentially no difference once curvature ETD
   is in place — they would be complementary in principle (FSSA
   stabilizes Stokes, ETD stabilizes the integrator), but in
   practice the integrator fix subsumes most of the benefit.

4. **The estimator is parameter-thin.** Window half-width $W$ is
   the only tunable. For $N_{\text{boundary}} = 76$ and mode 10, $W = 4$
   gave clean results. The lower bound $k_s \geq 1/r_o$ is the only
   regularization. No $\varepsilon$ for zero crossings (not needed with the
   regression form).

## Phase I-2D-buoyancy: the actual geodynamics case

**Geodynamics free surfaces are always driven.** The surface moves
because there is flow towards an equilibrium that is itself
determined by interior dynamics — gravitational potential
redistribution, buoyant plumes, descending slabs, changes in
rotation rate, glacial loads, eustatic forcing. The pure-relaxation
problem (initial perturbation, no internal forcing) is rare in
production geodynamics; it's a useful pedagogical case for showing
the drunken-sailor instability in isolation, but it isn't the
working regime.

The kinematic BC at the surface is the same in either case:

$$\dot h = u_n^{\text{(surf)}}$$

where $u_n^{\text{(surf)}}$ is the Stokes-solve velocity normal to
the surface. In the pure-relaxation case, $u_n \approx -\gamma h$
to leading order; in the driven case, $u_n$ contains both the
relaxation-toward-equilibrium part and the bulk-forcing part.

### The kinematic ETD update — the workhorse

The integrator that handles both regimes uniformly is

$$
\boxed{\;
h^{n+1} = h^n + \frac{1 - e^{-\gamma\Delta t}}{\gamma}\, u_n^{\text{(surf)}}
\;}
$$

where $\gamma$ is the local relaxation rate from the
mean-curvature wavenumber estimator. Two limits make this concrete:

- $\gamma\Delta t \to 0$: $(1-\alpha)/\gamma \to \Delta t$, recovering forward-Euler advection.
- $\gamma\Delta t \to \infty$: $(1-\alpha)/\gamma \to 1/\gamma$, the surface saturates at the
  driven steady state $h_{\mathrm{eq}} = u_n/\gamma$. Drunken-sailor
  is impossible because the prefactor cannot grow beyond $1/\gamma$.

This is the same structural pattern as the VEP exponential update
that motivated the whole investigation: an integrating-factor
prefactor that turns an explicit step into a bounded one whose
limit is the local steady state.

### Pure-relaxation limit (Form A): a tutorial case

In the pure-relaxation special case where $u_n = -\gamma h$ exactly,
the kinematic ETD reduces to

$$h^{n+1} = h^n + \frac{1-\alpha}{\gamma}\,(-\gamma h^n)
       = h^n - (1-\alpha)h^n = \alpha h^n.$$

Substitute and confirm: it's just $h^{n+1} = e^{-\gamma\Delta t}h^n$,
the closed-form exponential decay. This is a useful demonstration
that the scheme respects the analytical relaxation curve when no
forcing is present, but it isn't a separate method — it's the
$u_n = -\gamma h$ limit of the kinematic ETD. We refer to it as the
"closed-form" or "homogeneous-limit" form for clarity, but the
implementation is one rule: the boxed update above.

### Why the homogeneous-limit form fails on driven problems

If the implementation hard-codes $u_n = -\gamma h$ instead of
consulting the actual Stokes velocity, then on any driven problem
the scheme misses the forcing entirely. With $h(0) = 0$ and a
buoyant interior, the curvature is zero, $\gamma h$ is zero, and
the scheme predicts no surface motion — even though the bulk flow
is pushing the surface upward. This is exactly what the buoyancy
test below shows; it isn't a bug, it's the expected behaviour of
that closure choice on a problem outside its assumed regime.

### Test: buoyant blob in the annulus

Setup: same annulus geometry, but with $h(0) = 0$ (no initial
perturbation) and an internal density anomaly — a Gaussian buoyant
blob at radius 0.7, angle 0, $\sigma=0.08$, magnitude 0.6 of background
gravity. Body force becomes $-(1 - \text{blob})\,\hat r$. The blob rises and
deforms the surface above it.

| scheme | $A_{\max}$ final ($t\approx 160$) | behaviour |
| --- | ---: | --- |
| FE | 3.11e-1 | linear growth, would blow up |
| FE + FSSA | 1.63e-1 | linear growth at half rate |
| Closed-form / homogeneous-limit ETD | 1.11e-16 | misses forcing (by construction) |
| **Kinematic ETD (workhorse form)** | **5.77e-2** | **bounded, saturating** |
| BDF-2 (homogeneous-limit form) | 1.11e-16 | misses forcing (by construction) |
| ETD-2 (homogeneous-limit form) | 1.11e-16 | misses forcing (by construction) |

Plot: `output/phase_i2d_fs_buoyancy.png`.

### Reading

The buoyancy test is the actual geodynamics regime — surface
driven by interior dynamics, not relaxing from a perturbation
imposed by hand. Three observations:

- **The kinematic ETD** integrates the Stokes velocity with the
  $(1-\alpha)/\gamma$ saturation factor. Surface displacement
  remains bounded toward the driven steady state. This is the
  workhorse form.
- **FE / FE+FSSA** integrate the Stokes velocity directly with
  no saturation, so they grow linearly in time without bound
  (slower with FSSA's Robin term, but still unbounded).
- **The closed-form / homogeneous-limit** schemes hard-code
  $u_n = -\gamma h$ and never consult the actual Stokes
  velocity. With $h(0) = 0$ and zero curvature, they predict
  no motion — by construction, not by failure. They are the
  $u_n = -\gamma h$ limit of the workhorse form, useful for
  visualising the pure-relaxation case but not for production.

### Implication for the publication

The kinematic ETD update

$$
h^{n+1} = h^n + \frac{1 - e^{-\gamma\Delta t}}{\gamma}\,u_n^{\text{(surf)}}
$$

is the proposed scheme. Two-piece narrative for the paper:

1. **Pedagogical case (initial perturbation, no internal forcing).**
   Show drunken-sailor in FE / FE+FSSA on single-mode, multi-mode,
   and lateral-$\eta$-contrast tests. Demonstrate that the kinematic
   ETD (which in this regime reduces to the closed-form
   $h^{n+1} = \alpha h^n$) decays cleanly along the analytical
   relaxation curve at any $\Delta t$. This isolates the
   integrator question from the forcing question.

2. **Production case (initial flat, bulk forcing).** Buoyant blob
   in the annulus interior, driven surface uplift. FE / FE+FSSA
   grow without bound. The kinematic ETD saturates at the driven
   steady state. This is the regime production geodynamics codes
   actually run in — convection, subduction, glacial unloading,
   self-gravitation — and it's where the new scheme earns its keep.

The closed-form / homogeneous limit appears in the discussion
only as the special case where $u_n = -\gamma h$ holds exactly,
and as a sanity check that the kinematic ETD respects the
analytical relaxation curve when forcing is absent. It isn't a
separate method.

## Order-2 variants (BDF-2 and predictor-corrector ETD-2)

Two second-order extensions of the curvature-τ scheme were tested for
completeness:

- **BDF-2 (curv-$\tau$).** Per node:
  $h^{n+1}(\theta) = (4 h^{n}(\theta) - h^{n-1}(\theta)) /
  (3 + 2\Delta t \,\gamma_n(\theta))$, with $\gamma_n$ from current
  curvature. First step bootstraps with backward Euler (BDF-1).
- **ETD-2 (predictor-corrector).** Predict
  $h^* = e^{-\Delta t\,\gamma_n}\,h^n$, recompute curvature $\to \gamma^*$, average
  $\gamma_{\text{avg}} = (\gamma_n + \gamma^*)/2$, correct: $h^{n+1} = e^{-\Delta t\,\gamma_{\text{avg}}}\,h^n$.
  This is a 2-stage exponential-RK form that captures
  rate-variation across the step.

| scheme | single $A_{\max}$ | multi $A_{\text{mode}}$ | visc-V $A_{\max}$ |
| --- | ---: | ---: | ---: |
| ETD-1 curv | 5.02e-3 | +5.93e-5 | 1.26e-2 |
| BDF-2 curv | 5.02e-3 | -3.56e-5 | 1.28e-2 |
| ETD-2 curv | 5.02e-3 | +5.89e-5 | 1.26e-2 |

**Findings:**

1. *ETD-2 $\approx$ ETD-1 to ~5 significant figures.* Cox–Matthews ETD-2's
   correction terms vanish for the homogeneous problem
   $\dot h = -\gamma h$ (zero source $N(h,t)$); the predictor-corrector
   form's $\gamma$-averaging only matters when $\gamma$ varies mid-step, and the
   curvature $\tau$ doesn't change that fast over one $\Delta t$.
2. *BDF-2 stays sober.* No drunken-sailing, identical stability to
   BDF-1 (L-stable), with second-order truncation accuracy. The
   sailor gets slightly *more sober*, not more drunk: BDF-2 has
   smaller phase error than BDF-1 in problems with rapidly-varying
   solutions (e.g., load-induced surface motion).
3. *Order-2 doesn't reveal new physics here.* For pure linear
   relaxation with locally-frozen γ, ETD-1 is already exact along
   the integral curve. Order-2 corrections would matter for forced
   problems — moving load, oscillatory boundary perturbation,
   shoreline migration, eustatic forcing — anything where there's
   a non-trivial source term in the surface ODE.

**Decision:** ETD-1 curv is the publication-track scheme. BDF-2 curv
is included in the comparison plot for thoroughness; no need to
prefer it. ETD-2 documented as zero-cost upgrade if/when source
terms appear.

## Empirical $\gamma$ from history: dropping the dispersion assumption

The curvature-derived $\gamma = \rho g/(2\eta|\mathbf{k}|)$ assumes a
specific dispersion form (half-space) and requires knowing the local
material parameters $\eta, \rho g$. For real geodynamics applications
the dispersion may not be half-space (finite layer, layered
viscosity, compressibility, self-gravitation), and material
parameters can be heterogeneous in ways that aren't reflected in a
single $\eta$-times-curvature product.

A history-based estimator drops both assumptions while keeping the
exponential-interpolation form of the integrator intact.

### Estimator

The kinematic ODE is $\dot h = -\gamma h + s$. Two natural
empirical estimators of the slope $-\gamma$:

**Spatial regression (preferred).** At a single timestep, fit
$u_n = -\gamma h + s$ across a small window of neighbouring
boundary nodes by least-squares regression. $\gamma$ is minus the
slope; $s$ is the intercept (which we don't need but get for
free). No history required — the spatial samples themselves
provide the data points for the linear-response regression.

$$
\gamma_i^{\text{emp}} \;=\; -\frac{\sum_{j \in W_i} (h_j - \bar h)(u_{n,j} - \bar u_n)}
{\sum_{j \in W_i} (h_j - \bar h)^2}
\quad\text{(window means subtracted)}
$$

**Temporal regression.** Use $(h^{n-1}, u_n^{n-1})$ and
$(h^n, u_n^n)$ at each node:

$$
\gamma_i^{\text{emp}} \;\approx\; -\frac{u_n^{(n)}_i - u_n^{(n-1)}_i}{h^{(n)}_i - h^{(n-1)}_i}.
$$

Mathematically equivalent to the spatial form *if* the dynamics
are locally linear and steady. In practice the temporal version
has poor SNR because $\Delta h$ and $\Delta u$ over one timestep
can both be small, especially at the start of a buoyancy-driven
problem (small $h$, slowly-varying $u$). The spatial version
samples across nodes that are simultaneously at different
points on the linear-response curve (one window contains a full
mode-10 oscillation in our annulus tests at $W = 4$), giving
much better signal-to-noise.

We adopt the spatial form. Both have the same window-regression
form; the difference is whether the samples are temporal or
spatial.

Plug into the kinematic ETD update,

$$
h^{n+1} = h^n + \frac{1 - e^{-\gamma^{\text{emp}}\Delta t}}{\gamma^{\text{emp}}}\,u_n^n,
$$

and the integrating-factor structure, the small-$\gamma\Delta t$ FE
limit, and the saturation at $h_{\text{eq}} = u_n/\gamma$ are all
preserved. Only the source of $\gamma$ has changed.

### Why empirical beats curvature for production

- *No dispersion assumption.* Half-space, finite-layer, layered, or
  weakly nonlinear — $\gamma^{\text{emp}}$ tracks whatever rate is
  actually present in the data.
- *No material-parameter extraction.* Local $\eta$ and $\rho g$
  effects fall out of the slope automatically.
- *Self-adaptive to time-varying forcing.* As long as $s$ is
  approximately constant over the inter-step interval (or
  approximately linear, with three or more history steps), the
  estimator catches the local rate the system is currently
  exhibiting.

### Numerical robustness

At nodes where $\Delta h$ is small — near equilibrium, near zero
crossings of the perturbation — the ratio is noisy. Two fixes,
both transferred from the curvature-windowed-regression approach:

1. *Windowed regression.*
   $\gamma_i^{\text{emp}} \approx
   -\langle \Delta u_n, \Delta h\rangle_W /
    \langle \Delta h, \Delta h\rangle_W$
   over a small spatial neighbourhood $W$. Both numerator and
   denominator vanish together near equilibrium, so the ratio
   remains finite.
2. *Floor on $\gamma$.* A $\gamma_{\min}$ corresponding to the
   longest-wavelength mode the geometry supports prevents runaway
   when $\Delta h \to 0$.

### Bootstrap

First step has no history. Two options:

- *Curvature fallback.* Use the curvature-derived $\gamma$ on step
  one, switch to $\gamma^{\text{emp}}$ from step two onwards. Clean
  if the curvature machinery is in place.
- *FE seed.* Take a deliberately small first step in pure FE,
  accepting a small transient error to seed the $(h, u_n)$ history
  buffer. Independent of any dispersion assumption.

### More history → richer estimator

With three or more recent $(h, u_n)$ pairs per node, fit a local
linear model $u_n = -\gamma h + s$ by least-squares regression.
Recovers both $\gamma$ (slope) and $s$ (intercept), capturing
slowly-varying sources without the "constant $s$ over a step"
approximation. Worth doing if the forcing is fast — moving loads,
convective cells with significant time-variation, oscillatory
boundary perturbations.

### Implementation and test results

Implemented as the `empE` update mode in
`_phase_i_fs_etd_annulus.py` using spatial-regression $\gamma$.

**Cross-test summary** ($A_{\max}$ at final step, $\Delta t/\tau \approx 0.25$
or as noted):

| Test | analytical | curv | curvS | empE | FE+FSSA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-mode IC | 5.4e-3 | 5.0e-3 | 2.9e-2 | 1.3e-2 | 4.2e-2 |
| Multi-mode IC | n/a | 8.2e-5 | 6.4e-2 | 9.3e-2 | 1.1e-1 |
| Visc contrast (10·dt) | n/a | 1.3e-2 | 1.0e-1 | 3.3e-1 | 3.9e-1 |
| Buoyancy | n/a | 0 (by construction) | 5.8e-2 | 1.6e-1 | 1.6e-1 |

**Single-mode IC.** Both curv and empE recover the analytical
exponential decay; spatial regression gives $\gamma \approx 0.04$
matching the half-space modal value to 5%. empE bounces past zero
slightly more than curv but stays in the right regime.

**Multi-mode and visc-contrast.** empE degrades — its $A_{\max}$
sits between FE+FSSA and curvS, far from curv's clean decay.
The spatial regression picks up an *effective* $\gamma$ that
mixes contributions from multiple modes (or from spatially varying
$\eta$), and the kinematic update $(1-\alpha)/\gamma$ prefactor
amplifies small errors in $u_n$ by a factor $\Delta t$ when
$\gamma\Delta t \ll 1$. The closed-form ($\alpha h$) update is
not exposed to this amplification.

**Buoyancy.** empE essentially matches FE+FSSA (linear growth) —
it fails to deliver the saturation behaviour curvS achieves. The
problem is structural: at flat IC, $h \approx 0$ everywhere in
the window, the regression can't extract a slope, $\gamma$ falls
to the floor and the update is FE-like. After a few steps when
$h$ has been imprinted by the source, $h$ and $u_n$ become
spatially correlated *with the same sign* (both peaked above the
buoyant blob), the regression sees a positive slope and again
can't extract a relaxation rate.

**The structural lesson.** Spatial regression of $u_n$ vs $h$
implicitly assumes the linear-response form $u_n = -\gamma h + s$
where $s$ is *spatially constant*. When that assumption holds
(single-mode pure relaxation), the regression gives the right
$\gamma$. When $s$ is spatially correlated with $h$ (any
bulk-driven case where the source has imprinted itself on the
surface), the regression confounds source structure with
relaxation rate.

### Practical lessons from the implementation

- The temporal $-\Delta u/\Delta h$ form has bad SNR. With one
  timestep of history, $\Delta h$ is the per-step displacement
  (small) and $\Delta u$ is the per-step change in velocity
  (smaller still). The spatial form takes its samples from
  nodes that are *simultaneously at different states* on the
  linear-response curve — much better signal.
- The "floor" $\gamma_{\min}$ exists only to prevent
  division-by-zero from numerical noise. It must be much
  *smaller* than the smallest physical $\gamma$ in the
  problem. We initially conflated $\gamma_{\min}$ with the
  curvature estimator's $k_{\min} = 1/r_o$ floor, which forced
  $\gamma_{\min} = 0.5$ — way above the mode-10 $\gamma
  \approx 0.04$. Setting $\gamma_{\min} = 10^{-4}$ recovers
  correct behaviour.
- Spatial regression assumes $\gamma$ is roughly constant
  across the window. For lateral viscosity contrast, smaller
  windows preserve locality at the cost of regression
  conditioning. $W = 4$ (window of 9 nodes) gave a good
  balance for all our tests.

### Publication-track narrative (revised after empirical tests)

The spatial-regression empirical $\gamma$ works for the
single-mode validation case but is **not the production form**:
it confounds source spatial structure with relaxation rate, and
the $(1-\alpha)/\gamma$ prefactor amplifies $u_n$ noise at
small $\gamma\Delta t$. Better candidates:

1. **Curvature-derived $\gamma$ + kinematic ETD update.** The
   curvature gives a robust per-node $\gamma$ from the
   geometric structure of $h$, independent of $u_n$ contamination.
   The kinematic ETD update uses $u_n$ for the actual surface
   advection, with the curvature $\gamma$ supplying only the
   step-stabiliser. This is the form that worked across all
   four tests in this investigation (curv-style $\gamma$,
   curvS-style update with $u_n$).

2. **Empirical $\gamma$ from temporal regression with multi-step
   history.** Avoids the spatial-regression confounding: each
   node's $\gamma$ comes from the time-evolution of its own
   $(h, u_n)$ pair. Requires more history-buffer machinery and
   has SNR challenges at the start of a transient. Possible
   future direction.

3. **Hybrid: curvature for the prefactor, $u_n$ for the
   advection.** The simplest production scheme.

For the publication, lead with **kinematic ETD using
curvature-derived $\gamma$**. The empirical-$\gamma$ idea
is documented as an attractive next-step exploration that
turned out, on testing, to need more sophistication than a
single-shot spatial regression to be production-ready.

## What's still unproven / open

- **Long-term accuracy.** The tests above are short (8 large-dt
  steps). For a 100-step climate-of-relaxation run, do the per-step
  small errors accumulate? Need a longer-time run with a small-dt
  baseline.
- **Different dispersion.** Half-space dispersion $\gamma = \rho g/(2\eta|k|)$
  is hard-coded. For finite layers, $\gamma = \rho g \sinh(2kH)/(2\eta(2kH+\sinh(2kH)))$
  reduces at long wavelength. Should be straightforward to swap in.
- **3-D.** $\partial^2 h/\partial s^2$ becomes the surface Laplacian. Computing this
  via a submesh on the upper boundary is a UW3-API question, not a
  physics one.
- **Self-gravitation, loading.** Beyond linear relaxation. Needs
  an additive forcing term in the ETD update — analogous to the VE
  source-term update in the Phase B work.
- **Production deployment.** Currently a one-file demonstrator. To
  land in `uw.systems`, the curvature step needs to be a method on
  a free-surface mesh-deformation helper, with the $\tau$ estimator
  parameterized by viscosity expression and dispersion choice.

## Decision for the publication

The proposed integrator is the **kinematic exponential update**

$$
h^{n+1} = h^n + \frac{1 - e^{-\gamma\Delta t}}{\gamma}\,u_n^{\text{(surf)}}
$$

It is the natural generalisation of forward-Euler advection
($\gamma\Delta t \ll 1$ limit) that also handles
$\gamma\Delta t \gg 1$ gracefully by saturating at the local
driven equilibrium $h_{\mathrm{eq}} = u_n/\gamma$. This makes it
appropriate for the *general* free-surface evolution problem in
geodynamics: a relaxation toward equilibrium where the timestep
naturally grows as the system approaches steady state.

**$\gamma$ as the open knob.** Multiple estimators for $\gamma$
are possible. We have tested:

- *Curvature-derived $\gamma$* (half-space dispersion via mean
  curvature). Robust geometric estimator, no $u_n$ contamination,
  defensible whenever the local linear response is well-described
  by the half-space form.
- *Empirical $\gamma$ via spatial regression*. No dispersion
  assumption, but vulnerable to (i) small numerical components
  of $u_n$ that aren't captured by the linear-response model
  (mode-0 from pressure penalty, FSSA artefacts) and (ii)
  spatial-source confounding when forcing has imprinted itself
  on $h$.

The investigation has not yet identified a single $\gamma$
estimator that is best across all regimes. Curvature-derived
$\gamma$ is the most reliable starting point; better empirical
estimators (multi-step temporal regression, mode-filtering on
$u_n$) are open methodological work.

**FSSA is additive, not competing.** FSSA modifies the Stokes
solve to make $u_n$ at each step consistent with the upcoming
surface motion. The kinematic ETD then uses that velocity to
take a saturated step. The pair is the natural deployment.

**Section title for the paper**: "Exponential time-stepping for
the free-surface kinematic boundary condition: relaxation toward
equilibrium at arbitrary $\Delta t$."

## Working hypothesis for the publication angle

(Superseded by "Decision for the publication" above. Earlier
framing positioned ETD-curv as a refinement of FE advection
in pure-relaxation problems and asked whether it could match
FSSA's variational stabilisation. The buoyancy test settled the
question differently: the kinematic ETD update is the
generalisation of forward-Euler advection itself, and the
pure-relaxation case is a degenerate limit. The publication
narrative leads with the kinematic ETD as the actual scheme,
not with a comparison-against-FSSA framing.)

## Open questions

- **$\tau$ estimator robustness.** Does the curvature-derived form match
  the modal form to within (say) 10% for a well-resolved single mode?
  Out-of-the-box, or only after smoothing of $\partial^2 h/\partial\theta^2$?
- **Zero-crossing fallback.** What's the right behaviour when $h \approx 0$
  locally? Freeze, FE fallback, or something analytic?
- **3-D extension.** $\partial^2 h/\partial s^2$ becomes the surface Laplacian on the
  bounding manifold. Discussed in detail in the next section.
- **Coupling with mesh deformation.** Currently ETD is applied at the
  boundary, then the diffuser propagates the displacement increment
  inward. Is that the right thing, or should the ETD operator be
  applied to the whole mesh-deformation field with $\alpha(x)$ decaying
  inward?
- **Lagrangian particles.** If the free surface is tracked by
  particles rather than by mesh deformation (e.g. ALE or
  level-set), how does ETD plug in?

## Extension to 3D: mean curvature as the unifying scalar

The 1D surface curvature $\partial^2 h/\partial s^2$ used in the 2D
demonstrator is, after subtracting the unperturbed-circle background
$1/r_o$, the deviation of geometric curvature. In 1D-on-2D the
curvature is a scalar by dimensionality.

In 3D the analogue is **mean-curvature deviation**. The surface
Hessian has two principal curvatures $\kappa_1, \kappa_2$, and to
leading order in $h$,

$$2(H - H_{\text{ref}}) \;=\; \Delta_S h + O(h^2)$$

So the surface Laplacian *is* (twice) the mean-curvature deviation.
The estimator that worked in 2D,

$$|\mathbf{k}|^2 = -\frac{\Delta_S h}{h}\,,$$

generalises immediately. The half-space dispersion
$\gamma(|\mathbf{k}|) = \rho g/(2\eta|\mathbf{k}|)$ is already a
scalar function of $|\mathbf{k}|$, so once we have the surface
Laplacian the rest of the scheme — per-node $\gamma$, per-node
$\alpha = e^{-\gamma\Delta t}$, the source-aware update
$h^{n+1} = h^n + (1-\alpha)/\gamma\,u_n$ — is unchanged.

**Physical framing.** Mean curvature is the same scalar invariant
that drives Laplace pressure for a bubble, $\Delta P = 2\sigma H$.
For a viscous free surface under gravity the driving force comes
from gravitational pressure $\rho g h$ rather than surface tension,
but the *response* in both cases is wavenumber-dependent and
scales with mean curvature deviation. Gaussian curvature
($\kappa_1 \kappa_2$) doesn't enter the linearised dispersion in
either problem.

### Discretisation: the cotangent Laplacian

For a triangle mesh, the discrete Laplace–Beltrami operator at
vertex $i$ is

$$(\Delta_S h)_i = \frac{1}{2A_i}
   \sum_{j\in N(i)}
   \bigl(\cot\alpha_{ij} + \cot\beta_{ij}\bigr)(h_j - h_i)$$

where $\alpha_{ij}$ and $\beta_{ij}$ are the **interior angles
opposite the edge $ij$** in the two triangles sharing it, and
$A_i$ is the Voronoi/barycentric area at vertex $i$. This is
*exactly* the P1 FEM stiffness assembly — each edge's weight is
the sum of opposite-angle cotangents. The operator depends only
on edge lengths and angles; it is coordinate-free and intrinsic
to the embedded surface (Pinkall–Polthier / Meyer–Desbrun).

UW3 uses **geometrically linear elements** throughout (straight
edges, flat faces), even when field basis functions are higher
order — a deliberate restriction that enables fast point-location
and navigation shortcuts. The mesh-geometric quantities the
cotangent formula uses (edge lengths, opposite angles, vertex
areas) are therefore exact on UW3 surface meshes regardless of
the field-basis degree. No curved-element generalisation is
required.

### UW3 implementation: no new geometry code needed

The existing FEM Poisson machinery on a 2D submesh of the upper
boundary computes $\Delta_S h$ directly. The workflow:

1. Extract the upper boundary as a 2D submesh
   (`mesh._create_submesh` / `DMPlexFilter`-based — already used
   for the existing free-surface workflow).
2. On the submesh, declare $h$ as a scalar `MeshVariable`.
3. Ask for `mesh.div(mesh.grad(h))` symbolically; UW3's JIT
   codegen produces the surface-Laplacian operator with the
   correct (cotangent-equivalent) weights. The submesh inherits
   the metric from the embedding.
4. Evaluate at boundary nodes for the per-node $|\mathbf{k}|^2$
   estimator. Plug into $\gamma$, $\alpha$, source-aware update.

For the windowed-regression robustness (the 2D trick that handled
zero-crossings of $h$), apply the same logic per-node in 3D: take
a small neighbourhood (geodesic disk or $k$-nearest mesh
neighbours), compute
$|\mathbf{k}|^2 \approx -\langle \Delta_S h, h\rangle_W /
\langle h, h\rangle_W$. Both numerator and denominator vanish
together near zero crossings of $h$, and the ratio remains
well-defined.

### Suggested 3-D demonstrators

1. **Spherical shell, point-load relaxation.** Outer surface
   $r_o$, inner $r_i$, viscous Stokes, body force $-\hat r$.
   Initial $h = A_0\,\exp(-\theta_g^2 / 2\sigma^2)$ where
   $\theta_g$ is the great-circle angle from a chosen pole.
   Watch the Gaussian relax. Equivalent of the single-mode 2D
   test. Spherical-harmonic decomposition gives the analytical
   reference: each $Y_{\ell m}$ component decays at
   $\gamma_\ell = \rho g \ell/(2\eta)$ on a half-space-like
   approximation, which the curvature estimator should recover
   pointwise.
2. **Spherical shell, multi-scale IC.** Sum of two Gaussians
   (or two $Y_{\ell m}$ at different $\ell$). The 3D analogue
   of the multi-mode test that broke scalar-mode ETD in 2D.
3. **3-D Cartesian box, buoyant blob.** Forced/source-aware
   demonstrator. Box with periodic horizontal BCs, free upper
   surface, Gaussian buoyant blob in the interior. Tests the
   curvS form in 3D where the surface is genuinely 2D.

### Open complications worth flagging

- *Anisotropy.* For TI rheology (different viscosities in
  fault-parallel vs fault-normal directions) the response
  to a perturbation is direction-dependent. Mean curvature
  alone is insufficient — both principal curvatures matter
  and the dispersion becomes a tensor relation. Out of scope
  for the first paper; cleanly extends the same machinery.
- *Spectral baseline on the sphere.* Spherical harmonics give
  $\Delta_S Y_{\ell m} = -\ell(\ell+1)/R^2\, Y_{\ell m}$,
  which is a perfect ground-truth for the local estimator.
  Decompose $h$ in $Y_{\ell m}$, apply analytical
  $\gamma(\ell)$ per mode, compare against the per-node
  curvature-derived $\gamma$.
- *Per-quad cost.* In 2D we evaluated curvature at 76 boundary
  nodes; in 3D a typical mesh has $O(10^4)$ surface vertices,
  but each evaluation is local. The Stokes solve still
  dominates total cost — same as in 2D.

## Phase I-2D-integrator: RK family vs ETD-prefactor schemes

Following the kinematic-ETD investigation, we ran a structured
sweep comparing five integrators against the small-dt FE-noFSSA
reference (FE-noFSSA at $\Delta t = 0.05\,\Delta t_{\text{est}}$,
$n = 200$, $t_{\text{final}} = 18.4$, fitted
$\gamma_{\text{eff}} = 0.0457$, in agreement with the half-space
mode-10 prediction $\gamma = \rho g/(2\eta|k|) = 0.05$).

Five schemes:

1. **FE-noFSSA** — no corrections, stability by Δt alone.
2. **RK2 (no γ)** — pure midpoint method. Two Stokes solves per
   step. Stable for $\gamma\Delta t \le 2$.
3. **RK4 (no γ)** — classical 4-stage Runge-Kutta. Four Stokes
   solves per step. Stable for $\gamma\Delta t \le 2.78$.
4. **curvS-FSSA** — kinematic ETD with curvature-derived γ
   prefactor. One Stokes solve per step.
5. **midpoint-FSSA** — RK2-sampled $u_n$ combined with the
   curvature-γ ETD prefactor. Two Stokes solves per step.

For each scheme we ran at $\Delta t$-factor in
$\{1, 2, 5, 10, 20\}$ on the same internal-boundary mesh
(res=20, single-mode IC at $\sin(10\theta)$, amp 0.05). The
$\Delta t$-factor multiplies the PETSc estimate_dt, giving
$\gamma\Delta t \in \{0.09, 0.18, 0.46, 0.92, 1.84\}$ across
the sweep — spanning the FE-stable regime through to the
near-RK2-instability boundary.

Number of timesteps adjusts with $\Delta t$ so each run reaches
$t \in [60, 150]$, giving roughly 1–3 e-folds of mode-10
relaxation.

### Results

**Error vs reference** ($A_{\text{ref}}$ at each run's final $t$
from the fitted exponential):

| scheme | dtf=1 | dtf=2 | dtf=5 | dtf=10 | dtf=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FE-noFSSA | 1.4e-3 | 8.6e-4 | 1.4e-3 | 1.0e-3 | 9.9e-3* |
| RK2 (no γ) | 6.5e-4 | 5.4e-4 | 5.4e-4 | 1.6e-3 | 1.1e-2* |
| RK4 (no γ) | 7.5e-4 | 7.5e-4 | 2.4e-4 | 3.0e-4 | **3.5e-5** |
| curvS-FSSA | 7.4e-4 | 1.7e-3 | 4.7e-3 | 1.3e-2 | 1.7e-2 |
| midpoint-FSSA | 7.4e-4 | 2.7e-3 | 7.2e-3 | 1.7e-2 | 2.2e-2 |

\* drunken-sailor onset (mode-10 amplitude inflates beyond
reference).

**Final $A_{\max}$** (blow-up if $\gg$ initial 0.05):

| scheme | dtf=1 | dtf=20 |
| --- | ---: | ---: |
| FE-noFSSA | 3.5e-3 | 3.3e-2 (drunken-sailor) |
| RK2 | 3.8e-3 | **8.1e-2 (blown up)** |
| RK4 | 3.8e-3 | 2.5e-2 (bounded) |
| curvS-FSSA | 5.1e-3 | 1.9e-2 |
| midpoint-FSSA | 5.5e-3 | 2.2e-2 |

### Reading

1. **The kinematic ETD with curvature γ is a robustness hack,
   not an accuracy improvement.** curvS-FSSA / midpoint-FSSA
   never blow up across the dt range tested, but their
   trajectories *systematically overshoot* the reference at
   moderate-to-large $\gamma\Delta t$. Error grows from
   $\sim 7\times 10^{-4}$ at dtf=1 to $\sim 1.7\times 10^{-2}$
   at dtf=20 — accuracy degrades rapidly with $\Delta t$.

2. **RK4 is the accuracy winner at any cost.** Even at
   dt-factor=20 (γΔt ≈ 1.84) where pure FE drunken-sailors and
   RK2 has blown up, RK4 stays bounded *and* gives the
   smallest error against reference. At 16 Stokes solves total
   (dtf=20, n=4) RK4 gives error $\sim 4\times 10^{-5}$ — an
   order of magnitude better than RK2 at the same total cost.

3. **Midpoint-FSSA loses to curvS-FSSA.** Adding the RK2
   midpoint sampling on top of the (1-α)/γ ETD prefactor
   compounds the overshoot bias from the prefactor. The two
   "improvements" don't combine — the prefactor is already
   doing what midpoint sampling would do, and stacking them
   over-corrects. This was the priority-1 candidate in the
   handoff and is now empirically rejected.

4. **The ETD prefactor's value is L-stability**, not better
   step-by-step accuracy. It buys you absolute boundedness
   when γ is unknown / spatially varying / when you can't
   guarantee γΔt < 2. For homogeneous problems where the
   physics tells you γΔt is comfortable, RK4 with no γ
   estimate is strictly better.

### Decision (revised)

The publication-track scheme is **RK4 (no γ) for accuracy** when
the user can ensure $\gamma\Delta t < 2.78$, with **curvS-FSSA
as the safe-default fallback** when γ is unknown. The midpoint
hybrid is dropped.

Cost analysis: RK4 takes 4× Stokes solves per step vs FE's 1×,
but admits 4× larger Δt while remaining accurate and stable.
Net cost-per-target-accuracy is *lower* for RK4 across the
range tested. The kinematic ETD with curvature γ is not
needed for accuracy on the homogeneous test; its role is
robustness in regimes where γ varies (viscosity contrasts,
internal forcing, finite-layer dispersion) — to be confirmed
on the buoyant-block / isostatic-relaxation test where
$\gamma$ is spatially structured.

**Code:** `_phase_i_fs_etd_internal.py` schemes `fe`, `rk2`,
`rk4`, `curvS`, `midpoint`. Plotter
`_plot_phase_i_integrators.py` produces the trajectory and
cost-vs-accuracy figures.

**Plots:** `output/phase_i2d_fs_integrators_trajectories.png`
(5 schemes × 5 dt-factors), `output/phase_i2d_fs_integrators_cost.png`
(error vs total Stokes solves).

## Phase I-2D-isostasy: where the ETD prefactor earns its keep

The relaxation tests above use a fixed $\Delta t$ chosen up front
from `stokes.estimate_dt()` × dt-factor. That choice masks the
production failure mode the kinematic ETD was designed to fix:
**adaptive $\Delta t$ growing as velocities decay.** A real code
re-evaluates `estimate_dt()` each step. The CFL bound is set by
the largest velocity in the domain. As the surface relaxes
toward equilibrium, $|\mathbf{u}|$ falls, so estimate_dt grows.
Eventually $\gamma\Delta t$ crosses the explicit-method
stability boundary even though the system is "almost done"
relaxing — drunken-sailor onset right at the end of the
simulation.

### Test setup

Internal-boundary annulus (same as before), but with:
- **Flat IC.** No initial perturbation; the surface starts
  level at $r_o$.
- **Internal forcing.** Eulerian buoyant blob at $(r,\theta) =
  (0.85, 0)$, Gaussian $\sigma = 0.06$, density anomaly
  $\alpha = 0.5$. The surface bulges upward to isostatic
  equilibrium.
- **Adaptive $\Delta t$.** `delta_t.sym = dt_factor *
  stokes.estimate_dt()` re-evaluated each step.

Reference: FE-noFSSA at fixed dt-factor=0.1, n=80 (small enough
that adaptive doesn't matter). Final $h_{\text{eq}} \approx
0.0268$.

### Results (16 steps each, adaptive $\Delta t$)

Final $h_{\text{pole}}$ (radial rise above the blob;
equilibrium $\approx 0.0268$):

| scheme | dtf=1 | dtf=2 | dtf=5 |
| --- | ---: | ---: | ---: |
| FE | +0.018 (low) | +0.038 (high) | +0.067 (DS) |
| RK2 | +0.018 | +0.006 (asymm) | -0.042 (sign-flip) |
| RK4 | +0.024 | +0.001 (asymm) | +0.009 (asymm) |
| **curvS-FSSA** | **+0.027** | **+0.025** | **+0.019** |
| **midpoint-FSSA** | **+0.026** | **+0.025** | **+0.019** |

$\Delta t$ growth in 16 steps:
- curvS at dtf=5: $\Delta t \to 151$, total $t \to 2420$
- RK4 at dtf=5: $\Delta t \to 28$, total $t \to 447$

The curvS / midpoint schemes ride the adaptive $\Delta t$ all
the way out — saturation prevents drunken-sailor regardless of
how big $\Delta t$ grows.

### Reading

1. **The (1-α)/γ prefactor is L-stability**, not better
   per-step accuracy. The earlier comparison (fixed $\Delta t$,
   sized for the worst-case CFL) hides this entirely — every
   scheme is in its stable regime, and RK4 wins on
   higher-order accuracy.

2. **In the adaptive-$\Delta t$ regime that production codes
   actually use, curvS / midpoint dominate.** They saturate at
   $h_{\text{eq}} = u_n/\gamma$ regardless of $\Delta t$.
   FE / RK2 / RK4 don't saturate; once $\gamma\Delta t$ crosses
   their respective stability boundaries (2 for FE/RK2, 2.78
   for RK4), the per-step displacement grows without bound.

3. **At fixed $\Delta t$ pushed past the RK4 boundary**
   (dtf=5, $\gamma\Delta t \approx 5$), RK4 *catastrophically
   blows up* ($h_{\max} = 18.6$ vs initial 0), while curvS /
   midpoint stay bounded at $\sim 0.02$. The prefactor
   structurally cannot blow up.

4. **The midpoint hybrid (RK2 + ETD prefactor) ≈ curvS** in
   this regime. Its accuracy disadvantage from the
   homogeneous-relaxation test doesn't penalise the isostasy
   case, where saturation matters more than per-step
   truncation. Both are good production candidates.

### Decision (final)

Two-track recommendation for the publication:

- **Lead scheme: kinematic ETD with curvS / midpoint
  prefactor.** This is the production case — adaptive
  $\Delta t$, unknown $\gamma$, possibly varying $\eta$. The
  saturation buys the user "set $\Delta t$ from the velocity
  CFL and forget about the relaxation timescale" — a property
  no explicit-RK scheme has.
- **High-accuracy alternative: RK4 (no γ).** When the user
  knows $\gamma$ stays bounded and wants the highest accuracy
  per unit cost (e.g. benchmarks, validation runs).

The midpoint hybrid is *not* dominated by curvS in the
isostasy regime — they're functionally equivalent — so we keep
it as an option. The earlier "midpoint loses to curvS" finding
was specific to the homogeneous-relaxation test where
adaptive-dt isn't engaged.

**Plots:** `output/phase_i2d_fs_isostasy_trajectories_adt.png`
(per-scheme trajectories at dt-factor 1, 2, 5),
`output/phase_i2d_fs_isostasy_dthistory_adt.png` ($\Delta t$
growth over the run, log axis), `output/phase_i2d_fs_isostasy_profile_adt.png`
(final boundary $\delta r(\theta)$).

**Code:** `_phase_i_fs_etd_isostasy.py` (CLI flag
`--adaptive-dt`); `_plot_phase_i_isostasy.py --adaptive`.

## Phase II-2D-continent: isostatic block and the volume-conservation regime

The kinematic-ETD investigation (Phase I) settled on curvS-FSSA as
the production-track scheme for relaxation problems. Phase II
moved to a **driven-isostasy benchmark** to stress-test the
schemes on a structurally different problem: a Lagrangian
buoyant block (a "continent") embedded in the heavy fluid, with
the surface above the block bulging upward to isostatic
equilibrium.

### Setup

- **Mesh.** Annulus, $r_i = 0.5$, $r_o = 1.0$. Two variants:
    - *Sticky-air*: `AnnulusInternalBoundary` with internal
      boundary at $r_o$ as the "free surface" and an air layer
      $r_o < r < r_{outer}$ with $\eta_{air}/\eta_{fluid} = 0.1$.
    - *True free-surface*: `Annulus` (rock-only), with $r=r_o$
      as the free surface — no air, no internal-boundary trick.
- **Mesh resolution.** Both unstructured-triangle and
  transfinite polar-quad ("structured") versions written.
  Structured version uses 2 half-annuli per radial layer with
  `setTransfiniteCurve` + `setTransfiniteSurface`. Critical:
  `useMultipleTags=True, useRegions=True` in the UW3 `Mesh`
  constructor — without it, Stokes silently segfaults on quads.
- **Continent block.** Lagrangian P0 indicator $B$, set to 1 on
  cell centroids inside the sector $|\theta| < \theta_{block}$
  AND $r > r_{min}$. Default $\theta_{block} = 0.4$ rad,
  $r_{min} = 0.7$, $\beta = 0.2$ (block is 20% lighter than
  ambient rock).
- **Body force.** Full buoyancy:
  $\mathbf{f} = -(M - \beta B)\hat r$. Heavy rock (M=1) feels
  full gravity; block (B=1, M=1) feels reduced gravity
  $-(1-\beta)$; air (M=0) is weightless.

### Body-force formulation (lessons)

- The earlier "anomaly form"
  $-(M - M_{\text{ref}}(r))\hat r$ — Lagrangian $M$, Eulerian
  $M_{\text{ref}}$ as a Heaviside step at $r_o$ — is *non-physical*.
  It introduces a step-jump in body force when cells cross
  $r_o$, which RK4's trial stages stumble on. With full
  buoyancy ($M$ alone, no subtraction) the Stokes solver finds
  the hydrostatic pressure self-consistently and the artefact
  goes away.
- Sticky-air's $\eta_{air} = 0.1\,\eta_{fluid}$ adds spurious
  lateral viscous drag against the bulge, suppressing the
  natural lateral spreading that real isostasy would show.
  True free-surface (no air) gives the cleaner physics.

### What the schemes do (free-surface, structured mesh, dtf=1)

Final $h_{\text{pole}}$ at $t \approx 600$, target uniform-bulge
analytic estimate $h_b \approx 0.051$, peak-from-fine-Δt $\approx
0.04$ (regional compensation):

| scheme | h_pole (uncapped, adaptive Δt) |
| --- | ---: |
| FE | drift down, drunken-sailor |
| RK2 | +0.022 (declines from peak ~0.025) |
| RK4 | +0.017 (peaks ~0.034 then **wild oscillations**) |
| curvS-FSSA | +0.039 (settles near peak) |
| midpoint-FSSA | +0.040 (settles near peak) |

curvS / midpoint converge to ~0.040; RK schemes show
*declining* $h_{\text{pole}}$ that turns out to be a
volume-conservation artefact rather than physical relaxation
(see next subsection).

### The headline finding: volume loss, not stability, is the binding constraint

Volume change ΔA/A₀ over the run (initial annulus area
$A_0 = \pi(r_o^2 - r_i^2) = 0.75\pi$):

| scheme | halfway ΔA/A₀ | final ΔA/A₀ |
| --- | ---: | ---: |
| RK2 (uncap) | -0.08% | **-1.52%** |
| RK4 (uncap) | -0.08% | **-4.46%** |
| curvS | +0.20% | -0.08% |
| midpoint | +0.11% | -0.28% |

At halfway, all schemes are within 0.3% of perfect volume
conservation. By final time, RK4 has lost **4.5%** of mass —
that's the "decline of $h_{\text{pole}}$" we see. The bulge
isn't physically receding; mass is leaking out through the
incompressibility-projection error.

### Capping Δt smooths the trajectory but doesn't fix volume loss

Capping $\Delta t$ at the value reached at the halfway snapshot:

| scheme | Δt cap | n_steps | final h_pole | final ΔA/A₀ |
| --- | ---: | ---: | ---: | ---: |
| RK2 uncap | adaptive | 24 | +0.022 | -1.52% |
| RK2 cap=18 | min(adaptive, 18) | 30 | **+0.030** | -1.83% |
| RK2 cap=9 (half) | min(adaptive, 9) | 60 | +0.031 | -1.85% |
| RK4 uncap | adaptive | 24 | +0.017 | -4.46% |
| RK4 cap=20 | min(adaptive, 20) | 30 | **+0.030** | -2.03% |
| RK4 cap=10 (half) | min(adaptive, 10) | 60 | +0.029 | -2.07% |

Two distinct effects:

1. **Capping Δt → smoother trajectory.** RK4's wild
   oscillations between Δt=11 and Δt=33 disappear. Final
   $h_{\text{pole}}$ rises from 0.017 to 0.030 — the bulge
   stays where the physics says it should, instead of
   oscillating mass in and out.
2. **Halving Δt → no improvement in volume conservation.**
   Total volume drift is ≈ (per-step error) × n_steps, and
   per-step error scales linearly with Δt. Halving Δt halves
   per-step error but doubles n_steps; the total cancels.

So **stability is not the binding constraint** at moderate
$\gamma\Delta t$ on this problem; the binding constraint is
the cumulative pressure-projection (incompressibility) error
that compounds over time. To improve volume conservation
beyond a few percent at this discretisation, the spatial
discretisation has to change — not the timestep.

curvS / midpoint stay below 0.3% volume drift the whole way
because once $h \to h_{\text{eq}}$ the saturation prefactor
$(1-\alpha)/\gamma$ → 0, so they take effectively zero
displacement per step at equilibrium and don't accumulate
further compressibility error.

### Implication for the publication

The kinematic-ETD prefactor's value comes from **two** sources,
not one:

1. **L-stability under arbitrary $\Delta t$** (the original
   motivation): bounded by $u_n/\gamma$ regardless of how big
   $\Delta t$ grows.
2. **Bounded volume drift at long times** because the per-step
   displacement self-limits as $u_n \to 0$. Explicit RK schemes
   without the prefactor keep advancing per step, so their
   compressibility error keeps compounding.

For driven problems with adaptive Δt, the second property is
arguably more important than the first. Production codes care
about long-term mass conservation as much as they care about
not blowing up.

**Code:** `_phase_i_fs_etd_continent.py` (sticky-air),
`_phase_i_fs_continent_fs_snapshots.py` (free-surface,
captures UW3 HDF5 + pyvista VTU + profile npz checkpoints),
`_structured_annulus.py` (transfinite polar-quad mesh in two
variants: with and without internal boundary).

**Plots:** `output/phase_i2d_fs_continent_fs_topo_vs_t.png`
(h_pole vs t, uncapped/capped/half-cap),
`output/phase_i2d_fs_continent_fs_capped_vs_uncapped.png`
(surface profile and ΔA/A₀, halfway and final),
`output/phase_i2d_fs_continent_fs_snapshots*.png`
(deformed mesh + density per scheme).

## Phase III-2D-continent: kinematic-update redesign closes volume conservation

The handoff at the end of Phase II had two leading hypotheses for the
1.5–2% volume drift of the explicit RK schemes on the continent
benchmark: (a) the pressure space is too coarse (V2/P1 → V3/P2),
or (b) the Stokes solver tolerance is too loose (1e-5 → 1e-7).
Phase III tested both. **Both null results.** The problem was
elsewhere — in the kinematic update itself.

### Diagnosis: the radial-only kinematic update

The original RK schemes (`fe`, `rk2`, `rk4` in
`_phase_i_fs_continent_fs_snapshots.py`) sample only the
**radial** component of velocity at the surface, decompose it on
the boundary in Fourier modes, diffuse it as a scalar via Poisson
into the interior, and map it to a purely radial mesh deformation.
**The tangential velocity is discarded.** Mass that should be
redistributed laterally by the surface flow has nowhere to go —
each step compresses or stretches the rock radially without
balancing it tangentially, accumulating as a bias toward
contraction.

This explains a long-standing puzzle from Phase II: total drift
scales linearly with simulated time × Δt and is *independent of
mesh resolution*. The error is structural in the kinematic
discretisation, not in the spatial or temporal resolution of the
Stokes solve.

### Three candidate fixes (Phase III subsections)

#### Phase III(a): V3/P2 element pair → null

Bumping the Stokes pair from V2/P1 to V3/P2 (Q3-Q2 LBB-stable on
quads) on the same continent benchmark, capped at Δt=18, n_steps=30:

| pair | tol | halfway curved-ΔA | final curved-ΔA |
| --- | --- | ---: | ---: |
| V2/P1 | 1e-5 | −0.21% | −1.78% |
| V3/P2 | 1e-5 | −0.22% | −1.86% (slightly worse) |

V3/P2 is essentially indistinguishable on volume conservation —
mildly *worse*, in fact, on this run. The pressure space is not the
bottleneck.

#### Phase III(b): Stokes solver tolerance → null

Tightening from 1e-5 to 1e-7 with V2/P1, same Δt and n_steps:

| pair | tol | halfway | final |
| --- | --- | ---: | ---: |
| V2/P1 | 1e-5 | −0.21% | −1.78% |
| V2/P1 | 1e-7 | −0.21% | **−1.78%** (bit-identical) |

The two runs produce identical numbers to all printed digits. The
saddle-point solve is solving the discrete system to enough
precision; the compressibility error is in the *discretisation*,
not in the solve.

#### Phase III(c): the kinematic update is the culprit

Two replacements were tested.

**Full-velocity advection** (`rk2_full`): deform every interior
node by $\Delta t \cdot \mathbf{v}_{\text{node}}$ (both
components), bypassing the radial-only diffuser. Same 30-step
capped run:

| scheme | final h_pole | final curved-ΔA |
| --- | ---: | ---: |
| `rk2` (radial-only) | +0.030 | −1.78% |
| **`rk2_full`** | +0.020 | **−0.21%** |

A factor-of-8.5 improvement. Volume conservation is now within the
0.5% pass criterion. But the trajectory drops to h_pole=0.020, well
under the equilibrium estimate of ~0.04. Either real lateral
spreading or temporal under-resolution at Δt=18; a Δt=0.5x
convergence run gives the same h_pole=0.0195 at total t=540, so
the trajectory shape is dt-converged. After step ~40 the mesh
distorts enough that the Stokes SNES line-search starts diverging
— ALE-style mesh quality limits the practical run length and would
require periodic regridding.

**Semi-Lagrangian-horizontal surface advection** (`rk2_sl`,
`rk4_sl`, `fe_sl`): keep the interior radial-only smoothing
(mesh-friendly), but write the kinematic free-surface BC in its
material-derivative form:

$$\frac{\partial h}{\partial t} + \frac{u_t}{r_o}\frac{\partial h}{\partial \theta} = u_n$$

Discretise the tangential transport semi-Lagrangianly: trace each
surface point back along $u_t \Delta t / r_o$, sample $h$ at the
trace-back angle via Fourier interpolation, and add the radial
uplift:

$$h^{n+1}(\theta) = h^n(\theta - u_t \Delta t / r_o) + \Delta t \cdot u_n$$

Implemented as an *effective* normal velocity that the existing
diffuser-Poisson path can take as its boundary condition:
$u_n^{\text{eff}} = u_n + (h^n_{\text{traced}} - h^n)/\Delta t$.
One Fourier evaluation per stage; no other code changes.

| scheme | final h_pole | final curved-ΔA |
| --- | ---: | ---: |
| `rk2` (radial-only) | +0.030 | −1.78% |
| `rk2_full` | +0.020 | −0.21% |
| **`rk2_sl`** | **+0.036** | **−0.19%** |
| **`rk4_sl`** (Δt cap=18) | +0.038 | **−0.004%** |
| `rk2_sl` (dtf=0.5, cap=18) | +0.038 | −0.027% |

The SL formulation is the strongest candidate: comparable volume
conservation to `rk2_full`, the *correct* equilibrium height
(~0.038, matching the Phase I curvS-FSSA equilibrium estimate),
*and* it preserves mesh quality. RK4-SL hits 0.004% drift —
beyond the publication-track stretch goal of 0.1% — without any
saturation prefactor, so no systematic undershoot from a wrong-γ
guess.

### Phase III(d): the Δt-cap criterion — relaxation CFL with monotone γ history

The cap=18 used in Phase II was empirically chosen from the
"halfway-Δt" snapshot. For a production scheme the cap has to be
**derived from the surface state**, with no hardcoded $\eta$ or
$\rho g$.

**The criterion.** For a relaxing system $\dot h = -\gamma h$,
$\dot h / h \equiv u_n / h$ is exactly the relaxation rate $\gamma$
— and crucially, it is **invariant to the system's proximity to
equilibrium**: both numerator and denominator scale linearly with
amplitude. The L²-weighted least-squares estimator that picks the
dominant mode robustly is:

$$\gamma_{\text{eff}} = \frac{\bigl|\langle u_n, h\rangle_S\bigr|}{\langle h, h\rangle_S}$$

Absolute value handles both regimes (positive correlation during
forced rise toward equilibrium; negative during pure relaxation).

**The monotone-history fix.** The L² estimator gives the *dominant*
(largest-amplitude) mode's $\gamma$, not the *fastest*. When the
slow dominant mode lets $\Delta t$ grow large enough to destabilise
a fast mode that's currently small in amplitude, that fast mode
oscillates with growing amplitude — and only "shows up" in the L²
estimator a step or two later, by which point the trajectory has
been corrupted. The fix is to retain the maximum $\gamma_{\text{eff}}$
ever observed:

$$\gamma_{\text{used}} = \max\bigl(\gamma_{\text{eff,now}},\; \gamma_{\text{history}}\bigr)$$

Once a fast mode reveals itself in the surface state, it stays
binding. This is the **damping-requirement** form of the criterion:
the cap is whatever $\Delta t$ keeps every previously-observed mode
in the L-stable region of the integrator.

**The cap.** $\Delta t \le c / \gamma_{\text{used}}$ with safety
factor $c$. RK4 is stable to $\gamma \Delta t \approx 2.78$;
truncation is ~5% per step at $\gamma\Delta t = 0.5$.

**Empirical sweep on the continent benchmark, RK4-SL:**

| c | derived Δt | final ΔA |
| --- | ---: | ---: |
| 0.5 | 8.11 | +0.007% |
| 1.0 | 16.22 | **+0.001%** |

$c=1.0$ recovers the empirically chosen cap=18 to within 10% from
the surface state alone — and produces the best volume
conservation we've seen on this benchmark. $c=0.5$ is more
conservative; the marginally-larger drift comes from doubling the
step count (per-step error × n_steps).

### Generalisation — the recipe is field-agnostic

The criterion is not specific to free surfaces. For any
state-rate pair $(\phi, \dot\phi)$ that an integrator owns:

$$\gamma_{\text{eff},\phi} = \frac{|\langle \dot\phi, \phi\rangle|}{\langle \phi, \phi\rangle},\quad \gamma_{\text{used},\phi} = \max(\gamma_{\text{now},\phi},\; \gamma_{\text{history},\phi})$$

For VEP, $\phi = \sigma$ and $\dot\phi$ is the Jaumann rate (or
$\Delta\sigma/\Delta t_{\text{prev}}$ from the trajectory) — the
criterion captures the Maxwell time $\tau = \eta/G$ as a *measured*
property of the current stress field, not a hardcoded constitutive
constant. Multiple state-rate pairs combine as

$$\Delta t = c \cdot \min\bigl(\Delta t_{\text{bulk-CFL}}, 1/\gamma_{\text{used},h}, 1/\gamma_{\text{used},\sigma}, \dots\bigr)$$

This may explain Phase II of the VEP work, where BDF-1 was the
only stable integrator for tight-yield TI faults: the
"first-order-dissipation" finding may have been compensating for
an under-conservative Δt; with the relaxation CFL on stress,
higher-order schemes might be recovered.

### Phase III decision — production scheme

**Production scheme (continent isostasy and similar relaxation
problems on a free surface):**

- **Spatial:** V2/P1 Taylor–Hood, structured polar-quad annulus
  (or unstructured-triangle); free surface (no sticky-air, no
  Heaviside body-force trick).
- **Kinematic update:** RK4 with semi-Lagrangian-horizontal
  surface advection (`rk4_sl`).
- **Δt:** $\Delta t = \min(\Delta t_{\text{bulk-CFL}}, c/\gamma_{\text{used}})$
  with $c = 1.0$ and $\gamma_{\text{used}}$ tracked as the running
  maximum of the L² regression of $|\langle u_n, h\rangle|/\langle h, h\rangle$.
- **No FSSA prefactor required.**

Stability constraint, not volume conservation, is now the binding
constraint. The cap criterion provides the stability automatically
from a single observable, with no hardcoded $\eta$ or $\rho g$.

**Code:** `_phase_i_fs_continent_fs_snapshots.py` (schemes
`rk4_sl`, `rk2_sl`, `fe_sl`, with `--dt-cap-mode relax`
and `--dt-cap-c <c>`).

**Plots:** `output/phase_i2d_fs_continent_sl_suite_topo_vs_t.png`
(h_pole(t) for all variants, labelled by final ΔA),
`output/phase_i2d_fs_continent_sl_suite_profiles.png`
(surface profile dr(θ), halfway and final).

## References

- Kaus, Mühlhaus & May (2010), PEPI: original FSSA paper.
- Andrés-Martínez et al. (2015): refined FSSA with explicit treatment
  of surface load.
- Cox & Matthews (2002): general ETD framework.
- Cathles (1975): viscous half-space surface relaxation analytical
  result.
- Pinkall & Polthier (1993): cotangent Laplacian on triangle meshes.
- Meyer, Desbrun, Schröder & Barr (2003): "Discrete differential
  geometry operators for triangulated 2-manifolds" — the
  Laplace–Beltrami / mean-curvature discretisation now standard
  in DDG.
