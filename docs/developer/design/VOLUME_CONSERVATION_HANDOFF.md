# Handoff — Volume conservation in free-surface deformation

> **Read first**: `EXPONENTIAL_FREE_SURFACE.md`, especially the
> "Phase II-2D-continent" section. This document is the focused
> handoff for the next investigation: improving volume conservation
> on the continent-isostasy benchmark.

## What we know

The continent-isostasy benchmark (free-surface, structured polar-quad
mesh, β=0.2 Lagrangian sector block) showed that **volume
conservation, not stability, is the binding constraint at long
times**. Even with sensible Δt caps, RK2/RK4 lose 1.5–2% of mass
over the run. curvS / midpoint stay under 0.3% because their
saturation prefactor stops them from advancing once
$u_n \to 0$ at equilibrium.

The volume drift compounds linearly in total simulated time:
halving Δt while doubling n_steps gives the same total drift.
That's a strong hint that **the per-Stokes-solve compressibility
error is the bottleneck**, not the time integration.

## Current setup (already in the worktree)

- **Stokes element pairing**: V (P2 vector) / P (P1 scalar) —
  Taylor-Hood. UW3 default.
- **Stokes solver tolerance**: `stokes.tolerance = 1.0e-5`.
- **Pressure space**: continuous P1.
- **Mesh**: transfinite polar-quad annulus (rock-only,
  `AnnulusStructured` in `_structured_annulus.py`); also
  unstructured-triangle `Annulus` for cross-checking.
- **Diagnostic**: `_volume_check_continent_fs.py` reads pyvista
  VTU snapshots (no re-run) and prints ΔA/A₀ per
  scheme × (halfway, final).
- **Test runner**: `_phase_i_fs_continent_fs_snapshots.py` with
  `--dt-cap`, `--snap-suffix`, `--schemes`. Already saves UW3
  HDF5+XDMF, pyvista VTU, and surface-profile npz at halfway and
  final.

## Things to try (priority order)

### 1. Higher-order pressure space — V3/P2 Taylor–Hood

The most likely win. Current V2/P1 has 3 pressure DoFs per cell;
V3/P2 has more, capturing finer-scale pressure gradients near the
deformed boundary. The hypothesis: most of the per-step
compressibility error comes from the P1 pressure space being
unable to represent the body-force step at the block edges and
the curvature of the bulge.

The change is one parameter: increase the pressure
`MeshVariable`'s `degree` from 1 to 2. The velocity may need to
go from `degree=2` to `degree=3` to keep the inf–sup pair
(LBB-stable Q3-Q2 in 2D, or P3-P2 on simplices).

**Verify with**: rerun the continent test with V3/P2 (capped at
the halfway Δt). Check whether ΔA/A₀ at final drops below
~0.5%.

### 2. Tighten the Stokes solver tolerance

`stokes.tolerance = 1.0e-5` was set early in the investigation.
If the per-step KSP error is contributing significantly, tightening
to 1e-7 should reveal it. Run the same continent test, same
discretisation, just change tolerance.

If the volume drift is *unchanged* with tighter solver tolerance,
the error is coming from the discretisation (element pairing,
spatial resolution) not the solver — supports the pressure-space
direction above.

### 3. Explicit volume-correction projection per step

After the kinematic update (mesh deformation), apply a small
post-processing projection that pushes velocity onto a
discretely-divergence-free space and re-deforms. This is a
"divergence cleanup" pass.

In 2D, the displacement field has a stream function $\psi$ such
that $u = \partial_y \psi, v = -\partial_x \psi$, automatically
$\nabla \cdot u = 0$. After the integrator step, recompute the
displacement from the closest stream function. This is a
projection, not a free parameter, and should be cheap.

May be tricky to get right with deformed meshes / Lagrangian
markers. Lower priority than (1) and (2) but worth trying if
they don't help.

### 4. Conservative formulation of the kinematic update

Instead of advecting the surface displacement field via a
diffuser-smoothed increment, formulate the surface evolution as
a flux equation:
$\partial_t h + \nabla_S \cdot (h u) = 0$ (where $\nabla_S$ is
the surface divergence) and discretise it conservatively.

This is a substantial rework of the kinematic update. Save for
later — only if the simpler fixes don't get below 0.5% drift.

## Test protocol

For each candidate fix, run on the **continent free-surface
structured-mesh case** with **β=0.2, θ_block=0.4, r_min=0.7**.
Schemes tested: rk2, rk4 (the explicit cases that show the
compressibility issue most clearly).

Capping: use the established halfway-Δt cap (≈18 for rk2,
≈20 for rk4 on the structured mesh) so we're comparing fixes
in the cleanest regime.

Diagnostic: ΔA/A₀ at halfway and final. Pass = below 0.5% at
final. Stretch goal = below 0.1%.

Reporting: the existing `_volume_check_continent_fs.py` reads
the saved VTU files; just add the new snapshot dirs to its
`SNAP_DIRS` map and rerun.

## What to keep, what to change

**Keep**:
- The continent-isostasy setup (the benchmark itself).
- The structured polar-quad mesh — clean cell alignment with
  the block edges, no Stokes segfault.
- The free-surface formulation (no sticky air, no
  Heaviside body-force subtraction).
- Δt-cap mechanism on the runner.
- UW3 HDF5 + pyvista VTU + profile npz checkpointing for
  re-visualisation without re-simulation.

**Out of scope for this next session**:
- Time-integrator improvements (covered in Phase I).
- Sticky-air variants (we know they're worse).
- Mesh-resolution sweeps (we know it doesn't help — error
  scales with cumulative time × Δt, not cell size).
- Pressure-projection of curvS / midpoint (they already
  conserve volume well).

## Open question for the next session

Is the V3/P2 pairing's volume-conservation gain enough to
flip our integrator preference? At the moment curvS / midpoint
win on volume because their saturation prefactor stops them
moving. If V3/P2 brings RK2 below 0.3% drift, then RK2 capped
at the halfway Δt becomes a viable production scheme — its
2nd-order accuracy would beat curvS's 1st-order saturation in
the regions of phase space where saturation isn't engaged.

The publication-track decision tree might end up:

- **Stiff / driven problems with non-saturating dynamics**:
  RK2 capped + V3/P2.
- **Long-time relaxation toward equilibrium**: curvS-FSSA
  (saturation makes the volume drift question moot).
- **Unknown / mixed**: curvS-FSSA as the safe default; switch
  if the user knows the problem is non-saturating.

## Direction from lmoresi (2026-05-08)

The user is **not satisfied with the stabilised integrators**
(curvS, midpoint) as the production answer. The reason: they
**introduce systematic undershoots on simple physics**. The
saturation prefactor $(1-\alpha)/\gamma$ caps the per-step
displacement at $u_n/\gamma$, which is the local linearised
equilibrium estimate using a *guessed* $\gamma$ (from
curvature, with hardcoded $\eta_{\text{eff}}$). When the
guessed $\gamma$ is wrong by a factor of 2 — easy to do for
heterogeneous viscosity, finite layers, or non-half-space
dispersion — the saturation locks the surface at the wrong
height. We saw this clearly in the homogeneous-relaxation
test where curvS overshot the small-dt reference by 10–20%
even with eta_eff corrected.

**The user's preferred path** is to keep the *accurate* explicit
integrators (RK2 / RK4 with their natural high-order Taylor
match to the exponential) and derive a **principled local Δt
limit** from surface-state properties — not from a global
$\gamma\Delta t$ stability bound. Candidate inputs to that
limit:

- Surface curvature $\partial^2 h/\partial s^2$ (or its
  dimensionless form $\partial^2 h / \partial s^2 \cdot R$)
  → wavenumber proxy.
- Surface gradient $|\nabla_S h|$ → slope, encodes how steep
  the bulge is.
- Surface uplift rate $u_n / h$ (where it's well-defined) →
  empirical relaxation rate.
- Possibly per-cell volume-drift budget from the most recent
  Stokes solve, as an a-posteriori indicator of when to throttle.

The hypothesis: **some combination** of these locally sets the
"safe" Δt — tighter than the linear-stability γΔt bound, but
*derived from observable surface state* rather than from a
hardcoded $\eta$ + $\rho g$. The exact combination is unknown.
Possibilities to explore:

1. $\Delta t_{\max}^{(local)} = c \cdot (\partial s)^2 / (\eta / \rho g)$
   where $\partial s$ is the local arclength of curvature
   features — i.e., a curvature-CFL.
2. $\Delta t_{\max}^{(local)} = c \cdot |h| / |u_n|$
   (relaxation-rate Courant). Robust at zero crossings of $h$
   only with a window-regression form (cf. Phase I empirical γ).
3. Hybrid: take the min of (1), (2), and the bulk-velocity CFL.

The investigation needs:

- A clean surface-curvature / surface-gradient extractor on the
  internal-boundary nodes (we have curvature regression already in
  `_phase_i_fs_etd_internal.py`).
- A test problem where the "right" $\Delta t$ varies in space
  (e.g., the continent-isostasy test, where the bulge curvature
  near $\theta=0$ is tighter than the long-wavelength tails).
  The local Δt should track that variation.
- A direct comparison: RK2 with the proposed surface-derived
  cap vs RK2 with the empirical "halfway Δt" cap vs curvS.

**Pass criterion**: RK2 with the surface-derived cap should match
or beat the small-dt reference trajectory on the continent test
*and* keep ΔA/A₀ below ~0.3% (assuming V3/P2 has been adopted
to fix the discretisation-level volume issue).

This direction is **complementary to the V3/P2 work**, not
alternative: V3/P2 fixes the per-Stokes-solve compressibility;
the surface-derived Δt cap fixes the per-step kinematic step
size. Both are needed for explicit RK schemes to be production-
ready.

If both work, the publication story changes from "kinematic ETD
is the production scheme" to "RK2 + V3/P2 + surface-derived Δt
cap is the production scheme; ETD is the safe fallback for
unknown-γ problems". That would be a stronger result.
