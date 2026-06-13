# Stress-equilibrium free-surface integrator (`etd_topo`)

> Status (2026-06-14): **integrator validated and landed** on the upper-surface
> isostasy problem, running on the **diffuser** with the **SL lateral
> correction**. The MMPDE-mover swap is implemented and validated at parity
> (`--mover`) but its payoff refinements + convection are a separate follow-up.
> Driver: `_phase_i_fs_isostasy_upper.py`. Probes: `_probe_heq_*.py`.

## Idea

Free-surface evolution is a relaxation toward an equilibrium topography. Instead
of estimating the relaxation rate γ and inferring the end-state (`curvS`:
`h_eq = u_n/γ`), **measure the equilibrium dynamic topography `h_eq` directly
from the stress field** and integrate the surface exponentially toward that
known end-state:

```
h^{n+1} = h_eq + (h^n − h_eq) · e^{−γΔt}
```

`h_eq` is the "infinite-time relaxation" topography; measuring it from stress
factors the source structure out of the rate estimator (the confounding that
defeated the earlier empirical-γ scheme).

## The validated recipe (external upper free surface)

1. **Held-lid solve** — a *second* Stokes solve on the current mesh with a
   free-slip (Nitsche, γ=10) lid on the surface and the **blob-only / driving
   body force**. Project `n·σ·n` (radial: `r̂·σ·r̂`).
2. **`h_eq = −(σ_nn − mean(σ_nn)) / (Δρ g)`** — mean-relative.
3. **Free solve** → surface radial velocity `u_n` (the rate).
4. Per-node `γ = u_n_rel / (h_eq − shape)`, clamped ≥ 0; exponential step of the
   *shape* toward `h_eq` (mean preserved → area conserved).
5. **Lateral transport** — radial nodes don't follow tangential flow, so add the
   SL trace-back along `u_t` (`h_back = shape(θ − u_t Δt/r)`); the increment is
   `(h_back − shape) + relax_inc`.
6. **Propagate** the radial increment into the mesh (diffuser, or `--mover`).

### Non-negotiable details (each cost a debugging round)

- **External surface only.** On an *internal* interface the held-lid stress
  can't be measured cleanly — the pure penalty has no consistency term, so the
  constitutive `n·σ·n` projects to ≈0 and only the penalty-dependent reaction
  carries the load. Nitsche/penalty/constraint are one-sided domain-boundary
  operators. (Free-slip on the external boundary *is* consistent.)
- **Blob-only (driving) body force in the held-lid solve.** On a deformed mesh
  the topography is represented *geometrically*; adding the `ρ_ref` topographic-
  load shell **double-counts** the load and flips `h_eq` negative.
- **Mean-relative `σ_nn`.** The held free-slip solve (no-slip inner + free-slip
  outer) has a pressure null space → the *absolute* `σ_nn` has an arbitrary
  constant (flips to −0.15). The surface mean is the correct isostatic datum.
- **Radial prediction, tangential transport.** `h_eq` is predicted *radially*
  (lateral stress adds no vertical force); only the *evolution* needs the
  tangential velocity (SL term). Projecting the stress onto the velocity
  direction (`û·σ·û`) is **worse** than radial — off-pole `û` is dominated by
  tangential flow.
- **Noise: smooth the stress, not the velocity.** `topo_proj.smoothing_length
  ≈ 1 cellsize` cuts high-k topography ~3–10× (stress ~ ∇v amplifies high-k).
  P1-projecting / smoothing the velocity does essentially nothing.

## Validation (isostasy, res 16)

- `h_pole → 0.0232–0.0240`, matching the kinematic (curvS/rk4) equilibrium
  ≈ 0.0227 to within a few %.
- **L-stable**: at Δt = 4·Δt_est, rk4 drunken-sails (`h_pole = −0.019`,
  blown up) while `etd_topo` holds the equilibrium. Cost: 1 free + 1 held solve
  per step vs rk4's 4.
- Area conservation `ΔA/A ≈ +0.1%` (mean-relative formulation); the SL term is
  correct but negligible *here* (near-symmetric, steady) — it earns its keep on
  asymmetric/convecting problems.
- Figures: `~/+Simulations/freesurface_stress_equilibrium/`.

## MMPDE mover vs diffuser

`--mover` replaces the Poisson diffuser with `smooth_mesh_interior(method=
'spring', pinned=[Upper,Lower])`: move the surface ring radially, relax the
interior with the mover. Isostasy: `h_pole 0.0232 / ΔA/A +0.097%` vs diffuser
`0.0240 / +0.104%` — **at parity**. It currently relaxes every step (same
cadence as the diffuser); the *advantage* (occasional restore → minimal
interpolation; boundary-slip tangential redistribution that retires the SL
term; `follow_metric` refinement) is the next, focused piece of work.

`--tangent-slip` (lateral node motion via `project_to_slip_surface` FREE mode)
is **WIP/experimental**: it cannot be bolted onto the radial diffuser (whose
Fourier reconstruction assumes a fixed angular grid) — it needs the 2D mover.

## Open / next

- MMPDE mover **advantage**: occasional interior restore (not every step);
  boundary-slip tangential redistribution (retires SL); `follow_metric` near the
  surface.
- **Convection** — where SL / tangent-slip / problem-dependent stability bite.
- Internal-interface revisit; rung-2 pure relaxation (`h_eq=0` → exact decay).
