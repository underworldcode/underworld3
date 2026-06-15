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

## Convection (mechanism ported into the zoo)

The validated mechanism — interior **mover** + **tangent-slip** (height-field
carry) + **surface smoother** (Taubin low-pass via `extract_surface`) — is wired
into `_phase_i_fs_convection_zoo.py` as three flags (`--mover`,
`--tangent-slip`, `--surface-smooth N`), running **end-of-step, before the ALE
`v_mesh` delta**, so all node motion (tangential slide + surface re-snap +
interior relax) is captured by `v_mesh` and absorbed by SLCN's `V_fn = v −
v_mesh`. ALE stays load-bearing.

**Tangent-slip now works with the diffuser path.** The isostasy note (above) flagged
that tangent-slip "cannot be bolted onto the radial diffuser (fixed angular grid)."
The zoo port fixes that: after the FREE-mode `project_to_slip_surface` slide, it
**re-sorts `internal_idx`/`internal_th` by current angle** so the Fourier
reconstruction stays ordered; the surface-smoother map is node-identity (passed the
current index ordering), so it follows the re-sort for free.

**Goal-1 baseline (rk4 + ALE, res 20, Ra=ρg=1e5, θ=0.5, monotone=clamp).**
Reproduces the May-14 reference: `Nu = 13.6 / 16.8` at steps 5/10 (exact), same
Nu-growth curve thereafter with a 1–2 step phase shift in the steep transient
(expected for a chaotic fast-growth phase; not a regression). Plain rk4 (no `_sl`
term) drifts in volume and blows up past step ~25 (`T` overshoot, ΔA/A −16%) — the
deficiency the mechanism below addresses.

**Goal-2 A/B (rk4 + tangent-slip  vs  rk4_sl/SL; res 16, Ra=ρg=1e4, mode-5 IC,
both with mover + surface-smooth + ALE, 50 steps):**
- All three pieces run **clean** — no tangling, `T ∈ [0,1]` throughout,
  `ΔA/A < 0.07%` in both (vs plain-rk4's drift). Surfaces are smooth mode-5
  profiles (no sawtooth → smoother working).
- **tangent-slip ≈ SL** in the early symmetric transient (Nu agrees to **<0.5%**
  through step ~20). They diverge slowly thereafter (Nu 11.2 vs 10.5 ≈ 7%,
  `h_max` 3.97e-3 vs 3.22e-3 ≈ 23% by step 50): tangent-slip carries topography
  *exactly* while the SL FE trace-back is mildly diffusive, so tangent-slip retains
  more amplitude → marginally hotter. Both clean and bounded.
- Caveat: 50 steps ≈ `t=0.03` ≪ thermal time ~1, so this is the *early* transient,
  not steady state. The genuine discriminator (net, time-varying lateral transport)
  needs the harder regimes — Goal 3.
- Figure + work-logs: `~/+Simulations/fs_convection_goal4/`.

## Open / next

- **etd_topo held-lid in convection** (not yet ported): the L-stable surface
  integrator (held free-slip solve → `h_eq` from mean-relative `σ_nn/ρg`, relax
  toward it) for the high-Ra ringing regime. The zoo currently evolves the free
  surface with its existing rk4/rk4_sl integrators; etd_topo is the Goal-4 piece.
- **Goal 3 A/B** — break symmetry/steadiness (asymmetric IC, higher Ra,
  plume-shedding, longer runs) so lateral transport is net/time-varying; only then
  does tangent-slip-vs-SL genuinely bite. Compare h(θ), Nu, vrms, ΔA/A.
- MMPDE mover **advantage**: occasional interior restore (not every step);
  `follow_metric` surface refinement.
- Internal-interface revisit; rung-2 pure relaxation (`h_eq=0` → exact decay).
