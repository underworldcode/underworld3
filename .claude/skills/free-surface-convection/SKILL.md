---
name: free-surface-convection
description: The Underworld3 free-surface convection method we are hardening — the THREE-NUMBER pointwise topography integrator (held-lid stress equilibrium h_∞ + L-stable exponential relaxation), NOT FSSA. Reach for THIS before touching any free-surface / dynamic-topography convection run, choosing a surface-update scheme, or "stabilising" a free surface. It records the method, why FSSA is explicitly rejected, and the failure modes.
---

# free-surface-convection

The free-surface scheme used in `~/+Simulations/FreeSurface/convection/fs4_compare.py`
and the design doc `docs/developer/design/FREESLIP_DYNAMIC_TOPOGRAPHY_FREESURFACE.md`.
**This is the method we are HARDENING — do not replace it; do not add FSSA.**

> The 3-number integrator is necessary but NOT sufficient — see **Hardening strategies
> (2026-06)** below for the material-surface advection, tangential topography term,
> free-slip-inner nullspace, and graded/higher-order-mesh fixes that make it actually
> work. Reference impl + diagnostic tools live in `~/+Simulations/FreeSurface/convection/`.

## ⚠️ NOT FSSA

The `docs/examples/free_surface/advanced/Annulus*FS.py` examples use **FSSA**
(`add_natural_bc(δt·(Γ·v)Γ/2, "Upper")`). **That is NOT our method.** FSSA buys
stability by adding an implicit surface traction that **UNDER-deforms the surface**
— it trades accuracy for stability. Our scheme is designed to be stable **and**
accurate. If you find yourself adding `FSSA`, an `add_natural_bc` traction on the
free surface, or a `Gamma.dot(v)` stabiliser — STOP, you have the wrong method.
(Those example files are a template for a *different* approach, not this one.)

## The three-number pointwise integrator (THE method)

Two Stokes solves per step on the SAME mesh, then a pointwise surface update:

1. **Free solve** — stress-free top (NO velocity BC on `Upper`; pressure datum is
   pinned by the stress-free condition → no pressure nullspace). The surface
   normal velocity `u_n` of this solve IS the kinematic rate `ḣ`.
2. **Held-lid solve** — a second Stokes solve with a RIGID free-slip held lid
   (`u_n = 0`, via `add_nitsche_bc(Upper, local_h=True)` — see [[project_nitsche_local_h_pr275]])
   and a DRIVING-ONLY body force. Its surface normal stress `σ_nn` gives the
   equilibrium topography `h_∞ = -(σ_nn - mean)/ρg`. (The free solve forces
   `σ_nn = 0`, so the equilibrium MUST come from the held-lid stress.)
3. **Pointwise exp step**, per surface node, from THREE numbers (`h`, `ḣ=u_n`,
   `h_∞`):
   ```
   γ = ḣ / (h_∞ − h)        # local relaxation rate, clamp γ ≥ 0
   h ← h_∞ + (h − h_∞)·exp(−γ·dt)
   ```
   L-stable: the step is bounded between `h` and `h_∞`, so it **cannot overshoot**
   regardless of a noisy local `γ` (no "drunken sailor"). 1 extra solve/step;
   beats RK4 at large dt. NO per-node freeze-clamp (that was the old `relax` bug).
4. The nodal surface increment is **carried inward by a Laplacian diffuser**
   (smooth, minimal mesh deformation — NOT full mmpde adaptation), then
   `mesh.deform()`. Uniform meshes are fine; adaptivity is NOT required.

Reference impl: `fs4_compare.py` → `_surface_step`, `_h_inf` (held-lid σ_nn via a
`Projection`), `_surf_un`, `_carry_diffuser`. The free-slip RIGID-top run (no
surface motion) is the reference; the free-surface run is the same driving solve
PLUS this surface update. See [[project_fs4_adaptive_2x2]],
[[project_stress_equilibrium_freesurface]], [[project_freeslip_topo_freesurface]].

## Performance

- Free-slip (rigid top) stagnant-lid runs are FAST. The free-surface cost is the
  extra held-lid solve **plus** that moving the surface forces a COLD-START Stokes
  each step (can't warm-start across a deformed mesh).
- Use **uniform** meshes for this problem — the diffuser gives minimal deformation,
  no mmpde needed. (FMG works on a uniform `refinement=N` hierarchy; scalar solvers
  must avoid FMG — PETSc err62, issue underworldcode/underworld3#276.)

## Hardening strategies (2026-06) — the integrator alone is not enough

The 3-number integrator moves the surface correctly, but several *other* things must
be right or it runs away / tangles. All implemented in `fs4_compare.py` (flags noted).

### 1. Material-surface advection — THE key fix (`--advect-velocity`)
The runaway (`u_n` 42→125→285→445, cold lid leaking in, plumes punching through) was
NOT an `h_∞`/BC bug (`h_∞` is verified correct, even in the stagnant FK lid — held-lid
free-slip is the EASY case there). The bug: the surface moves by the L-stable relaxed
rate `ũ_n = Δh/Δt ≤ u_n`, but T was advected with the **stress-free solve velocity**
(surface-normal = full `u_n`). Net material then crosses the surface. A free surface is
a MATERIAL boundary: advect T with a velocity whose surface-normal = `ũ_n`. Modes:
- `consistent` (the right way): a THIRD Stokes solve, same buoyancy, `v·n̂ = ũ_n`
  PRESCRIBED at the surface (penalty), tangential stress-free. `ũ_n = (shape_new−shape0)/dt`
  = the full ∂h/∂t at fixed θ (correct ALE target).
- `blend`: `α·v_free + (1−α)·v_held`, `α = φ1(γΔt) = (1−e^{−γΔt})/(γΔt)` (the exp-decay
  time-average). By Stokes LINEARITY this *is* the prescribed-`ũ_n` solve for UNIFORM α
  (and free for FK, which is linear in v). BUT the single mean-α collapse is NOT close
  enough once γ varies per surface node — the planform diverges (mode-3 vs mode-1/2),
  throughflow ~23 vs ~0.08. Per-node α breaks div-free (∇α·(v_free−v_held)). So the
  per-node `consistent` 3rd solve is REQUIRED for structured planforms.
- `free`: advect with stress-free v (the inconsistent baseline — the runaway).

### 2. Tangential topography advection (`--no-tangent-advect` to disable; default ON)
The pointwise relaxation omits the `v_t·∂_s h` term — a surface rotation/convergence
should carry the topography pattern along the surface; without it you get edge artefacts
where ∂_s h is large (plume-bulge edges). Fix = operator split per step: (1) departure-
point semi-Lagrangian transport of the surface shape in θ by `ω = v_t/r`, then (2) the
L-stable normal relaxation. Lowers throughflow + improves mesh quality.

### 3. Free-slip inner boundary — rotation nullspace (`--inner freeslip`)
The rigid rotation `[-y,x]` is a velocity nullspace ONLY while the boundary is CIRCULAR.
Once the free surface DEFORMS, do NOT attach it to the held/consistent solves (→ held
22 s/`DIVERGED_LINEAR_SOLVE`, throughflow blow-up). Keep `petsc_use_pressure_nullspace`;
strip the gauge with the exact post-solve projection `_project_out_rotation` on `v`,
`v_cons` (drives advection) AND `v_h` (one consistent non-rotating frame). The undeformed
free-slip *reference* (`--surface freeslip`) is fine WITH the nullspace attached.

### 4. Graded / higher-order meshes (drive node movement consistently)
- **Surface-ring detection**: tie the tolerance to the FINEST cell
  (`0.5·mesh.get_min_radius()`), NOT the nominal `cellsize`. On a gmsh-graded mesh
  (`cellSizeOuter`) the old tolerance scoops the first interior ring → a 2%-thick
  "surface band" → tangling (looks like the surface "destroying itself"; it isn't —
  the diffuser was fed a corrupt surface). BETTER (TODO): build the ring from the DMPlex
  `Upper` label (`dm.getLabel("Upper").getStratumIS`), removing the tolerance entirely.
- **Node movement**: the solve velocity is P2; the mesh geometry is P1. Drive `u_n` and
  the tangential transport from a P1 length-smoothed `Vector_Projection` of V (`v_p1`),
  NOT a point-evaluation of the P2 field.
- **Stress smoothing**: `topo_proj.smoothing_length` = a fixed PHYSICAL length
  (`--smooth-length`), not cell-count, so `h_∞` is mesh/order-independent.

### 5. Cost — there is no acceleration win (don't chase it)
3 Stokes solves/step (free→u_n, held→h_∞, consistent→advect). Warm-start does NOT help
(outer KSP already 1 iter; FMG supplies its own nested guess — measured SLOWER). Blend-
skip rarely fires (α-spread always large). Operator/PC reuse: already reused across
`solve()`s (first 5.5 s setup, steady ~945 ms = irreducible FMG solve; RHS-only resolve
same cost). The per-step cost is the geometric FMG hierarchy REBUILD on the deforming
mesh — intrinsic to moving meshes, "live with it." The UNIFIED-PENALTY single solver
(`penalty·(v·n̂ − V₁·n̂)·n̂`; penalty=0→free, V₁=0→held, V₁=ũ_n→consistent — held &
consistent share the matrix) is the cleanest formulation (no recompile on the constant)
but doesn't cut the irreducible solve.

### Elastic-plate flexure `h_∞` — IMPLEMENTED (`--flexure-D`)
Generalizes the LOCAL Airy `h_∞ = −σ_nn/ρg` (the D=0 limit) to a flexed plate
`(D ∂_s^4 + ρg) h_∞ = −σ_nn`, solved SPECTRALLY on the ring (serial Fourier — the
feasible substitute for UW3's blocked 1D-manifold FE solve): per mode
`h_∞(m) = −σ_nn(m)/(ρg + D(m/r_o)⁴)`. `D` sets the flexural wavelength `(D/ρg)^{1/4}`
and damps short-wavelength loads — the physically-grounded, mesh-independent length-
smoothing. In `_h_inf` (h_∞-ONLY — the stable form). **PROTOTYPE — amplitude response correct
(stiffer plate → less deflection) but it does NOT low-pass the SURFACE**: filtering `h_∞` only
sets a smooth set-point; the surface still picks up short-wavelength content from the SL
tangential transport + partial relaxation. "Filter every surface number (h, ḣ, h_∞)" was
TRIED and REJECTED — filtering the GEOMETRY `h` injects a spurious smooth-the-mesh motion into
`ũ_n` → flow runs away (Vrms 50→345); filtering `ḣ` alone is stable but elevates Vrms with no
benefit. So making flexure a TRUE surface low-pass without destabilising is OPEN/hard
(`_flex_filter` helper is in place). Examples: `stagnant_lid_mode1_study/{figures/flexure_*.png,
runs/flexure_D*}`.

### Open / next (not yet done)
- **Label-based surface ring** (replace the radial heuristic with the `Upper` stratum).
- **Flexure D calibration** to a realistic lithospheric flexural wavelength.

### Diagnostic tools (`~/+Simulations/FreeSurface/convection/stagnant_lid_mode1_study/scripts/`)
- `heldlid_hinf_check.py` — verify `h_∞` via 4 independent free-slip enforcements × Δη sweep
- `stitch_compare.py` — side-by-side montage of per-run dirs (`--dirs a,b,c`)
- `unified_penalty_solver.py` — the one-solver penalty formulation probe
- `resolve_timing_probe.py` — repeated-solve / lag-Jacobian / reuse-PC timing

## Failure modes — symptom → cause

| Symptom | Cause |
|---|---|
| Surface deforms but `u_n` RUNS AWAY (e.g. u_n 42→125→285→445), cold lid leaks in, plumes punch through | **RESOLVED**: material-surface advection inconsistency — T advected with stress-free `u_n` while surface moves by relaxed `ũ_n`. Fix = `--advect-velocity consistent` (Hardening §1). NOT an h_∞ bug, NOT fixed by FSSA. |
| `held` solve 22 s / `DIVERGED_LINEAR_SOLVE`, throughflow blows up, with `--inner freeslip` | rigid-rotation `[-y,x]` attached as a nullspace on the DEFORMED (non-circular) surface — invalid. Don't attach it on the moving surface; use the post-solve projection (Hardening §3). |
| Graded-mesh surface "destroys itself" (q→0.2, h_max 2% at step 1) | surface-detection tolerance scooped the first interior ring → 2%-thick band, NOT real deformation. Tie tolerance to finest cell (Hardening §4). The diffuser is innocent. |
| Stress-free top but surface not updated each step | nothing stops throughflow (the stress-free top is an open boundary unless the integrator moves the surface to track `u_n`) |
| Nu decays when it should be steady (kinematic free surface) | LAG: the SL foot reaches beyond an under-moved surface → cold pump. Fix = the h_∞ relaxation, not more smoothing |
| Surface "mountain" / one-step spike on adaptive mesh | held-lid Nitsche penalty over-stiffened by GLOBAL h; use `local_h=True` (default, PR #275) = `mesh.cell_size()` |

## Dead ends (already tried — do NOT repeat)

- **FSSA** signed-traction free-surface: diverges / under-deforms — rejected.
- **High-k post-smoothing** of the surface: the instability is low-m, smoothing
  the wrong band.
- Per-node freeze-clamp in the relaxation (the old `relax` fatal bug).

## Diagnose by

`h_max` (deflection, as % of r_o), `u_n` / `vhmax` (surface throughflow — should NOT
grow unbounded), `hinf_max` (the equilibrium target), `vrms`, `Nu`. Compare the
free-surface run against the free-slip RIGID-top reference at matched physical time.
