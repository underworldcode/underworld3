# Adaptive meshing + faults in convection — ideas to preserve & re-implementation plan

Status: **spec** (2026-06-22). The exploratory driver + diagnostic scripts in
`scripts/fault_*.py` were written *before* the mmpde-holes root cause was found
and fixed (PRs #259 + #264 + #266 on `development`). Their **ideas are sound**;
their **quantitative results are suspect** (they ran on a mover that evaluated a
mis-located / negative metric — see `project_mmpde_holes_real_root_cause` and the
`adaptive-meshing` skill). This note captures the approach so it can be
**re-implemented cleanly and re-validated** on the fixed base.

## What "fixed base" means now
`development` has: deformed-mesh point-location fix (#264), monotone RBF metric
bake (#266), SPD metric floor (#259). On it, mmpde is rock-solid: forced adapt
every step at R=5 → folded=0, cell-area-ratio flat (~14), vigorous convection
(vrms→18, Nu→1.86). The mover, `accel`, and `refinement=R` are all fine.

## Validated approach (the recipe to re-implement against)
- **Mover:** `uw.meshing.smooth_mesh_interior(mesh, metric, method="mmpde",
  method_kwargs=dict(step_frac=0.2, accel=None|"cg", momentum=0.0),
  slip_surfaces=True, skip_threshold=...)`. Adapt as often as wanted (forced
  every step is fine now). See the `adaptive-meshing` skill for the full recipe.
- **Thermal metric:** `metric_density_from_gradient(mesh, T, refinement=R,
  coarsening="auto", metric_choice="front-following")`. R≈5 is fine.
- **Fault metric:** hand-built anisotropic SPD tensor
  `M = ρ·I + (Rf²−1)·exp(−(d/w)²)·n nᵀ` (thin ACROSS the fault normal n), with a
  DIRECT unsigned distance field `d` (geometry_tools), NOT the signed
  `Surface.distance` (bleeds along the line extension).
- **Resolved faults:** gmsh `Annulus(refine_lines=[xy], refine_size_min=...)`
  base (places nodes the mover can't create from uniform), then mmpde MAINTAINS;
  **carrier** (windowed rigid rotation / per-fault windowed displacement) to
  transport the refined cluster with a migrating fault. Judge by fault/bulk
  nearest-neighbour spacing RATIO (misalignment is global/useless for the fault).
- **dt:** larger is fine — SLCN is unconditionally stable; do NOT let the
  smallest/median adapted cell govern dt. `estimate_dt(percentile=50) × mult`.
- **Free-slip:** penalty (`add_natural_bc(KFS·v·n·n)`) or Nitsche γ=10.
- **Rheology:** isotropic floored weak fault (`ksponly`, cheap) is the tractable
  path; TI weak fault is expensive (~20 Picard, anisotropic-Jacobian+Nitsche bug
  → use penalty). Viscosity fields P0/P1 only.

## Diagnostics worth re-implementing (the valuable scratch ideas)
- **Mesh quality:** folded-element count + cell-area-ratio + aspect ratio per
  step (the definitive "is the mesh OK" check — NOT the render alone).
- **`_refine_quality`:** per-step fault/bulk + thermal-BL/bulk NN-spacing ratios
  (cKDTree) → npz time series.
- **`mmpde_metric_proof`-style harness:** fixed-metric convergence + R-sweep +
  feedback-loop, to catch mover/metric regressions early.
- **`adapt_vs_uniform_compare` / `multi_history_plot`:** compare runs at MATCHED
  physical time t (dt differs); build a resolved arbiter (finer space+time) when
  unsure whether behaviour is physical.
- **`bc_leak_check`:** v·n on a free-slip boundary (caveat: nodal v·n overstates
  Nitsche "leak" — Nitsche is weak; use vrms/KE as the real indicator).
- **Rendering:** the `uw-visualisation` skill — T+mesh, and T + V **streamlines**
  (sparse seeds, thin lines) for convection; render ALL frames.

## Re-implementation plan
1. **Clean driver** for stagnant-lid annulus convection with adaptive mesh
   (no fault first): mmpde + |∇T| metric + penalty free-slip + SLCN, larger dt,
   render-all + mesh-quality + Nu/vrms history. Validate vs a resolved arbiter.
2. **Add faults**: gmsh refine_lines base + anisotropic tensor metric + carrier;
   re-measure fault/bulk ratio under live convection (the numbers from the
   buggy-mover era must be re-taken).
3. **Diagnostics module** (not scattered scripts): mesh-quality, _refine_quality,
   matched-t comparison, streamline render — as reusable helpers.
4. Land via PRs to `development` (driver + diagnostics; the mover/metric fixes are
   already in).

## Pointers
Skills: `adaptive-meshing`, `uw-visualisation`. Memories:
`project_mmpde_holes_real_root_cause`, `project_fault_refine_fixed_topology_cap`
(gmsh+carrier recipe — ideas sound, numbers re-take), `project_fault_convection_working_settings`.
Reference scratch (this worktree, untracked): `scripts/fault_convection_adapt_loop.py`,
`scripts/mmpde_metric_proof.py`, `scripts/adapt_vs_uniform_compare.py`,
`scripts/bc_leak_check.py`, `scripts/multi_history_plot.py`.
