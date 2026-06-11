# Handover — FMG vs GAMG benchmark figure + adaptive-convection animation

Produced 2026-06-11 from the boundary-slip / anisotropic-mover branch (PR #228).
Two assets: a solver-scaling **benchmark figure** (for `docs/advanced/`) and an
**animation** of the same run (for the PR description / a docs gallery).

## Absolute locations (the worktree may differ — use these)

- **Figure + CSV + this note** — currently *uncommitted* in the feature worktree:
  `/Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/gmg-geometric-interp/docs/advanced/figures/`
  Once PR #228 merges they live in the main checkout at
  `/Users/lmoresi/+Underworld/underworld3-pixi/docs/advanced/figures/`
  (repo-relative `docs/advanced/figures/` — stable in any checkout).
- **Animation, per-step PNGs, render scripts** — on disk, not in the repo:
  `/Users/lmoresi/+Simulations/StagnantLid/anim_full_Ra1e7_dEta1e3_res32_R8_mode1/`
  (frames + GIFs) and `/Users/lmoresi/+Simulations/StagnantLid/` (`_render_clean.py`,
  `_render_watch.py`).
- **Harness** — `scripts/stagnant_lid_adapt_loop.py` in the repo (currently
  `/Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/gmg-geometric-interp/scripts/stagnant_lid_adapt_loop.py`).

The repo-relative paths used in the docs-reference snippets below are intentional
— image references resolve relative to the docs page, not the filesystem.

---

## 1. Benchmark figure — `bench_fmg_vs_gamg_velocity_ksp.svg`

**Files (here, `docs/advanced/figures/`):**
- `bench_fmg_vs_gamg_velocity_ksp.svg` — two-panel figure (text is `svg.fonttype=none`, so selectable/scalable).
- `bench_fmg_vs_gamg_velocity_ksp.csv` — the underlying per-step data:
  `adapt_step, fmg_vel_ksp_its, fmg_snes_reason, fmg_t_stokes_s, gamg_vel_ksp_its, gamg_snes_reason, gamg_t_stokes_s`.

**What it shows:**
- *Top* — inner velocity-block KSP iterations vs adaptation step. FMG (geometric full multigrid) holds a **mesh-independent ~5**; GAMG (algebraic) is **volatile ~64–131 (≈22×)** and does **not** cliff at R=8 over 50 steps.
- *Bottom* — Stokes-solve wall time. GAMG is only **~1.7× FMG**; FMG even spikes to match GAMG around the hardest adapts (steps 25–32) — that spike is the **cold-start Stokes solve** after an adapt (common to both engines), not the preconditioner.
- The **outer Schur-complement KSP converges in 1 iteration for both** — the entire difference lives in the inner velocity block.

**Suggested caption:**
> **Velocity-block solver scaling under adaptive remeshing.** Inner velocity-block
> KSP iterations (top) and Stokes-solve wall time (bottom) versus adaptation step
> for a Ra = 10⁷, Δη = 10³ annulus convection model adapted every timestep with the
> MMPDE mover (res 32, resolution-ratio R = 8, np = 5). Geometric full multigrid
> (FMG) keeps a mesh-independent ≈ 5 inner iterations as the cells stretch, where
> algebraic multigrid (GAMG) runs a volatile ≈ 64–131 (≈ 22×) without cliffing at
> this anisotropy. The wall-clock gap is only ≈ 1.7×: each GAMG V-cycle is far
> cheaper than an FMG F-cycle, and the cold-start Stokes solve after each adapt
> (common to both) dominates the time. The outer Schur KSP converges in a single
> iteration for both — the difference is entirely in the inner velocity block. The
> value of geometric FMG here is *predictability and mesh-independence* (which widen
> with problem size and multigrid levels), not a large raw speed-up.

**Docs wiring** (Markdown/MyST under the benchmark table):
```markdown
![Velocity-block solver scaling under adaptive remeshing](figures/bench_fmg_vs_gamg_velocity_ksp.svg)
```
(Typst equivalent: `#image("figures/bench_fmg_vs_gamg_velocity_ksp.svg")`.)

---

## 2. Animation companion — adaptive convection GIF

**File (NOT in the repo — multi-MB — in the simulation directory):**
`/Users/lmoresi/+Simulations/StagnantLid/anim_full_Ra1e7_dEta1e3_res32_R8_mode1/`
- `anim_clean_Ra1e7_dEta1e3_res32_R8_mode1.gif` — **docs/PR size**: 8 MB, 63 frames (every 4th step), 480 px, clean style (T field + adaptive-mesh edges; no scalebar, streamlines, or text).
- `anim_clean_…_HQ.gif` — 24 MB, 125 frames, 600 px.
- `frame_clean_step0001…0250.png` — all 250 clean PNGs, kept for re-cutting cadence/size.

**What it shows:** the 250-step run — a degree-1 perturbation breaking symmetry into
vigorous, irregular stagnant-lid convection (Nu 1 → 4.15), with the mesh
redistributing every step to track the inner thermal boundary layer and the plumes.
It is the *same run* whose Stokes solves stay flat under FMG in the figure above.

**Suggested caption:**
> Adaptive-mesh convection in an annulus (Ra = 10⁷, Δη = 10³): temperature with the
> MMPDE-adapted mesh redistributing every timestep to follow the thermal boundary
> layer and plumes, mesh-owned tangent-slip keeping boundary nodes on the curved
> surface. 250 steps, np = 5 — the same run whose velocity solves stay
> mesh-independent under geometric FMG (see the benchmark figure).

**Placement:** for the PR description, drag-drop the 8 MB GIF into the #228 comment
box via the GitHub web UI (under the 10 MB limit; `gh` can't upload binary
attachments). For docs, host/link it or commit a lighter cut — don't commit the
multi-MB GIF to the repo.

---

## Reproduction

Run (boundary-slip / mover branch; `scripts/stagnant_lid_adapt_loop.py`):
```bash
REFINE=2 PCVEL=gmg MG_TYPE=full MOVER=mmpde MMPDE_ACCEL=cg MOVER_SLIP=ring \
mpirun -np 5 python scripts/stagnant_lid_adapt_loop.py \
  --from-perturbation --pert-mode 1 --Ra 1e7 --delta-eta 1e3 \
  --dt-cell-percentile 50 --skip-threshold 99 --adapt-every 1 \
  --resolution-ratio 8 --n-steps 250 --res 32 --snapshot-every 1 --out-tag <name>
```
- **FMG vs GAMG** (figure data): same config; `PCVEL=gmg MG_TYPE=full` (FMG) vs
  `PCVEL=amg` (GAMG), 50 steps. Inner velocity iterations captured with a temporary
  `FMG_DIAG` env-gated diagnostic in the harness
  (`stokes.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getIterationNumber()`,
  plus `len(mesh.dm_hierarchy)` and the converged reason) — reverted after the runs.
- **Prerequisite** for FMG: the mesh must be built with `refinement` (so
  `len(mesh.dm_hierarchy) > 1`) and adapted with a *coordinate-deforming* mover
  (MMPDE preserves topology, so the hierarchy survives — confirmed `levels=3` on all
  50 steps). A checkpoint-resumed mesh has **no** hierarchy (it isn't persisted), so
  `--resume` cannot be used for the FMG arm.
- **Render:** `_render_clean.py <run_dir>` (clean — no scalebar/streamlines) or
  `_render_watch.py <run_dir>` (default — with scalebar + streamlines), both in
  `/Users/lmoresi/+Simulations/StagnantLid/`. GIF assembled with Pillow (crop white margin →
  resize → sample frames).

## Status
- SVG + CSV + this note: **uncommitted** in `docs/advanced/figures/` — commit them
  with the docs reference.
- The harness is at its committed state (the `FMG_DIAG` diagnostic was temporary).
- PR #228 (movers consume `mesh.boundary_slip` + the anisotropic-mover work) is
  green and mergeable; this benchmark is its "why this works" companion.
