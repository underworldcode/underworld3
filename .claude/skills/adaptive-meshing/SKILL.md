---
name: adaptive-meshing
description: The canonical, workable recipe for Underworld3 moving-mesh / adaptive-mesh convection (annulus stagnant-lid, faults, free surface). Reach for THIS first when setting up any model with a deforming or adapted mesh — it encodes the combination that does not blow up, tangle, or inject spurious energy, and explains the failure modes so you don't re-derive them. Use before choosing movers, free-slip BCs, restart, or field-transfer options.
---

# adaptive-meshing

The one workable combination for UW3 moving/adaptive-mesh convection, distilled
from many sessions that each re-picked options and stomped on each other's
defaults. **Start from this recipe; change one thing at a time and verify.**

Reference implementation (current, validated): the **`underworld3.workflows`
adaptive-convection example** —
`docs/examples/workflows/adaptive_convection/` on the `feature/adaptive-convection`
worktree (`config.py`+`simulate.py` no-fault; `fault_config.py`+`fault_simulate.py`
fault; `diagnostics.py`, `render.py`, `compare.py`). Express adaptive runs as a
WORKFLOW (a `WorkflowConfig` + `@workflow_step` DAG + `Run`), NOT a monolithic
driver. The older `scripts/fault_convection_adapt_loop.py` (feature/fault-convection)
is superseded — its ideas are folded into the workflow + this skill.
Companion: the `uw-visualisation` skill for rendering results.

**Choosing the paradigm:** THIS skill is the **mover** (node movement /
equidistribution, `smooth_mesh_interior`) — the mesh deforms to follow a field. For
**local refinement** instead (`mesh.adapt(engine="nvb")` returns a refined CHILD; a
fault resolved by a fine band + custom-P FMG + rotated free-slip + dynamic topography
+ advection-diffusion), use the **`adapt-on-top-faults`** skill. Different tools —
don't mix them.

---

## Mover quick-start (copy-paste — this is the hard-to-discover bit)

The mesh mover is `uw.meshing.smooth_mesh_interior`. Minimal correct setup to
adapt a mesh to a field `T` each step:

```python
import underworld3 as uw

# metric from |grad T|: refinement = finest:coarsest cell-size ratio (~5).
# Use refinement=R, NOT strategy= (strategy caps at ~2 and under-grades).
rho = uw.meshing.metric_density_from_gradient(
    mesh, T, refinement=5, coarsening="auto", metric_choice="front-following")

# move the mesh — mmpde: variational, non-folding, clusters AND aligns cells.
# It OWNS field transfer (remaps T + SLCN history, fires on_remesh hooks).
uw.meshing.smooth_mesh_interior(
    mesh, metric=rho, method="mmpde",
    method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0),  # mmpde's OWN kwargs
    slip_surfaces=True,        # boundary nodes slide tangentially (parallel-safe)
    skip_threshold=0.9)        # skip the move when the mesh is already aligned
```

For a **sharp feature (fault)** pass an anisotropic SPD TENSOR metric instead of
the scalar `rho` (thin ACROSS the feature normal n) and bake a gmsh base:

```python
import sympy
n = sympy.Matrix([nx, ny])                       # constant fault-normal unit vector
d = dfac.sym[0]                                  # DIRECT unsigned distance field (P1)
M = rho * sympy.eye(2) + (Rf**2 - 1.0) * sympy.exp(-(d/w)**2) * (n * n.T)
# mesh built with: uw.meshing.Annulus(..., refine_lines=[xy], refine_size_min=smin)
uw.meshing.smooth_mesh_interior(mesh, metric=M, method="mmpde",
    method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0),
    slip_surfaces=True, skip_threshold=None)     # tensor metric: do the skip check yourself
```

Pitfalls that make it "not work": `method="anisotropic"`/`"ot"`/`"spring"`/`"ma"`
(RETIRED 2026-07 — they now raise ValueError; mmpde is the default and only
metric mover); injecting `relax`/`n_outer` (starves mmpde's CG); `strategy=` instead of
`refinement=R` (under-grades); a scalar bump for a fault (refines a fat corridor,
leaves the centre coarse); signed `Surface.distance.sym` for `d` (bleeds along the
line extension — use a direct unsigned distance). Full rationale + the rest of the
recipe (BCs, restart, field transfer, cadence) below.

---

## The canonical recipe (defaults that work)

### 1. Mover — mmpde
`uw.meshing.smooth_mesh_interior(mesh, metric=..., method="mmpde",
method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), slip_surfaces=True)`.
- mmpde = Huang–Kamenski variational, **non-folding** (energy → ∞ as detJ → 0),
  clusters AND aligns cells. It is the only clean mover.
- **NOT** the `anisotropic` mover (shreds/freezes on static features) or `OT`
  (slivers — OT is optimal transport, sliver-prone, not a route around the cap).
- Do **not** inject `relax`/`n_outer` into mmpde (those are the anisotropic
  mover's knobs and starve mmpde's internal CG).

### 2. Metric
- Thermal: `metric_density_from_gradient(mesh, T, refinement=R,
  metric_choice="front-following")` — `refinement=R` (≈5) is the finest:coarsest
  grading ratio; named `strategy=` caps at ~2 and under-grades. R≈5 extracts ~all
  the grading the node budget/layout allows; don't over-tune R (benign no-op above
  budget).
- Fault / sharp feature: a **hand-built anisotropic SPD tensor**
  `M = ρ·I + (Rf²−1)·exp(−(d/w)²)·n nᵀ` (thin ACROSS the feature normal n). A
  scalar bump refines a fat isotropic corridor and leaves the centre-line coarse.
- Use a DIRECT unsigned distance field (geometry_tools) for `d`, NOT the Surface's
  signed `.distance.sym` (its zero-contour bleeds the metric along the line
  extension).

### 3. Creation vs maintenance (the cap) — and the gmsh base COMPOUNDS
mmpde **cannot create** strong refinement from a uniform mesh — it saturates at a
fixed-topology cap (~1.8× on the fault, re-measured), because the mover has a FIXED
node budget (it redistributes, never adds nodes). To go finer you need MORE NODES,
which only gmsh can add at construction. **Bake refinement into the gmsh base**
(`Annulus(refine_lines=[xy], refine_size_min=...)` — now real, see the Faults
section) and the mover doesn't just MAINTAIN it: the extra gmsh nodes lift it off
its budget cap so it **compounds** (gmsh f2 base 0.44 → mover 0.29; f3 base 0.30 →
mover 0.19 ≈ 5× finer). Measured (Ra1e6 Rf8 res24, all folded=0):
uniform 0.55 (~1.8×) → gmsh-f2 0.29 (~3.4×) → gmsh-f3 0.19 (~5×). Judge by
fault/bulk nearest-neighbour spacing RATIO, never by global misalignment.

### 4. Cadence — adapt as often as you like (forced every step is FINE)
Adapt every step or every few — on a CORRECT build (see §5) the mover converges
dead-flat and forced every-step adaptation under vigorous convection is stable
(validated: 39/39 forced adapts, mesh folded=0, area-ratio ~14 flat). Skipping
when aligned just saves cost. (An earlier claim that forcing adaptation
"tangles / over-injects energy" was WRONG — that was the deformed-eval bug in §5.)

### 5. THE bug that wrecked adaptation (holes) — and the fix
SYMPTOM: giant empty cells / holes in the adapted mesh, intermittent area-ratio
spikes, convection wrecked. ROOT CAUSE (proven): `uw.function.evaluate`
**mis-locates points on a deformed mesh** (the nav kd-tree `mesh._nav_coords` was
captured from the ORIGINAL coords and never refreshed), so the metric — built with
strictly-positive nodal values — evaluates to **NEGATIVE garbage** (even at its own
DOFs) → non-SPD → the mover wrecks the mesh. FIX (both):
- **`8a9d2ff2`** (refresh `_nav_coords` + projected normals on every deform) —
  the real fix; makes `function.evaluate`/`points_in_domain` track deformation.
- **Monotone RBF metric bake** (in `_mmpde_mover`, formerly `_winslow_mmpde`): Shepard-interpolate the
  metric from its **positive nodal values** — a convex average is guaranteed ≥0
  (monotone) + fast (no cell-location). Use RBF for the metric; it doesn't need
  high-precision eval.
With these, the metric stays positive/SPD; #259's SPD-floor never fires (harmless).
The mover, `accel`, and `refinement=R` (e.g. R=5) are all FINE — they were
red-herring symptoms of the eval bug. Always confirm a CLEAN BUILD first
(`./uw build`; `md5 site-packages/.../smoothing.py == src`); a stale build is its
own cause of holes.

### 6. Stokes free-slip
Penalty (`add_natural_bc(KFS·v·n·n)`, KFS≈1e6) or Nitsche γ=10 both work on a
UNIFORM mesh. Nitsche preferred for sharp fault corners. Do NOT diagnose free-slip
with nodal v·n: Nitsche enforces v·n=0 weakly, so nodal v·n is large even when
correct — use vrms (kinetic energy). (An earlier "warm-restart × Nitsche → blow-up,
use cold restart" diagnosis was WRONG / confounded by the §5 eval bug + a stale build.)

**On an ADAPTIVE mesh, Nitsche free-slip γ=10 is TOO SOFT → intermittent vrms
spikes at adapt steps** (under-enforcement, NOT a blow-up: single-step velocity
garbage that T survives because the solve is cold-restarted — e.g. vrms
144→8887→144). The mover pins the boundary and refines just beneath it, creating
high-aspect-ratio near-boundary cells whose Nitsche inverse-estimate constant needs
a bigger penalty. Two fixes, BOTH good (corroborated across sessions):
- **Nitsche γ=100** (not 10) for a free-slip top on an adaptive mesh — clean, smooth
  signal. This is INDEPENDENT of the penalty's mesh-size scaling: local vs global `h`
  are equivalent here (both garbage at γ=10, both clean at γ=100).
- **Don't fully pin the boundary in the mover** — let it tangential-slip
  (`slip_surfaces=True`, the §1 default), NOT `pinned_labels=[Upper,...]`. Avoiding
  the distorted near-boundary cells lets γ=10 work. Pin a boundary only when you must
  hold a prescribed shape (e.g. a free-surface height the integrator just set).

Aside — free-SURFACE held-lid is the OPPOSITE problem: the GLOBAL `h =
get_min_radius` drifts BELOW the surface cell as the interior refines → the Nitsche
penalty OVER-stiffens → a spurious one-step surface "mountain" (vhmax spike). Fix =
a LOCAL per-cell `h` via `mesh.cell_size()` (deformation/adaptation-tracking);
`add_nitsche_bc(..., local_h=True)` is the default (PR #275). So held-lid wants LESS
penalty (local-h), free-slip-adaptive wants MORE (γ=100) — different knobs.

### 7. Advection + timestep
`AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V.sym, theta=1.0, monotone_mode='clamp')`.
theta=1.0 (backward Euler) for stability; monotone clamp bounds SL overshoot;
`V_fn=V.sym` is the PHYSICAL velocity. **dt can be LARGE — SLCN is unconditionally
stable; the smallest/median adapted cell does NOT have to govern dt.** Use
`estimate_dt(percentile=50)` × a multiplier (e.g. dt_mult 3–5) to advance physical
time faster (needed to develop convection in stiff stagnant-lid regimes).

### 8. Rheology
- **isotropic** (linear) ⇒ `snes_type=ksponly` (one exact KSP solve; default
  newtonls rejects steps on vigorous flow). Cheap. Use for resolved features.
- **TI / anisotropic** weak fault (`TransverseIsotropicFlowModel`:
  `shear_viscosity_0=η_FK`, `shear_viscosity_1=η_weak`, `director`=fault normal):
  keep the **default newtonls** (NOT ksponly — ksponly converges to the WRONG
  answer); use **penalty** free-slip (Nitsche trips the anisotropic-Jacobian bug);
  **GAMG** (FMG ~7× slower). Measured on the gmsh-resolved + one-sided base
  (Ra1e6 Δη1e3): ran clean to t=0.06, folded=0, vrms→20, **~10×** the isotropic
  cost per step (GAMG eats the 1000× anisotropic contrast — well under the feared
  20×). The TI fault visibly steers the flow (persistent recirculation at the trace).
- Viscosity-bearing fields are **P0/P1 only** (positivity; higher order overshoots).
  FK viscosity `η = exp(θ(1−T))`, θ = ln(Δη). Floor a weak zone, don't multiply → 0.

### 9. Build discipline
- `./uw build` after ANY source change; verify `uw.__file__` is in site-packages
  and (when debugging) that `site-packages/.../smoothing.py` matches `src/...`. A
  **stale mover build is a top cause of "giant empty elements / holes"** in the
  adapted mesh.
- **NEVER** `pip install -e .` (contaminates all envs). Run from inside the worktree
  (`pixi run -e amr-dev`).

---

## Faults — the gmsh-resolved, on-fault, one-sided recipe (validated 2026-06-23)

The full fault recipe, hard-won. Reference implementation: the
`underworld3.workflows` adaptive-convection example
(`docs/examples/workflows/adaptive_convection/{config,fault_config}.py`, on the
`feature/adaptive-convection` worktree — built on the FIXED mover base). Use the
WORKFLOW system, not a monolithic driver.

### gmsh line refinement (`refine_lines`) — NOW IMPLEMENTED in `Annulus`
```python
uw.meshing.Annulus(radiusOuter=1, radiusInner=0.5, cellSize=1/24,
    refine_lines=[xy],            # list of (N,2) polylines (model coords)
    refine_size_min=cellSize/3,   # cell size ON the line (factor 2-3 is plenty)
    refine_dist_min=0.02, refine_dist_max=0.12)   # size ramps back to cellSize
```
A gmsh **Distance + Threshold** field along the polyline; INTERIOR points are
embedded so nodes land ON the line. Backward-compatible (default `None`). It is a
core meshing change → lands as its own small meshing PR, separate from the workflow
example. (Before this session it was only *called* in old scripts and never existed
— don't trust `refine_lines` on any branch but the one carrying this commit.)

### Keeping refinement ON the fault (the metric-composition trap)
The fault metric is `M = ρ·I + (Rf²−1)·exp(−(d/wₙ)²)·n nᵀ` with the isotropic SIZE
density `ρ`. **Do NOT fuse the fault density into ρ_T by PRODUCT** — the cold
surface thermal BL (ρ_T ~ R^d ~ 20–25) out-competes the fault near its top and the
refinement drifts ABOVE the fault, starving the deep fault ("seems to repel"):
- Use `ρ = max(ρ_T, fault_ρ)` (NOT `ρ_T · fault_ρ`).
- Make `fault_ρ = 1 + amp·gauss` with **amp > ρ_T** (~25 at R=5) so the fault wins
  the max along its whole length.
- DIAGNOSE this by comparing the gmsh BASE (step 0, refinement centered on the
  fault — correct) vs the DEVELOPED mesh (drifted above) → it's the MOVER's metric,
  not gmsh. Render the mesh with the **fault trace overlaid** (`render.py --fault`).

### One-sided fault influence (the clean control)
Even with max+amp, the symmetric metric DEMANDS both flanks while realized nodes
drift to the hanging wall. Make it one-sided:
- Store the **SIGNED** distance in `dfac` (the gaussians square it, so magnitude is
  unchanged — the sign only feeds a `0.5(1+tanh(m·d/w))` gate). Probe the
  radially-outward side once to define "upper" regardless of the distance tool's
  orientation convention.
- `fault_metric_side` (both/upper/lower) gates the refinement; `fault_rheology_side`
  gates the weak zone. **`both=upper`** is the physical recipe: a one-sided
  hanging-wall damage zone with the mesh refined on the same side (refinement and
  rheology coincide; gmsh-f3 → fault/bulk ~0.19, folded=0). `metric_side=lower`
  instead pulls refinement onto the footwall to counter the upward drift.

### The wedge fill (anti-collision)
The fault pull and the surface-BL pull compete for the coarse cells in the radial
sliver BETWEEN them. `fault_wedge=True` gmsh-fills that wedge (sample radial
segments from each fault point up to the surface, add as a second `refine_lines`
point set) so both pulls have their own budget and merge into one coherent fine
wedge instead of colliding.

### Weak zone (rheology) — geometric blend + gaussian PEAKED on the fault (2026-06-24)
The weak-zone viscosity blends `η_FK` (background) and `floor` (fault) by the
influence `f`∈[0,1] (P1, positive). Get it right with TWO rules (verify by
reconstructing + rendering the REALIZED η field — `render_fields.py` — and the
combined `η_FK(T)^(1−f)` field; never assume the floor is reached):
- **GEOMETRIC blend `η_weak = η_FK^(1−f)·floor^f`** (NOT arithmetic
  `η_FK·(1−f)+floor·f`, whose `(1−f)` term leaks the stiff-lid background through
  → η≈8 at f=0.97 in a 1000× lid). Geometric reaches the floor genuinely (η≈1.2 at
  f=0.97) AND ties the contrast to the LOCAL η_FK — so the fault automatically
  bites hardest where it cuts the cold stiff lid (physically correct), nothing in
  the hot interior.
- **`f` must be PEAKED on the fault (gaussian), NOT a TOPHAT block.** A top-hat
  makes a uniform weak BLOCK; its sharp edges mean strain follows ∇f (TWO parallel
  lines at the block edges, not on the fault), and a one-sided block is offset to
  the hanging wall (η_1=1 core sits ABOVE the drawn line). Use a **gaussian
  `f=exp(−(d/w)²)`** (peak f=1 ON the fault → η_1=1 on the drawn line, strain
  localizes INTO the slot as a single feature). side=both = symmetric; side=upper =
  hanging-wall halo (gaussian taper up, sharp footwall recovery — NO halving gate).
  Centre the METRIC too (`metric_side=both`) so nodes refine on the line.
THERMAL CONTRAST (Louis's insight, confirmed): even a symmetric gaussian gives a
much bigger η-contrast on the COLD (upper/surface) side than the warm (footwall)
side, because η_FK rises ~1000× toward the surface — so the fault's dynamical
prominence naturally concentrates in the cold lid. This is a feature, not a bug.
Verified (gmsh-f3, gaussian width=0.025, side+metric=both): weak zone centred on
the drawn fault on the adapted mesh, folded=0. TUNE AT STEP 0 (build mesh+fields,
no solve — fast; plot the 1D η_1(d) profile + the 2D field). COST: genuine 1000×
TI contrast ~25–145 s/step (cold-start steps slow, ~25 s once developed; GAMG).
For TI see §8. (`fault_config.py`: `fault_profile=gaussian` (default still tophat —
pass gaussian), geometric blend in `create_solvers`; `render_fields.py` light maps.)

---

## Failure modes — symptom → cause → fix

| Symptom | Cause | Fix |
|---|---|---|
| Giant empty cells / holes in adapted mesh; intermittent area-ratio spikes | `function.evaluate` mis-locates on deformed mesh → metric → negative/non-SPD (the real bug). OR stale build. | `8a9d2ff2` + monotone RBF metric bake; `./uw build` + verify md5 site-packages==src |
| Decays when it should convect | over-diffusion OR under-resolution OR dt too small to develop | check a resolved arbiter (finer space+time); raise node budget; **larger dt** (dt_mult 3–5) |
| Fault won't refine under convection | mmpde creation cap; field gives no signal in cold lid | gmsh `refine_lines` base — the gmsh nodes let mmpde COMPOUND past the cap (~5×) |
| Refinement drifts ABOVE the fault / "repels", deep fault starved | fault density fused by PRODUCT with ρ_T → thermal BL out-competes the fault near the surface | `ρ = max(ρ_T, fault_ρ)`; `amp > ρ_T` (~25); or one-sided `fault_metric_side`; +`fault_wedge` |
| Refinement on both flanks but you want one side | symmetric (unsigned-distance) metric | signed `dfac` + `fault_metric_side`/`fault_rheology_side` tanh gate (`both=upper` = physical) |
| TI fault solve: ksponly gives wrong answer | ksponly skips the Picard the inexact GAMG inner solve needs | keep default newtonls; penalty free-slip; GAMG (~10× isotropic cost, fine) |
| free-slip ADAPTIVE: vrms spikes at adapt steps (e.g. 144→8887→144), T stays bounded | Nitsche γ=10 too soft on the distorted near-boundary cells the pinned-top+interior-refine creates (under-enforcement) | **Nitsche γ=100** (not 10); OR don't pin the boundary (tangential-slip `slip_surfaces=True`). h-scaling (local vs global) is moot here |
| free-SURFACE held-lid: spurious one-step surface "mountain" / vhmax spike | global `h=get_min_radius` drifts below the surface cell as interior refines → Nitsche OVER-stiff | LOCAL per-cell h: `add_nitsche_bc(local_h=True)` (default, PR #275) = `mesh.cell_size()` |

**Diagnose by:** vrms (KE), Nu (`BdIntegral` surface flux), fault/bulk NN-spacing
RATIO (cKDTree), folded-element count + min cell area. Compare runs **at matched
physical time t** (dt differs between meshes), not by step number. When unsure
whether behaviour is physical, build a **resolved arbiter** (uniform mesh finer in
BOTH space and time) — if it agrees with one candidate, that's the truth.

## Diagnostics (in the workflow example, reusable)
`diagnostics.py`: `mesh_quality` (folded / area-ratio / aspect), `nn_spacing_ratios`
(BL + fault/bulk), `NusseltSurface`, `vrms`, `History`. `render.py`: T+mesh+
streamlines on the `Run` layout, **`--fault`** overlays the fault trace (read from
the run manifest) + **`--focus-fault`** auto-crops on it + **`--mesh-only`** for the
clean mesh, **`--all`** for every frame. `compare.py`/`fault_refine_plot.py`:
matched-physical-time comparison + fault/bulk-ratio time series.
**Rendering long runs as they go:** a completion-only Monitor is NOT enough — arm a
Monitor that POLLS for new `run.mesh.NNNNN.xdmf` checkpoints and emits the index, so
you render each step as it lands.
**Checkpoint an ADAPTIVE mesh with `meshUpdates=True`** (per-step geometry) or the
saved frames pair deformed fields with stale step-0 geometry.

## The verified canonical command

Requires the fix (§5): `8a9d2ff2` (deformed-mesh point-location) + the monotone RBF
metric bake in `smoothing.py`. Validated 2026-06-22 at vigorous Ra1e6/Δη1e3
stagnant-lid: forced adapt EVERY step (39/39), larger dt, → vrms→18, Nu→1.86,
|v|max 61, mesh CLEAN every frame (folded=0, area-ratio ~14 flat), no abort.

```bash
# no-fault baseline (the workflow CLI; one flag per config field)
pixi run -e amr-dev python docs/examples/workflows/adaptive_convection/simulate.py \
  --output-dir ~/+Simulations/<study>/baseline \
  --rayleigh 1e6 --delta-eta 1e3 --cellsize 0.0417 \
  --resolution-ratio 5 --adapt-every 1 --dt-mult 4 --max-steps 80 --max-t 0.06

# resolved fault: gmsh base (factor 3) + on-fault + one-sided hanging wall + TI
pixi run -e amr-dev python docs/examples/workflows/adaptive_convection/fault_simulate.py \
  --output-dir ~/+Simulations/<study>/fault \
  --rayleigh 1e6 --delta-eta 1e3 --cellsize 0.0417 --resolution-ratio 5 \
  --fault-base-smin 0.0139 --fault-anisotropy 8 \
  --metric-combine max --fault-refine-amp 25 \
  --fault-rheology-side upper --fault-metric-side upper \
  --rheology ti --dt-mult 4 --max-steps 40 --max-t 0.06
```

Key choices, verified:
- **`--resolution-ratio 5`** (R=5) is fine — the mover handles it on a correct build.
- **`--adapt-every 1`** force adapt every step (the strongest mesh test).
- **`--dt-mult 4`** — larger dt for STABILITY is fine (SLCN unconditional); but it
  costs transient ACCURACY (over-diffusive backward-Euler DELAYS the convective
  onset vs a resolved arbiter — dt×1.5 recovers it). Use small dt_mult for faithful
  transients, large to reach quasi-steady fast.
- **`--freeslip penalty`** (default) — REQUIRED for TI (Nitsche trips the
  anisotropic-Jacobian bug). It's the raw velocity penalty `kfs·(v·n)n`.
- Fault: `--fault-base-smin` (gmsh resolve), `--metric-combine max` +
  `--fault-refine-amp 25` (keep refinement ON the fault), `--fault-*-side` (one-sided),
  `--rheology ti` (real fault). Drop the fault flags for the no-fault control.

Render with `render.py` (`--fault --focus-fault` to see the trace + refinement
coincide; `--mesh-only`; `--all`). Judge the mesh by folded/area-ratio, the physics
by vrms/Nu, ALWAYS at matched physical time.

## Related memory
`project_adaptive_convection_as_workflow` (THIS session: workflow port, gmsh
refine_lines, on-fault/one-sided/wedge, TI, dt-accuracy), `project_mmpde_holes_real_root_cause`,
`project_fault_refine_fixed_topology_cap`, `project_uw_workflow_landing`,
`feedback_debug_adaptive_solver_method`, `project_fault_convection_working_settings`.
