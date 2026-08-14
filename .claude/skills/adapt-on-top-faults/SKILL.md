---
name: adapt-on-top-faults
description: Recipe for Underworld3 FAULT models on an NVB adapt-on-top mesh — resolve a fault Surface by LOCAL refinement (mesh.adapt(metric, max_levels=...) returns a child; NVB is the 2D default engine), drive it from the fault's EXACT signed distance, use rotated strong free-slip (composes with transverse-isotropy where Nitsche does not), recover dynamic topography from the constraint reaction, and run advection-diffusion on the adapted mesh with field transfer across re-adaptation. Reach for THIS for instantaneous/coupled fault-flow problems. For MMPDE node-movement convection use the `adaptive-meshing` skill instead; for rendering use `uw-visualisation`.
---

# adapt-on-top-faults

The validated recipe for **fault problems on a locally-refined (adapt-on-top) mesh**.
Distilled from the annulus fault study (2026-07, `feature/adapt-on-top`).

**This is the REFINEMENT paradigm**, not the mover one:
- `mesh.adapt(metric, max_levels=...)` bisects the base finest **locally** and returns
  a **new child mesh** (`child.parent is mesh`). It is *adapt / re-adapt*, NOT node
  movement — non-cumulative (each call re-marks from the static base). The child owns
  a custom-P geometric-MG (FMG) tail so solvers on it get multigrid for free.
- For **MMPDE / equidistribution node movement** (deforming the same mesh to a field)
  use the **`adaptive-meshing`** skill instead. Different tool; don't mix them up.

Reference implementations (copy from these — all run np1/2/4):
`~/+Simulations/nvb_parallel_fault_study/` (weak-fault Stokes + FMG),
`~/+Simulations/shear_box_fault_study/` (iso vs TI, orientation sweeps),
`~/+Simulations/annulus_fault_study/` (rotated free-slip + topography + moving fault +
advection-diffusion). Companion skills: `uw-visualisation`, `adaptive-meshing`.

---

## The core loop (fault → metric → adapt → child)

```python
import underworld3 as uw, numpy as np, sympy

# Base mesh MUST be built with refinement>=1 (supplies the coarse MG tail that NVB
# extends). Base cellSize ~ 2x the target h_near is the SWEET SPOT (see gotchas).
base = uw.meshing.Annulus(radiusInner=0.55, radiusOuter=1.0, cellSize=0.08,
                          refinement=1, qdegree=3)          # or UnstructuredSimplexBox(..., refinement=2)

# fault as a Surface (polyline control points, N x 3 with z=0 in 2D)
fault = uw.meshing.Surface("fault", base, fault_pts, symbol="F")
fault.discretize()

# METRIC = a CALLABLE built from the fault's EXACT signed distance. This is the key
# to clean, non-patchy grading: it is evaluated at each refined level's centroids,
# so it resolves itself at the new resolution (no P1-field aliasing).
metric = fault.refinement_metric_function(h_near=0.02, h_far=0.08, width=0.05,
                                          profile="linear")
child = base.adapt(metric, max_levels=3)     # -> graded child (NVB is the 2D default)
```

- NVB (the 2D default engine) = graded newest-vertex bisection (bounded closure,
  parallel via the native `uwnvb` transform; bit-confluent serial↔parallel);
  `engine=` is the advanced selector. `engine="sbr"` = uniform
  patch (the default; not graded). NVB is 2D only for now.
- `max_levels` is the isotropic-equivalent depth (NVB runs `2*max_levels` bisection
  passes). The metric shape decides the grading; `max_levels` just caps it.

### Metric options (all accepted by `adapt`)
1. **callable** `metric(centroids)->M` — **preferred for faults**. Evaluated per level.
2. MeshVariable / sympy expression — sampled via `uw.function.evaluate` from the BASE
   mesh → a peaked `M=1/h²` aliases → *patchy* levels. Avoid for thin features.

**Custom refinement shape** — pass any callable. For a *fat, uniformly-fine* band
(not just a thin line at the fault), a flat-core metric:
```python
def metric(pts, _f=fault):
    d = _f.unsigned_distance(pts)               # EXACT distance at arbitrary points
    core, ramp, hn, hf = 0.02, 0.05, 0.01, 0.08
    h = np.where(d < core, hn, np.minimum(hn + (hf-hn)*(d-core)/ramp, hf))
    return 1.0 / h**2
```

`Surface` distance API (all exact, arbitrary query points):
`fault.signed_distance(coords)`, `fault.unsigned_distance(coords)`,
`fault.director` (unit normal = normalised ∇(signed distance); the TI weak-plane
director), `fault.refinement_metric_function(...)`.

---

## Engines: `nvb` vs `edge_split` — and what runs in parallel

`engine="nvb"` (default) is graded newest-vertex bisection: bounded conforming
closure, a similarity-class bound that keeps child quality tied to the base, and
**partition-independent** output. `engine="edge_split"` splits the **longest edge**
of every cell coarser than the metric asks for and needs **no conforming closure
at all**, because splitting an edge divides every incident cell at the same new
vertex. Consequences:

- refinement **cannot escape the marked region** — the band hugs the feature
  instead of a halo around it;
- it marks on the cell **DIAMETER**, not `(dim!·vol)^(1/dim)`. The volume proxy
  reported the target met while the mesh was **3.2× coarser** across the feature;
- it gives up the similarity-class bound, so quality at depth is not guaranteed
  the way bisection's is — that is what `repair=` and `relax()` are for.

**Both run in parallel, 2-D and 3-D, and both are bit-confluent** (identical mesh
at any communicator size). `edge_split` drives the same compiled `uwnvb_bisect`
transform as NVB, so it inherits star-forest propagation, co-partitioning, labels
and coordinates. Verified at np=1/2/3/4 up to 56k cells.

```python
child = base.adapt(metric, max_levels=3, engine="edge_split")
child = base.adapt(metric, max_levels=3, engine="edge_split", repair=True)
```

`repair=True` runs a **reconnection (Lawson flip) pass** after each generation —
2-D and `edge_split` only; it raises rather than silently doing nothing otherwise.
It gates on **reducing the largest angle**, NOT on Delaunay: Delaunay maximises the
*minimum* angle while P1 interpolation depends on the *maximum* (Babuška–Aziz), and
flipping a gmsh mesh toward Delaunay was measured to RAISE the 99th-percentile max
angle 126.8° → 129.3°. gmsh optimises shape, not the empty-circle property.

- **worth it on a POOR base** — anisotropic, graded, relaxed, or read from a file:
  99th-pct max angle 156° → 115°, slivers below q=0.1 3.84 % → 0.00 %. On a clean
  gmsh base it moves 124.7° → 120.5° and the error not at all.
- ⚠️ **it gives up bit-confluence.** Which cavities may be flipped depends on where
  the partitioner cut (no cavity may contain a cell incident on a shared point).
  Conformity, orientation, volume, labels and the SF stay exact at every rank
  count; only the choice of flips near a seam differs. Hence opt-in.
- Seam cost is small and **shrinks with resolution**: frozen repair sites 0.9–3.5 %
  at 56k cells, np=2..8, halving with every halving of the target size. In a fault
  band specifically, 5.5 % at np=2 and 13 % at np=4 on a 4k-cell mesh.
- ⚠️ the 99th-pct angle recovers under a frozen seam but the **absolute max does
  not** — a few worst cells sit on the seam (148° vs 123° serial).
- it invalidates the cell-parent map for the any-degree MG transfer (a flipped
  cell can straddle two coarse cells), so degree ≥ 2 falls back to the geometric
  prolongation builder. The exact vertex prolongation survives — flips move no
  vertex.

## Relaxing an adapted fault mesh — PIN THE BAND

`child.relax()` on a mesh refined onto an interface **makes things worse**. The
MMPDE mover optimises element shape against an equilateral reference and knows
nothing about where the material changes, so it slides the small cells that
refinement placed on the interface *off* it. Measured on a step-edged fault:
manufactured stress across the interface **+77 %**, and it stopped being confined
to the fault. Counter-intuitively it *reduces* the number of straddling cells
(1343 → 965) and is still worse, because the survivors are bigger.

```python
child.relax(pin_bands=[fault])                    # interface = the surface itself
child.relax(pin_bands=[(fault, 0.02)], pin_halo=2)  # weak zone of half-width 0.02
```

Leak unchanged to five decimal places (0.03075 → 0.03076), confinement preserved,
straddling count identical — while the mover still reshapes the rest of the
domain. `pin_halo` (default 1) pins extra rings; pinning only the cut cells lets
the mover pull on them from outside. `pin_bands` **merges** with the auto-pinned
boundaries, so it cannot silently release the domain edge.

## Fault as a constitutive weak zone (iso and TI)

The metric only needs the fault GEOMETRY (pure distance). The constitutive weak zone
needs the fault's distance/normal ON THE CHILD. Two ways:

```python
# (A) re-home the SAME fault onto the child (cleanest; distance recomputes on child)
fault.remap_to(child)
eta = fault.influence_function(width=0.04, value_near=1e-3, value_far=1.0,
                               profile="smoothstep")     # isotropic weak zone

# (B) or build a child-side Surface (needed if the base fault is still in use for a
#     base-mesh metric — symbol disambiguation refuses a base-mesh symbol in a child
#     solver). fault_c = uw.meshing.Surface("fault_c", child, fault_pts); fault_c.discretize()
```

Isotropic weak zone:
```python
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta            # drops near fault
```

Transverse-isotropic weak PLANE (the physical fault; low fault-parallel shear):
```python
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_bulk        # normal viscosity (constant)
stokes.constitutive_model.Parameters.shear_viscosity_1 = eta             # weak near fault -> bulk far
stokes.constitutive_model.Parameters.director          = fault.director  # unit fault normal
```
The **TI Jacobian is the full consistent tangent** (not isotropic + defect
correction — that framing is WRONG). The TI velocity FMG V-cycle needs a few more
iters than iso (~8 vs ~2) because of a directional near-null mode the isotropic
point-smoother doesn't damp — bounded and contrast-independent, not a bug. TI needs
~2 elements across the weak zone to resolve (iso ~1).

---

## Rotated strong free-slip (the reason to use this, not Nitsche)

Nitsche free-slip is INCOMPATIBLE with the TI model (its penalty scales by an
isotropic viscosity). Rotated strong free-slip imposes `u·n̂=0` as an ESSENTIAL
constraint in a per-node (n,t) frame → machine-zero leakage AND composes with TI.

```python
# normal=None (the default) is measure-weighted and consistent with the assembly —
# prefer it. An analytic nhat is exact for the TRUE circle but keeps a consistency
# error against the faceted integral (#560); use it only when the constraint must
# follow the geometry rather than the mesh.
nhat = mesh.CoordinateSystem.unit_e_0                    # exact radial normal (annulus/sphere)
stokes.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
stokes.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
stokes.petsc_use_pressure_nullspace = True               # enclosed -> pressure gauge
stokes.solve()                                            # rigid-rotation gauge auto-removed
# Convergence status: read stokes._rotated_freeslip_info = {ksp_reason,
# nonlinear_iterations, rotation_gauge_removed, reaction} — NOT s.snes.getConvergedReason()
# (the rotated solve is a manual loop, not snes.solve). Also sanity-check v·n leakage (~1e-16).
```

**Nonlinear rheology / warm-start / timestepping (PR #298, `feature/rotated-snes`):**
rotated free-slip now works *inside* the nonlinear iteration — a nonlinearity probe
auto-dispatches power-law / VEP / TI-with-yield models to a Newton/Picard loop
(`solve_rotated_freeslip_nonlinear`), so warm-started time loops are correct. On the
worktree state *before* #298 lands, the rotated path is a SINGLE linear solve — it
silently returns one Newton linearisation from `u=0` for a nonlinear model. If you
run nonlinear TI + timestepping with rotated free-slip, make sure #298 is in.

**Dynamic topography** from the constraint reaction (the reason to bother):
```python
h = uw.discretisation.MeshVariable("h", mesh, 1, degree=1)
stokes.dynamic_topography("Upper", h, buoyancy_scale=rho_g)   # h = -(σ_nn - mean)/ρg
xs, sig = stokes.boundary_normal_traction("Upper")            # or the raw σ_nn (lumped-mass)
```

---

## FMG under rotated free-slip

`rotated_bc.solve_rotated_freeslip` builds its OWN fieldsplit KSP, but it resolves
the multigrid hierarchy through the same `custom_mg.build_transfers` rule as the
standard path: an explicit `set_custom_fmg` registration wins, otherwise a
mesh-owned adapt tail is picked up opportunistically. So:
- On an `adapt()` **child**, the mesh-owned custom-P tail is auto-picked-up for the
  velocity block → FMG for free, **rotated free-slip included** (the old
  unreachability — rotated solves silently falling back to GAMG on adapt
  children — was #467, fixed).
- On a plain **refined `Annulus`** with **rotated** free-slip, the native
  `dm_hierarchy` is still not read; to get FMG you must build a coarse-mesh tail
  and call:
  ```python
  from underworld3.utilities.custom_mg import set_custom_fmg
  set_custom_fmg(stokes, [Annulus(cs=0.16), Annulus(cs=0.08)], field_id=0)   # velocity block
  ```
  (~4 velocity iters vs ~26 GAMG). Otherwise it falls back to GAMG (fine, just slower).

The default velocity-block preconditioner and the pressure Schur (`1/η`) are already
near-optimal for TI — do **not** hand-roll a "TI-aware Schur"; measured, the default
`1/η₀` beats every alternative (the weak TI mode is fault-parallel shear, which is
volume-preserving, so pressure sees η₀).

---

## Moving fault + re-adaptation + field transfer

`adapt()` is non-cumulative and deterministic. Move the fault, re-adapt from the
static base, carry any field by interpolation:

```python
for step in range(N):
    fault_pts = move(fault_pts, step)                    # kinematics
    fault = uw.meshing.Surface(f"fault{step}", base, fault_pts, symbol="F"); fault.discretize()
    child = base.adapt(fault.refinement_metric_function(...), max_levels=3)
    # verify: folded=0 (all cell |vol|>0), base unchanged (non-cumulative)
    # carry a field child_{k-1} -> child_k by interpolation:
    T = uw.discretisation.MeshVariable(f"T{step}", child, 1, degree=1)
    if T_prev is None:
        T.data[:,0] = uw.function.evaluate(T0_expr, T.coords)
    else:
        T.data[:,0] = uw.function.evaluate(T_prev.sym, T.coords)   # mesh->mesh interp
    T_prev = T
```
Transfer error is **non-accumulating** for a smooth field (bounded by the per-mesh P1
representation floor, ~0.2% on a fine base, ~3% on a coarse base) — repeated
re-meshing does not diffuse a smooth field away. Sharp fronts lose more per transfer.

---

## Advection-diffusion on the adapted mesh

```python
T = uw.discretisation.MeshVariable("T", child, 1, degree=2)
adv = uw.systems.AdvDiffusionSLCN(child, u_Field=T, V_fn=stokes.u.sym, order=1,
                                  monotone_mode="clamp")     # clamp = bounded, no overshoot
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 2e-4
adv.f = 0.0
dt = 0.015                       # FIXED dt — SLCN is semi-Lagrangian (unconditionally
                                 # stable). estimate_dt() reports the tiny fault-band
                                 # Courant limit; do NOT use it to size the step.
for step in range(nsteps):
    adv.solve(timestep=dt)
```
- **SLCN dt is NOT Courant-limited** by the fine fault-band cells — pick dt from the
  coarse-region advection, verify accuracy.
- The scalar AD auto-FMG-injection bug (velocity-block custom-P mismatching a scalar
  operator on adapt children → PtAP error 60) is **FIXED**: `auto_inject_custom_mg`
  now calls `snes.setUp()` before reading the finest reduced map (so the DM section
  is the finalized space the operator lives on) and validates that map against the
  assembled operator. Custom-P installs successfully on scalar SLCN AdvDiffusion on
  NVB adapt children (no skip-guard, no workaround) — `bugfix/custom-mg-parallel`.

---

## Sizing the band, and how the fault margin is represented

The artefact that matters for a fault is **stress manufactured by elements that
straddle the weak-zone margin** — high strain rate at one end, high viscosity at
the other. It is exactly

```python
leak = 2 * (eta.mean(axis=1) * edot.mean(axis=1) - (eta * edot).mean(axis=1))
```

per cell (vertex values), i.e. `−2 Cov(η, ε̇)`: **zero** for any cell wholly inside
or wholly outside the weak zone, positive only across the transition. It lives
strictly *inside* elements — plotting nodal `2ηε̇` cannot show it, because at a node
the two fields are sampled at the same point and are consistent by construction.

Measured guidance, all at matched cell count:

- **Band width.** The answer depends on what you are minimising, and the two
  objectives disagree. *Total* leak: narrower is better (concentrate cells where
  ∇η is steepest). Leak **into the matrix** (what usually matters): an optimum at
  core half-width ≈ the **influence width**, 2.6× better than a narrow band.
  Straddling-cell count: wider is monotonically better.
- **Don't invent a marking rule.** Marking on within-cell η variation is
  intuitive and measurably *worse* per DOF than the plain distance size field:
  N^-0.37 (absolute jump) or a complete stall (log ratio) against **N^-1.04**. The
  leak is spread across the whole transition, not concentrated in a few cells, so
  there is nothing for a targeting rule to target. ⚠️ The log ratio is largest
  where η is *smallest* — it refines the fault core, the opposite end from the
  problem.
- **A step-edged margin confines it.** `influence_function(profile="step")` (the
  DEFAULT profile) plus marking on the distance level set puts essentially **0 %**
  of the leak beyond d=0.03, against 11.4 % for a smooth blend, and converges
  slightly faster (N^-1.32). The price: total leak 2.5× higher and the **worst
  single cell 20× worse** (21.4 vs 1.04) — concentrated into a one-cell collar
  welded to the interface rather than spread. For a viscous solve that is a clear
  win; for a yielding model the worst cell is what reaches yield first, so weigh
  it. Mark geometrically on the level set: once the edge is sharp, sampled η
  depends on which side a vertex happens to fall.
- **Exact fixes.** An element-wise constant (P0) viscosity makes `Cov(η, ε̇) ≡ 0`
  on any mesh — not reduced, zero. So does aligning the interface with element
  boundaries — and there are now primitives that do exactly that: `place_sheet` /
  `place_thin_volume` / `remove_embedded` in `utilities/place_surface.py`
  (#517–#526). Both fixes move the error from *inside* elements to *where the
  element boundaries fall*, which makes `relax(pin_bands=...)` the lever rather
  than shape repair. ⚠️ P0 also breaks any within-cell marking rule (contrast is
  identically zero) — it would have to be reposed on the facet jump.

## Gotchas / rough edges (candidates to fix as we go)

| symptom / edge | cause & handling |
|---|---|
| patchy along-fault refinement (level 4 here, 2 there) | P1-interpolated `M=1/h²` aliasing. Use the **callable exact-distance** metric. |
| child solver rejects `fault.distance.sym` (foreign-mesh error) | symbol disambiguation. `fault.remap_to(child)` or build a child-side `Surface`. **Rough edge**: needing two Surface objects. |
| adapt added-cell count jitters ±30% step-to-step | small-number geometry on a thin band. Deterministic; quality (near-fault h) is constant. Base ≈ **2× target** minimises it (CoV ~10% at 0.08 base vs 19% at 0.14, 24% at 0.05). |
| tiny AD timestep / slow advection | `estimate_dt` returns the fault-band Courant limit. Use a fixed dt (SLCN is unconditionally stable). **Rough edge**: `estimate_dt` not adapt-aware. |
| surface-breaking fault: huge local vmax; topo peak keeps growing | real stress singularity at the outcrop. Topo peak **saturates** (integrable/log, bounded by finite buoyancy) — far field is fine; only the pointwise outcrop value is mesh-dependent. |
| rotated free-slip Stokes "not converged" | `s.snes` isn't the solving object (manual loop). Read `stokes._rotated_freeslip_info['ksp_reason']` / `['nonlinear_iterations']` (PR #298); sanity-check v·n leakage. |
| nonlinear TI/VEP + rotated free-slip + timestepping gives a wrong (frozen) answer | pre-#298 the rotated path is ONE linear solve (one Newton step from u=0). PR #298 runs it inside a Newton/Picard loop — ensure it's merged for nonlinear/warm-start runs. |
| FMG under rotated free-slip on a plain (non-adapt) refined mesh | the rotated KSP resolves hierarchies via `custom_mg.build_transfers`: an adapt child's mesh-owned tail is picked up AUTOMATICALLY (#467 fixed the old silent GAMG fallback), but the native `dm_hierarchy` is still not read — on a plain refined mesh, `set_custom_fmg(..., field_id=0)`. |
| NVB at np>1 raises NotImplementedError | native `_nvb_transform` extension not built (needs the custom-PETSc/amr env). Both `nvb` and `edge_split` are otherwise fully parallel, 2-D and 3-D. |
| high stress appears in the matrix beside the fault | elements STRADDLING the weak-zone margin: one end sees high strain rate, the other high viscosity. The FE forms `mean(η)·mean(ε̇)`; the honest cell average is `mean(η ε̇)`, and the difference is `−2 Cov(η, ε̇)` across the cell. Zero for any cell wholly in or wholly out. See the band-width section below. |
| refinement band narrower than the fault's INFLUENCE | measured: η still 0.07 at d=0.06 while the mesh has already coarsened 4×, so the artefact peaks on the transition flank, not on the fault. **85 % of it sits at d>0.01.** Size the flat core from the *influence* width, not the fault. |
| `uw.function.evaluate` fails "Total components 8 != 6" | cached-interpolation mismatch on a mesh already carrying several solver variables. Sample the field numerically from `surface.unsigned_distance` instead. |
| bare SIGSEGV, no traceback, after building a Mesh from a raw DM | `uw.discretisation.Mesh(dm, ...)` TAKES THE DM OVER. Read geometry from `child.dm`, never the handle you passed in. |
| `KeyError: 'Left'` from a Mesh built on a refined DM | `Mesh(dm)` without `boundaries=` loses the boundary ENUM even though the labels are on the DM. Pass `boundaries=base.boundaries`. |

**Build/run**: this lives on the `feature/adapt-on-top` worktree; env
`.pixi/envs/amr-dev/bin/{python,mpirun}`; `./uw build` after source changes.
