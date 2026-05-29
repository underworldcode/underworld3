# Fault refinement — the simplification

```{note}
Design note, 2026-05-28. Captures the convergence after the
feature/elliptic-ma fault-meshing work: one mover, one metric form, one
slip, 2D *and* 3D. The pieces this collapses (the anisotropic tensor
mover and the analytic-Eulerian per-segment machinery) remain present
for the moment but are scheduled for deprecation.
```

## The recipe

```python
import sympy, underworld3 as uw

rho_T = uw.meshing.metric_density_from_gradient(mesh, T, metric_choice="arc-length")
rho_F = uw.meshing.fault_comb_metric(mesh, faults, cell_size=dx, n_across=N)

uw.meshing.smooth_mesh_interior(
    mesh, method="ma",
    metric=[(rho_T, 1.0), (rho_F, w_F)],        # composable list (max-on-excess)
    boundary_slip=True,                          # generic topology slip — required
    method_kwargs=dict(n_outer=1, n_picard=25))  # single-shot
```

One mover (single-shot Monge–Ampère), one metric form (scalar density), one
composition operator (weighted max on the excess), one slip (topology-based
vertex normals). Works in **2D and 3D**, on Cartesian boxes, annulus,
sphere, polyhedra, curved surfaces.

```{note}
``boundary_slip=True`` is part of the recommended recipe, not optional. For
any feature that **touches the boundary** (a thermal BL that runs full
width, a fault that reaches the wall, …), pinning the boundary effectively
wastes the budget at the edges: the refined band visibly fades as it
approaches the wall. With the generic topology slip enabled, boundary face
nodes slide along the face to cluster where the metric demands them, and
the refinement runs uniformly to the wall (corners stay pinned, box
shape exactly preserved). See ``fault_compose_demo2.py``.
```

## Why each piece

### Single-shot MA and what `n_outer` actually does

`smooth_mesh_interior(method="ma", n_outer=1)` is the Caffarelli-clean
Monge–Ampère map: one solve, untangled by construction, **composable**
(see below — repeated calls compose correctly toward the equidistribution
fixed point). For most metrics this is also the right default: one
solve gives a clean band, and `n_outer=1` is what `fault_metric(...)`
wraps.

`n_outer>1` performs `n_outer` outer Picard iterations *within a single
`smooth_mesh_interior` call*, each recomputing the source density on
the current deformed mesh. With the lumped-V projection fix (see the
"Composable iteration" section below), `n_outer>1` is now equivalent
to calling the mover `n_outer` times in sequence — both paths converge
to the same equilibrium. The "patch-aware composition" language in
the original design note was describing the *intent*; the original
implementation didn't reliably deliver it because of the bug fixed in
this update.

**The honest update**: iterated calls now compose correctly, so the
choice between "call `smooth_mesh_interior` once with `n_outer=k`"
and "call it `k` times with `n_outer=1`" is a stylistic one — same
trajectory, same equilibrium. Use whichever fits the surrounding
code structure. The pre-placement recipe below uses repeated calls
because it varies the *metric width* between calls.

For composed metrics including a Lagrangian field (gradient(T), a
`Surface.distance`-field comb): keep `n_outer=1` and don't iterate
manually either — the feature would convect each pass and the bands
would smear. The honest path to more refinement there is finer base
mesh or `mesh.adapt`.

**Don't use `target_side_rho=True`.** It exists in `_winslow_elliptic`
as an experimental option (query ρ at the target position
`x + ∇φ(x)` rather than the source). The Picard fixed-point coupling
is much tighter than the default and the default `n_picard=25` is
typically under-converged (it needs ~100+ iters for moderate-to-strong
demand) — silently producing inconsistent results. Even when fully
converged, it doesn't deliver sharper realised refinement than iterated
source-side. Treat it as an internal experiment, not a user-facing
lever.

### Scalar comb metric

`fault_comb_metric(mesh, faults, cell_size=dx, n_across=N)` places narrow
teeth at `d = 0, dx, 2 dx, …` from each fault's distance field.
Equidistribution drops a node row at each tooth → evenly-spaced rows ⇒ a
band of `~ N` roughly-uniform cells across each fault, **with the `d=0`
tooth pinning a row on the fault line** (so close faults centre to
~0.0002 — better than h-adapt with `mesh.adapt`).

For 2D faults the per-segment min-distance is analytic. For curved or
**3D triangulated** fault surfaces (`FaultSurface.compute_distance_field`,
kdtree-based), the comb is built directly on the precomputed distance
**field** — segment-count-independent JIT cost, and the natural input
for 3D where analytic point-to-triangulated-surface distance is hard.

### Composable list of metrics

`smooth_mesh_interior(metric=[(m_i, w_i), …])` composes internally via

$$\rho_{\text{combined}}(x) = 1 + \max_i\, w_i\,\big(\rho_i(x) - 1\big)$$

— "refine wherever any feature demands it," with weights scaling each
feature's demand cleanly. Scalar densities compose by `max` trivially;
metric *tensors* would need Alauzet metric intersection (much more
involved) — another reason scalar-MA is the convergence point.

### Generic topology-based tangent slip

`_boundary_vertex_normals(mesh)` computes outward unit normals at each
boundary vertex *geometrically* from the cell coordinates (boundary
facets identified topologically, normals area-weighted averaged). It
classifies each vertex as **face-slip** (all incident facet normals
within ~15° of the average — slides tangentially) or **pinned**
(corners, 3D edges between faces). Works on **any** simplicial mesh.

This replaces the old `Gamma_P1`-based slip, which evaluated PETSc's
`petsc_n` quadrature symbol at *vertices* (undefined off boundary
quadrature points) — radial mesh classes worked around it by
redefining `Gamma` as the analytic radial unit vector, but Cartesian
got garbage normals and was silently pinned.

### Dimension-general MA

`_winslow_elliptic` is now dimension-general (bit-identical at `cdim=2`):

* **Normalisation `c`** branches on the source's leading term:
  `c = 1/⟨b^{-1/2}⟩²` for the 2D convex radical, `c = 1/⟨b^{-1}⟩` for
  the 3D simple Picard. Wrong `c` made the source non-zero-mean and the
  pure-Neumann φ-Poisson unsolvable (the constant nullspace fixes
  *solution* ambiguity, not *RHS* inconsistency) — the actual cause of
  the previous 3D failure.

* **3D source**: `f_src = tr(H_s) + g − det(I+H_s)`
  (`H_s` symmetrised), restoring the 2×2 principal-minor terms the old
  `(g−1) − det(H)` dropped in 3D. Reduces to the 2D simple-Picard form
  exactly.

* **Tet signed-volume backtrack**: `_tri_cells` returns `None` for tets,
  so 3D previously had no anti-tangle guard. Added `_tet_cells` +
  `_signed_volumes` and a tet branch in the backtrack.

Validated on a 3D slab and spherical-shell adapt (refines toward the
feature, 0 inverted tets) and a 3D disk fault (the recipe above).

## What this collapses

The following remain in the codebase for the moment but are scheduled for
deprecation once external users have migrated:

| Component | Replaced by |
|---|---|
| `_winslow_anisotropic` (anisotropic tensor mover) | single-shot MA + comb |
| `fault_metric_tensor` (analytic 2×2 supplied tensor) | `fault_comb_metric` |
| `_winslow_anisotropic.supplied_D` entry point | (no need — comb is scalar) |
| Per-segment analytic min-distance for curved faults | `Surface.distance` / `FaultSurface.compute_distance_field` |
| Ring-projection slip on annulus + geometric box-slip | topology-based generic slip |

The `fault_metric` facade keeps `method="anisotropic"` and `method="adapt"`
(MMG) for the moment as documented alternatives — the recommended default
is `method="ma"`.

## Composable iteration: lumped V_T projection

```{note}
Update 2026-05-28 (late session). Replaces the earlier
`_patch_volumes` source density in `_winslow_elliptic`. Makes
repeated calls to `smooth_mesh_interior(method="ma", ...)`
properly **composable**, which in turn unlocks the
*pre-placement* recipe in the next section.
```

### The bug that wasn't documented

`_winslow_elliptic` solves the convex-branch Picard for the
Caffarelli-Brenier displacement potential. The right-hand side
contains a **source density `V(x)`** representing the current mesh —
in continuous form `V` would be `det(I + ∇²φ_current)`, i.e. the
local Jacobian of the deformed mapping at every point. Per-vertex
discretisation of `V` is what tells the solver "this region is
already partially adapted, don't pull it further."

The previous code did one of two things:

```python
if tris is not None and n_outer > 1:
    patch = _patch_volumes(...)      # Σ_{T ∋ i} |T| / 3  per vertex
    patch /= float(np.mean(patch))
else:
    patch = np.ones(n_verts)          # assume mesh is uniform
```

Both were wrong, in different ways:

1. **`patch = ones` at `n_outer=1`** (the default) — assumed the input
   mesh is uniform regardless of how it actually looked. Calling
   `smooth_mesh_interior` a second time from a previously-adapted
   mesh produced the same displacement that the first call would
   have produced from cold, applied on top of the existing
   deformation. Composition broke: every call started from scratch
   conceptually, so iterated calls compounded biases instead of
   correcting them. This is why the design note above had to
   recommend `n_outer=1` "single-shot, don't compose."

2. **`_patch_volumes` at `n_outer>1`** — returned `Σ_{T ∋ i} |T| / 3`,
   which is the **lumped mass diagonal** `M^lumped_ii = ∫ ψ_i dx`,
   an *integral* with units of area. The code then used it as a
   *density*. On an unstructured Delaunay mesh of equal-area cells
   `M^lumped_ii = d_i · |T_0| / 3` (proportional to vertex valence
   `d_i = 5..7`), so the equation saw a ~30 % spurious source
   non-uniformity from FE bookkeeping, not from any actual mesh
   deformation. The conservative behaviour of `n_outer>1` under
   the old code was the mover *trying to flatten that valence
   noise* and giving up.

### The fix

`V(x)` is fundamentally a **cell** quantity: `V_T = |T|` in 2D,
`|Tet|` in 3D. The Caffarelli equidistribution invariant is
*cell-wise*: at equilibrium `ρ_T · |T| = const` over all cells.
The FE-natural projection of this cell field into the P1
`vol_field` storage that the solver expects is a **lumped L2
projection**:

$$V_i = \frac{\sum_{T \ni i} V_T\,|T| / k}
            {\sum_{T \ni i} |T| / k}
       = \frac{\sum_T |T|^2}{\sum_T |T|}$$

(`k = 3` in 2D, `k = 4` in 3D — the per-vertex weight per incident
cell). This is the *area-weighted average of incident cell
volumes*, strictly local, no neighbour mixing, valence-independent
on uniform meshes (`Σ|T|² / Σ|T| = |T_0|` exactly when all `|T|`
are equal regardless of valence).

It is implemented inline in `_winslow_elliptic` with two
`np.add.at` accumulators (numerator and denominator) and one
division.

```{note}
An intermediate attempt used the consistent-mass `uw.systems.Projection`
to project `V_T → vol_field`. That introduces an intrinsic L2
smoothing kernel of ~one element width. Cell-density signals
narrower than the kernel get smoothed into a halo around refined
bands, and the next solve reads the halo as "over-refined" and
*undoes* the refinement — iteration becomes regressive. The
lumped form has zero kernel scale and behaves correctly.
```

### What this changes for users

The mover is now **composable**: each call to
`smooth_mesh_interior(method="ma", ...)` produces a displacement
*from the actual current mesh state* toward the target metric.
Repeated calls iterate the same fixed point, with `|Δo|` decreasing
monotonically. Single-shot remains the recommended **default**;
iterated calls are now safe to use when more refinement is wanted
than a single solve delivers, and — more importantly — when the
*metric itself changes between calls*. That second case is the
pre-placement recipe below.

```{note}
**TODO (parallel)**: the lumped projection accumulators are
rank-local (`np.add.at`). At MPI partition boundaries, vertices
owned by one rank under-count contributions from cells owned by
neighbouring ranks. Same parallel deficit as the old
`_patch_volumes` had. The fix is to assemble the two numerators
into PETSc Vecs with `ADD_VALUES` so the assembly ghost reduction
sums them correctly. Required before parallel use of the MA mover
on adapted meshes.
```

## Pre-placement and redistribution recipe

```{note}
Recommended when single-shot MA leaves the band off-line — the
classic case is two or more faults closer to each other than the
band width can comfortably resolve from cold.
```

### Why single-shot is centroid-biased for close faults

For two faults at half-separation `a` and a metric built as a
**sum** of per-fault Gaussians,

$$\rho(x) = 1 + A\,\sum_i \exp(-d_i(x)^2 / w^2)$$

the two Gaussians overlap when `w > a√2`. Past that crossover the
sum has a **single maximum at the midpoint** between the faults
rather than two maxima on the faults. The mover faithfully
equidistributes to whatever the metric's actual maximum is, and
ends up clustering nodes at the centroid — not because of any
mover deficiency, but because the metric construction *told it
to*. With `a = 0.030`, `w_crit = 0.030√2 ≈ 0.042`; anything at
or above the crossover puts the metric peak in the gap.

Starting cold from a uniform mesh and applying any single-call
narrow-`w` solve produces a converged equilibrium where `ρ · V`
is balanced even though many refined cells sit in the gap and not
on the lines — a *degenerate* equidistribution. With the mover
now composable (above), iteration on a fixed metric stays at this
equilibrium; the local minimum of the equidistribution functional
is genuine.

### The recipe — MAX, wide pre-place, narrow redistribute

Use a **max** combination of per-fault Gaussians, not a sum:

$$\rho(x) = 1 + A\,\max_i \exp(-d_i(x)^2 / w^2)
         = 1 + A\,\exp(-d_{\min}(x)^2 / w^2)$$

Pick the closer fault at every point. The metric is constant
amplitude `A` on any fault, falls off independently to either
side, and **no centroid pile**, however wide `w` is.

Then a two-stage iterated call:

```python
# Stage 1 — wide pre-place (a few iters)
for _ in range(n_wide):
    rho = max_of_gaussians(mesh, faults, w=w_wide)
    smooth_mesh_interior(mesh, method="ma", metric=rho,
                         boundary_slip=True,
                         method_kwargs=dict(n_outer=1, n_picard=25))

# Stage 2 — narrow redistribute (more iters)
for _ in range(n_narrow):
    rho = max_of_gaussians(mesh, faults, w=w_narrow)
    smooth_mesh_interior(mesh, method="ma", metric=rho,
                         boundary_slip=True,
                         method_kwargs=dict(n_outer=1, n_picard=25))
```

The wide stage pre-clusters cells *around the entire fault
system* without piling them in any specific spot (the MAX
amplitude is flat over the broad neighbourhood). The narrow stage
inherits a mesh that *already has refined cells in the right
neighbourhood* of every fault, and the equidistribution at the
narrow width simply pulls those cells onto the lines.

### The width-vs-separation knob

`w_wide` is the single design knob and it scales with the **fault
separation**, not with the mesh resolution:

| `w_wide / a` (a = half-separation) | Behaviour |
|---|---|
| ≈ 1 (just the gap) | Mild improvement over cold-narrow; still some centroid bias |
| **≈ 4 (≈ 2× full separation)** | **Sweet spot — bands land on lines to ≤ 1/10 cell** |
| ≫ 4 (very wide) | Refinement too diffuse; pre-placement doesn't localize |

Two-fault test case (gap `2a = 0.060`, target band `w_narrow = 0.015`,
60×60 base mesh):

| Schedule | `f0` offset | `f1` offset |
|---|---|---|
| Cold → `w=0.015` × 10 | −0.0109 | +0.0103 |
| `w=0.060` × 2 → `w=0.015` × 8 (MAX) | −0.0040 | +0.0021 |
| **`w=0.120` × 4 → `w=0.015` × 8 (MAX)** | **−0.0005** | **−0.0014** |
| `w=0.200` × 4 → `w=0.015` × 8 (MAX) | −0.0069 | +0.0035 |

`w_wide = 0.120` (`= 2 × 0.060`, i.e. `2 × full separation`) wins:
both bands within `≤ 8 %` of one mesh cell of the actual lines.
The recipe genuinely *places* nodes on the close-paired fault
lines that cold-narrow iteration could not reach.

### Convergence diagnostic and why `n_picard=25` is the right default

The equation-natural residual is the **coefficient of variation of
$\rho \cdot V$ over cells**:

$$\mathrm{cv}(\rho V) = \frac{\mathrm{std}(\rho_T \cdot |T|)}
                          {\mathrm{mean}(\rho_T \cdot |T|)}$$

At equilibrium $\rho \cdot V = K$ constant, so $\mathrm{cv}(\rho V) = 0$.
On a discrete mesh against a continuous metric, the minimum achievable
$\mathrm{cv}$ is non-zero — but the *relative* value across iterations
and schedules cleanly distinguishes which equilibrium the mover settled
into. For the two-fault recipe at gap=0.060, the centroid-local-minimum
sits at $\mathrm{cv} \approx 1.07$, the bands-on-lines equilibrium at
$\mathrm{cv} \approx 0.79$.

```{important}
Crucial finding: **the inner Picard iteration count is not a "more is
better" knob**. At `n_picard=50` and `n_picard=200` the trajectory
becomes *bit-identical* (inner Picard is fully converged at 50) — but
the recipe **gets stuck in the centroid local minimum and never
escapes**. At `n_picard=25` the inner Picard is mildly under-converged,
and that residual non-equilibrium acts as **numerical annealing**: it
occasionally kicks the system out of shallow local minima into deeper
ones. The bands-on-lines result we report for the two-fault gap=0.060
case is *only* reachable with `n_picard=25`; tightening to 50+ locks
the centroid-bias floor.

This is counter to the usual "tighter inner solve is better" intuition
and is the reason `n_picard=25` was chosen as the default in
`smooth_mesh_interior(method="ma", ...)`. **Don't increase it for
"convergence."**
```

The geometric `|Δo|` we used in the diagnostic plots is a poor stopping
signal because it reads ≈ 0 immediately when the mover hits the
*centroid* local minimum (locally converged, just to the wrong place).
`cv(ρV)` reads ≈ 1.07 there and only drops to ≈ 0.79 when the recipe
escapes — so it's a much better measure of actual equidistribution
quality.

A practical stopping rule:

```python
prev_cv = float("inf")
plateau = 0
for outer_iter in range(MAX_OUTER):
    smooth_mesh_interior(mesh, method="ma", metric=rho_target,
                         method_kwargs=dict(n_outer=1, n_picard=25))
    cv = cell_cv_of_rho_V(mesh, rho_target)
    if abs(prev_cv - cv) < 0.001 * cv:
        plateau += 1
        if plateau >= 3 and outer_iter > MIN_OUTER:
            break
    else:
        plateau = 0
    prev_cv = cv
```

`MIN_OUTER` should be at least the wide-stage iteration count plus a
few — the system has to be given a chance to escape the wide-stage
local minimum.

### When this matters

* Stationary fault-pair problems — geometry once, iterate to
  equilibrium, use the resulting mesh as the substrate for the
  rest of the simulation.
* Moving-fault problems — the long-term aim. When the fault
  positions evolve, redoing the schedule each adaptation step is
  expensive. *Open question (next session)*: can the converged
  equilibrium for time `t` serve as the wide-pre-placed state for
  time `t + Δt`? The mover being composable suggests yes — the
  narrow-stage iteration should be sufficient to track small
  motion.

* Faults farther apart than `w_wide` becomes irrelevant: single-shot
  with `n_across = 1` (a single Gaussian per fault) is already
  centred on the line. The pre-placement recipe is specifically
  for the close-paired regime where overlap matters.

## Update 2026-05-29: smooth-aid, plain Picard, fat band for moving faults

```{note}
This section supersedes the earlier "n_picard=25 is a feature" and
"Anderson acceleration" framings further up. Those findings were
*directionally* correct (under-converged Picard helps escape local
minima, Anderson does accelerate per-iteration descent) but the
recipe that actually works robustly across geometries — and so
qualifies as a *user-facing default* — turns out to be different.
```

### The mover misses one fault without a smooth aid

The sharpest finding of this update. Cold-start with a single sharp
narrow Gaussian per fault (the previous design), and no wide
pre-pass, the mover **catastrophically misses one of two close
faults** — only the first one gets a refinement band, the second
is uniformly meshed. Adding a low-amplitude wide Gaussian on top of
the sharp narrow one provides a non-trivial $\nabla \rho$ everywhere
in the fault neighbourhood. With the smooth aid, both faults get
their bands; without it, the mover's equidistribution invariant is
satisfied by a one-band solution.

The recommended target metric is therefore **always** a sum of two
Gaussians: a sharp peak for localisation, a smooth halo for "find
the fault" direction:

$$\rho(x) = 1
  + A_{\text{sharp}} \cdot \max_i \exp(-d_i(x)^2 / w_{\text{target}}^2)
  + A_{\text{smooth}} \cdot \max_i \exp(-d_i(x)^2 / w_{\text{smooth}}^2)$$

With $A_{\text{sharp}} \approx 6$, $A_{\text{smooth}} \approx 2$,
$w_{\text{smooth}} \approx 5 \cdot w_{\text{target}}$ as
reasonable defaults.

### Plain Picard is geometrically equivariant; Anderson isn't

Anderson acceleration on the outer fixed-point map gives a 3× per-step
speedup *on a fixed geometry* but is **not equivariant under
translation of the fault** — the basin Anderson converges into depends
on the post-phase-1 cell distribution, which depends on fault position.
A uniform translation of all faults that should give a uniformly
translated solution does not, with Anderson: shifted geometry can
land at $\mathrm{cv} \approx 1.08$ where the original geometry lands
at $\mathrm{cv} \approx 0.39$. The deeper basin exists for the shifted
geometry too, Anderson just can't find it.

**Plain Picard does not have this problem.** On both initial and
shifted geometries it reaches the same basin ($\mathrm{cv} \approx
0.57$), takes 10–15 outer iterations to plateau, and is the
*reliable* default. Anderson is opt-in for speed when the user can
afford to verify it reached a good basin.

The displacement residual $\|X_{k+1} - X_k\|/(\sqrt N \cdot h)$ is
the natural fixed-point convergence signal — clean monotone descent
under plain Picard, machine zero at the fixed point — and is the
right stopping criterion. `cv(ρV)` is a *quality* measure (lower =
deeper basin) and is useful to compare which basin you landed in but
*not* a convergence test.

### The right recipe (current state)

```python
def fault_metric_iterate(mesh, faults, w_target, *,
                         w_smooth=None,
                         amp_sharp=6.0, amp_smooth=2.0,
                         w_wide=None,
                         n_pre=4, n_combined=16,
                         tol_disp=1e-4):
    """Recommended recipe for two-fault (and multi-fault) refinement.

    Phase 1 — marshalling (wide sharp pre-pass): a wide MAX-of-Gaussians
        metric at sharp amplitude pulls cells into concentrated clusters
        around each fault. Not a smooth metric; the concentration is
        the *point*.

    Phase 2 — localisation (sharp + smooth combined target): the sharp
        narrow peak localises onto each line; the smooth halo provides
        non-trivial gradient direction everywhere in the fault
        neighbourhood (without it, cold-start can miss a fault entirely).
        Plain Picard, no Anderson — geometric equivariance > speed.

    Termination: |ΔX|/(√N · h) < tol_disp, OR n_combined iters exhausted.
    """
    if w_smooth is None:
        w_smooth = 5.0 * w_target
    if w_wide is None:
        # Heuristic: 2 × estimated fault separation
        w_wide = 0.120        # for the canonical test geometry

    # Phase 1 — wide sharp pre-pass
    rho_pre = max_of_gaussians(mesh, faults, w_wide, amp=amp_sharp)
    for _ in range(n_pre):
        smooth_mesh_interior(mesh, method="ma", metric=rho_pre,
                             method_kwargs=dict(n_outer=1, n_picard=10))

    # Phase 2 — sharp + smooth combined, plain Picard
    rho_target = (
        max_of_gaussians(mesh, faults, w_target, amp=amp_sharp)
      + max_of_gaussians(mesh, faults, w_smooth, amp=amp_smooth))
    X_prev = mesh.X.coords.flatten()
    for k in range(n_combined):
        smooth_mesh_interior(mesh, method="ma", metric=rho_target,
                             method_kwargs=dict(n_outer=1, n_picard=10))
        X = mesh.X.coords.flatten()
        h = median_min_edge(mesh)
        disp = np.linalg.norm(X - X_prev) / (np.sqrt(len(X) // 2) * h)
        if k > 4 and disp < tol_disp:
            break
        X_prev = X
```

### Moving faults: fat band + deferred re-meshing

For a fault that moves through several mesh-cell widths per simulation
step, the realistic strategy is **not** to re-mesh every step but to
build a refinement band wide enough to contain the fault over multiple
timesteps, then re-mesh only when the fault is about to exit. Picking

$$w_{\text{target}} \approx v_{\text{fault}} \cdot \Delta t_{\text{remesh}}$$

(where $v_{\text{fault}}$ is estimated fault drift speed per timestep
and $\Delta t_{\text{remesh}}$ is the desired re-mesh interval in
timesteps) gives a fat refined band that the fault stays within for
$\Delta t_{\text{remesh}}$ steps. Trade-off: a 1.8 × $h$ wide band
(instead of sub-element) buys ~3 timesteps of fault motion at the
cost of a slightly more diffuse band on the t=0 fault (offset
~1/2 cell instead of ~1/10 cell). Total cost goes from ~20 mover
calls per timestep to ~7 amortised — a ~3× speedup.

Warm-starting from the previous timestep's converged mesh does **not**
work as a substitute. The cells inherit the old fault positions and
plain Picard from that state finds a *different, suboptimal* local
fixed point rather than tracking the moving fault. Cold restart per
re-meshing event with plain Picard is the reliable approach.

### Future: SNES wrap with approximate Jacobian

The remaining major efficiency lever — left as a follow-up session —
is wrapping the fixed-point map $F(X) = X - \mathrm{mover}(X)$ in
PETSc's `SNES` framework, with either matrix-free JFNK
($J \delta X$ via finite-difference) or an approximate analytic
Jacobian using the per-vertex $\partial V / \partial X$ from the
lumped-L2 projection. Expected gains: quadratic convergence rate
near the fixed point, line search for global robustness, and
standard SNES tooling for convergence tests. The mesh deformation
inside the outer loop is what makes the present Picard slow — folding
it into a Newton step is the natural next move.

## Honest limits

* **Budget cap**: `r-adapt` (any mover, including MA) redistributes a *fixed*
  set of nodes — `cell_size` in `fault_comb_metric` is a *target*, not a
  guarantee. The realised cell sizes are roughly `~1.5–2.5×` finer than the
  base mesh per feature. To honour an absolute `cell_size`, use
  `mesh.adapt` (MMG) via `fault_metric(method="adapt")` — but that *adds*
  nodes (topology changes, disturbing particle workflows).

* **Composed multi-feature budgets compete**: composing gradient(T) with a
  fault sends a fixed budget over two extended demands. Weights tune
  *who* wins; the base mesh resolution controls the absolute resolution
  each can reach.

* **Multi-iteration metric convection**: at `n_outer>1` the MA mover
  re-queries the target metric on the deformed mesh. Analytic metrics
  re-evaluate correctly (Eulerian); a frozen *field* metric (the field
  comb) convects and degrades. The recommended single-shot recipe
  sidesteps this entirely.

* **3D MA is the simple Picard, not a convex branch**: it converges
  cleanly on gentle metrics (validated on the slab, sphere shell, and
  disk fault) but could be fragile on very strong/sharp ones. The 2D
  convex-branch (BFO) path stays in place at `cdim=2`.

## Migration

For users of the now-deprecated paths:

* `smooth_mesh_interior(method="anisotropic", supplied_D=M, ...)` →
  `smooth_mesh_interior(method="ma", metric=fault_comb_metric(...))`
  (or via the list-of-metrics composition).

* `fault_metric_tensor` → `fault_comb_metric` (or `fault_metric(method="ma", ...)`).

* Hand-built `sympy.Max(...)` composition → pass `metric=[m1, m2, …]`
  to `smooth_mesh_interior`.

* Custom box-face slip code → just enable `boundary_slip=True`; the
  generic slip handles any geometry.

## References

* `src/underworld3/meshing/surfaces.py` — `fault_metric_tensor`,
  `fault_comb_metric`, `fault_metric`, `compose_metrics`.
* `src/underworld3/meshing/smoothing.py` — `_winslow_elliptic` (now
  dimension-general), `smooth_mesh_interior(metric=[...])`.
* `src/underworld3/meshing/_ot_adapt.py` — `_boundary_facets`,
  `_boundary_vertex_normals`, generic `_build_slip_projector`.
* `tests/test_0762_fault_metric_tensor.py` — 17 tier-A tests locking
  the new layer.
