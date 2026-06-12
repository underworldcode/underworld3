# Anisotropic MMPDE Mesh Mover (variational, Huang–Kamenski)

**Status:** design + validated numpy prototype (2026-05-30); UW3 port
pending as `smooth_mesh_interior(..., method="mmpde")`.

## Why this mover exists

The existing movers cannot produce a **thin refined strip aligned to a
codimension-1 feature** (a fault, an interface):

- **`"ma"` (Monge–Ampère, `_winslow_elliptic`)** genuinely *clusters*,
  but only isotropically. A scalar value-metric peaked on a fault
  refines a *disk around the densest part* — the cluster lands at the
  metric's centre of gravity, not *on* the line (measured: only
  ~60–80 % of refined cells within ~0.75 h of a fault, the rest pulled
  toward the middle/root). Right tool for *tangentially-uniform*
  features (thermal boundary layers, plumes); wrong *shape* for a fault.
- **`"anisotropic"` (decoupled forward-Winslow, `_winslow_anisotropic`)**
  uses a tensor metric but solves a *linear* M-weighted Laplacian for
  each physical coordinate independently. That is a **smoother, not a
  clusterer** — it reshapes/aligns cells but does not concentrate them
  (measured band median-area/global ≈ 0.9 ≈ uniform). It also has no
  non-folding guarantee: at fault-grade anisotropy (≳6:1) the map folds
  one cell, and the *global* signed-area backtrack then throttles the
  whole step to protect it, so the mesh freezes (`scale → 0`) while
  reporting "converged" — the paradoxical "at the stability limit but
  nothing moves." The inner solve is exact (MUMPS LU, SPD); the failure
  is the *formulation/strategy*, not the linear solve.
  See `ADAPTIVE_MESHING_DESIGN.md` and `mesh-adaptation-formulation.md`.

The MMPDE mover fixes this structurally. It is the **variational moving
mesh** method of Huang & Russell, in the direct simplex discretization
of Huang & Kamenski. It generates the physical mesh as the image of a
**fixed computational (reference) mesh** under the *inverse* coordinate
map, minimizing a meshing functional that combines **equidistribution**
and **alignment** to a metric tensor `M`. Because the functional has a
barrier (`G → ∞` as `det 𝕁 → 0`) it is provably **non-folding** (Huang
& Kamenski 2018), and because it is the inverse map of a convex
computational domain it genuinely *clusters and aligns* — a thin strip
*on* the feature line.

Validated (numpy prototype): refinement **on the fault line**
(on-fault fraction 0.95–0.99 vs 0.6–0.8 for `"ma"`), **0 crushed
cells**, **monotone-convergent, never folds** (`scale = 1` throughout),
generalizes to **multiple / crossing / curved** faults and to an
**evolving (moving) metric**.

## References

- W. Huang & L. Kamenski, *A geometric discretization and a simple
  implementation for variational mesh generation and adaptation*,
  J. Comput. Phys. 301 (2015) 322–337. **doi:10.1016/j.jcp.2015.07.015**
  (arXiv:1410.7872). — the discretization and the analytic nodal
  velocities implemented here.
- W. Huang & L. Kamenski, *On the mesh nonsingularity of the moving mesh
  PDE method*, Math. Comp. 87 (2018) 1887–1911. **doi:10.1090/mcom/3271**
  (arXiv:1512.04971). — the non-folding guarantee.
- W. Huang, *Variational mesh adaptation: isotropy and equidistribution*,
  J. Comput. Phys. 174 (2001) 903–924. **doi:10.1006/jcph.2001.6878** —
  the functional and its `p`, `θ` parameters.
- W. Huang & R. D. Russell, *Adaptive Moving Mesh Methods*, Springer AMS
  174 (2011). **doi:10.1007/978-1-4419-7916-2**.

## Formulation (general `d`; simplex meshes)

Per element `K` with local physical vertices `x0, x1, x2` and the
corresponding **fixed computational** vertices `ξ0, ξ1, ξ2`:

```
E    = [x1-x0,  x2-x0]          physical edge matrix (columns)
Ehat = [ξ1-ξ0,  ξ2-ξ0]         computational edge matrix  (FIXED reference)
𝕁    = Ehat · E^{-1}            Jacobian of the inverse map ξ(x)   (eq 17)
r    = det 𝕁 = det Ehat / det E
M    = M(x_K)                   SPD metric at the element centroid
S    = tr(𝕁 M^{-1} 𝕁^T)
```

**Huang's functional** (eq 3; with `d = 2`, `dp/2 = p`):

```
G = θ · √det(M) · S^p  +  (1 - 2θ) · 2^p · r^p · det(M)^{(1-p)/2}
I_h = Σ_K |K| · G_K                                              (eq 6)
```

`θ ∈ (0, ½]` balances **alignment** (1st term) vs **equidistribution**
(2nd term); `p ≥ 1`. Coercive/polyconvex (unique minimizer) for
`0 < θ ≤ ½`, `dp ≥ 2`, `p ≥ 1`.

**Derivatives** (eq 16):

```
∂G/∂𝕁 = 2 p θ √det(M) · S^{p-1} · M^{-1} 𝕁^T
∂G/∂r = p (1 - 2θ) 2^p · det(M)^{(1-p)/2} · r^{p-1}
```

**Physical-coordinate nodal velocity** (Appendix A, eqs 39–41). The
descent velocity `v_i = −∂I_h/∂x_i` is assembled from per-element local
velocities; for the local non-`0` vertices,

```
[v1; v2] = −G E^{-1} + E^{-1} (∂G/∂𝕁) Ehat E^{-1} + (∂G/∂r) r E^{-1}
v0       = −(v1 + v2)
∂I_h/∂x_i = − Σ_{K ∋ i} |K| v^K_{i}
```

### The metric-variation term is ESSENTIAL (key gotcha)

Equations 39–41 as written treat `M = M(x_K)` as moving with the cell
centroid, contributing a **`∂G/∂M : ∂M/∂x`** term. Dropping it
("frozen-M") is *wrong wherever `M` varies sharply* — i.e. **on the
fault**, which is exactly where it matters. Measured: frozen-M gradient
is **65–330 % wrong** vs finite differences for a sharp fault metric,
and the resulting flow does **not cluster** (band ≈ uniform, energy
wanders). Including it restores agreement to **1e-8**.

```
∂G/∂M = θ √det(M) [ ½ S^q M^{-1} − q S^{q-1} M^{-1} 𝕁^T 𝕁 M^{-1} ]
      + (1-2θ) d^q r^p · (1-p)/2 · det(M)^{(1-p)/2} · M^{-1}     (symmetric)
```
(general `d`, `q = dp/2`; for `d=2`, `q=p`, `d^q=2^p`.)

assembled per vertex as `∂I_h/∂x_i += Σ_{K∋i} (|K|/(d+1)) · tr(∂G/∂M ·
∂_c M)` (the `1/(d+1)` because `∂x_K/∂x_i = 1/(d+1)`), with `∂M/∂x` from
the analytic metric (centroid finite difference is fine).

**Lesson:** finite-difference-validate any hand-derived mesh gradient
before trusting it. The prototype's `mmpde.py __main__` does exactly
this (const-M and varying-M, all `p/θ`).

## Time integration (the MMPDE)

Gradient flow `∂ξ/∂t = −(P/τ) ∂I_h/∂ξ`, rewritten in physical
coordinates (eq 39):

```
dx_i/dt = (P_i / τ) Σ_{K ∋ i} |K| v^K_i ,   P_i = det(M(x_i))^{(p-1)/2}
```

`P_i` (eq 24) makes the flow invariant under `M → cM` (scale-free
node concentration). Discretized as **explicit Euler with two
safeguards** (validated):

1. **Per-node step cap**: limit each node's move to `step_frac · h_i`
   (`h_i` = min incident edge). Prevents single-step overshoot
   (the boundary-crush mechanism); `step_frac ≈ 0.2`.
2. **Energy line-search backtrack**: accept a step only if it produces
   **no fold** *and* **decreases `I_h`** (halve `scale` up to ~20×).
   This makes the descent **monotone** — it reaches the true minimizer
   instead of oscillating around it (an early non-monotone version
   produced run-to-run-variable, over-stated refinement).

`τ` sets the move scale; with the line-search, `τ` is non-critical
(`τ = 1`). Convergence ≈ a few hundred explicit steps for `cellSize`
0.04; the line-search crawls to small `scale` near the minimum (a
candidate for acceleration in the port).

## Boundary conditions

- **Pinned** interior-only boundary: boundary nodes excluded from the
  move (`free = ~is_bnd`). Simplest; the ring cannot open to admit a
  surface-reaching feature.
- **Tangential slip** (recommended for surface-reaching faults): include
  boundary nodes in the move but **remove the outward-normal component**
  of their velocity, then snap them back onto the surface (`project`,
  e.g. fixed `|r|` for an annulus). `free = (~is_bnd) | slip`.
  - Trade-off (measured, surface-reaching fault, `across=100`): slip
    lets the ring **open** to admit the fault (finer band at the
    surface, 0.30 → 0.26) **but** costs boundary-row angle quality
    (min angle 20° → 9°, on-fault 0.88). It is a real knob, **not
    free** — localize slip to the fault root and/or temper its strength.
  - Use the projected PETSc/`Gamma_P1` normal in UW3 (the generic
    boundary normal used for slip BCs), consistent with the existing
    `_build_slip_projector`.

## Behaviour and tuning (validated, numpy prototype)

| Knob | Effect |
|---|---|
| `across`-ratio of `M` | primary strength; 9 → 100 deepens band 0.79 → 0.44, on-fault → 1.0, 0 crushed. `≳400` over-shoots (refinement drifts off-fault). Sweet spot ~100. |
| `p` (with `θ`) | `p` 1.5 → 3 sharpens the band; pair higher `p` with smaller `θ` (e.g. `θ = 1/6`). |
| node count (base `h`) | sets **absolute** on-fault cell size (≈ linear in `h`): `cellSize` 0.04 → 0.013 gives `h_fault` 0.017 → 0.006. Use to get real resolution; the *ratio* is capped by the fixed budget (~1.5–2× median, the standard r-adapt cap). |
| cumulative reference-reset | re-running with `ref = current mesh` pushes past the single-run cap (band 0.62 → 0.19 over 3 rounds) but **degrades quality** (min angle 24° → 0.9°, crushed cells appear). Use sparingly. |

**Discriminant:** judge with `n_crushed` (cells with area < 0.02 ·
global median) and the metric-aware *radial/tangential* extent — **not**
min-angle, which over-counts legitimate thin anisotropic cells (a
resolved strip looks like "slivers" to an isotropic detector). See
`alignment.py` in the prototype scratch.

## UW3 port plan (`method="mmpde"`)

**Architectural rule: PETSc-native and parallel-safe by construction.**
The numpy prototype was a *validation* vehicle only. The element-level
algebra (per-`K` `d×d` matrices `E`, `Ehat`, `𝕁`, and `G`, `∂G/∂𝕁`,
`∂G/∂r`, `∂G/∂M`) is genuinely local and may stay vectorised NumPy over
the rank-local element block — that is not a parallel hazard. Everything
that **couples across vertices or ranks** must go through PETSc `Vec` /
DM operations, never a rank-local `np.add.at` into a global array. The
prototype's `np.add.at` assembly is serial-only and must NOT be ported
as-is.

1. Add `_winslow_mmpde(mesh, metric, pinned_labels, verbose, **kw)` to
   `src/underworld3/meshing/smoothing.py`, **dimension-general (`d = 2`
   and `3`) from the start** — the method and all formulas above are
   general `d` (paper validates 3D), and UW3 already has the 3D
   infrastructure (`_tet_cells`, `_signed_volumes`, and a 3D branch in
   `_boundary_vertex_normals` / `_build_slip_projector` for tangent-plane
   slip). Use `cdim` everywhere: `(d+1)`-vertex cells, `d×d` edge
   matrices / metric, `det`/`inv` via batched `numpy.linalg` (not a
   hand-coded `2×2`), signed *volume* via `_tet_cells`/`_signed_volumes`
   in 3D. Do **not** raise `NotImplementedError` for 3D the way
   `_winslow_anisotropic` does — that 2D-only limitation is what this
   mover supersedes. Element terms (eqs 16, 40–41 + `∂G/∂M`, with
   `q = d·p/2`) computed per rank-local cell; the **velocity assembly is
   a PETSc Vec**, not a numpy array:
   - assemble `Σ_{K∋i} |K| v^K_i` into a global `Vec` with `ADD_VALUES`
     (DM section / `petsc_dm.localToGlobal(..., ADD_VALUES)`), so cells
     straddling a partition boundary correctly contribute to off-rank
     vertices. This is the same ghost-summation pattern flagged for the
     lumped-V source in `_winslow_elliptic` (whose `np.add.at` is a known
     serial-only TODO) — do it right here from the start.
   - the `P_i` balancing and the final coordinate update act on the
     assembled global Vec, then scatter back to the local (ghosted)
     coordinate vector.
2. **All scalar tests/norms are collective reductions** (`uw.mpi.comm`
   `allreduce`):
   - energy `I_h = Σ_K |K| G_K` — sum over owned cells then `allreduce`
     (count each cell once; use the owned-cell mask, not ghosts);
   - the line-search predicates — *"min signed area > floor"* and *"I_h
     decreased"* — must be **global** (`MIN` / the globally-summed `I_h`),
     so every rank takes the same accept/backtrack branch in lockstep
     (otherwise ranks desynchronise on `scale`);
   - the convergence norm `max|Δx|` — `allreduce(MAX)`.
3. Reuse existing parallel-aware infrastructure: `_tri_cells`,
   `_signed_areas`, `_min_incident_edge`, and `_ot_adapt._build_slip_projector`
   / `_resolve_slip` for the slip normal (`Gamma_P1`). The slip
   projection is per-vertex local (no coupling) once the velocity Vec is
   assembled and ghost-updated. `Gamma_P1` is already projected/parallel.
4. **Metric `M` and `∂M/∂x`** via `uw.function.evaluate` at element
   centroids (already parallel-aware) — do **not** hand-roll a coordinate
   loop. `M` is a `d×d` sympy matrix / `VarType.TENSOR` MeshVariable via
   the existing `supplied_D`-style entry routed by `smooth_mesh_interior`.
   Eulerian re-eval each step is safe (`M` anchored to fixed feature
   geometry).
5. The **fixed computational reference** = mesh coordinates at the first
   call, cached as a *ghosted* coordinate Vec on the mesh (like
   `_ot_adapt_reference_coords`), so each rank has its halo. For an
   *evolving* feature, keep the uniform reference and re-relax each adapt
   event (validated serially: tracks a moving fault cleanly).
6. `method_kwargs`: `p` (1.5–2), `theta` (1/3), `tau` (1), `n_steps`,
   `step_frac` (0.2), `slip` (bool/mask), `area_floor_frac` (0.01).
7. Cross-link from `ADAPTIVE_MESHING_DESIGN.md` /
   `mesh-adaptation-formulation.md`. **Regressions must cover both
   dimensions and parallel** (decision 2026-05-30: port 2D+3D, validate
   both directly in UW3 — no separate 3D numpy prototype):
   - **2D serial** (Tier-A): uniform `M` ⇒ near no-op; single fault ⇒
     on-fault band, 0 crushed.
   - **3D serial**: uniform `M` ⇒ near no-op on a tet mesh; a planar
     fault ⇒ on-plane refined slab, 0 crushed. 3D is *derived* here but
     **not yet numerically validated**, and its decoupled non-folding
     margin is tighter than 2D — treat this as a first-class acceptance
     test, not an afterthought.
   - **`np>=2` in each dimension**: matches the serial result to solver
     tolerance (same final coords up to partition-independent reduction
     order) — the assembly/ghost path is exactly where serial-only bugs
     hide.

## Open items

- Acceleration: the line-search takes tiny end-steps near convergence;
  an accelerated / semi-implicit step could cut iteration count. Any such
  scheme must keep its global-reduction predicates collective (item 2).
- Slip localization: temper/localize slip to the fault root to keep the
  finer-at-surface benefit without the global boundary-angle cost.
- Parallel correctness is a **release gate**, not an open item: the port
  is not "done" until the `np>=2` regression (item 7) matches serial.
