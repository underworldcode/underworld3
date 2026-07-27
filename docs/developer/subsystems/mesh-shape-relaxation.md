# Mesh shape relaxation

Refinement chooses where new nodes go from **combinatorics**, never from
geometry: newest-vertex bisection puts the node on whichever edge the tagging
rule nominated, centroid (Alfeld) refinement puts it at the cell centre. Neither
rule knows anything about the problem being solved. A refined mesh therefore
carries needles and slivers that reflect the base mesh's arbitrary choices —
elements organised around the base grid's axes rather than around the feature
you refined towards.

**Relaxation** moves those nodes to where the geometry wants them, keeping the
resolution the refinement installed:

```python
child = mesh.adapt(metric, max_levels=3)
child.relax()
```

or, in one call:

```python
child = mesh.adapt(metric, max_levels=3, relax=True)
```

## Why the mover could not already do this

The obvious tool was the MMPDE mover (`mesh.redistribute_nodes`), and it moved
*nothing*: `redistribute_nodes(1)` on a distorted mesh is exactly a no-op.

The reason is the mover's **reference frame**, not its metric. MMPDE generates
the physical mesh as the image of a fixed *computational* mesh, and that
reference is set to the mesh's own coordinates on the first call. "Distortion"
therefore means *deviation from the mesh I was handed* — so a sliver-ridden mesh
is its own optimum, and under a uniform metric the identity map already
minimises the functional.

Relaxation swaps the reference for a **regular simplex**. Distortion now means
*deviation from equilateral in the metric*, so the distortion a mesh arrived
with is no longer self-justifying. Everything else is unchanged: same
functional, same non-folding guarantee, same parallel assembly. (This is the
target-matrix idea from the TMOP literature.)

Three frames exist on the mover's `reference=` argument:

| `reference` | reference element | what it does |
|---|---|---|
| `"mesh"` | the mesh at first call | **redistribution** — the classical MMPDE. Default. |
| `"ideal"` | regular simplex at *each cell's own* volume | **shape repair at fixed size**. `r = 1` at entry, so the size term starts at its optimum and only shape moves. |
| `"ideal-metric"` | regular simplex at one uniform volume | shape repair **plus** re-grading; the metric sets size. |

`mesh.relax()` selects `"ideal"`; `mesh.relax(metric)` selects `"ideal-metric"`.

## Relaxing is not the same as relaxing to a metric

```{warning}
Passing a metric to `relax()` changes what is being optimised. The
ideal-metric frame re-grades as well as reshapes, and it will **trade element
shape away** to chase the size field. Measured on a 4-level graded box, the
99th-percentile maximum angle went 117.9° → 113.8° with no metric, but
117.9° → **127.4°** with one.
```

Use `relax()` when you want a shape guarantee. Use `relax(metric)` when the
sizes themselves are wrong and you accept shape is no longer the objective.

## Where to put the relaxation

Two placements, and **neither dominates** — which you prefer depends on which
property you weight:

```python
adapt(metric, max_levels=N, relax=True)              # once, at the end (default)
adapt(metric, max_levels=N, relax="per-generation")  # inside the refinement loop
```

Measured in 2D on a gmsh base with the NVB refiner, a dipping fault segment:

| property | unrelaxed | at end | per generation |
|---|---|---|---|
| P1 interpolation error | 2.75e-2 | 2.56e-2 | **2.38e-2** |
| 99th pct max angle | 112° | 110° | **96°** |
| on-fault size spread (p90/p10) | 2.80 | **2.31** | **2.31** |
| far-field closure halo (excess cells) | 42.8 | **27.7** | 43.9 |
| cells | 1110 | 1110 | 1186 |

Relaxing **at the end** cannot change the topology, so the cell count is
identical to the unrelaxed mesh, and it cuts the far-field closure halo most.
Relaxing **per generation** gives the cleanest element shapes and the lowest
error, because each generation marks from already-relaxed coordinates — but it
changes which cells get marked, so it spends ~3% more cells.

Every column beats no relaxation. The default is at-end because it is the
cheaper and less invasive of the two.

## Multigrid: fine in 2D, can break in 3D

In **2D** relaxation is transparent to the custom-P geometric-MG tail. The
operators need only the *topological* hierarchy, and the coarse levels keep
their own coordinates. Measured with the Stokes velocity block and with
Poisson: full `pc=mg` in every configuration, no fallback, solutions agreeing
to four significant figures, and the at-end case one Krylov iteration
*cheaper*.

### Why relaxation can disturb it, and why it no longer costs you the hierarchy

The custom-P prolongation is built **geometrically, not topologically**. The
`barycentric` builder re-triangulates the coarse DOF *point cloud*
(`Delaunay(coarse_coords)`) and locates each fine DOF in one simplex of it; it
never consults the parent/child refinement relation. That is deliberate — it is
exactly what lets custom-P accept non-nested coarse/fine pairs — but it means a
preserved topology buys the transfer nothing.

So the builder has **local support**: a coarse DOF is reached only if some fine
DOF lands in a simplex touching it. Move the fine coordinates, which is what
relaxation does, and a coarse DOF can lose every fine image. Its column in `P`
goes to zero, the Galerkin coarse operator `PᵀAP` acquires a zero row and
column, and it is singular. 3D is more exposed: a Delaunay triangulation of a
graded 3D cloud is full of slivers, each simplex has only four vertices, and
proportionally more DOFs sit near the hull.

The builder now **repairs orphaned coarse DOFs directly**: each is given its
nearest fine DOF as a pure injection (weight 1), which is the same fallback
already used for fine points outside the coarse hull. Partition of unity and
sparsity (~d+1 non-zeros per row) are both preserved, and the column is no
longer empty. Measured on a relaxed 3D adapt child: `pc=mg` at 2 iterations,
where before the repair it fell back to GAMG at 23.

If a pathological pair still leaves an orphan, the build retries with the
global-support `rbf` builder before giving up. That is a **last resort, not a
scalable transfer**: the RBF prolongation is fully dense (measured
`nnz/row == n_coarse` exactly — 200 of 200, 800 of 800), so `PᵀAP` is dense
too. It rescues correctness on small problems; it is not something to rely on
at production sizes.

## What relaxation cannot do

- **It cannot rescue centroid/Alfeld refinement.** Those slivers are
  *structural* — an interior point wired to a fixed parent face, producing a fan
  of near-degenerate cells radiating from one vertex. Node motion cannot unpick
  a topology. Measured: cells with `q < 0.3` stay at 25–28% and the worst angle
  stays at 176–178° whatever the placement. Centroid refinement is a
  shallow tool only — see {doc}`../design/NVB_GRADED_ADAPT`.
- **It cannot flatten the size steps along a feature.** Bisection generations
  are discrete (each halves cell area), so on-feature sizes come in
  powers-of-two classes. Relaxation compresses the spread (p90/p10 2.80 → 2.31)
  but two size classes remain populated.
- **It does not concentrate cells on the feature.** That is refinement's job.
  Relaxation holds the size distribution by construction, so the share of cells
  in the band barely moves (48.9% → 51.1%).

## 3D

Relaxation improves 3D mesh **quality** but has not been shown to improve
accuracy. On an adapted 3D mesh it halves the near-degenerate population (cells
with `q < 0.1`: 3.6% → 1.8%), lifts median quality 0.32 → 0.39 and pulls the
99th-percentile dihedral angle back from 153° to 146° — but the interpolation
error of an *isotropic* feature is unchanged (+0.5%).

That is consistent rather than contradictory: relaxation holds the size
distribution, so its only route to lower error is by *aligning* cells with the
feature, and an isotropic metric gives it no reason to align anything. Whether
an anisotropic (tensor) metric recovers the 2D-style accuracy gain in 3D is
open — see the tracking issue.

Use it in 3D for element quality and conditioning, not expecting an accuracy
win.

## Diagnostics

Element quality alone is a poor guide here, and can actively mislead: on a
structured base the *unrelaxed* mesh scores best on median quality (every cell
is an identical right triangle) while resolving the feature worst. Four
properties are worth reporting together:

- **gather** — share of cells within the metric band;
- **clean** — fraction of near-degenerate cells and the 99th-percentile max
  angle (not the worst single element);
- **uniform** — spread of cell size along the feature;
- **leakage** — refinement added where the metric did not ask for it.

Leakage in particular resists being reduced to one number: it is a *spatial*
property (large coherent over-refined blocks versus small scattered patches),
and per-cell maps of `log2(h / h_asked_for)` are more informative than any
scalar we have found.
