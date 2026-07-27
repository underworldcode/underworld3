# Two MG transfer paths, each used properly

**Status**: PROPOSED (2026-07-26)

## The problem

`mesh.adapt()` maintains an exact refinement hierarchy — that is the entire
point of newest-vertex bisection with conforming closure, and it is what the
"escape"/halo cost buys. The multigrid transfer then **throws that hierarchy
away** and re-derives an approximation to it geometrically:

```python
def barycentric_prolongation(coarse_coords, fine_coords):
    tri  = Delaunay(coarse_coords)          # re-triangulate the coarse DOF CLOUD
    simp = tri.find_simplex(fine_coords)    # locate each fine DOF in one simplex
```

The coarse mesh's own cells are discarded; the parent/child relation is never
consulted. We pay for exact hierarchy maintenance and then do not use it.

This is not merely inelegant — it is the direct cause of issue #424. The
barycentric builder has only **local support**: a coarse DOF is reached only if
some fine DOF happens to land in a Delaunay simplex touching it. Move the fine
coordinates — `mesh.relax()`, surface snapping, free-surface deformation — and
a coarse DOF can lose every fine image. Its column in `P` goes to zero,
`PᵀAP` acquires a zero row and column, and the coarse operator is singular.

## Why the geometric builder exists (and should stay)

It was built for genuinely **non-nested** pairs, where no topological relation
exists at all:

* a moved base mesh against an adapted child,
* two independently generated meshes,
* the planned non-hierarchical variant with newly added seed points.

That capability was asked for and is worth keeping. The error was applying it
to the *nested* case as well, where the exact answer is already known.

## The nested transfer is trivial and cannot go singular

For a bisection hierarchy every fine vertex is exactly one of two things:

* an **inherited** coarse vertex → weight 1 on itself;
* the **midpoint of a coarse edge** `(a, b)` → weights ½, ½ on `a` and `b`.

Properties that matter:

* **Exact** for the un-deformed nested mesh (it *is* the FE embedding).
* **Structurally full rank** — every coarse DOF appears with weight 1 in its
  own inherited fine DOF, so a zero column is *impossible by construction*.
  #424 cannot occur in this formulation.
* **Immune to node motion.** The relation is topological, so relaxation,
  snapping and deformation do not disturb it.
* **Cheap** — 1 or 2 non-zeros per fine row, no Delaunay, no point location.

The engines already record exactly this. `nvb.py` keeps

```python
self.edge2mid = {}     # (a, b) -> midpoint vertex id
```

for both the 2D `NVBMesh` and the 3D `TaggedBisectionMesh`, and the native
`nvb_transform.c` is a real `DMPlexTransform`, whose split-edge bookkeeping
carries the same information.

Note also that `petsc_dm_set_regular_refinement` — our own wrapper for the flag
PETSc checks to take its exact nested interpolation path — appears exactly once
in the repository: its own definition. Nothing calls it.

## The one hard part: *when* the relation can be captured

The relation must be recorded in **DM point numbering**, and the bridge from
engine-internal vertex ids to DM points is coordinate matching (this is how
`to_dm` already transfers labels). Coordinate matching is exact only while
midpoints are still exact float averages of their parents.

Both snapping and relaxation destroy that:

* **native path** — `_nvbx.refine()` produces the DM, *then*
  `snap_level_boundaries()` moves boundary vertices. There is a clean window
  between the two where coordinates are pristine.
* **cell-list path** — the snap is applied to `nvb.coords` *before* `to_dm()`
  is called, so by the time a DM exists the window has closed.

So the capture must be designed in, not bolted on:

1. capture the parent/child map immediately after refinement, **before** any
   snap or relaxation, and store it on the child;
2. or have `to_dm()` return its engine-id → DM-point map directly, removing the
   dependence on coordinate matching altogether (preferable — it is exact
   regardless of ordering).

Option 2 is the more robust and is the recommended route.

## Proposed shape

* Each generation stores a sparse vertex-level prolongation on the child, e.g.
  `child._adapt_prolongation[k] = (parents, weights)` with `parents` of shape
  `(n_fine, 2)` in DM vertex numbering (an inherited vertex is `(c, c)` with
  weights `(1, 0)`).
* `custom_mg` gains a `"topological"` builder that consumes it and expands from
  vertices to DOFs for the field's basis.
* Selection: **topological when the map is present** (nested adapt hierarchy),
  geometric otherwise. The RBF retry added for #424 then becomes a fallback
  that should essentially never fire on adapt children.

## Caveat worth measuring, not assuming

With snap-every-generation a midpoint is moved onto a curved boundary, so it is
no longer at ½(a+b) physically. The ½,½ transfer is then not the FE-exact
interpolation of the coarse space at that point — the geometric builder is.

Multigrid does not require FE-exactness, only a full-rank prolongation that
represents smooth functions well, and ½,½ on the topological hierarchy is the
classical geometric-MG choice. But on strongly curved or strongly relaxed
meshes the geometric transfer may be the more *accurate* one. The trade is
**accuracy versus robustness**, and it should be measured on the annulus and
spherical-shell cases before the default is settled — not assumed either way.

## Related

* #424 — zero-column failure and the RBF retry (the symptom this addresses).
* `custom_mg.py` — `barycentric_prolongation`, `rbf_prolongation`,
  `CustomMGHierarchy`.
* `nvb.py` — `edge2mid` in `NVBMesh` and `TaggedBisectionMesh`.
* `petsc_discretisation.pyx` — `petsc_dm_set_regular_refinement` (uncalled).
