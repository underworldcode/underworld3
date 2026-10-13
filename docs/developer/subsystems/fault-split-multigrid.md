# Split-level multigrid for split-node faults

Governing document for the fault multigrid hierarchy: `utilities/fault_mg.py`, the
split-pair branches of `utilities/custom_mg.py` and `utilities/rotated_bc.py`, the
level replay in `Mesh.add_fault`, and the fault-carrying tail of `meshing/extrude.py`.
The exploration that chose this design is
[FAULT_MULTIGRID_HIERARCHIES](../design/FAULT_MULTIGRID_HIERARCHIES.md).

## What it is

A split-node fault lives on the finest mesh. Its multigrid levels come from the add-on
hierarchy: the static background, which is never cut, and the add-on levels refined
from it toward the fault. `mesh.add_fault` now carries the fault through **every
add-on level** of the inherited tail, cut and split exactly as the finest is, so the
coarse spaces contain the discontinuity. The background levels are the base's own
wraps, untouched (`_mg_static`).

```python
child = base.adapt(metric, max_levels=k)            # add-on levels toward the fault
mesh = child.add_fault([("Flt", trace)])            # finest AND every add-on level split
stokes = uw.systems.Stokes(mesh)
stokes.add_fault_bc(0, "Flt")
stokes.solve()                                      # geometric multigrid, no setup
mesh = child.add_fault([("Flt", trace)], split_levels=False)   # the welded tail
```

In 3-D a fault is a triangulated sheet (`("Flt", points, triangles)` or a
`FaultSurface`), placed with the sheet placement and split; the levels are replayed
the same way. The extruded thick twin of a faulted sheet (`mesh.extrude`) extrudes the
split sheet and its split levels, so the layer's hierarchy is the sheet's one dimension
up.

## The three pieces

**Side-aware transfer** (`cell_located_prolongation`, `cell_located_transfer_parallel`).
For each fine node one fine cell containing it is taken, and that cell's *centroid* is
located in the coarse mesh. A centroid is never on the fault, so the coarse cell is on
the node's own side; a node on the slit interpolates from its own side. The coarse
cell's Lagrange interpolant is evaluated at the node by the dual-basis identity
$W = BM^{-1}$ (as the nested native-pair transfer does), batched over cells. The levels
are conforming but not nested near the trace, so a fine node can lie slightly outside
its located cell: the evaluation point is clamped into the cell (barycentric
coordinates clipped and renormalised), which keeps every weight within the Lagrange
basis on the simplex. Every pair that touches a split level takes this transfer; the
node-cloud builders triangulate the coarse nodes and degenerate on a slit.

**Per-level pair frame** (`level_pair_rotation`). Each split level carries its own pair
rotation $Q_l$ (mean and jump in the fault frame, the same construction as
`build_rotation`, from the level's own pairing and facet normals), built on the
solver's discretisation copied onto the level DM. The rotated transfer is
$\hat P_l = Q_{l+1} P_l Q_l^{T}$ with the level's opening rows removed (and, on a
surface mesh, the jump and mean along the surface normal).

**Unreached unknowns dropped** (`rotated_split_transfers`, `restrict_unreached`). A
coarse unknown no fine unknown reaches (a few near the tips) is a null mode of the
Galerkin coarse operator. It is removed from the level, fine to coarse, never repaired
by injection: the injection duplicated columns and made the coarse operators singular
in parallel.

## Parallel

Every split level is redistributed so its fault star is rank-interior (the split's own
rule), so the pair blocks are rank-local on every level. The transfer locates each
owned fine node in the *local* coarse mesh (owned plus ghost cells); the nodes whose
coarse cell lives elsewhere are all-gathered as query points, answered by the rank
holding the cell with global column indices, and the asker keeps the best answer. The
levels need not be co-partitioned.

A coincident pairing rides a redistribution as a pair of DM labels
(`fault_split.stamp_point_pairs` / `collect_point_pairs`): one value per pair on its
Minus and Plus points. The migration star forest PETSc returns for an *initial*
distribution does not index the stratified numbering (measured: three quarters of the
points mapped to the wrong depth), so the forest is not used. `extrude_sheet` builds the
layer on one rank, distributes it with the labels on, redistributes the fault stars
rank-interior and reads the pairing back; a balanced partition readily puts the whole
Minus side on one rank and the whole Plus side on another, since the slit disconnects
them.

## Measured (box in simple shear, straight fault, 2026-10-13)

Velocity iterations per solve, split levels (C) against the welded tail (A):

| add-on levels $k$ | near-fault $h$ | C | A |
|---|---|---|---|
| 2 | 0.0125 | 6 | 13 |
| 3 | 0.00625 | 5 | 18 |
| 4 | 0.0031 | 4 | 23 |

Viscous fault at $k = 3$: 5 against 17. The peak slip agrees to six figures. In
parallel, C at $k = 2, 3$: 6, 5 iterations at np = 2 and 7, 6 at np = 4; A runs at np = 2
too (15), where before it returned NaN.

A moving fault at $k = 4$ (three positions, the add-on levels rebuilt and re-split at
each): 4 to 5 iterations in serial, 7 at np = 8, the slip equal to six figures. Per
step at np = 8: adapt 0.8 to 1.3 s, cut and split of every level 1.5 to 2.1 s, solve
11 to 14 s (serial: 2.7, 4.5 and 26 to 30 s). In parallel `adapt` keeps one fewer
intermediate level (a known difference in its subsampling), which the transfer does
not mind.

The faulted thick plate of `test_1083` (two layers, 6024 tetrahedra, 261 pairs):
11 velocity iterations in serial and 13 at np = 2, slip 0.19356 on both; with one
add-on level in the sheet (7524 tetrahedra, 681 pairs): 16 in serial and 20 at
np = 2, slip 0.19382 on both. The thin layer's iteration count still grows with
refinement; the through-thickness coupling and the pressure Schur complement on a
thin layer are their own question, outside this hierarchy.

## Tests

`tests/test_1087_fault_split_multigrid.py` (tail structure, iterations falling with
refinement against the welded tail, viscous fault, a moving fault, the transfer's side
awareness and quadratic exactness) and
`tests/parallel/test_1089_fault_split_multigrid_parallel.py` (np = 2 and 4: split and
welded tails converge with the serial slip; the thick twin extruded with its pairing).

## Not yet done

- The 3-D sheet placement refuses an NVB-refined child (its volume guard fires), so the
  general 3-D path is exercised only on meshes the placement accepts.
- Co-locating the add-on levels with the finest in parallel is an optimisation the
  query path makes unnecessary for correctness; not measured at scale.
- The one-rank build of `extrude_sheet` bounds the layer size by one rank's memory.
