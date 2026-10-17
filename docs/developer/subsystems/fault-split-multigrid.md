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
by injection. This is now the rule on every custom-P hierarchy, split or not
(`CustomMGHierarchy.drop_orphans`): the nearest-node injection put a weight-1 entry
on a fine row that already carried one, which is two identical columns and a singular
coarse operator. Measured: NaN on a two-mid placed-ladder chain in serial (four
orphans, with and without a split finest) and 512 duplicated columns at np = 2. The
coarse levels' FAC patch rows do not survive the renumbering and are cleared; the
finest's are unchanged.

## Rotated walls on every level

A hierarchy under rotated free-slip walls (an annulus, a spherical shell) rotates
its coarse levels too (`level_wall_rotation`, composed with the pair rotation of a
split level by `level_rotation`; `_build_rotated_custom_Pl` routes any hierarchy
with rotated walls through `rotated_split_transfers`). Each coarse level takes the
same per-node frame as the finest, built on the level's own wall nodes from the
solver's boundary specifications (the analytic normal evaluated at the level's
nodes, or the level's own measure-weighted facet normals), and its normal rows are
removed. Before this the coarse levels were Cartesian and only the finest's normal
rows were zeroed in the transfer. A coarse wall node's normal Cartesian component
was then reached only through the fine tangential rows, weighted by the sine of the
angle between the two nodes' normals: a Galerkin nodal block of 1e-3 against 2.5 at
a node near an axis, 1e-20 on the axis. The inode SOR smoother inverts those 2x2
blocks and reported a zero pivot, and the velocity solve failed before its first
iteration (measured on the annulus of the Zhong & Gurnis analogue; a box with
Dirichlet walls never showed it because an axis-aligned wall's sines are exactly
zero). Two further rules came out of the same case:

- the reach test of the orphan rule is relative (`UNREACHED_REL`, 1e-8 of the
  level's largest column, squared norms) and runs after the finest's constrained
  rows are zeroed, not before;
- the Galerkin operators of the rotated coarse levels do not use PETSc's inode
  block grouping (`MAT_USE_INODES` off, set after the velocity PC's set-up; the
  finest operator is untouched). A rotated wall node keeps one tangential row
  beside the two-row nodes, and on the developed temperature field of the
  annulus run at four ranks the inode SOR's 2x2 block inversion met a zero pivot
  on the base's two finest levels with every scalar diagonal healthy (minimum
  2.6 against a maximum of 3e4); without the grouping the same solve converged
  in 9 iterations to the two-rank answer. Which rows the grouping pairs there is
  not yet identified (a row-level inspection is in the driver's history);
- the free-slip gauge mode must verify for the SVD coarse solve to be selected,
  which on a faceted circle needs the exact radial normal
  (`normal=mesh.CoordinateSystem.unit_e_0`); and that SVD needs a small coarsest
  level (14k unknowns took over 30 minutes, 900 is instant), so a base with three
  halvings rather than one.

## Parallel

The add-on is partitioned **once, before any surgery**, as one hierarchy
(`utilities/addon_partition.partition_addon`, design note
[parallel-addon-distribution-2026-10](../design/parallel-addon-distribution-2026-10.md)).
`add_fault`, `add_conforming_sheet`, `split_fault` and `split_faults` call it in
parallel: the fault regions — each fault's cell star plus growth layers, marked on
every add-on level because a coarser level's star reaches further — are glued and
sent to the rank already holding most of them, the load is balanced by diffusion
over the seams (free cells move from a heavier rank to a lighter neighbour, the
cells farthest from the faults first), and the one assignment is applied to every
add-on level through the parent-cell relation. The background levels never move.
The cut, the placement and the split then run on rank-interior regions and move
nothing; the replay of the faults on the add-on levels (`split_tail_levels`) arrives
on levels already partitioned with the finest.

Every add-on pair is therefore co-partitioned exactly and the cell-located transfer
finds every node in its local coarse cells. The one pair the partition separates —
the base's finest static level against the first add-on level — carries the exact
relation: each level records, per cell, where its parent on the level below lives
(provenance riding the move through `distributeField`, the new addresses resolved
through two star forests), and `provenance_transfer_parallel` fetches each fine
cell's parent through one star forest. Nothing is located across partitions and
nothing is all-gathered. The transfer's record on the hierarchy
(`level_transfer_stats`) shows `misses` zero on every pair and `fetched_remote` on
the moved one.

A coincident pairing rides a redistribution as a pair of DM labels
(`fault_split.stamp_point_pairs` / `collect_point_pairs`), and a per-cell field as
a cell-wise section through `distributeField` on the migration star forest
(`redistribute_by_assignment(fields=...)`). `extrude_sheet` builds the layer on
one rank and distributes it with the labels on; its fault stars are then made
rank-interior by the older per-fault move, which remains for that path.

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

A planar fault patch in a 3-D box (sheet placed on the NVB-refined child with
`clearance=0.9`, 35712 tetrahedra at one add-on level): 5 velocity iterations, slip
0.18113. One isotropic add-on level in 3-D is three bisection passes, which the
tail's one-level-per-halving rule folds into the finest, so there is no intermediate
level to split at k = 1 and the two tails coincide there; k = 2 is the first with a
split 3-D level.

## Tests

`tests/test_1091_place_outcrop_curved_wall.py` (the placer on an annulus: loop
orientation, the dipping band through the curved wall, the outcropping ladder on
both placement routes) and
`tests/test_1087_fault_split_multigrid.py` (tail structure, iterations falling with
refinement against the welded tail, viscous fault, a moving fault, the transfer's side
awareness and quadratic exactness) and
`tests/parallel/test_1089_fault_split_multigrid_parallel.py` (np = 2 and 4: split and
welded tails converge with the serial slip; the thick twin extruded with its pairing).

## Not yet done

- A regression test for the coarse-level wall rotation itself (a rotated free-slip
  annulus with a placed band and the custom tail), beyond the example scripts
  (`scripts/sessions/fault_mg/zg95_probe.py`, `zg95_annulus.py`).
- The 3-D sheet placement on an NVB-refined child needs `clearance` of about 0.9 at
  one add-on level (its volume guard fires at the default 0.6) and refuses the
  two-level child at every clearance and sheet size tried (volume change 4e-6 to
  5e-5 against a guard of 1e-9). General 3-D patches on add-on meshes wait on the
  placement; `add_fault` passes the placement options through.
- A placed band's spines (`cut=False`) are not replayed on the levels; such a fault
  keeps the welded tail and the FAC patch of the fault-network work.
- The replay runs from the finest add-on level down and stops, with a warning, at the
  first level the faults cannot be carried onto; that level and those beneath it stay
  welded. A network whose junction ligament is one fine cell wide closes at the next
  coarser spacing (the two labels meet), so such a network gets split levels only where
  the ligament survives — on `test_0866`'s kissing Y with one add-on level, none.
- Co-locating the add-on levels with the finest in parallel is an optimisation the
  query path makes unnecessary for correctness; not measured at scale.
- The one-rank build of `extrude_sheet` bounds the layer size by one rank's memory.
