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

A second `add_fault` call on a mesh that already carries split faults keeps them
(#1099). The earlier faults' pairing is recorded by mesh point, and a cut renumbers
the points, so the call marks every recorded point with a label before it cuts, reads
the label back afterwards, and hands the renumbered pairing to the new mesh and to
each add-on level that carried it. If a recorded point was refined or removed, or a
recorded node moved, the new fault has changed an earlier one and the call is refused:
the check is collective. Only the faults the call adds are replayed onto the tail.
`add_conforming_surface` and `add_conforming_sheet` carry the record through their cut
the same way. In 3-D the sheet clip leaves the two sides of an earlier fault out of the
domain boundary it clips against; they stay walls to the cavity, which is what keeps a
new sheet clear of them. The outcrop frame skips their vertices: they are on no wall
region, so an outcrop neither deletes them nor counts them as near. Both apply only to
a mesh that carries a split fault (`place_sheet(split_faults_are_interior=True)`); on
any other mesh a wall vertex missing from the boundary complex is still an error.

The add-on partition restamps the pairing through its move. It does not carry the
seam normals of a fault split through the partition seams (`across_seams`), and
refuses to move a mesh that holds any: a second call on such a mesh is refused, not
handed normals that name other points.

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

`CustomMGHierarchy.level_patch_rows` is rank-local, and its form differs between
ranks: per level None, one `(owned, subdomain)` pair, or a list of pairs, and a rank
holding no part of a zone has an empty pair. Read the patch through
`hierarchy.patch_summary` instead. It is one entry per level, None where no rank has
a patch, otherwise the blocks, owned and subdomain rows summed over the ranks, the
largest block, and each rank's block sizes. It is made once, with one allgather
(`custom_mg._fac_patch_summary`), wherever the patch is installed:
`CustomMGHierarchy.install` and the rotated path's two `_configure_pcmg` calls.
Reading it makes no collective call, so any rank can read it alone. A driver that
looped a reduction over its own rank's blocks hung at np 4, because the ranks made
different numbers of calls. With no zone keyed, the summed patch (owned) rows are the
serial count at any rank count (`test_1133`), and the subdomain rows follow the
structural halo, which can be a few rows short of serial (#896). With a zone keyed,
its blocks and the structural block beside them hold the same rows as patch and
subdomain, so both counts carry that halo. Each block of the finest level is held
whole by one rank (see below, #925), and `block_rows_per_rank` shows where each sits:
the structural LU on one rank. The record is each block as handed to the installer, before PCASM
adds its overlap layer at set-up.

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
- the free-slip gauge mode must verify for the rotation to be removed, which on
  a faceted circle needs the exact radial normal
  (`normal=mesh.CoordinateSystem.unit_e_0`). The coarse solve is the shared
  redundant LU with a zero-pivot shift (#901).

## The FAC patch smoother

A level with a FAC patch (the rows its transfer cannot represent, plus a one-cell
halo, or a keyed `fac_zone`) is smoothed by a multiplicative composite: the level's
SOR, then a basic ASM over the patch's blocks with one operator-sparsity overlap
layer, each block solved by LU with a nonzero factor shift and quotient minimum
degree ordering (`custom_mg._configure_pcmg`). This is the one configuration; the measurement knobs
that varied it are gone, and `test_1090` checks the live objects on every install
path.

- **Composite, not replacing.** Blocks that replace the level's SOR starve whatever
  they leave out: 60 velocity iterations against 10 on the fine S-fault rig (#670).
- **Basic ASM.** The correction must land on the halo too; restricted ASM stalled at
  80-375 outer iterations on a banded Poisson tail.
- **LU with a shift.** PCASM's default, unshifted ILU(0), takes a zero pivot on the
  rotated Galerkin block or gives no band solve (344 velocity iterations against
  67, #892). Inexact block solves were measured on the add-on matrix's box (kaiju,
  np 4, 2026-11). They gave the same Newton counts and slip as LU, and took longer:

  | box | LU | ILU(2), 5 steps | GAMG, 5 / 10 steps | hypre, 5 / 10 steps |
  |---|---|---|---|---|
  | S 0.004, one block | 603 s | 1299 s | 1750 / 2102 s | 2360 / 2179 s |
  | S 0.004, blocks of 2000 rows | 433 s | 1404 s | 2089 / 2058 s | 2248 / 2981 s |
  | S 0.015625, one block | 516 s | 766 s | 931 / 869 s | 923 / 1207 s |

- **Quotient minimum degree.** A fault band's block is long and narrow, and under
  PETSc's default ordering (nested dissection) its fill followed the subdomain's row
  numbering: the same block took 8.1e9 flops to factor at np 4 and 2.35e10 at np 8
  and 16. Measured on the add-on matrix's row 5 networks (kaiju, np 4 to 16, 2026-11),
  with the same iterations and slip in every case (reverse Cuthill-McKee was worse
  than nested dissection):

  | case, np | ordering | LU flops | MatSolve, max | solve, first / again |
  |---|---|---|---|---|
  | pair, 8 | nd | 2.35e10 | 77.2 s | 59.7 / 46.6 s |
  | pair, 8 | qmd | 4.08e8 | 12.5 s | 15.1 / 12.6 s |
  | pair, 16 | nd | 2.35e10 | 75.8 s | 60.3 / 43.6 s |
  | pair, 16 | qmd | 4.10e8 | 15.4 s | 18.5 / 16.0 s |
  | junction, 4 | nd | 7.24e9 | 39.5 s | 32.5 / 27.0 s |
  | junction, 4 | qmd | 2.94e8 | 11.1 s | 13.8 / 12.4 s |

  The block's triangular solves, which every smoother application pays, are most of
  its cost: the factorisation is a few per cent of it.

The size of the LU blocks is bounded by `fac_block_rows` for keyed zones. The
structural block is not bounded: it is one LU, held by one rank in parallel (#895).

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
add-on level through the parent-cell relation. The coarse background levels never
move. When the base has no add-on levels (one built with `refinement`, say), the
level that moves is the base's own finest, the one the band is placed on. That
level is still background: `partition_addon` marks the moved copy `_mg_static`,
because the base mesh itself carries no mark (only the wraps of its levels do). The
replay therefore leaves it uncut, as in serial. A later partition of a mesh placed on
it (the cut that `add_fault` makes) takes that moved level as its base and does not
move it. The first level's recorded parents on it are its current addresses, and the
record of the level's own earlier move does not apply. The cut, the placement and the
split then run on rank-interior regions and move nothing. The replay of the faults on
the add-on levels (`split_tail_levels`) arrives on levels already partitioned with
the finest.

Every add-on pair is therefore co-partitioned exactly and the cell-located transfer
finds every node in its local coarse cells (`level_transfer_stats` records `misses`
zero; the query exchange still runs, with nothing to send). The one pair the
partition separates, the static level under the add-on (a coarse wrap, or the moved
finest of a base with no add-on levels) against the first level above it, carries
the exact relation: each level records, per cell, where its parent
on the level below lives (provenance riding the move through `distributeField`, the
new addresses resolved through two star forests), and `provenance_transfer_parallel`
fetches each fine cell's parent through one star forest (its record carries
`fetched_remote`). Nothing on that pair is located across partitions; a fine node in
no recorded cell is a collective refusal.

The record is read in the geometry the mesh was made in. A move of the mesh
(`mesh.deform`, the MMPDE mover, the Laplacian smoother) changes the finest level
and none of the levels below, so the parents are located from the vertices the
mesh was built with (`mesh._as_built_vertices`, taken when the mesh is constructed
and read once it has moved), and the transfer takes each node back to those
vertices before it weights it: the transfer of a moved pair is the one it had
unmoved. A cut, placed or split mesh made on a moved one reaches the level below
through its host's map (`mesh._addon_reference`, set by `inherit_provenance` with
the level it is valid against). A periodic mesh keeps no such record (#1091).

A coincident pairing rides a redistribution as a pair of DM labels
(`fault_split.stamp_point_pairs` / `collect_point_pairs`), and a per-cell field as
a cell-wise section through `distributeField` on the migration star forest
(`redistribute_by_assignment(fields=...)`); a per-vertex field rides the same way
(`vertex_fields=...`), which is how a moved level keeps its as-built coordinates. `extrude_sheet` builds the layer on
one rank and distributes it with the labels on; its fault stars are then made
rank-interior by the older per-fault move, which remains for that path.

## A moved level keeps its transfers

A move of the mesh that keeps its topology (`mesh.deform`, the MMPDE mover, the
Laplacian smoother) changes where the nodes of the finest level sit. It changes no
cell, and no node's parent cell on the level below. The transfer between two levels
states that relation, so it is the same before and after the move.

The hierarchy is built again at the first solve after a move, and
`CustomMGHierarchy.build` reads a moved level on the geometry the level was made
in. `fault_mg.as_built_nodes` reads the nodes on the vertices the mesh was made
with (`Mesh._transfer_vertices`), through the interpolation that gives the mesh
its nodes where it stands. The interpolation holds the place of every node in its
cell and no coordinate, so it is the same operator on the same vertices as before
the move and the positions are the built ones bit for bit, at any degree and for a
discontinuous space as well. The node-cloud builders need that. They are not
continuous in the nodes they are given: a node on a face of the triangulation is
read in either neighbour, and with the positions recovered only to round-off one
weight of a pair of tetrahedral levels changed by $1.4\times10^{-4}$ on 2 and 4
ranks and by $1.1\times10^{-4}$ on 8. Those positions are the fine side of the
pair on three routes: the node-cloud builders, the cell-located transfer between
split levels, and the embedding of a native refine pair. The native pair
identifies a fine vertex with a coarse vertex or an edge midpoint by exact
equality of coordinates, so it reads those vertices themselves; on the moved
coordinates it found no match and the pair fell to the node-cloud builder. In 3-D
the embedding carries the wrong entries of #449, and a moved pair now carries
them as the pair at rest does. The parallel builders are handed the same
positions. The recorded route between add-on levels on several ranks
(`provenance_transfer_parallel`) is not: it locates nothing and takes each node
back through its own cell (#1091). Only simplex levels are kept: a quadrilateral
or hexahedral level is read on its moved coordinates, and its transfers are
rebuilt from them.

Built from the moved coordinates, a moved node was located among the unmoved
coarse nodes and its weights followed the displacement. On the fixtures of
`tests/test_1176_custom_mg_transfer_after_a_move.py` (serial, displacements of
0.02 on a unit box) a weight changed by 0.65 to 1.1, and the faulted Stokes solve
went from 5 velocity iterations to 13 after each of three moves, with 62 to 137
coarse unknowns of the level under the fault merged by the dependent-column
repair of #1101. With the level read where it
was built the transfers after a move equal those before it entry for entry, and
the same solve takes 6 iterations after each of three moves, as a solver built on
the moved mesh does. On 2 to 5 ranks it took 6 at rest and 7 after each move.
`tests/parallel/test_1176_custom_mg_transfer_after_a_move_parallel.py` holds those
counts. On 7 and 8 ranks the adapt keeps one add-on level on that fixture instead
of two (#865), so the level under the fault is welded: the finest level carries a
FAC patch at rest, a move drops it, and the solve takes 3 iterations at rest and
15 after each move with the transfers unchanged to round-off.

The kept transfers do not lower the count on every move. Under a mover that
shifts interior nodes by one to four cell widths while the coarse levels still
fill the domain, the count changes by one in either direction. With
`node_redistribution` on a twice-refined box (P2 Poisson, three moves, serial) it
went from 3 to 2 at one contrast of the metric and was unchanged at two others; on
a box refined three times it went from 2 to 3 at the strongest contrast and was
unchanged at the others; the Stokes velocity count was 3 either way. Where the
boundary moves the transfers kept do better: with the top wall of a box
raised by 5 % in each of three steps the velocity count stays at 3, against 5, 6
and 7, and `preconditioner="fmg"` on a deformed box stays at 2 against 3.

A strong redistribution is the limit. On the box refined three times the mover
carries nodes up to 3.8 cell widths at the strongest contrast, and 3.1 at the one
below it. A native pair takes the node-cloud builder on more than one rank, and a
serial run takes it with `UW_CUSTOM_MG_DISABLE_NESTED_NATIVE=1`. Measured that
way, the serial Stokes velocity count at the strongest contrast was 4 after each
move with the transfers kept and 3 with them rebuilt; at the contrast below it was
3 either way. On 2 to 8 ranks the strongest contrast took 5 to 6 iterations with
the transfers kept and 3 to 4 with them rebuilt. These are measurements and no
test holds them. A run that redistributes this hard pays one or two iterations a
solve for the kept transfers.

A move still drops the structural patch (#1017). The split-level fixture
carries no patch on one rank. The placed fixture does, and its solve goes from 3
iterations to 5 at the first move with the transfers kept. On 8 ranks the
split-level fixture carried a patch at rest and solved in 3 iterations; the first
move dropped it and the solve took 15, with the transfers unchanged, and the same
with them rebuilt. The patch comes back when the hierarchy is made again from the
moved fault: re-refine the base from it and register the new levels
(`set_custom_fmg`). Registering the old coarse levels again does not bring it
back. On a level with a `fac_zone` registered the rows of the zone stay in the
patch across the move and are cut into blocks again from the moved coordinates;
a split level then loses only the structural rows of its fault. The rows are
those of the cells the mask named at registration. Where a mover holds the zone
field in space the band moves off those cells, and registering again with the
zone recomputed puts the patch back on it (counts in the
[fault-multigrid guide](../guides/fault-multigrid.md)).

This is a mode for movers. A change of topology (an adapt, a remesh, a fault
placed or moved) makes a new mesh, whose as-built geometry is its own, and its
hierarchy is built from that mesh as before. Only a level's own moves are undone.
A fault cut into a mesh that had already moved lies on its trace in the moved
geometry, on the new mesh and on the split level under it alike, so the new mesh
is read where it is: 6 velocity iterations on the fixture of the test, whose host
moved by 0.002 before the cut, and 5 with the host at rest. The split level under
the new mesh is cut where the host's coarse level stands, which did not move. On
that fixture a host that moved by 0.005 or more (0.4 of its finest cell size)
leaves that split level with dependent coarse unknowns, which the repair of
#1101 merges (2 at 0.005, 6 at 0.01, 62 at 0.02), with the transfers kept or
built from the moved coordinates; the solve takes 6 or 7 iterations. The finest level of a refined base that moved before an adapt or a cut is
likewise carried into the child's hierarchy as a new mesh on the moved
coordinates, and the pair under it is rebuilt from them: with the top wall of a
box raised by 5 % three times before the adapt, the solve on the child takes 4
iterations against 2 with the box at rest, and 2 when the base itself is given as
that level (`set_custom_fmg`).

Whether a level has moved is read off its coordinates, which are compared with
the ones the mesh was made with. Coordinates that were written and put back
(`ephemeral_coords`, a snapshot restore, a deform back to rest) are not a move.
Nor is a `relax()` of a mesh that stands where it was made, whether
`adapt(relax=True)` calls it or the user does, once or several times. It repairs
the shapes the refinement left and is part of making the mesh: the transfers that
are built from coordinates are built on the relaxed ones (2 iterations on the P2
Poisson fixture of the test, 3 when read on the coordinates from before the
relax), and a later move is taken back to the relaxed coordinates. The parent of
a cell on the level below is still read where the mesh was refined. A `relax()`
of a mesh that has moved is one more move. So is the stamp of a reload: the
levels are rebuilt by refinement and the finest is then given the saved
coordinates, so a mesh saved after a move comes back as a moved level, and so
does a mesh whose refinement put new boundary nodes on a curved wall. Either is
read on its rebuilt refinement. A refined annulus takes 3 iterations as built and
2 reloaded.

A mover redistributes the nodes a mesh already has, which limits the refinement
it can reach; it does not replace refinement that adds nodes. A periodic mesh
keeps no as-built record (#1091), and its transfers are still built from the
moved coordinates.

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
`tests/test_1176_custom_mg_transfer_after_a_move.py` (the transfers after
`mesh.deform` and after a mover are those before it, on a placed level, a native
refine pair and a split level; the iterations of the faulted solve after a move,
and of a fault cut into a mesh that had moved).

## When a level takes the FAC patch

The FAC patch is a level's rows that its transfer cannot represent, plus a one-cell
halo. It is solved with LU blocks on top of whole-level SOR. The cover gate declines
it, and keeps whole-level smoothing, when the patch and halo exceed a fraction of the
level's rows (`_FAC_COVER_MAX` and `_FAC_COVER_MAX_SPLIT` in `custom_mg`). On a
uniform refinement, cut or not, the halo reaches every row.

- **A level without a split: 75 %.** On a wide adapt child the patch and halo reach
  95-98 % of the rows (a widened NVB metric), and taking them would mean one LU over
  nearly the whole level.
- **A 2-D level that carries a split fault: 98 %**, that is, it declines only when
  under 2 % of its rows lie outside the patch and halo. A 3-D split level keeps
  75 %: on test_1097's sheet (an adapt child of the 3-D base) its patch and halo are
  84 435 of 89 520 rows (94 %), and taking them was one 3-D LU over them, 463 s and
  6.3 GB serially against 35 s and 1.6 GB with whole-level SOR (`test_1097` asserts
  the decline). The 3-D threshold is a placeholder until the 3-D add-on is measured:
  that measurement decides whether 3-D needs a different threshold or a bounded
  block (#895). The cases below are 2-D.
  - *A placed fault level* keeps an untouched background there. On UWTN 2026-022's
    linear fixture at h_b/S = 24, 48 and 97, the patch and halo were 62 %, 76 % and
    86 % of the rows, with 38 %, 24 % and 14 % outside. The 75 % gate declined the
    last two. Whole-level SOR then converged the band, 24-97 times finer than the
    coarse level, slowly: 909 and 1487 velocity iterations summed over 17
    applications (about 53 and 87 each), against 66 (about 4) with the patch (#918).
  - *A fault cut into a refined base* (`add_fault` in 2-D, `split_fault` on
    test_1095's 3-D patch box) keeps the base's coarse levels, so its finest pair is
    a uniform refinement plus the cut, and its patch and halo are the whole level.
    That level declines, and whole-level SOR converges it: 8 velocity iterations on
    the 2-D `base_2d` of `tests/parallel/fault_fixtures.py`, 4 on the 3-D box.
  - *A welded add-on* (an adapt child, then `add_fault(split_levels=False)`) is
    split over unsplit coarse levels, and its patch and halo grow with the metric's
    width: 30 % at width 0.08, 82 % at 0.6, 96 % at 1.0. Up to 98 % it takes the
    patch. At 96 % that is one LU block over 96 % of the level, for 2 velocity
    iterations against 13 with whole-level SOR.

The background outside a placed band shrinks as the band refines, so a band fine
enough would decline again. By extrapolation that happens near h_b/S = 200 on a 1/16
base and 800 on a 1/32 base in 2-D; in 3-D the band's rows grow as (h_b/S)². Bounding
the block's size instead of declining it is #895.

The verdict is recorded per level on `hierarchy.fac_gate` (rows, patch and halo rows,
taken), from the global counts in parallel. It is the gate's verdict: install drops
a renumbered coarse level's patch, and the rotated path drops every coarse level's
patch, afterwards. `set_custom_fmg(..., verbose=True)` prints it on either path. The
mesh-owned and auto-injected hierarchies do not print, and their record is at
`solver._custom_mg["hierarchy"].fac_gate`. Serial builds gate every level the split
runs on, which leaves out pairs that touch a split coarse level. Parallel builds gate
only the finest. Tests: `test_1134` in `tests/` and `tests/parallel/`.

With `fac_zone` masks keyed, the masks' cells give the patch's blocks. On a finest
level that carries a split fault, the structural rows are added as one more block,
whatever the masks cover, under the split level's 98 % gate. On a zone-keyed level,
the gate's verdict in `fac_gate` is the structural block's alone: the zone's blocks are
installed either way. On a placed split fault the structural rows are the band, the
split duplicates and the fill around them, which the coarse level cannot represent
and whole-level SOR converges slowly. On the add-on matrix's row 5 (a TI band keyed
beside a split fault at h_b/S = 24), the velocity solve took 820 iterations serially, summed
over 17 applications, with the TI blocks alone and 94 with the structural block added
(#921). Keying the fault's band mask as well is not enough (344): the mask
holds the band cells but not the fill around them. A level without a split keys its
patch on the masks alone. That is #670's default, measured on a placed weak plane
when the patch replaced the level smoother (`test_1135`).

In parallel the finest level's blocks are built from every rank's rows. Each mask's
rows are gathered and segmented along strike as one band, with the two nodes of a
split pair kept in one piece, so the zone's pieces are the same at any rank count.
The structural rows outside every mask are one block, and with no zone keyed the
structural patch is one block; its halo can differ from the serial one by a few rows
where a partition boundary runs along the band (#896). Each block goes whole to the
rank that owns most of its rows, so the structural block is one LU on one rank, as in
serial. That rank carries the factorisation the others do not
(`patch_summary`'s `block_rows_per_rank`); balancing it is #895's question. Built
rank by rank, a rank's blocks held seam rows that its neighbour's blocks also held,
and rank boundaries cut the band and the structural block. On row 5 at n_rungs 80
the velocity solve then took 4 iterations serially, 118 at np 2 and more than 200 at
np 4 (#925). Gathered, it takes 4 at np 1, 2 and 4 (`test_1139`). The structural
patch alone, one block per rank, converged but took 3 iterations serially, 4 at np 2
and 6 at np 4; gathered it is one block at any rank count (`test_1139`, `test_1133`).
On row 5 with no zone at n_rungs 320 (30 048 rows in the block; kaiju, 2026-11), both
take 67 iterations serially, where the code is the same. Cut per rank, the velocity
solve took 128, 131, 316 and 445 iterations at np 2, 4, 8 and 16; as one block, 76 to
86. The owning rank's peak memory was 512-541 MB at np 2-16; the others'
fell from 379 MB at np 2 to 248-284 MB at np 16 (1.35 times at np 2, 1.6-2.2 times
at np 4-16).

## Not yet done

- A regression test for the coarse-level wall rotation itself (a rotated free-slip
  annulus with a placed band and the custom tail), beyond the campaign's annulus
  drivers.
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
- The one-rank build of `extrude_sheet` bounds the layer size by one rank's memory.
