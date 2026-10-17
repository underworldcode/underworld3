# Distributing the add-on once (October 2026)

Design note, 2026-10-17. The decision on how the add-on — the finest mesh built on a
static base hierarchy by `adapt`, a cut, a placement or a split, together with its
add-on levels — is partitioned in parallel, and the measurements behind it. It
follows the September ruling in
[fault-parallel-placement-2026-09](fault-parallel-placement-2026-09.md) (the base is
never rebalanced; load fixes go on the child; every transfer to the child must be
partition-agnostic) and the hierarchy of
[FAULT_MULTIGRID_HIERARCHIES](FAULT_MULTIGRID_HIERARCHIES.md). Implementation:
`utilities/addon_partition.py`; tests `tests/parallel/test_1096_*`.

## The problem

The base hierarchy is partitioned once and refined in place, so its levels are
co-partitioned, and `adapt` (the in-place `uwnvb` transform) keeps the add-on levels
co-partitioned with it. Every divergence comes from fault surgery, and each surgery
moves cells on its own terms:

- `split_fault` / `add_fault` move each fault's cell star to the rank already holding
  most of it (`_redistribute_fault_interior`), on the finest and again on each
  replayed add-on level, each level to *its own* majority rank; a second pass
  (`colocate_split_level`) then moves every level's star to the finest's owner.
- `place_sheet` / `place_thin_volume` gather the surgery region to one rank
  (`_gather_region(s)`); the 3-D sheet's mask does not cover the cavity's growth
  rounds, so on a partitioned adapt child the carve touched a shared point and the
  placement refused where the same placement passes serially (`TODO(BUG)` at the mask).
- `adopt_hierarchy` moves *copies of the base's static levels* to follow a placed
  mesh (`_colocate_level`).
- The 2-D `add_fault` cut constructs its child without the parent's tail, so solvers
  on it fell to GAMG silently (`TODO(BUG)` at the constructor).

The transfers then pair levels that are not co-partitioned by point location: the
cell-located builder all-gathers the fine nodes it cannot place locally as queries,
and the node-cloud builders needed two fixes on 2026-10-17 (f21f3d85) to stop
producing wrong rows on such a pair. Pair diagnostics, checkpoints and the placement
each carry their own partition handling.

## Measured (hyperion, f21f3d85, under a 16-rank study)

Refined patch box, split (`BoxInternalPatch(refinement=2)` then `split_fault`,
19072 cells, P2 velocity, tolerance 1e-4; `scripts/sessions/parallel_addon/probe_scaling.py`):

| np | velocity iterations | finest P2 nodes per rank (max / mean) |
|---|---|---|
| 1 | 4 | 28274 |
| 2 | 5 | 17118 / 12501 (1.16) |
| 4 | 5 | 9695 / 7789 / 7599 / 5529 (1.27) |
| 8 | 6 | 7500 / 3555 / 4113 / 3461 / 3399 / 3837 / 3762 / 2147 (2.12) |

The iteration count is now near serial; the owner rank carries twice the mean at
eight ranks, and 3026 of its 7500 nodes lie outside its local coarse hull, so the
finest transfer on that rank is built from remote answers.

Adapt-on-top fault, 2-D, two add-on levels, np = 4
(`scripts/sessions/fault_mg/parallel_load.py 2`): finest cells per rank
240 / 292 / 243 / 695 (max / mean 1.89); the base-to-first-add-on transfer misses
265 of 2528 owned nodes (10 %) and asks the other ranks for them; the add-on pair
misses 14 of 2992 after the levels are co-located. Six velocity iterations.

A per-cell field rides a shell redistribution exactly through
`DMPlex.distributeField` on the migration star forest `distribute()` returns
(`probe_migration_sf.py`, np = 2: zero error). The earlier reading that the migration
forest "maps points to the wrong depth" came from reading remote point numbers
against the local chart; the forest is a point map, and `distributeField` is the API
for riding it.

## Measured after (same fixtures, the partition in place)

Refined patch box, split: velocity iterations 5 / 6 / 6 at np 2 / 4 / 8 (4 in
serial), wall 65 / 34 / 25 s against 72 / 38 / 30 s before. Adapt-on-top fault, 2-D,
np 2: finest cells per rank 742 / 618 (max / mean 1.09, from 1.38); every add-on pair
co-partitioned, every recorded parent contains its cell; the base-to-add-on pair
fetched, no query. At np 4 and 8 on that 1360-cell fixture the fault region itself
(544 cells with its growth layers, lifted to units) exceeds the mean load, so the
owner keeps it whole and the balance stops there: a region is the indivisible unit,
as the September note's granularity remark says, and the test bounds the load by
the region's weight. The 3-D sheet placement on a partitioned adapt child, refused
before ("the gather mask under-reached" at np 2 on the disc fixture of test_0867),
places with nothing gathered and splits with nothing moved (test_1097).

## The decision

**One assignment for the whole add-on, computed on the finest level before any
surgery, applied to every add-on level through the parent-cell relation, once.** The
base levels never move. Concretely:

1. **Units.** The unit of assignment is a cell of the base's finest level together
   with all its descendants on the add-on levels (nested refinement: every add-on
   cell has exactly one parent on the level below, found by centroid containment
   while the levels are still co-partitioned, which is exact for a bisection
   hierarchy). Every cell of a unit goes to the same rank, so every add-on pair stays
   co-partitioned exactly, and the base-to-add-on pair is co-partitioned except
   where a unit moved.
2. **Fault regions.** The cells the surgery will touch — the star of the vertices
   within reach of the fault geometry, plus two growth layers so every point the
   carve or the split touches is unshared — are lifted to units and glued: a region
   is one indivisible block, sent to the rank already holding most of it, or left
   where it is when it is already interior. Regions whose shells touch merge (the
   existing `_gather_regions` rule). The split's pairs and the fault cell fans are
   therefore rank-local by construction, and never because a later pass moved them.
3. **Balance.** The owner of a region is overloaded by the region's weight (its
   finest cells). Units that belong to no region are peeled from an overloaded rank
   to an underloaded neighbour across their common seam, one layer per sweep, until
   every rank is within a tolerance of the mean or no seam can give. The load fix is
   on the child; the base's partition is read, never written.
4. **Provenance rides.** Each level carries, per cell, its parent cell on the level
   below and its own pre-move identity, as cell-wise fields moved by
   `distributeField`. After the move every fine cell knows where its parent now
   lives, and the transfer for every pair — the add-on pairs and the one
   base-to-add-on pair — is the coarse cell's Lagrange basis evaluated at the fine
   node (the existing dual-basis evaluation of `fault_mg`), with the parent's node
   columns and vertex coordinates fetched through one star forest. Co-partitioned
   pairs fetch locally; the base boundary fetches the moved regions' parents from
   their ranks. No point location across partitions, no all-gathered query cloud.
5. **Surgery afterwards.** Cut, placement and split run on the partitioned levels
   and move nothing; their seam tests become assertions of the precondition. A level
   created by surgery inherits provenance from the cells it re-represents (the split
   is one-to-one; a cut cell or a fill cell takes the pre-surgery cell holding its
   centroid, a local test since the region is rank-interior).

What this retires: `colocate_split_level`, `_colocate_level` and the per-level
`_redistribute_fault_interior` calls; the cross-partition query path of the transfer
builder becomes unreachable for an add-on built this way and is kept only for a
hierarchy registered by hand (`set_custom_fmg` with independently built meshes).

## What was weighed and rejected

- **Partition the base around the faults at construction.** Perfect co-partition and
  balance, nothing ever moves. Rejected: it is a fix on the base, a moving fault
  invalidates it every step, and the base is built before the model knows its faults.
- **Keep the base's partition and split through the seams** (the 2-D `conform` mode
  one dimension up). Nothing moves and balance is the partitioner's. Rejected for
  the split: the rank-interior fan is what the contact solve's pair block and the
  fault-normal accumulation are built on, the 3-D cut through a seam needs the global
  patch and a geometric side rule at every shared rim vertex, and the along-strike
  seam — the generic case on a base graded toward the fault — is refused today.
  The `conform` mode stays as it is for the 2-D band.
- **Leave the gathers and keep repairing the transfers.** The iteration counts are
  already near serial after f21f3d85, but every consumer still handles the
  partition separately, the owner rank carries twice the mean, and the next
  surgery (the 3-D placement on an adapt child) refused for a reason that the one
  partition step settles.

## Stages and tests

Every stage ships its regression test first, shown failing (Charter §8), under
`tests/parallel/` at 2, 4 and 8 ranks, and the serial reduction.

1. The 2-D `add_fault` cut child carries its parent's tail (the `TODO(BUG)`): a
   refined box, `add_fault`, the velocity preconditioner is the custom FMG and not
   GAMG, serial and np = 2.
2. `addon_partition`: after `add_fault` / `split_fault` on an adapt child and on the
   refined patch box, every add-on pair reports zero misses, every fault star is
   rank-interior with no pair dropped, the finest cells per rank are within the
   balance tolerance of the mean, and the velocity iteration count is within a few
   of serial. The finest transfer reproduces the coordinate field (test_1095 stands).
3. The provenance transfer replaces location for every pair of such a hierarchy:
   the base boundary's moved units are fetched, the transfer stats show no query.
4. The 3-D sheet placement on a partitioned adapt child (the refused fixture) places
   and splits at np = 2, 4, 8 with the same cell and pair counts as serial up to the
   fill's noise, and the gather reports nothing moved.

Out of scope here, recorded for later: `extrude_sheet` still builds on one rank;
checkpoint reload of a partitioned add-on; the 3-D split through a seam.
