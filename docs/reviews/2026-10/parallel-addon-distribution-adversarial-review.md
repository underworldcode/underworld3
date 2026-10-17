# The add-on partitioned once: adversarial review record

Branch `feature/parallel-addon-distribution`, from `feature/tidal-shell` at `f21f3d85`.
One round of three independent reviewers (parallel semantics and numerics; the wiring
into the cut, placement and split pipelines; tests, style, documentation and
contracts), read-only against the branch diff, 2026-10-17. Design note:
[parallel-addon-distribution-2026-10](../../developer/design/parallel-addon-distribution-2026-10.md).

## What the branch does

| Change | Why |
|---|---|
| `utilities/addon_partition.partition_addon`: one assignment for the add-on hierarchy, computed before any surgery and applied to every add-on level through the parent-cell relation; fault regions glued and sent to their majority rank, the load balanced by diffusion over the seams; provenance riding the move | every surgery moved cells on its own terms, so no two levels shared a partition and every consumer carried its own cross-partition handling |
| `add_fault`, `add_conforming_sheet`, `split_fault`, `split_faults` partition first; the level replay arrives partitioned | the split moved each level's star to its own majority rank; the 3-D placement refused on a partitioned adapt child ("the gather mask under-reached") |
| `fault_mg.provenance_transfer_parallel`: the pair the partition moved apart is built from the recorded parent, fetched through one star forest | the cell-located transfer all-gathered the misses as queries |
| the 2-D `add_fault` cut child takes the base's coarse levels | every solver on it fell back to GAMG |
| `colocate_split_level`, `fault_star_owner`, `_fault_star_cells` removed; `_redistribute_fault_interior` kept for `extrude_sheet` only | superseded |

## Findings and their state

Blocking:

- **A second partition on a partitioned lineage was refused** (wiring review). The
  "partition once" guard read only its argument; a placed or cut child of a
  partitioned host fell through to a co-partition check that fails by design, so the
  fault-network route (`add_conforming_sheet` per sheet), sheet-then-fault and two
  `split_fault` calls all broke in parallel. Fixed: the first add-on level's parent
  on the base is read from the recorded provenance when the pair is no longer
  co-partitioned (`_base_parent_addresses`); the guard is gone. Tests:
  `test_1096::test_a_second_partition_keeps_the_first_fault_whole`,
  `test_1097::test_two_sheets_placed_in_turn_then_split_together`.
- **A split fault was not glued by a re-partition** (semantics review). Its facets
  are single-sided after the split, so a support-based interior test marked nothing,
  and the balance could peel the first fault's star across a seam. Fixed: glued
  surfaces are named (`embedded_surface_names`: every pairing's label and sides, every
  placed sheet), not detected; this also stops a domain-spanning internal boundary
  being glued (should-fix in the same review).

Should-fix, fixed:

- Empty region ids gave a NaN fault centre and disordered the balance; regions with
  no cell anywhere are dropped.
- The provenance branch skipped the finest level's FAC patch; it now sits inside the
  builder branches.
- The provenance lookup located fine centroids in an ancestor's possibly stale
  geometry; the relation is recorded when the surgery makes the mesh
  (`inherit_provenance` in the cut, placement and split), and a mesh carrying its own
  record needs no location at transfer time.
- `locate_in_cells_exhaustive` scanned every cell per miss; it widens to the nearest
  128 and 1024 cells first and scans only what still misses, and with the record
  taken at surgery time it runs once per surgery, not per transfer. The two
  inside-tolerances now agree.
- `mark_polylines` read a rank-local cell size at seam vertices; reconciled over the
  star forest.
- The balance could overshoot by a whole unit and trade it back; a unit heavier than
  twice the remaining gap is skipped.
- A level assigned wholly to one rank would be repartitioned by the Mesh constructor
  under the provenance; refused collectively.
- Two non-collective raises inside collective sequences made collective.
- Serial and parallel tails differed for a sheet fault on a refined base with no
  registered tail; the 3-D path takes the native coarse levels as the 2-D one does.
- Stale docstrings (`split_faults`, `split_fault`, `stamp_point_pairs`,
  `_fault_labels_touch_seam`, `Mesh.add_fault`), the dead `groups` / `pairs`
  parameters of `_redistribute_fault_interior`, the uncommented swallow in
  `_face_other_rank`, the clearance default duplicated from `place_sheet`.
- Tests: the provenance pair's record (`fetched_remote`) and the per-level lineage
  asserted; `test_1097` compares facets and pairs with its own serial run and the
  tail by identity; the serial tail test by object identity; the fixtures shared in
  `tests/parallel/fault_fixtures.py` (and `test_1089` moved onto them).
- Documentation: the design note's unit definition, its retirement list (honest about
  `_colocate_level` in `adopt_hierarchy` and the extrude move), the query exchange
  that still runs empty; supersession notes in the September placement note,
  `FAULT_MULTIGRID_HIERARCHIES.md` and the conforming-surfaces subsystem page.
- Probes: the broken and unrelated ones deleted, a README dating the rest.

Left open, recorded:

- `place_thin_volume` (the 3-D band of `FaultNetwork.build`) still gathers its own
  regions, and `adopt_hierarchy` still co-locates the tail behind it
  (`_colocate_level`). The placed-band route is the next one to put on the partition.
- A direct `place_sheet` call keeps the under-reaching mask (`TODO(BUG)` at the mask).
- Python loops per cell, face and vertex in the partition and the section fill:
  seconds per level at 1e5 cells per rank, tens at 1e6. Not measured at scale.
- The balance's seam adjacency is the original partition's; received units are not
  passed on. It converges on the fixtures; not a bound.
- `_face_other_rank` and `place_surface._propagate_vertex` return early on a
  `getGraph` exception before a star-forest collective (pre-existing pattern).
- Only np = 2 runs in CI; the 4- and 8-rank runs here are by hand.

## Attacks that found nothing

- Collective ordering in `partition_addon`, the balance sweeps, `_region_cells`,
  `_merge_regions`, the provenance transfer (local work only inside the guards).
- `_resolve_parents`: the root count is each rank's own pre-move cell count; one leaf
  per root, so REPLACE is the reduction.
- The base is never redistributed in place; the cached wraps keep their identity, so
  `provenance_source` matches by identity, newest record first.
- The l2g packing through doubles is exact; constrained columns survive and drop.
- Serial behaviour of every path is unchanged except the two tail fixes.
- CI reach: every new test file is in a `scripts/test.sh` glob; markers complete.
