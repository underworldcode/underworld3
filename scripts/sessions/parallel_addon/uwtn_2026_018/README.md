# Figures for UWTN 2026-018, "Faults in parallel: partitioning the add-on once"

Made on hyperion, 2026-10-25, branch `feature/coarse-solve-nullspace`, library
source `e135429f` (later commits touch `scripts/sessions/` only):
`feature/parallel-addon-distribution` 8fe9ad76 with the empty-rank fixes
(aed17e23, bfae3e65) and the coarse-solve prototype, inactive unless
`UW_ROTATED_COARSE` is set (it was not). `dev` environment, PETSc 3.25.3.

## parallel-pipeline.png (`fig_pipeline.py`)

The pipeline of `../../fault_mg/fig_schematics.py` redrawn for partition-once
(docs/developer/subsystems/fault-split-multigrid.md, "Parallel"): base
distributed once; add-on levels refined in place; the add-on partitioned once
(fault regions to one rank, balance by diffusion, one assignment on every
level); cut and split of the finest and every add-on level, rank-interior;
transfers found locally, the base-to-first-add-on pair by provenance; per-level
pair frames rank-local. No gathered stars, no misses asked of other ranks. The
extrusion box is unchanged: `extrude_sheet` still builds on one rank and keeps
the per-fault move.

## load-per-processor.png (`fig_load.py`)

Refined patch box, split (`BoxInternalPatch(refinement=2)` then `split_fault`,
19072 cells, P2 velocity). Finest P2 nodes per rank:

| np | before (f21f3d85, recorded) | after (re-measured 2026-10-25) |
|---|---|---|
| 2 | 17118 / 12501 (1.16) | 13683 / 16090 (1.08) |
| 4 | 9695 / 7789 / 7599 / 5529 (1.27) | 8170 / 8470 / 7766 / 7697 (1.06) |
| 8 | 7500 / 3555 / 4113 / 3461 / 3399 / 3837 / 3762 / 2147 (**1.89**) | 5899 / 4062 / 4007 / 4165 / 3772 / 3642 / 3949 / 3907 (1.41) |

The ratio is the largest rank's load over the mean. **Correction:** the design
document and the note's draft give 2.12 for np 8 before; from its own per-rank
numbers it is 7500 / 3972 = 1.89. The other five ratios agree with their
per-rank numbers. "Before" cannot be re-run on the current library; "after" was
re-run with `../probe_load_3d.py` and is identical rank for rank to the
record. The partition's own record (cells): np 2, 3 sweeps, 2451 units moved;
np 4, 4 sweeps, 3314 moved; np 8, 13 sweeps, 4515 moved; at np 8 the owner
holds exactly the fault region (3524 of 19072 cells), the indivisible unit.

## band-through-the-boundary.png (`fig_band_seam.py`)

**Stand-in fixture.** The note asks for the "S-fault rig"; its script is not on
this machine (only its measurements are cited in the library). The figure uses
an S-shaped trace `y = 0.5 + 0.04 sin(2 pi (x - 0.2) / 0.6)`, x in [0.2, 0.8],
network spacing h = 0.02, band width 0.04, base cell 0.08 with refinement 1,
`FaultNetwork.build(realisation="ti", seams="conform", max_levels=1)`, np = 2.
Cells per rank 1062 / 1106; band cells per rank 118 / 154; 4 band vertices
held by both ranks (the seam inside the band).

**A finding.** The same S builds serially at amplitude 0.15 (2188 cells, 272
band cells), but at np 2 with `seams="conform"` it fails: amplitude 0.15,
"a skin edge is not an edge of the sewn mesh; the gap was not sewn onto the
layer"; amplitude 0.08, "place_thin_volume failed on the surgery rank: the
fill ring crosses itself 1 time(s)". Only the gentle 0.04 builds. A curved band
through a partition seam is not yet robust in the conform build.

## Parallel split-level iterations after partition-once

The box in simple shear with the straight frictionless fault of
fault-split-multigrid.md (`../../fault_mg/library_c.py`), velocity iterations
per solve and peak slip:

| np | k | C split | A welded | peak slip C | peak slip A |
|---|---|---|---|---|---|
| 1 | 2 | 5 | 8 | 0.1595808977 | 0.1595810283 |
| 1 | 3 | 5 | 13 | 0.1589860839 | 0.1589859512 |
| 2 | 2 | 6 | 8 | 0.1595810551 | 0.1595810398 |
| 2 | 3 | 5 | 13 | 0.1589864514 | 0.1589860564 |
| 4 | 2 | 6 | 8 | 0.1595811119 | 0.1595810333 |
| 4 | 3 | 5 | 13 | 0.1589862215 | 0.1589860003 |

Before partition-once (13 October): C 6, 5 at np 2 and 7, 6 at np 4. Now np 4
matches np 2. The moving fault at k = 4, np 8 (UWTN 2026-022's figure 3
fixture): 7 velocity iterations at each of the three positions (serial 4, 4,
5); per step at np 8, refine 0.26-0.58 s, conform and split 0.75-0.89 s,
solve 4.5-4.8 s.

## NOTES

- 2026-10-25 (hyperion): three figures at library `e135429f`. The np 8 "before"
  ratio is 1.89, not 2.12. The band figure is a stand-in until the S-fault
  rig is supplied, and the conform build fails on stronger curvature at np 2.
