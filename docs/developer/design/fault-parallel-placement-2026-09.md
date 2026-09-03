# Fault networks in parallel: what moves, why, and the decision (September 2026)

This note records how the fault-network machinery behaves in parallel,
the measurements that established it, the design decision taken with
Louis on 2 September 2026, and the test that keeps the numbers honest.
It is the governing document for the parallel placement of faults; the
subsystem description is `../subsystems/conforming-surfaces-and-fault-zones.md`
and the multigrid side is `fault-patch-multigrid-2026-08.md`. Issues:
#670 (the gather), #671 (the tail), PRs #669 and #672.

## The frame: adapt-on-top

The base mesh is the persistent, distributed object. It holds the bulk
fields, it is refined in place by newest-vertex bisection co-partitioned
with itself, and its coarse levels form the multigrid tail. A fault
network is a *child* of that base: the band is placed into the base by
surgery, the faults are split on the placed mesh, and the child is
solved. The child is ephemeral. It is rebuilt when the base is adapted
or rebalanced, and it carries no state of its own: the fault's history
lives on the `FaultSurface` object, the bulk state on the base.

Two consequences follow. The child does not have to inherit the base's
partition, only the two transfers must be partition-agnostic: bulk
fields base to child and back, and the multigrid tail from the base's
coarse levels to the child's finest. And the accepted trade for a fault
is **extra load on the rank that owns it, in exchange for no
communication** while it is solved.

## Three parallel strategies coexist in the stack

1. **The edge cut** (`line_cut`, 2-D): fully distributed. Each rank
   splits its own edges where the surface crosses them; every crossing
   is a pure function of the coordinates and the surface, every
   tolerance is a global length, and nothing moves.
2. **Placement** (`place_sheet`, `place_thin_volume`, the 3-D band):
   *gather-first*. Placement deletes the base vertices in the way and
   asks gmsh to refill the cavity with the band's own mesh embedded in
   it. gmsh is serial and a cavity cannot be carved or filled across a
   partition seam, so the region around the band is moved onto one rank,
   the surgery runs there, and every rank rebuilds its chart
   collectively. The result is partition-independent by construction.
3. **The split** (contact): redistribute the fault's thin cell star onto
   one rank, then split with serial topology, so every cut pair is born
   on one rank.

What the gather moves is **base mesh, never the fault**: the band is
generated from a few patch coordinates on the surgery rank. The seam
rule that forces the move is that no point the surgery deletes or
creates may be shared, because the rebuild carries the old star forest
over by renumbering. That needs exactly three layers of base cells on
one rank: the cells the carve drops, their vertex star (the ring the
fill attaches to), and one more layer so the ring's own points are
unshared. A shell about three cells thick around the band. The third
layer was tested for necessity: with the gather stopped at the star, the
placer's own gate ("the gathered region touches a shared point; the
gather mask under-reached") fires at np=2, 3 and 4 on the thin-volume
suite, so the carve drops cells beyond the marked vertices' star and the
layer stays. Two shells therefore touch at about six base cells'
separation, and two zones closer than that are one region.

## What was wrong, measured

All on the crossing-patches fixture of `test_0863` (unit cube, far-field
cells of about 0.126 after one refinement, band cells 0.08, width 0.04)
unless stated; the wide box is the same patches in `[-0.5, 1.5]^3`.

**The fill is small; the gather was not.** The base has 5592 cells, the
placed mesh 8405, of which 1663 are band: the fill adds about 1150 cells,
all within a cell of the patches. But the gather moved 5087 of the 5592
base cells onto one rank before any surgery, and at np=4 the surgery
rank ended with 8012 of 8405 cells. The mark reached two cell widths
beyond the carve *and then* `_gather_region` grew the star and the layer
from it: the margin was paid twice.

| region on the base | cells | share |
|---|---|---|
| carve reach | 334 | 6% |
| reach + 1 cell (the dropped cells) | 1313 | 23% |
| reach + 2 cells (the old mark) | 3486 | 62% |
| old mark + star + layer | 5592 | 100% |

The unit cube is four far-field cells across, so three layers from any
band reach the walls: **that fixture measures correctness and can never
show balance**. On the wide box the same patches gather 9831 cells under
the old mark and 5046 under the new one (CROSS patches; 8961 and 4196 for
the P_A/P_B pair), identical at np=2 and np=4.

**One target for the whole network, and no return.** The placer took the
network as one assembly and gathered the union of every patch's region to
the single rank holding the most of it; two faults each interior to a
different rank were both moved to one of them. The moved cells stayed
there. The extra load therefore landed on the rank that was already
heaviest in that neighbourhood — anti-balanced — and was paid for with a
redistribution: neither half of the accepted trade.

## What was done (PR #672)

- **The mark covers only what the carve drops** (victims plus the crossed
  cells' vertices, one cell diameter out), at every placement path. The
  gather's own star-and-layer growth supplies the seam freedom.
- **Regions.** `_gather_regions` marks a chart of region ids, claims each
  region's star and layer, merges regions whose shells touch (a collective
  union-find), and sends each to the rank already holding most of it — or
  leaves it where it is when it is interior to one rank. One shell
  partition moves them all.
- **Per-region surgery.** One region per connected component of the
  assembly; the owning ranks carve and fill their own components
  concurrently, each on its compacted share of the assembly and skin, and
  the collective rebuild sews them at once. Outcrop and ladder paths keep
  one region (their bowl, cap and extrusion are single-rank).
- **The split follows the regions.** `split_faults(..., groups=...)`
  redistributes per group through the same gather; the network passes it
  the regions the placement reported (`info["embedded_regions"]`).
- **The tail on a distributed band** (#671). The rotated prolongation
  zeroed its constrained rows by *local* block index where PETSc's
  `zeroRows` takes global ones, so every rank but the first zeroed the
  wrong rows; that is the "new nonzero caused a malloc" in `PCSetUp_MG`
  (PETSc repairs a zero Galerkin row with an identity diagonal that was
  never allocated). And the `cross_partition="auto"` rule replaced a
  co-partitioned transfer with 3 orphan columns by a cross-partition one
  with 16,791 of 96,009, which the repair then filled with nonsense. The
  rows now carry the ownership offset, and the rebuild is kept only where
  it leaves fewer orphans.

Measured after: two zones a domain apart at np=2 move 304 cells to two
owners where 8451 moved to one; two faults in one network at np=2 are
both interior to their ranks, nothing moves, 14,339 / 14,332 cells; at
np=4 each region goes to a different rank. The contact solve on that
distributed band converges on the geometric tail in one Newton step at
np=2 and np=4, slips within 0.5% of serial (the gap fill's node count
varies by one or two with the partition, so the meshes are not
identical).

## The decision

Two designs were weighed:

1. **Cut the faults at the partition boundary and solve locally.** Best
   balance and no build communication, but every cost sits on the seam:
   a fill conforming to a polyhedral seam that the band skin crosses (two
   discrete surfaces to intersect consistently on two ranks), the split's
   pair nodes on shared points (the one seam problem the multigrid note
   lists as open), and a decomposition that changes with every rebalance.
2. **Faults as separate objects, distributed at will.** The surgery stays
   serial per object, the pairs are rank-local by construction, an object
   can be sent where the load is light, and the only communication is its
   shell, once. Its limit is granularity: an object is a connected fault,
   so one long fault is one rank's load, proportional to its area.

**Ruling: the second is the primitive** — it is what is built — and the
first's cut moves from the partition seam into the CAD when a long fault
needs it. Cut a long fault along strike at chosen planes before meshing:
OCC cuts the band by a plane exactly, so the two skins meet on a
triangulated plane generated once, broadcast, and embedded verbatim by
both owners (the fill already gates for verbatim constraints); the base
cells straddling the plane are victims on both sides, so each rank's
cavity ring closes onto the same plane triangulation and nothing
discrete is intersected with anything discrete. The pieces are then
ordinary objects. What still crosses ranks is the split: the fault trace
runs through the cut plane, so the pair nodes there sit on shared
points, and the strip along the plane — a few cells wide — is gathered
to one of the two owners before splitting. Communication is then
proportional to the cut planes, not the fault.

None of that is built, and it should not be until a model needs a fault
longer than one rank comfortably holds. The first thing to test when it
is: the split across one cut plane at np=3.

Two further rulings from the same discussion:

- **Do not rebalance the child after placement.** It fixes the solve
  imbalance by destroying co-partition everywhere, and every transfer
  then becomes a cross-partition search. Balance the base instead: refine
  it in the broad region where faults can exist (the lithosphere, say) in
  the *coarse* gmsh mesh, so the grading nests through the whole tail,
  and let the partitioner balance that.
- **A rebalance of the base means a rebuild of the child.** That is
  consistent with the faults being ephemeral; what must be verified is
  that the base's own state (mesh variables, swarms) migrates cleanly
  through a redistribution of a live mesh, which has not been exercised.

## Still open

- **np=3 answer on the crossing fixture** (#671): the smallest piece's
  slip is 10% low. It no longer crashes; it is a wrong number.
- **The cross-partition transfer builder** loses 17 to 45 percent of the
  coarse columns on a pair that is actually co-partitioned. It is only
  prevented from being chosen when it is worse.
- **Slicing a long fault** (above).
- **The seam-straddling gauge**: the weak-plane slip gauge omits a probe
  pair the rank does not own on both sides; unreachable while the
  gathered region holds the probes, and worth a collective count if a
  band is ever partitioned.

## The throughput test

The numbers above say what moves. The question that matters for a
time-stepping model is what the movement *costs* in the solve, and the
only fair measurement holds everything fixed except where the faults sit
relative to the seams. `fault_parallel_layouts.py` (beside this note)
does that on a 6 x 1 x 1 box whose np=3 partition is three slabs along x
with seams near x = 2 and 4, each slab about fifteen base cells long.
Four faults, 0.3 long and 0.2 tall, at least 1.5 apart so that no two
shells touch: one alone, a junction-connected pair as one cluster
(ligament 1.5; 1.0 degenerates the junction cut on this mesh), and one
more that the layout moves:

- **local**: one fault per slab, the pair in the middle slab — nothing
  should move;
- **straddle**: the fourth fault shifted onto the seam near x = 4 — it
  is gathered to the majority rank, that rank carries its shell, and the
  tail's transfers for the shell go cross-partition;
- **gathered**: every fault forced into one region — what the code did
  before #672;
- **serial** and **GAMG instead of the tail** on the same faults, as the
  baselines.

Two fixture lessons, both found the hard way: the 6:1 box meshes with
cells up to 0.4 across, and a fault 0.4 tall in a unit-high box leaves
0.28 of clearance, so the carve reached the floor and refused ("the
cavity reached the domain wall"); and faults about eight cells apart
merge into one region, since two three-cell shells touch at six.

The slips must agree across layouts to the fill's noise; the cost is in
the build time, the cells moved, the per-rank imbalance, the iteration
counts and the cold and warm solve times. "Warm" is a second full solve
from zero with the tail and the rotation reused — a repeat from the
converged solution takes no Newton step and measures nothing. Warm solve
time is what a time-stepping model pays per step.

### Results (2–3 September 2026, 16-core workstation, 20,778-cell base)

The junior of the crossing pair is consumed whole by the ligament cut at
1.5 on this mesh, so the fixture as run is three faults, one per slab,
and the "cluster" is a single fault. Warm is the second full solve from
zero; every solve took one Newton step; the tolerance is 1e-5.

**The first pass was uninterpretable, and why matters.** With the rotated
path's default pressure sub-solve tolerance (a tenth of the solver
tolerance) every configuration ran the pressure Schur solve to its
200-iteration cap. The monitor showed the mechanism: the pressure
residual falls by 5e-3 in twenty iterations and then creeps — 2.6e-8 at
20, 5.5e-9 at 50, 8e-10 at 199 — against a target of 1.2e-10. The floor
is set by the inexact velocity solve inside the Schur application (the
velocity sub-solve stops at 3.3e-7 of its own residual, and the Schur
complement amplifies that to about 7e-6 relative in the pressure
residual), so the margin rule "inner converges well below outer" is
inverted between the two inner solves: the pressure cannot converge
below the velocity's floor. Each wasted pressure iteration is a
velocity-preconditioner apply, which is why in that pass GAMG (26
velocity iterations, cheap applies) was 2.6 times faster in wall time
than the tail (4 iterations, dear applies). Recorded on #625. The
pressure tolerance is now an attribute knob (`solver._rotated_pres_rtol`,
a `TODO(MEASURE)`); the table of record uses 1e-4, ten times the solver
tolerance, at which the pressure solve converges in 26 iterations and
the outer FGMRES does the rest. The default is unchanged: what it should
be is a solver-configuration decision, not a placement one.

| layout | regions | cells moved | cells per rank (max/mean) | cold s | warm s | velocity its | pressure its | slips A / B / D |
|---|---|---|---|---|---|---|---|---|
| serial, tail | 3 | 0 | 20778 (1.00) | 31.4 | 14.0 | 3 | 26 | 0.11182 / 0.13117 / 0.11110 |
| np=3 local, tail | 3 | 122 | 6727 / 6733 / 7317 (1.06) | 13.5 | 6.8 | 4 | 26 | 0.11181 / 0.13176 / 0.11108 |
| np=3 straddle, tail | 3 | 834 | 5624 / 6733 / 8438 (1.22) | 16.6 | 8.7 | 4 | 26 | 0.11182 / 0.13115 / 0.10927 (D moved) |
| np=3 local, GAMG | 3 | 122 | 6727 / 6733 / 7317 (1.06) | 9.3 | 3.3 | 26 | 26 | 0.11182 / 0.13176 / 0.11108 |
| *capped pass, for the record:* | | | | | | | | |
| serial, tail, pressure at cap | 3 | 0 | 20778 (1.00) | 97.7 | 81.0 | 3 | 200 | same |
| np=3 local, tail, pressure at cap | 3 | 122 | (1.06) | 52.3 | 45.8 | 4 | 200 | same |
| np=3 straddle, tail, pressure at cap | 3 | 834 | (1.22) | 60.6 | 52.9 | 4 | 200 | same |
| np=3 gathered (pre-#672), pressure at cap | 1 | 5440 | 3514 / 13208 / 4054 (1.91) | 89.4 | 78.3 | 4 | 200 | 0.11182 / 0.13116 / 0.11111 |
| np=3 local, GAMG, pressure at cap | 3 | 122 | (1.06) | 24.9 | 17.6 | 26 | 200 | same |

What the table says:

- **The answer is layout-independent.** A and D agree to four digits
  across serial, local, straddle and gathered, and across both pressure
  tolerances; B to 0.5%, which is the gap fill's node-count noise
  between partitions. The straddle's D is a different fault position and
  legitimately a different number.
- **Per-region placement is what makes np=3 worth running.** The old
  single gather left one rank with 63% of the mesh and (in the capped
  pass) a warm solve of 78 s against 81 s serial: no parallel gain at
  all. Per region, the mesh is balanced to 6% and the warm solve is
  2.1 times serial (6.8 s against 14.0 s).
- **A straddling fault costs about 28%** (8.7 s against 6.8 s): 834
  cells moved, the owner at 1.22 of the mean, and that shell's transfers
  cross-partition. That is the price of one non-local fault on three
  ranks, and it is the number the design decision rests on. (In the
  capped pass the same difference read 15%, diluted by the wasted
  pressure iterations.)
- **GAMG is still twice as fast as the tail in wall time on this
  fixture** (3.3 s against 6.8 s warm) with the pressure solve fixed,
  at 26 velocity iterations against 4. The tail's application is dearer
  by more than the iteration ratio buys back here. This fixture is
  linear viscous, uniform viscosity, one Newton step — the regime GAMG
  is built for; the tail's measured advantage (#579, #576) is on banded
  contrast and nonlinear solves, which this test does not exercise. It
  is a red flag worth its own measurement on a contrast fixture, and it
  is not a placement matter.

The script beside this note (`fault_parallel_layouts.py`) regenerates
the table:

    mpirun -np 3 python -u fault_parallel_layouts.py -uw_layout local|straddle|gathered [-uw_tail 0]
