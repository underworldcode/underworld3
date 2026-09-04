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

**Revised on 3 September (Louis), after the hierarchy was seen and the
two realisations' requirements were separated.** The embedding is the
same for both realisations; only the split needs its pairs on one rank.
The weak and TI band need nothing beyond mesh continuity across a seam,
so with a seam-conforming fill a band can be cut anywhere, including on
a process boundary. The design line is therefore: embed the fault-zone
mesh with its mid-surface as reserved conforming faces and its cells
labelled, immediately usable by the TI formulation; then make the split
local by construction — cut only the mid-surface facets whose star is
interior to the rank and leave a ligament, a physical width the user
sets, at each seam crossing, which the band's weak rheology bridges.
That is the hybrid recipe applied at seams, and the stepover and
junction studies already measured it: a TI band bridging a cut gap
reproduces the continuous fault to 0.2%, and cuts plus a weak band cost
what the split costs and give its answer. What it gives up is
bit-identical partition independence, since the ligaments move with the
seams; that is a bound to measure at np=3, including a seam near a tip
or a junction. Build order: the seam-conforming fill in 2-D first, then
the bridging split mode (skip and report seam-touching facets rather
than refuse or redistribute), then the 2-D study of one-object cuts
against per-rank cuts with ligaments in both realisations. This
supersedes the CAD cut for the split; the per-region gather stays as
the mechanism for whole objects meanwhile.

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
- **The penalty.** The runs here use `penalty = 0`, the solver default,
  which the solver holds at zero because a grad-div penalty of 10
  corrupted the vertex-sampled dynamic topography recovered from the
  free-slip reaction (test_1018). That is a recovery defect, not a
  reason to forgo the penalty: the traction is the multiplier plus the
  augmentation, r(u·n − ũ_n), and at a viscosity contrast of 1e6 the
  multiplier alone carries almost none of it (Louis's note,
  https://www.underworldcode.org/boundary-conditions-on-non-planar-boundaries).
  The Coulomb fault law takes its normal stress from the same reaction,
  so the correction belongs there too before a penalty is used with
  frictional faults. Measured on this fixture: penalty 1 at tolerance
  1e-5 does not lift the pressure cap and shifts the slips by 0.6 to
  0.9% at this resolution.
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
the outer FGMRES does the rest. The default is unchanged, and the
better reading is Louis's: the fixture's tolerance of 1e-5 is one or two
orders stricter than a mesh of 0.08 to 0.2 cells deserves, and at 1e-3
the default margins converge the pressure solve in the same 26
iterations with no knob at all (the third block of the table). The
slips move by 0.2 to 0.7% between the two tolerances, which is the
loose tolerance's own noise, still below the fill's partition noise.

| layout | regions | cells moved | cells per rank (max/mean) | cold s | warm s | velocity its | pressure its | slips A / B / D |
|---|---|---|---|---|---|---|---|---|
| serial, tail | 3 | 0 | 20778 (1.00) | 31.4 | 14.0 | 3 | 26 | 0.11182 / 0.13117 / 0.11110 |
| np=3 local, tail | 3 | 122 | 6727 / 6733 / 7317 (1.06) | 13.5 | 6.8 | 4 | 26 | 0.11181 / 0.13176 / 0.11108 |
| np=3 straddle, tail | 3 | 834 | 5624 / 6733 / 8438 (1.22) | 16.6 | 8.7 | 4 | 26 | 0.11182 / 0.13115 / 0.10927 (D moved) |
| np=3 local, GAMG | 3 | 122 | 6727 / 6733 / 7317 (1.06) | 9.3 | 3.3 | 26 | 26 | 0.11182 / 0.13176 / 0.11108 |
| *tolerance 1e-3, default margins (the tolerance this resolution deserves):* | | | | | | | | |
| serial, tail | 3 | 0 | 20778 (1.00) | 26.1 | 10.1 | 2 | 26 | 0.11187 / 0.13104 / 0.11116 |
| np=3 local, tail | 3 | 122 | (1.06) | 15.1 | 5.4 | 2 | 26 | 0.11187 / 0.13152 / 0.11115 |
| np=3 straddle, tail | 3 | 834 | (1.22) | 12.8 | 5.3 | 2 | 26 | 0.11197 / 0.13069 / 0.10942 (D moved) |
| np=3 local, GAMG | 3 | 122 | (1.06) | 8.3 | 2.2 | 16 | 26 | 0.11164 / 0.13162 / 0.11104 |
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
- **A straddling fault costs between nothing and 28%.** At tolerance
  1e-5 it read 8.7 s against 6.8 s (28%); at 1e-3, 5.3 s against 5.4 s
  (none); in the capped pass, 15%. These are single runs on a shared
  workstation, so the honest statement is that one non-local fault on
  three ranks costs at most a quarter of the solve and may cost nothing
  measurable: 834 cells moved, the owner at 1.22 of the mean, and that
  shell's transfers cross-partition. The design decision rests on that
  bound, not on a point value.
- **GAMG is still twice as fast as the tail in wall time on this
  fixture** (3.3 s against 6.8 s warm at 1e-5; 2.2 s against 5.4 s at
  1e-3) with the pressure solve converging, at 16 to 26 velocity
  iterations against 2 to 4. The tail's application is dearer
  by more than the iteration ratio buys back here. This fixture is
  linear viscous, uniform viscosity, one Newton step — the regime GAMG
  is built for; the tail's measured advantage (#579, #576) is on banded
  contrast and nonlinear solves, which this test does not exercise. It
  is not a placement matter. Seen rather than inferred (the renders in
  `~/+Simulations/fault_network_3d_parallel/figures/`, 3 September): the
  tail here is ONE genuine coarsening — the gmsh box at 0.24 (2433 cells,
  the redundant LU coarse solve at about 13,000 velocity DOFs), its
  single bisection (19,464), and the placed mesh (20,778) which differs
  from the level below in 1,314 cells in three pockets, so the finest
  transfer is the identity on 94% of its rows. The band is two cells
  across its width inside fill cells four to eight times larger with no
  grading between, because the band builder hard-codes one refinement of
  the far-field box. A two-grid method with custom transfers and a large
  coarse LU has nothing to gain over GAMG on an isoviscous problem, and
  that is what the row shows. **Ruling (Louis, 3 September): the
  preconditioner question is deferred to 2-D**, where a real hierarchy
  (two or three doublings to the same finest cell) and a viscosity
  contrast in the band are cheap to build and to see; this 3-D fixture
  measures placement and balance, not multigrid.

The script beside this note (`fault_parallel_layouts.py`) regenerates
the table:

    mpirun -np 3 python -u fault_parallel_layouts.py -uw_layout local|straddle|gathered [-uw_tail 0]

## Interface sketch: seam-conforming placement and the bridging split

Written 3 September 2026 for implementation in a fresh session; debug in
2-D serial and parallel, validate in 3-D serial and parallel. Nothing
below is built.

### What changes and what does not

The carve, the gmsh fill, the collective rebuild and every gate stay.
What changes is the contract on *where* a cavity may lie: today a
cavity must be interior to one rank (the gather makes it so); with seam
conformance a cavity may straddle a seam, and the two ranks fill their
own sides against an interface they both hold exactly.

### The mechanism, in four steps

1. **Interface planes.** For each seam a band's cavity would cross, a
   plane (a line in 2-D) is chosen deterministically: through the mean
   position of the seam faces inside the cavity zone, normal to the
   band's strike there. Collective: every rank computes the same list.
   A cavity zone touching no seam gets no plane and proceeds as now.
2. **The strip alignment.** Cells adjacent to the cavity zone are
   reassigned by the side of the plane their centroid lies on, so that
   within the zone the partition boundary *is* the plane, up to the
   cells that straddle it, which are dropped. One shell partition, a
   few cells wide along each plane. This is the only redistribution,
   and it moves a strip, not a region.
3. **The interface surface.** On the lower rank of each pair, the
   plane region bounded by the ring's outline (the ring faces meeting
   the plane form a closed polygon) is meshed once in 2-D by gmsh with
   the band's cross-section outline embedded, and broadcast to the
   pair. In 2-D it is a chain of points along the line: the ring ends
   and the band's two skin crossings. Both ranks embed it verbatim;
   the fill's existing gating (zero moved constrained nodes, every
   constraint facet present) covers it.
4. **Two fills and one rebuild.** Each rank carves its own side (the
   band cells assigned by centroid side, the assembly cut by the plane
   in CAD so the two halves' skins meet on the interface exactly) and
   fills between its ring, its half of the skin and the interface. The
   rebuild is the existing collective one with one addition: the
   interface's new vertices are created on both ranks and must be the
   *same* points, so `_attach_uninterp_vertex_sf` gains entries for
   them — owner the lower rank, leaves on the higher, matched by the
   interface's own point numbering, which both ranks hold. That is the
   one deep change: today's rebuild asserts that no placed point is
   ever shared.

### Signatures

```
place_thin_volume(dm, patches, width, ..., seams="gather")
    seams : {"gather", "conform"}
        "gather"  — the per-region gather (today).
        "conform" — cavities may straddle seams; the four steps above.
    info["seam_crossings"] : list of dicts, one per interface —
        {"ranks": (r, s), "point": ..., "normal": ..., "n_strip": int}

split_faults(mesh, names, groups=None, at_seams="gather", ligament=None)
    at_seams : {"gather", "bridge"}
        "gather" — redistribute per group, cut everything (today).
        "bridge" — cut only facets whose closure is unshared AND farther
                   than ligament/2 from every interface plane; leave the
                   rest as ligaments for the band's rheology.
    ligament : physical width (same units as the band width); required
               with "bridge".
    info / the child's record: "ligaments" — one entry per seam crossing
        with the uncut facet count and the ligament's extent.

FaultNetwork.build(..., seams="gather", seam_ligament=None)
    passes both through; net.info gains "seam_crossings" and
    "ligaments"; the band's rheology (apply / ti_fields) must cover the
    ligaments, which it does already since the band is weak everywhere
    it is not cut.
```

### Collective discipline

Every decision above is taken from gathered data before any branch:
the plane list, the strip assignment, the interface point set, the
ligament list. A rank with no crossing still participates in each
collective. The refusals stay collective. Arm `UW_HANG_WATCHDOG` on
every run.

### Tests, in order

- **2-D serial**: `seams="conform"` on a single-rank mesh must reduce
  to the existing path bit-for-bit (no seams, no planes).
- **2-D np=2, one band across the seam**: `conform` against `gather`
  — same zone and skin counts, same volume and Euler gates, the TI
  solve identical to the fill's noise; then `at_seams="bridge"` — one
  ligament reported, its width the one asked for, slip along the fault
  within the stepover bound (0.2% of the continuous control away from
  the ligament).
- **2-D np=3**, a seam near a fault tip and one at a junction: the
  bound on the ligament's effect where it is not a small perturbation.
- **3-D serial** (reduction to the existing path), then **3-D np=2 and
  np=3** on the crossing fixture and the layout fixture: the same
  assertions, plus the interface surface's own gates (the plane
  triangulation present verbatim in both fills).
- The layout throughput test gains a `conform` column: cells moved
  should be the strips only.

### Traps known in advance

- The rebuild currently *asserts* no placed point is shared; the
  interface vertices violate that by design, and the SF extension must
  come before the interpolate (the #520 ordering).
- gmsh's fill is not bit-reproducible across inputs that differ only
  in ordering (the 611/612 placed-node effect); the interface surface
  must be generated once and broadcast, never regenerated per rank.
- A plane chosen from the seam's mean position can cut a band near
  its tip; the ligament rule must handle a crossing at a tip (the tip
  itself is never split, so a seam through the tip region simply
  enlarges the uncut margin).
- The 2-D thin volume has the one-region gather only; the per-region
  gather (`_gather_regions`) is wired into the 3-D thin volume alone.
  Wire it into 2-D first so `gather` and `conform` are compared like
  for like.

## What was built: the seam ligament (3 September 2026)

Louis's ruling on the morning of 3 September simplified the sketch above:
partition-crossing structures stay transversely isotropic. A band that
crosses a seam needs only a conforming mesh on each side; the splitting
machinery never runs across the decomposition. Split nodes live in the
interior of a rank, and across the seam the band's weak plane is the glue.
Splitting through the seam is a later experiment, not a requirement.

We built the cheapest version of that and measured it. Nothing is
gathered and no interface is meshed. Each rank carves its own cavities,
and a cavity stops one cell short of the seam: a base cell with a shared
vertex is never dropped and its vertices are never deleted. The band
assembly is clipped to what each cavity holds, less a margin of 0.4 band
cells from the seam-side ring so the fill has room, and the clipped subset
is made manifold (a bow-tie vertex or a triangle hanging by one vertex is
removed) so its skin is a set of closed loops. The base cells that the
clipped-away band cells covered are the LIGAMENT. They keep the base's
vertices, so the rebuild's star forest carries over unchanged, and they
are labelled both as zone and as `<label>_ligament`. `place_thin_volume`
exposes this as `seams="ligament"` (2-D), `place_fault_ribbon_2d` and
`FaultNetwork.build` pass it through, and in serial it reduces to the
gather path exactly: same cells, same coordinates, same pairings.

The split then never sees a seam. The cut is replaced by a label-only pass
(`add_fault(cut=False)`) over the mesh edges already lying on the trace,
excluding edges in ligament cells, and a fault that crosses a rank in and
out is split as several sub-chains, one collective pass per piece
(`split_fault` loops until no rank has a piece left). A fault lying wholly
in a ligament stays uncut with empty side labels. For the split
realisation `FaultNetwork.apply` paints the weak plane on the ligament
cells, so it now takes `eta_1` in both realisations.

### Measured, and what it means

One vertical fault of 35 cells (h = 0.02, band 2h) crossed once by the
np=2 seam and twice at np=3; the network of ptest_0859 turned vertical;
`ptest_0864` pins both. The gathered answer is the serial-topology one.

| realisation | np | cells per rank | ligament cells | peak slip | vs gathered |
|---|---|---|---|---|---|
| TI, gather | 2 | 1629 / 635 | 0 | 0.5688 | — |
| TI, ligament | 2 | 1216 / 1050 | 33 | 0.5605 | −1.5% |
| TI, ligament | 3 | 781 / 897 / 574 | 45 | 0.5484 | −3.6% |
| split, gather | 2 | 1629 / 635 | 0 (69 pairs) | 0.5094 | — |
| split, ligament | 2 | 1216 / 1050 | 33 (62 pairs) | 0.4421 | −13% |
| split, ligament | 3 | 785 / 897 / 574 | 51 (52 pairs) | 0.3869 | −24% |

The weak plane crosses the seam at the base's resolution and loses a few
per cent, with the profile ratio to the gathered answer between 0.96 and
1.00 along the whole fault, while the cells stay balanced instead of
sitting 72% on one rank. That is the design line, confirmed: for the TI
realisation the ligament is a resolution effect, and pre-refining the base
in the fault region shrinks it.

The split is different, and the reason is structural. Each rank-local
sub-chain ends in a tip, and a tip is a single vertex shared by both
sides, so slip is zero there by construction. The weak plane painted on
the ligament cells is correctly applied (33 cells across the band width,
director along the normal, TI engaged), and changing `eta_1` from 0.01 to
0.001 changes the peak by 0.5%: the loss is the pinned tips, not the
bridge. Near the ligament the slip drops to about 45% of the gathered
profile, and at the far ends of the fault it recovers to 92–94%. Each
crossing is a weld of about 13% of peak slip on this fixture.

### The along-strike seam

On the unit box at np=2 the seam runs along y = 0.5, and the horizontal
network of ptest_0859 lies on it. Its whole band is then ligament, and the
Splay, which leaves the seam, is the only piece embedded (the figure
below; the pieces' trace edges are drawn on the embedded band). This is
not the fixture being unlucky: a balanced partition of a base refined
along a fault puts its cut in the densest region, which is the band, so a
seam that follows a fault is the generic case. For the TI realisation
that stretch is a painted base-cell band. For the split it is uncut.

```{figure} figures/seam-ligament/along_strike_seam.png
:alt: A triangulated unit square split between two ranks, dark grey above and light grey below, with the seam running horizontally along the fault line at mid-height; the band cells along that line are red (ligament) on both ranks, and only a short inclined band above the seam is gold (embedded) with a green trace on it.
:width: 70%

The np=2 seam (black vertices) follows the horizontal network of
ptest_0859; the band along it is all ligament (red) and only the Splay is
embedded (gold, trace in green). `figures/seam-ligament/probe_along_strike.py`
and `plot_along_strike.py`.
```

```{figure} figures/seam-ligament/crossing_gather_vs_ligament.png
:alt: Two panels of the same vertical fault network on a two-rank mesh. Left, the gather: the whole band and its surround sit on the darker rank, with the traces drawn in green. Right, the ligament: the band's upper part is embedded on the dark rank with a green trace, its lower part on the light rank with a blue trace, and the seam crossing, including the junction and the inclined splay, is a strip of red base cells.
:width: 100%

The vertical network, gathered (left) and with the seam ligament (right).
The split's sub-chains carry their own traces on each rank; the junction
and the Splay fall in the ligament. `figures/seam-ligament/probe_cross.py`
and `plot_cross.py`.
```

### The band butted up to the seam (3 September, evening)

Louis's reading of the first diagram was that the design had always been
the band butted up to the partition boundary in elements, with the split
blind at the seam as it is at the free surface, and that the weak glue
ameliorates the tip stress only if the fault in the TI sense is
continuous through the tip. The first build fell short of that in one
rule: it protected every cell touching a shared vertex, so the band
stopped a base cell short of the seam on each rank, the gap between the
two tips was six band cells of base mesh, and each tip touched the weak
material on one side only.

The rebuild's contract is narrower: no shared vertex deleted, no placed
vertex shared. So the second build protects the shared vertices only,
carves the seam cells like any other, lets the cavity ring run along the
seam's own edges, and caps the band against them. What is left at the
seam is one strip of fill cells, the ligament. The cut stops one band
cell short of each clipped end (`add_fault(..., blind=1)`), and the weak
plane is painted on the ligament and on the band within one band width
of each blind tip (`FaultNetwork.bridge_cells`), so the tip is enclosed.

```{figure} figures/seam-ligament/fault_mesh_serial_vs_butted.png
:alt: Two zoomed panels of a vertical fault band on a triangulated mesh. Left, serial: a continuous gold band with a blue cut down its middle. Right, np=2 with the band butted to the seam: the gold band reaches the seam from both sides, a thin strip of red fill cells sits on the seam between them, the blue cut stops at a green tip one band cell inside each side, and a purple outline marks the band cells around both tips and the strip as painted eta_1.
:width: 100%

The seam crossing of the long-fault fixture, serial (left) and butted at
np=2 (right). `figures/seam-ligament/dump_fault_mesh.py` and
`plot_fault_mesh.py`; `fault_mesh_serial_vs_seamcell.png` is the same
view of the first build.
```

| realisation | np | seam-cell ligament | butted, blind tip | gathered |
|---|---|---|---|---|
| TI | 2 | 0.5605 (−1.5%) | 0.5684 (−0.07%) | 0.5688 |
| TI | 3 | 0.5484 (−3.6%) | 0.5643 (−0.8%) | 0.5688 |
| split | 2 | 0.4421 (−13%) | 0.4703 (−7.7%) | 0.5094 |
| split | 3 | 0.3869 (−24%) | 0.4352 (−15%) | 0.5094 |

For the TI realisation the crossing is now within the fill's own noise:
the band is at its own resolution everywhere except one strip of fill on
the seam. For the split the weld per crossing halves, and the profile
shows where the rest is: at the far ends of the fault the butted band
gives 0.95 to 0.98 of the gathered slip where the first build gave 0.90
to 0.96, and the drop is confined to the two band cells either side of
the ligament, where the pinned tips sit. The tips are now enclosed by
weak cells, and the loss that remains is the tips themselves: the blind
margin plus the ligament is three band cells the crack does not cross,
and the weak plane at eta_1 = 0.01 carries about half of the slip across
it. That is the number the free tip has to beat.

### The rig: a fine fault across a coarse seam (3 September, late)

Louis's direction for the parallel work is the TI realisation; the split
nodes wait for a later session. So the S-fault rig ran in parallel with
TI on the network path (`cases/s_fault_rig`, `-uw_seams ligament`; the
rig's gauges were rank-local and now reduce across ranks). Coarse
(w = 0.03) the ligament is within 3% on the Main, local to its crossing,
and the other strands within 1%, with the first speed-up the rig has had.
Fine (w = 0.01, the sensible width) it is not acceptable: the seam's cells
are three to six band widths across, the Main loses 8% (np=2) to 10%
(np=4) of peak slip along its whole length, Cont loses 4-5%, and the
Branch gains 6-20% because the Main's throughput drops at its junction
crossing and the slip partitions onto the Branch. One more refinement
level toward the traces halves the seam cell and halves the loss (Main
−3.1% at np=2, −0.8% at np=4; the serial answer is unchanged): the loss
is the seam cell's size, i.e. the ligament standing in for the band at
base resolution. The rig case's README carries both tables.

Two library consequences fell out. Several cavities on one rank are now
carved and filled independently in every mode, which lifts the "one
simple hole" refusal the fine rig hit on a graded base at two levels.
And the cells a cavity holds are the assembly cells whose vertices lie
inside it, except for one seamless cavity, which holds the whole
assembly (an outcrop vertex snapped to a curved wall lies just outside
the straight base cells).

### The decision: mesh the band through the seam

Louis: "That really will not do. Why is it not possible to mesh
consistently across the boundary?" It is possible. The band assembly is
identical on every rank already, so both sides of a seam can compute the
same interface; what is missing is that the rebuild carries the old star
forest over and assumes no placed point is shared. The conforming mode of
the interface sketch is therefore the next build, in this form:

1. Split each seam edge the band crosses at its two rail crossings; the
   crossing points are new vertices, shared by construction.
2. Assign the band's cells to sides by centroid; the chain of band edges
   between the two sets is the seam inside the band, and its vertices are
   shared.
3. Fill each side against its own ring with the crossing points in it;
   the band's cells on each side are made by that side as now.
4. Extend the vertex star forest with the crossing points and the chain
   vertices before the interpolate, keyed by the band's own point
   numbering (owner the lower rank; one small exchange of the owner's new
   indices).

The band then keeps its own resolution through the seam and the TI
realisation is partition-independent up to the fill's noise. The
ligament mode stays as the fallback and as the split's blind-tip form.

### Built: the band meshed through the seam (3 September, night)

`seams="conform"` is built for the 2-D ribbon, and it is what the
gather's partition independence was for. Nothing moves. The mechanism,
as it ended up rather than as sketched:

- **Victims on the seam are one decision.** A seam vertex is deleted
  only if the band itself reaches it (within 0.6 of a band width), never
  because it lies within the carve's clearance, and the decision is
  reconciled over the star forest so both sides delete it or neither
  does. A seam the band merely runs beside keeps its edges, and the
  fills on either side keep their common boundary there.
- **Ownership by cavity.** Every band cell belongs to the rank whose
  dropped cells hold its centroid; every band vertex likewise; every
  skin edge belongs to the rank whose cavity holds the point just
  outside it. Three global arrays, one reduction each. The boundary
  between the two ranks' band cells is a chain of band edges — the seam
  inside the band — and needs no construction of its own.
- **The fill boundary is a graph.** A rank's ring edges that are not
  crossed seam edges, its skin edges, and one connector from each
  surviving end of a crossed span to the nearest free end of its skin
  runs (a run ends where the skin changes hands, which is where the seam
  crosses a rail, so the connector is the same edge on both sides).
  Every vertex then has two edges; walking the graph gives the fill
  loops, outer ones filled with the loops they contain as holes. A
  junction loop the seam crosses twice, a band the seam skirts, a fault
  ending near the seam all come out of the same walk; the arc-splicing
  formulation tried first did not survive the rig's junction.
- **The rebuild's star forest gains the new shared points.** The band
  vertices more than one rank places are owned by the lowest rank; the
  others enter as leaves before the interpolate, keyed by the band's own
  point numbering with one exchange of the owners' first placed index.
  A shared vertex deleted on both sides drops out; deleted on one side
  only is refused.
- **The fill is graded from the band arcs in the ring** as it is from
  the holes, or the cavity meshes at band resolution end to end (the
  first conform run was a third larger than serial for that reason).

In serial it reduces to the gather path exactly.

| fixture | np | cells per rank | result vs gathered |
|---|---|---|---|
| long fault, TI | 2 | 1212 / 1042 | 0.5688 vs 0.5688 |
| long fault, TI | 3 | 784 / 891 / 571 | 0.5688 vs 0.5688 |
| S-fault rig fine, TI | 2 | 4781 / 2899 | Main 0.5649 vs 0.5683; Branch, Step, Cont within 0.1% |
| S-fault rig fine, TI | 4 | 4475 / … | Main 0.5648; the rest within 0.1% |

The rig's fine profiles sit within 2% of serial everywhere along the
Main (the fill's noise at the crossing) and within 0.1% along Cont, where
the ligament had lost 8-10% and shifted the partition onto the Branch.
The repeat solve on the fine rig is 5.0 s at np=2 against 7.4 s serial.
`ptest_0864` pins the long fault at np=2 and np=3; the rig case's README
carries the profiles and `sf_partition_fine_conform_np2.png` the
partition, with the seam running through the Main's band along its own
edges at the junction.

**Not handled, refused with a message:** a seam that runs inside the
band along strike (its vertices are all victims and the two fills have
no common boundary to keep); a crossing so oblique that the skin
changes hands at fewer than two vertices; and 3-D, where the thin volume
still gathers. A refused fill saves its inputs to
`place_fill_failure_rank<r>.npz` beside the script.

### The fill graded harder (4 September)

Louis: the fill around the faults sits at band resolution about three
band widths out and should be considerably less. Measured on the fine rig
(w = 0.01, band spacing 0.005, base 0.03 near the traces), median cell
size in band spacings by distance from the nearest trace:

| distance / w | linear fill | exponent 0.5 | exponent 0.35 |
|---|---|---|---|
| 0.5–1 | 1.18 | 1.38 | 1.29 |
| 1–1.5 | 1.34 | 1.93 | 2.23 |
| 2–3 | 2.15 | 3.03 | 3.49 |
| 4–6 | 4.03 | 4.52 | 4.71 |
| fill cells within 3w | 3924 | 2328 | 1931 |
| total cells | 8170 | 6278 | 5816 |
| worst angle | 17.7° | 17.4° | 17.2° |
| Main / Cont peak | 0.5683 / 0.3274 | 0.5682 / 0.3263 | 0.5681 / 0.3257 |

The fill's size function is now a power of the relative distance across
the cavity (`grading`, `FaultNetwork.build(fill_grading=)`), default
0.35. The three-band-width extent itself is the cavity, one base cell
wide, and cannot shrink: at a narrower clearance the fill is a sliver
between 0.005 and 0.03 edges and gmsh never settles (a run at 0.3 spun
for ten minutes inside `generate`). The ladder mesher keeps the linear
fill, since its sequential cavities need the room between close strands.
Two placement repairs came with the measurement: an island of kept
cells enclosed by a narrow cavity is dropped rather than refused, and a
band cell goes to the cavity holding its centroid when a rank carves
several. At np=2 with the band through the seam the fine rig is now
3363 cells on rank 0 and the repeat solve 3.7 s (serial linear: 7.0 s),
the answer unchanged.

### The FAC patch in parallel (4 September)

The patch smoother was serial only because its blocks were built in the
serial branch of the transfer builder; the installation already took
per-rank subdomains. The blocks are now built in the parallel branch too,
in global row numbering: the fault-zone blocks from the level's local
cells through the layout's ghost-resolved numbering, which puts a masked
cell's off-rank nodes into this rank's subdomain and so gives the seam
halo for free, and the structural block from this rank's owned rows of
the distributed transfer, with the cover gate one collective verdict.

Measured on the fine rig (TI, contrast 1, eta_1 = 1e-3, harder grading),
velocity iterations of the last apply and the repeat solve:

| configuration | serial | np=2, seams="conform" |
|---|---|---|
| whole-level smoothing (no patch) | 44 its, 4.2 s | 57 its, 3.7 s |
| fault blocks + structural block | 60 its, 5.2 s (block 20290 of 23134 rows) | 66 its, 4.2 s (rank 0: 11944 of 22590) |
| fault blocks alone | stalls (watchdog) | stalls (watchdog) |

That table says the replacing patch is a net loss and the fault blocks
alone starve the level. Louis: with a full hierarchy everywhere and a
plain mesh this should be the easy case. It is. The same mesh and
hierarchy take 4 velocity iterations with no viscosity contrast, 9 at a
tenfold weak plane, 44 at the thousandfold one, and one more base level
under the band changes nothing (42): the hierarchy is stitched correctly,
and the cost is the weak plane in a band thinner than the coarse cells,
which the Galerkin coarse operators average away. That is a smoother
problem on the band. The patch as first built replaced the level's
smoother, so the blocks had to cover everything the coarse level could
not represent, which on a placed mesh is the whole refined region. The
form that works is the composite one: the whole-level smoother, then the
fault blocks on top, with an exact LU sub-solve on the band block and one
overlap layer.

| finest-level smoother, fine rig, TI 1e-3 | serial | np=2 | np=4 |
|---|---|---|---|
| whole-level gmres/sor | 44 its, 4.2 s | 57 its, 3.7 s | 58 its |
| replacing patch (fault + structural) | 60 its, 5.2 s | 66 its, 4.2 s | — |
| composite, fault blocks, sor sub-solve | 33 its, 4.5 s | 46 its, 4.4 s | — |
| composite, fault blocks, LU sub-solve | 14 its, 2.6 s | 16 its, 1.9 s | — |
| composite, LU, one overlap layer | 10 its, 2.1 s | 11 its, 1.5 s | 12 its, 1.5 s |

The last row is the default now when a zone is keyed: `fac_zone` masks
give the blocks, the structural split is off, the sub-solve is LU with a
nonzero shift, the overlap one layer (`UW_FAC_MODE`, `UW_FAC_SUB_PC`,
`UW_FAC_OVERLAP`, `UW_FAC_STRUCTURAL` remain as measurement knobs). One
parallel trap on the way: the structural split's cover gate is a
collective, and a rank with no band cells decided alone to take it while
the others skipped it, a hang at np=4; the decision is now made from
what any rank holds.

### The band block's size (4 September, evening)

Louis: keep the band level manageable; in 3-D this will be large. The
block is the one cost that grows with the fault, so it was swept on the
fine rig (TI 1e-3, composite form, LU sub-solve, overlap 1). The band's
footprint is 11440 rows; with its overlap layer one block is 15736 rows,
two thirds of the finest level. Cutting it along strike into pieces of
at most N rows:

| rows per block | serial: blocks, its, first / repeat solve | np=2: blocks (rank 0), its, solve | np=4 |
|---|---|---|---|
| whole band | 1, 10, 6.1 / 2.2 s | 1, 11, 3.6 / 1.6 s | — |
| 6000 | 2, 12, 4.4 / 2.4 s | — | — |
| 3000 | 4, 12, 4.6 / 2.4 s | 3, 12, 5.0 / 1.6 s | — |
| 1500 | 8, 12, 4.7 / 2.5 s | 5, 12, 3.5 / 1.6 s | 5, 12, 4.0 / 1.6 s |
| 800 | 15, 13, 6.1 / 2.6 s | 9, 13, 3.7 / 1.8 s | — |
| 400 | 29, 15, 5.1 / 2.7 s | 17, 30, 5.6 / 3.4 s | — |

Segmenting is nearly free down to about a thousand rows per block: 12
iterations from 2 to 8 blocks in serial and at np=2 and np=4, with the
first solve faster than the single block's because the factorisations
are small. Below that the interfaces between pieces start to cost, and at
np=2 the 600-row blocks double the iteration count. Two other points:
ILU on the block instead of LU is 27 iterations against 10, so the
sub-solve must be exact; and a second overlap layer buys one iteration
for 2700 rows (9 against 10) while segmented blocks with overlap 2 did
not get through setup in serial within the watchdog.

**The 2-D defaults**, set in `set_custom_fmg` and the automatic path:
the composite form, LU sub-solve with a nonzero shift, one overlap layer,
`fac_block_rows=2000` (rows before the overlap), no structural block. On
the rig that is 6 blocks and 12 iterations in serial, 5 blocks and 12 at
np=2 and np=4, with the repeat solve at 1.6 s where the run began the day
at 7.4 s. For 3-D the arithmetic is the point: the band is a surface two
cells thick, its rows grow with the fault's area, and the cap turns that
into more blocks of the same size rather than a larger factorisation;
the iteration count is flat in the block count above the floor.

### What follows

1. **A free tip at a ligament end.** The split's weld is the pinned tip.
   Where a sub-chain ends because the band was clipped rather than because
   the fault ends, the tip vertex can be duplicated too, with its fan
   assigned by the side of the trace's extension; the discontinuity then
   ends inside the ligament's weak cells, which carry the jump as strain
   over one cell. That is the bridging split the sketch asked for, done
   at the tip rather than at the seam.
2. **Steering the seam off the fault.** A segmented gather: chunks along
   strike, each moved to the rank that holds most of it, so the seam
   crosses the band only where the owner changes. `_gather_regions`
   merges touching regions today; chunks need a `merge=False` form with
   a deterministic claim of the overlap. Without this, an along-strike
   seam leaves a fault's whole stretch as painted base cells.
3. **The conforming interface is built** (`seams="conform"`, above). What
   it leaves for the split is the cut through the seam: duplicating a
   shared chain vertex on both ranks with star-forest entries for the
   replica pair and a pairing record that agrees across ranks — the same
   bookkeeping, applied to a duplicated point instead of a new one.
4. **3-D.** The 2-D ribbon has the mechanism; the 3-D thin volume still
   gathers. The clip and the manifold clean-up carry over one dimension
   up (tets, faces), the multi-pass split already works in 3-D through
   `split_along_label_3d`.
