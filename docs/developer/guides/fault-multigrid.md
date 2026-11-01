---
name: fault-multigrid
description: Which multigrid hierarchy a faulted Stokes solve gets in Underworld3 and why — the base mesh is never cut or rebalanced, the fault is carried on an ephemeral add-on, the finest level takes a strong patch keyed on the fault physics, and the coarse levels must carry the fault the way the fine operator does. The three rulings, the production tail ruled on 2026-11-01 (the welded base, the split finest level, the FAC LU patch), what ships on development, the iteration counts to expect, where parallel imbalance comes from, and the split-level hierarchy as the measured alternative.
families: [Stokes, Stokes_Constrained]
kind: guide
---

# Multigrid for faulted meshes: the rulings

A fault concentrates the difficulty of the Stokes operator into a thin
cross-cutting region: a split-node contact, a painted weak band, or a painted
transversely isotropic band. Ordinary multigrid handles everything smooth,
refined regions and layered viscosity jumps included, and fails on the band,
because a coarse space that cannot represent slip corrects nothing there.
This page says what the solver does about it on `development`, what is settled
by measurement and ruling, and what is kept as the measured alternative. The construction of
the faulted mesh itself is on [Fault networks](../../advanced/fault-networks.md)
and [Split-node faults](../../advanced/split-node-faults.md); the general
multigrid controls are on
[Multigrid preconditioning](../../advanced/multigrid-preconditioning.md).

## The three rulings

These keep being re-derived in new sessions. They are decisions, taken with
Louis in August to October 2026, and they fix everything below.

1. **The base mesh is never cut and never rebalanced.** The base and its
   coarse levels are the persistent, distributed object that holds the fields.
   Everything the fault needs is built on top of it: the add-on levels are
   ephemeral, rebuilt whenever the fault moves, and may be partitioned any way,
   because every transfer to them is partition-agnostic. A load fix is applied
   to the add-on levels, never to the base.
2. **One hierarchy carries the fault as the model has it.** A split spine on
   every level that holds the fault, or a weak transversely isotropic band at
   each level's own resolution. The background stays welded either way. Not the
   cut on some levels and a painted band on others: a smeared band on a coarse
   level is a continuous shear that a split fine operator rejects, measured
   worse than no fault on the coarse levels at every band strength.
3. **No bespoke meshes per case.** The placed ribbon band along the trace,
   sub-sampled 2:1 so that every coarser spine vertex is a finer one, is the one
   add-on structure that serves both realisations. The ribbon is part of the
   multigrid design, not an alternative to it: it is the bridge between the
   background mesh and the fault, the authored patch key, and the same mesh
   discipline for split, weak and TI faults.

## What ships on `development`

The architecture is the fault-patch design (#629): **multigrid over everything,
strong patches where the physics is sharp.** The custom-prolongation Galerkin
ladder from the base mesh smooths every level; the finest level's smoother
gains one additive-Schwarz component whose subdomains are the fault zone, each
block solved directly once per smoother application. The patch is keyed on the
fault, never on the refinement structure: in a plate-scale mesh the refined
region is most of the node budget while the fault zone scales as the trace. The
blocks are solved by LU with a nonzero shift; on the rotated free-slip path,
which every fault model uses, they ran as unshifted ILU(0) from 2026-09-03 until
#912 (2026-10-31), see the configuration rules below.

With it, the velocity block converges in 4 to 9 iterations on the 2-D rig,
independent of viscosity contrast from 1 to $10^{6}$, of how many along-strike
blocks the trace is divided into, of a strong lid of either jump or smooth
profile, and of network topology. Split contact takes 6 to 7, a painted weak
band 9 falling to 7 as the contrast rises, a painted TI band 4. GAMG on the same
problems takes 40 to 178 and its iteration cap by $10^{6}$. In 3-D, pure
contact on a 20,000-cell ribbon takes 7 against GAMG's 56 with identical
physics.

How to get it:

- `FaultNetwork.build(width=w, realisation=...)` places the ribbon, adopts the
  base's refinement hierarchy onto the final mesh (so build the base with
  `refinement >= 1`), and sets the patch key itself: none for `"split"`, the
  band's cells for `"ti"`. Nothing more is needed for the finest patch.
- `base.multigrid_levels` (#915) returns a mesh's levels coarsest first with the
  mesh itself last, so a solver on a mesh placed on `base` takes
  `set_custom_fmg(solver, base.multigrid_levels, field_id=0)` and nothing else;
  a solver on `base` itself takes every level but the last.
- A hand-built hierarchy registers with
  `set_custom_fmg(solver, coarse_meshes, field_id=0, fac_zone=mask)`, where
  `field_id=0` is the velocity block of a saddle-point solver and `fac_zone`
  is a boolean cell mask or a list of per-segment masks;
  `mesh.cells_labelled(name, value)` builds one from a placement's label.
  `solver.preconditioner = "fmg"` is the *preference* form, which warns and
  falls back to GAMG; `set_custom_fmg` is the *demand* form and raises.
- On an `adapt()` child the mesh-owned custom-prolongation tail is picked up
  automatically, rotated free-slip included. On a plain refined annulus with
  rotated free-slip the native hierarchy is not read, and a coarse-mesh tail
  must be registered by hand (see
  [Adapt-on-top faults](adapt-on-top-faults.md)).

**What production runs do.** The campaign drivers (the zg95 annulus and slab
drivers, the box-fault driver) place the fault as a structured ladder band into a copy
of the base's finest level with `place_fault_ribbon_2d(..., mesher="ladder",
split=True)`, rung spacing $S$ and band $2S$; the base and its levels are never
modified, and no bisection is involved. The tail under it is ruled (Louis,
2026-11-01, on #892): **the welded base with the split $S$ level on top and the FAC
LU patch on that finest level**, `set_custom_fmg(stokes, base.multigrid_levels,
field_id=0)`. On UWTN 2026-022's fixtures with the LU blocks running it takes 66
velocity iterations over a linear solve at band ratios 6, 12 and 24, and 7 per Newton
increment on the yielding-lid box at ratios 8, 16 and 25, with the Newton path and
the slip unchanged. The chain of split ladders (the same ladder at $2S$ placed into a
copy of the base's finest level and at $4S$ into a copy of the level below, each split
along its rungs, the band widening with the level) is the measured alternative and
not the production tail: its levels never carry the patch, and on the same states it
takes 83 to 166 and 28 to 76. The ruling's condition, confirmation on the production
annulus at np 16, is met (kaiju, `development` 25d62812, four steps from a TAf
checkpoint): the welded base takes 5.9 velocity iterations per Newton increment
against the chain's 7.7, with the same Newton counts, vrms to 2e-6 and slip to 1 in
6000; the margin is smaller than on the fixtures because both arms carry the same
finest patch and only the coarse levels differ. The chain stays driver code
(`uw_chain`), whose default moves to 0 as the campaign builds move to `development`;
whether split coarse levels remain a library option is not decided. The adapt-and-cut route is the automated one and the
one the hierarchy strategies were compared on; it is not what production runs.
(Journal: `2026-10-28-which-fault-mesh-construction-is-used-in-practice`, the sweep
of 2026-10-31, and the decision and finding entries of 2026-11-01.)

**Which key for which model.** The zone key is for volumetric fault
representations only: a weak, TI or damage zone whose width is physics. A
split-node fault runs the structural patch alone, which is installed
automatically from the split's coincident pairs. The ribbon is never the key;
it is resolution, not rheology. Measured: under pure contact the zone block is
redundant and inflates the finest block from the trace to most of the band.

**What the width means** follows from the same choice. For a split fault
`width` is a resolution parameter, the band exists to give the cut its own
vertices, and nothing rheological may be keyed to it. For a weak or TI fault it
is the fault-zone thickness, a physical parameter that sets the slip rate and
wants two or three elements across it. Split nodes plus band-wide weakness with
the width chosen as a mesh number is neither model; it was built three times in
one campaign and is the error to recognise.

## Configuration rules, each learned from a failure

- The finest patch always contains the structural patch (the rows the coarse
  level cannot represent). A zone away from the cut leaves those rows smoothed
  nowhere and the velocity sub-solve caps on every application. The zone-block
  builder unions them in; do not defeat it.
- Basic additive Schwarz, never restricted: discarding the halo part of the
  subdomain correction stalls the outer Krylov at 80 to 375 iterations where
  basic runs 6.
- Shifted factorisation is mandatory on split-contact patches
  (`sub_pc_factor_shift_type nonzero`): the rotated Galerkin block carries
  near-zero pivots from the constraint-zeroed transfer rows, and an unshifted
  LU dies at zero iterations with `PC_FAILED`.
- The FAC blocks are LU with the nonzero shift; ILU there is a regression. From
  2026-09-03 (891c9bb0, which made composite mode and LU the defaults in one
  commit) until #912 (2026-10-31) the rotated path ran them as unshifted ILU(0):
  the block options were written to the database, the rotated path deleted them
  after forcing the smoother's set-up, and the composite's ASM child read the
  database only at its first apply (#892). `test_1090_mg_live_configuration_parallel`
  now reads back every level's live smoother against what was configured. Any
  faulted rotated solve measured between those dates was measured on ILU blocks.
- The robust (GMRES) smoother under contrast: the fast Richardson variant
  degrades already at $10^{2}$.
- Check the subdomain probe, not only the iteration count: a patch that
  declines silently leaves a plausible whole-level solve, and two bugs in the
  original campaign were caught only because the probe disagreed with the
  narrative.
- The coarsest level must be small. The free-slip annulus leaves a rotation
  null mode, so the coarse solve is a dense SVD on one rank: 900 unknowns are
  instant, 14,000 took over thirty minutes. Add a level rather than tolerate a
  large bottom; the general rule is on the multigrid page.
- The augmented-Lagrangian penalty at $\gamma = 1$ shortens the pressure Schur
  loop once the velocity preconditioner is healthy (22 to 17 iterations, warm
  solve ahead of GAMG outright); $\gamma = 10$ over-stiffens the velocity block
  and reverses the gain. The recovered pressure is then the multiplier, and
  peak slip drifts by about 1 % at rig resolution.

## Parallel: where the imbalance is, and is not

The base and its coarse levels are balanced at every rank count and nothing in
the fault machinery touches them. The imbalance is in the add-on levels, in two
parts, in this order of size:

1. **The add-on child inherits the base's partition.** Every refined cell stays
   on the rank that owned the cell it came from, so the ranks under the band
   collect most of the new cells. On a 26,000-cell box at 16 ranks the base
   puts about 930 cells on each rank and the child puts 6,800 on one. This
   exists before any fault is split, and the base is not the fix.
2. **The fault's cell star is gathered onto one rank**, the contact design's
   accepted trade: extra load on the owning rank for no communication inside
   the contact solve. A network's faults go to different ranks region by
   region, so the cost is a few stars per rank rather than all on one. Measured
   on the production annulus at np 16 (`development` 25d62812): the whole
   finest-level patch, 465 rows and all 105 split pairs, sits on the one rank
   whose partition holds the fault, so the patch's LU solve is serial there. At
   that size it costs little; at higher rank counts, or with several faults in
   one partition, it is the imbalance #895 tracks.

With a band of area fraction $f$ refined $k$ halvings on $P$ ranks, for a fault
of length $L$ in a domain of width $D$, the busiest rank sits roughly
$f\,4^{k}\sqrt{P}\,D/L$ above the mean. The $4^{k}$ is the term to control:
keep the band at the base's resolution or one halving below it, and grade the
base toward the fault corridor at construction when more is needed, since the
base is partitioned once for what it holds. Measured on the toy, the solve
stops scaling at 8 ranks because every smoother sweep waits for the rank that
holds the band; a plate-scale model puts the cap higher, but it is a cap, and it
is set by the add-on's inherited partition. Partitioning the add-on levels for
themselves, star-weighted, is the planned fix; rebalancing the base has been
proposed twice and ruled out twice.

## The split-level hierarchy: merged, and now the alternative

`fault_mg.py`, `add_fault(split_levels=True)`, the side-aware transfer that
locates each fine node through its own cell's centroid, the per-level pair frame
(unreached coarse unknowns dropped rather than repaired) and the thin-sheet
solver merged to `development` on 2026-10-31 (#889, #890). On an adapt-and-cut
child every add-on level is cut and split like the finest (strategy C); the
ribbon replay of a placed band onto the add-on levels is not in the library,
and since the ruling above it is not a production need. The measurements that
motivated the split levels were all made before #912, on the rotated path with
the welded hierarchy's patch running ILU(0) blocks and the split levels carrying
no patch at all:

- On the cut, iterations *fall* with refinement (6, 5, 4 at two, three and
  four add-on levels) where the welded hierarchy's counts rise (13, 18, 23); a
  Coulomb fault's Newton steps take 3 to 15 iterations against 15 to 50.
- A single placed ladder over the welded base is enough up to a ratio of about
  12 between the base's finest cell and the band spacing, for both
  realisations. Past that, the TI realisation gains from nested welded mid
  levels (8, 6, 5, 4 with zero to three mids) and the split realisation gains
  nothing from them: it needs the chain of split ladders, or a base refined
  under the band, which gives the same count.
- On a convecting annulus with a dipping trench (the Zhong and Gurnis 1995
  analogue), 10 velocity iterations against GAMG's 119, with one rank and two
  agreeing in slip to four figures.

With the LU blocks the welded base with the split finest takes 3 velocity
iterations per linear solve at every band ratio from 6 to 25 on the yielding-lid
box, so the chain's case in 2-D is gone. It remains the measured alternative
where the LU patch is unavailable or its block too large (the block grows with
the band's rows; 3-D is not measured), and whether split coarse levels stay a
library option is not decided. `FaultNetwork.build` gives the finest band only,
which is now the production tail. The record, with the dead ends (TI bands
under a split finest, node-cloud transfers to a split level, extrapolated
weights, nearest-node injection) is the journal topic
[Fault multigrid](https://github.com/underworldcode/underworld3-journal/blob/main/topics/fault-multigrid.md).

## Sources

- [Fault-patch multigrid: the measured skeleton](../design/fault-patch-multigrid-2026-08.md),
  the architecture and the configuration rules (#629).
- [Fault networks in parallel](../design/fault-parallel-placement-2026-09.md),
  the gathered star and the seam rules (#672).
- The journal topic page above, for the split-level work and the measurements
  behind the rulings; its entries of 2026-10-31 (the tail sweep) and 2026-11-01
  (the tail ruling; the history of #892) for the production tail.
- Louis's ruling on the tail, on
  [#892](https://github.com/underworldcode/underworld3/issues/892#issuecomment-6550186134),
  and the fix, [#912](https://github.com/underworldcode/underworld3/pull/912).
