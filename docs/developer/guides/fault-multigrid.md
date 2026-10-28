---
name: fault-multigrid
description: Which multigrid hierarchy a faulted Stokes solve gets in Underworld3 and why — the base mesh is never cut or rebalanced, the fault is carried on an ephemeral add-on, the finest level takes a strong patch keyed on the fault physics, and the coarse levels must carry the fault the way the fine operator does. The three rulings, what ships on development, the iteration counts to expect, where parallel imbalance comes from, and what is measured but still on a branch.
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
by measurement, and what is measured but not yet merged. The construction of
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
region is most of the node budget while the fault zone scales as the trace.

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
- SOR subdomain solves are the safe default; LU is what buys contrast
  constancy on the near-singular split patch.
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
   region, so the cost is a few stars per rank rather than all on one.

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

## Measured, not yet merged: the split-level hierarchy

On `feature/fault-split-multigrid` every add-on level is cut and split like the
finest (strategy C), with a side-aware transfer that locates each fine node
through its own cell's centroid, a per-level pair frame, and unreached coarse
unknowns dropped rather than repaired. The measurements that motivate it:

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

None of this is on `development`: `fault_mg.py`, the ribbon replay onto the
add-on levels and the per-level frames are branch work, and `FaultNetwork.build`
still gives the finest band only. The record, with the dead ends (TI bands
under a split finest, node-cloud transfers to a split level, extrapolated
weights, nearest-node injection) is the journal topic
[Fault multigrid](https://github.com/underworldcode/underworld3-journal/blob/main/topics/fault-multigrid.md).

## Sources

- [Fault-patch multigrid: the measured skeleton](../design/fault-patch-multigrid-2026-08.md),
  the architecture and the configuration rules (#629).
- [Fault networks in parallel](../design/fault-parallel-placement-2026-09.md),
  the gathered star and the seam rules (#672).
- The journal topic page above, for the split-level work and the measurements
  behind the rulings.
