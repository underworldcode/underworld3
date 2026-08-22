# Fault-patch multigrid: the measured skeleton (2026-08, #629)

The solver architecture for fault-system Stokes problems, settled by
measurement on a 2-D analysis rig and ready to carry into 3-D. One
sentence: **ordinary balanced multigrid solves the smooth problem —
refined or not, layered or not — and a set of small strong subdomain
solves, keyed on the fault physics and proportional to the fault trace,
restores the multigrid contract where coarse spaces cannot see.**

This note records the architecture, the evidence, the configuration
rules that were each learned from a failure, and the parallel design
rules for the 3-D deployment. The campaign log with every number is
`~/+Simulations/place_route_health/README.md` (items 17–22) and the
issue thread is #629; the code landed on `feature/fault-outcrop-3d-cap`
(commits `ec213489` → `d4b86ea0`).

## The problem

A fault zone — split-node contact, a painted weak band, or a painted
transversely-isotropic band — concentrates the difficulty of the Stokes
operator into a thin cross-cutting region. Two measured pathologies:

1. **Coarse corrections cannot see the band modes.** Level ablation at
   viscosity contrast 1e4: full / no-mid / two-level tails converge in
   20 / 19 / 24 iterations — tail depth is irrelevant. Even a level that
   carries the band geometry contributes nothing: an unsplit coarse
   space cannot represent slip, and coefficient-blind transfers smear
   the weak inclusion.
2. **Convergence is then proportional to fine-level smoothing work.**
   Doubling the smoother (gmres/8 for /4) exactly halves the iteration
   count. So the count grows with contrast — 7 → 10 → 36 over
   1e0 → 1e4 — where a healthy multigrid holds constant V-cycles.

GAMG fails the same way but harder (56 → 178 over the same range, its
iteration cap by 1e6): the sharp thin inclusion is the malignant kind of
coefficient structure for any coarsening that has to guess.

## The architecture

* **Multigrid over everything.** The custom-P Galerkin ladder (or GAMG
  where no hierarchy exists) with whole-level smoothing. This handles
  every smooth mode, including in refined regions and across *extended*
  (co-dimension-1, layered) viscosity jumps — measured: a sharp 1e4
  strong lid crossed by the fault leaves the iteration count unchanged,
  with the lid boundary entirely outside any patch.
* **Strong patches where the physics is sharp.** The fine-level smoother
  gains one ASM component whose subdomains are the fault zone(s): the
  slit trace for a split fault, the painted cells for a weak or TI band.
  Each block is solved directly (small shifted LU) once per smoother
  application. The patch is *never* keyed on the refinement structure —
  in a global subduction mesh the refined region is most of the node
  budget, while the fault zone scales as the trace,
  m ≈ N^(2/3) in 3-D.

With this in place the velocity block converges in **4–9 iterations
independent of viscosity contrast from 1 to 1e6**, of the number of
along-strike blocks the trace is divided into (k = 1…16 measured), of a
strong lid of either jump or smooth profile, of fault-network topology
(a V-junction runs identically as one merged block or per-segment
blocks overlapping at the apex — the junction is a non-event), and of
the fault representation: split contact (6–7), unsplit painted weak
band (9 → 7 over 1e4 → 1e6, GAMG 40+), and painted TI band (4, GAMG
43). One cross-subsystem observation: with the velocity preconditioner
made whole, the pressure sub-solve dropped from its 200-iteration cap
to 14 — part of the recorded #625 pathology is downstream of an
incomplete velocity PC, and the Schur diagnosis should be re-baselined
against a healthy velocity block.

## Configuration rules (each learned from a failure)

0. **The finest patch always contains the structural patch.** The patch
   smoother *replaces* whole-level smoothing, so it inherits every row
   the coarse level cannot represent — the cut/split-inserted DOFs that
   the transfer's non-identity rows identify — whether or not the
   physics zone covers them. A zone away from the cut leaves those rows
   smoothed nowhere and the velocity sub-solve caps on every
   application (measured three independent ways; the overlap-0 slit
   stagnation was the same omission). The zone-block builder unions the
   uncovered structural rows in as an additional block automatically.
1. **Basic ASM, never restricted.** Discarding the halo part of the
   subdomain correction stalls the outer Krylov at 80–375 iterations
   where basic runs 6 (whole-level baseline 4). `PC_ASM_RESTRICT` is the
   wrong variant for this use.
2. **Shifted factorization is mandatory on split-contact patches.** The
   rotated Galerkin patch block carries near-zero pivots inherited from
   the constraint-zeroed transfer rows (min diagonal ~1e-5); an
   unshifted factorization takes `NUMERIC_ZEROPIVOT` even as exact LU
   and the KSP dies at 0 iterations with `PC_FAILED`. Use
   `sub_pc_factor_shift_type nonzero`.
3. **SOR subdomain solves are the safe default; LU is the strong
   option.** On the near-singular split-contact patch the LU is what
   buys contrast constancy. On an unsplit painted band the two are
   indistinguishable at rig scale.
4. **Patch keying.** Split meshes: coincident fine pairs — the split's
   plus/minus nodes at bit-identical coordinates. (Not transfer identity
   rows onto coarse nodes: a cut inserts *new* vertices at edge
   crossings that coincide with no coarse node, and that detection
   declines silently.) Painted models: an explicit boolean cell mask —
   the modeler painted the band and knows the cells; nothing needs
   detecting.
5. **Blocks per fault-network connected component; junction cells in
   every adjacent block.** Distant segments couple through the medium,
   which is smooth and belongs to the multigrid. Along-strike division
   of one fault into blocks is measured free (7 iterations at every k),
   so blocks can be sized for convenience and parallel placement.
6. **The smoother variant matters under contrast.** The fast
   (richardson) smoother degrades where gmres holds (12 → 23 already at
   1e2); contrast work runs the robust variant.
7. **Check the ASM probe, not just the iteration count.** A patch mode
   that declines silently leaves a plausible-looking whole-level solve;
   two instrumentation bugs in this campaign were caught only because
   the probe line (subdomain sizes, failed-reason codes) disagreed with
   the narrative.

## Two transfer-level results the ladder depends on

* **Structural zeros fattened the Galerkin chain.** The point-located
  transfer builders emit ~1e-16 weights that are structural nonzeros,
  and PtAP fills by structure, compounding per level (90 → 265 → 481
  nnz/row measured down a 4-level tail). Dropping them
  (`_drop_structural_zeros`) collapses the chain to native-like density
  (138/119/173/90) and halved the warm solve. This was the largest
  single win of the campaign.
* **Exact nested transfers for native `refine()` pairs.** Level pairs
  tagged as native refinement get the true any-degree FE embedding
  (dual-basis `W = B·M⁻¹` under the parent's affine pullback), exact to
  1e-15 at P1/P2 in 2-D/3-D. PETSc's `DMCreateInterpolation` general
  path is **not** the embedding (row sums to 1.375, 1e-2 error on
  quadratics) and must not be used for transfers.

## Scaling

The patch reduces the only direct solve from N volume DOFs to the trace
m ≈ N^(2/3). Because iterations are flat in the number of along-strike
blocks, fixed-size segment blocks make the total factorization cost
**linear in the trace and embarrassingly parallel** — no superlinear
direct-solver scaling survives. Under nonlinear fault rheology the
economics improve further: the tangent changes in the patch, so the
per-Newton refresh is the small factor, and the background ladder can be
lagged. (The end-state for stiff friction laws is nonlinear Schwarz /
ASPIN on the same subdomains; nothing in the architecture has to change
to permit it.)

## Parallel design rules for 3-D

1. **Split in the local frame, after distribution — then pair
   co-residency is automatic.** A pair is the duplication of one owned
   facet's nodes and is born on that facet's rank; no partitioner
   constraint is needed (Louis's correction of an earlier draft of this
   rule). The two real items instead: (a) **never distribute a
   pre-split mesh** — the slit is a zero-cost cut in the mesh graph, so
   a partitioner would *preferentially* separate the sides while the
   contact coupling lives outside the graph; gate that pipeline rather
   than engineer around it. (b) **Seam consistency**: where the fault
   crosses a rank boundary along strike, on-rank duplication must
   update the star forest consistently (the known np>=3 line-cut issue;
   the independent-pass design is the template). Along-strike block
   cuts at seams are measured free (the k-ladder).
2. **PCASM is rank-local.** Blocks become per-rank lists with local
   indices; faultless ranks carry `nsd = 0`, and every IS-building path
   must survive empty sets (the empty-stratum `getIndices()` SEGV
   family, #589).
3. **Pair detection must go collective or be replaced by zone keying.**
   Rank-local coordinate matching misses pairs that straddle a boundary
   and can false-match ghosts; the zone mask is rank-local by nature and
   is the recommended production key.
4. **Test at np ≥ 4 with timing probes** — np = 2 is a special case
   that passes collectives which hang at 4 (#512/#596 discipline), and
   `mat_block_size` divisibility is BC-dependent and mixed across ranks
   (#584): gate collectively.
5. **Serial-only pieces that need parallel forms before 3-D-parallel:**
   the FAC patch split, the nested-pair transfer detection, and the
   zero-column repair (the parallel guard currently raises where the
   serial path repairs — placed levels with orphan columns would fail
   at np > 1).
6. **The pressure block is the wall-clock gate at contrast** (its gasm
   sub-solve caps at 200 iterations, silently unconverged) and gasm is
   the most parallel-fragile piece in the stack — #625, being addressed
   separately; whatever lands there should be chosen with parallel
   behaviour in mind.

## Status of the knobs

The campaign ran on TODO(MEASURE)-marked environment variables
(`UW_FAC_*`, `UW_CUSTOM_MG_*`, `UW_MG_SMOOTH_ITS`) plus one explicit
hook (`solver._fac_zone_cells`). Before 3-D production these settle into
API: the zone mask (or a fault-network object) as the patch key, the
basic-ASM/shifted-LU configuration as the default for fault patches, and
the A/B flags removed.

## What the fault ribbon is for (ruling, 2026-08-23)

The finite-width ribbon around a fault is a **modelling object in its
own right**, not just mesh scaffolding. Its sanctioned uses:

* **A damage zone** — when a damage *evolution equation* is part of the
  physics. Damage is then a solved field whose rheology follows from
  the field; it localises where the mechanics puts it. Damage does not
  exist in a model that does not evolve it.
* **A nonlinear plastic / yielding material** — the ribbon's material
  given a yield rheology, so that **failure patterns can emerge within
  the resolved band**. This is the mechanism by which junctions *form*:
  localisation finds its own geometry inside the ribbon instead of the
  modeller authoring it. (The gradient-plasticity caveats apply when
  this is built: the regularising length must enter the plasticity, and
  mesh objectivity needs more than one resolution — the ribbon width
  supplies a geometric length scale but does not by itself regularise.)
* **A permeable zone** — a localized pathway for fluid flow, carrying
  the permeability structure a fault zone has and the surrounding rock
  does not.

What the ribbon region is **not**: a hand-painted static weak
viscosity. "Fake damage" is neither gouge nor a fault — a wide soft
channel needs no split nodes and no fancy solver, and tells us nothing.
The fault itself is the **split surface in the mesh**, always; ribbon
physics complements the slip surface, never substitutes for it. In a
model with none of the three physics above, the ribbon carries
background rheology and is purely resolution.

These roles compose with the solver design below unchanged: whichever
physics the ribbon carries, the band is the authored patch key, the
levels are the FMG bridge, and the machinery is measured
contrast-robust for whatever coefficient structure the physics
produces.

## The ribbon is part of the FMG (ruling, 2026-08-22)

The placed ribbon — the finite band of fault-scale resolution around
each fault, with its modest adaptation — is not an alternative to the
patch methodology; it is **part of the multigrid design**, playing
three roles at once:

* **The bridge between the standard mesh and the patch.** The ribbon's
  band is the natural intermediate structure: coarse ladder below,
  band-resolved level(s) in the middle, the cut/split finest on top.
  Its cell burden is minimal by construction (the band is a few local-h
  wide), and its levels are exactly the composite-grid structure the
  FAC machinery was built for.
* **The damage-zone identifier — authored, not detected.** The
  placement machinery labels the band's cells, and the zone label
  survives the cut/split as cell children. The strong patch keys on
  that label directly: an exact, mesh-conforming, modeler-authored
  region — never the ragged staircase of a distance mask, never the
  one-cell zipper comb of ``cells_supporting`` around the split.
* **One mesh discipline for every fault representation.** The same
  ribbon carries split-contact faults (cut down the transfinite
  centreline — three nodes across, so nothing snaps, #595), painted
  TI, or painted weak rheology; the representation choice stops
  dictating the mesh strategy.

Junctions follow the revised ruling: split segments stop a short
distance apart and the **intact gap is the linkage** — no painted core
by default (a damage core is opt-in physics). The ribbon simply
encloses the whole stepping system, gaps included, at fault resolution.

The open engineering question is **parallel balance**: the ribbon
concentrates cells, so the partitioner must weight them. With split
surgery done in the local frame (rule 1), pair residency is automatic
and this reduces to ordinary cell-count weighting, uncoupled from any
fault-topology constraint.

## The 2-D rig

`~/+Simulations/place_route_health/line_mg.py`: 2-D box, interior line
cut, optional split + frictionless contact, painted weak / TI / lid
viscosity structures, V-junction fixture, ~2.5k cells, under a second
per warm solve. Every configuration reproduces with one flag. New
solver-design questions should be answered there first; 3-D is for
confirmation, not exploration.
