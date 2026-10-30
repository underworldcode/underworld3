# A coarse space for fault slip makes the cut band smoother scale (2026-10)

Status: measured in 2-D on a probe, implemented as an opt-in on
`feature/fault-slip-coarse-space`. Louis, 2026-10-30: lock the idea in and
implement it while the fault work is in progress. Measurements:
`~/+Simulations/fac_892/segment_sweep` (`slip_coarse.py`, `run_slip.sh`,
`records_slip/`). Background: `fault-patch-multigrid-2026-08.md` (the FAC patch
smoother) and `fault-parallel-placement-2026-09.md` ("rows per block").

## Cutting a split fault's band block costs iterations in proportion to the cuts

On a split fault the finest level's FAC patch is the structural subdomain: the
rows the welded coarse levels cannot represent, which is the band around the
fault. It runs as one LU block per rank. `fac_block_rows` caps only the blocks of
a `fac_zone` mask, so the structural block grows with the fault: 15.9k rows in
serial on the linear tail-sweep fixture at $h_b/S = 24$, 19.7k rows per rank at
np 4 on the yielding-lid box. In 2-D one LU of that size is cheap. In 3-D the
band's rows grow with the fault's area, and the block has to be cut.

Cutting it numerically (the measurement knob `UW_FAC_SEGMENT_STRUCTURAL`) makes the
band solve one-level overlapping Schwarz. Each piece is solved exactly with
Dirichlet conditions at its cut faces, and nothing carries slip from one piece to
the next. The welded coarse levels cannot carry it either: an unsplit coarse space
has no slip. Velocity iterations then grow with the number of pieces:

| linear fixture, $h_b/S = 24$, serial, all applications | one block | 2 | 4 | 8 | 16 | 32 pieces |
|---|---|---|---|---|---|---|
| pieces additive (PCASM basic) | 67 | 85 | 94 | 110 | 148 | 180 |
| pieces multiplicative (in turn) | 67 | 71 | 72 | 78 | 92 | 122 |
| overlap 2 / 3 / 4 at 8 pieces | | | | 99 / 91 / 98 | | |

The parallel decomposition makes the same cut. One block per rank is one piece per
rank, and a fault that crosses $p$ ranks is cut $p - 1$ times whatever the cap.

## The method: a second Schwarz level whose unknowns are slip

The fix is the standard one for one-level Schwarz, a coarse space. Here its
unknowns are slip on the fault. Let $S$ be the slip rows: in the rotated frame
each split pair carries one jump-tangent row per slip component, on the Minus
point (`fault_contact._InterfaceAssembler`), one row in 2-D and two in 3-D. Let
$I$ be every other row of the band. Each coarse column $z_q$ prescribes slip
$\chi_q$ on $S$ and carries the field that slip induces in the band, the discrete
harmonic (energy-minimising) extension

$$
z_q\big|_S = \chi_q, \qquad
z_q\big|_{I_q} = -A_{I_q I_q}^{-1} A_{I_q S}\, \chi_q, \qquad
z_q = 0 \text{ outside } S \cup I_q ,
$$

as GDSW and BDDC coarse functions are built. The coarse problem and its correction
are

$$
A_c = Z^T A Z, \qquad y = Z A_c^{-1} Z^T r ,
$$

with $A$ the finest level's velocity operator. The correction is the third member
of the finest smoother's multiplicative composite: SOR over the whole level, then
the ASM pieces, then the slip correction. The coarse levels of the multigrid are
unchanged.

The extension is what makes the column work. A bare indicator of slip on a piece
is a dislocation with no field around it, which has high energy: the diagonal of
$A_c$ was about 93 for the indicator against 5.5 for the extended column, and at 8
pieces the bare indicator gave 110 iterations against 81.

## Parallel: one global coarse space over every piece, rank and segment

The coarse space spans the whole fault. Each rank contributes columns for its own
pieces, and $A_c$ is assembled over all of them. A piece on one rank is coupled
to its neighbour on the next rank through $A_c$, as two pieces on one rank are.
The rank boundaries need nothing of their own, because they are cuts of the same
kind. A fault network contributes the pieces of every segment, and the segments
are coupled through $A_c$ where they meet. A junction is not a special case.

$A_c$ is small and sparse: one column per piece, with neighbouring pieces
coupled. Its size is the fault's length divided by the piece size. Every rank
holds a copy and factors it.

In the parallel split as it stands, the add-on partition gathers each fault onto
one rank. A single fault's columns therefore sit on one rank, at any length: a
fault across 90% of the box, and one from corner to corner, were both gathered at
np 2 and np 4. Columns span ranks in two cases, and both converge to the answer
without the correction:

- a network whose faults land on different ranks: two separated faults put their
  columns on ranks [2, 3] at np 2 and [2, 0, 0, 3] at np 4;
- a fault split through the partition seams (`FaultNetwork.build(seams="conform",
  realisation="split")`, ptest_0865's long fault): [1, 1] at np 2 with one block
  per rank, [7, 4] at np 2 and [5, 6, 0] at np 3 with 300-row pieces, peak slip
  0.5094, the gathered answer.

## The partition of unity, and how far each column extends

Which rows each column's extension may use is a choice between cost and quality.
The extension over each rank's whole band patch is the reference: one
factorisation the size of the uncut block, which is the cost we are cutting away.
Measured on the linear fixture, $h_b/S = 24$, serial, all applications (`records_slip/`):

| coarse space | extension domain | 8 | 16 | 32 pieces | extension rows, largest |
|---|---|---|---|---|---|
| none | | 110 | 148 | 180 | |
| slip indicator, not extended | | 110 | | | |
| indicator per piece | the rank's whole patch | 81 | 82 | 85 | 14.9k |
| indicator per piece, other slip held at zero | the piece + 1 ring | 90 | 117 | 152 | 2.1k |
| the same | the piece + 2 rings | 78 | 92 | 119 | 2.5k |
| indicator per piece, other slip free | the piece + 1 ring | 94 | 121 | 150 | 2.1k |
| partition of unity | the pieces its support touches | 79 | 83 | 104 | 5.6k |
| partition of unity | its support + 1 band half-width | 79 | 81 | 85 | 7.8k |

A column of indicator slip that stops at a piece's edge puts a step in slip there,
which is again a dislocation, and the cost grows with the number of steps. A
smooth partition of unity removes the step: weights $\chi_q$ on the slip rows that
sum to one, each falling to zero over the next piece. The coarse space is then a
coarse finite-element discretisation of slip on the fault, lifted into the band.
Its support must still be extended over the distance on which the field of a slip
decays, which is the band's half-width $w$. With a margin of $w$ the iterations are
those of the whole-patch extension at every piece count. The library measures $w_q$
per piece, as the largest distance from the piece's rows to the nearest slip node.
A single global value is wrong in parallel: a rank can hold structural rows far
from any fault, and at np 4 the global value made every extension cover the whole
band (17.5k rows where serial needs 7.8k). The extension domain is
bounded by the band width rather than by the fault's length, so the cost per
piece does not grow with the fault.

The weights are Shepard weights. Each piece $q$ has the centre $c_q$ and radius
$R_q$ of its slip nodes, and

$$
\chi_q(x) = \frac{\phi_q(x)}{\sum_p \phi_p(x)}, \qquad
\phi_q(x) = \max\!\left(0,\, 1 - \frac{|x - c_q|}{2 R_q}\right)^2 .
$$

They need only the centres and radii, so the same weights serve a network and a
3-D fault surface, although a 3-D column needs more than weights (see Open). The reach ($2R_q$ against $1.5R_q$) did not move the
iteration count (79/83/104 against 78/84/105 with the touched-pieces domain).
Adding a second column per piece, slip varying linearly along strike, bought
3–5 iterations with the whole-patch extension and was not carried forward.

## The implementation

- `set_custom_fmg(..., slip_coarse_space=True)` cuts the structural block at
  `fac_block_rows` and adds the slip correction to the finest level. The default
  is unchanged: one block per rank and no slip correction.
- The correction is a PETSc python PC, added to the finest smoother's composite
  after the FAC blocks are set up (`custom_mg._set_up_fac_smoothers`), on the
  rotated saddle-point path, which is where the slip rows exist.
- Its set-up runs whenever the operator's values change, which is every Newton
  iterate. It rebuilds the extension and $A_c$ from the current operator. A
  column extended with an old operator would still be a valid coarse function
  but a weaker one.
- Every collective call is made on every rank, including the ranks that hold no
  slip rows and so no columns. In the probe, a rank whose pieces held no slip
  skipped the collective submatrix extraction, and the other three ranks waited on
  it for two and a half hours. In the first library build, a rank with band rows
  and no slip rows raised an index error between two collectives, and np 2 hung
  the same way; `tests/parallel/test_1091` runs at np 2 and 4, where such ranks
  exist.
- The extension domains cross rank boundaries. Each rank owns its pieces'
  columns, and the band rows near a piece may belong to another rank. The rows
  are gathered through the matrix transpose of a membership matrix (band row
  against piece), and the operator blocks through `MatCreateSubMatrices`.
- $A_c = Z^T A Z$ is a PETSc `PtAP`. Each rank holds a redundant copy and factors
  it with LU.
- The membership matrix's transpose is taken into a new matrix,
  `M.transpose(PETSc.Mat())`: petsc4py's `transpose()` with no argument
  transposes in place.
- The slip rows are those the rotation carries: the registered contact faults'
  pairs, without the pairs that have a constrained DOF
  (`fault_contact._fault_slip_rows`, which shares the slip-DOF rule with the
  interface assembler).
- 2-D only. The standard solve path and the rotated vector solve refuse the
  flag; they have no slip rows.

## Measured with the library (`~/+Simulations/fac_892/slip_coarse_lib`)

| case | one block per rank | cut, no correction | cut + slip correction |
|---|---|---|---|
| linear, $h_b/S = 24$, serial, 8 / 16 / 32 pieces, all applications | 67 | 110 / 148 / 180 | 79 / 82 / 87 |
| linear, np 4, 8 / 16 / 32 columns | 70 | 125 / 158 / (not run) | 84 / 84 / 92 |
| yielding-lid box, np 4, $h_b/S = 15.5$, cap 2000: per Newton increment, total | 7, 192 | 11–12, 278 | 9, 239 |

The extension domains are at most 7.4k, 5.7k and 4.5k rows in serial and 10.2k,
8.3k and 7.4k at np 4. On the linear fixture the correction removes the growth
with the piece count and leaves an offset of 12–14 iterations over one block. On
the box it recovers about two thirds of the cut's cost per increment. With one
block per rank and the correction (one column, the fault on one rank) the box
takes 188 iterations, 7 per increment.

## Open

- 3-D needs one column per slip direction, and the direction must be smooth
  along the fault. The node-by-node tangent frame (`fault_contact._tangent_frame`
  picks its first axis from the smallest normal component) switches axis on a
  curved fault, so a column per frame component would prescribe a slip
  direction that jumps. A candidate is a Cartesian slip direction projected onto
  each node's tangent plane. The library refuses 3-D until this is built.
- 3-D cost: a band-wide extension per piece against one LU per rank, and whether
  `_segment_block`, which cuts along the leading principal direction only,
  should cut a fault surface into patches rather than strips.
- The residual offset over one block (12–14 iterations on the linear fixture, 2
  per increment on the box). A linear-in-strike column per piece bought 3–5 with
  the whole-patch extension.
- A test with a single fault crossing ranks under load: the seam-conforming
  fixture converges in 4–5 iterations with or without the correction, too easy to
  separate them.
- #896: at np 2 the parallel cover gate declines the structural patch on the
  linear fixture at $h_b/S = 24$ (640 iterations instead of 68), and the flag
  then refuses.
- Whether the correction should become the default once the 3-D numbers are in.
  In 2-D one uncut block per rank is still the faster solve (`segment_sweep/README.md`).
- #894 (an interface law's Newton loop leaves the finest smoother on the iterate-0
  operator): the correction is set up from the smoother's own operator, so it
  inherits that staleness and does not add to it.
