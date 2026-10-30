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

$A_c$ is small and sparse: one column per piece and slip component, with
neighbouring pieces coupled. Its size is the fault's length (2-D) or area (3-D)
divided by the piece size. Every rank holds a copy and factors it.

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
decays, which is the band's half-width $w$, measured as the largest distance from
a band row to the nearest slip node. With a margin of $w$ the iterations are those
of the whole-patch extension at every piece count. The extension domain is
bounded by the band width rather than by the fault's length, so the cost per
piece does not grow with the fault.

The weights are Shepard weights. Each piece $q$ has the centre $c_q$ and radius
$R_q$ of its slip nodes, and

$$
\chi_q(x) = \frac{\phi_q(x)}{\sum_p \phi_p(x)}, \qquad
\phi_q(x) = \max\!\left(0,\, 1 - \frac{|x - c_q|}{2 R_q}\right)^2 .
$$

They need only the centres and radii, so the same construction serves a 3-D fault
surface and a network. The reach ($2R_q$ against $1.5R_q$) did not move the
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
- $Z$ takes the membership matrix's pattern by `MatDuplicate` before the
  transpose. In PETSc 3.25, duplicating a matrix whose transpose has been
  destroyed is a segfault, and duplicating it while the transpose is alive loses
  the pattern.
- The library reproduces the probe exactly on the linear fixture in serial: 79,
  81 and 85 iterations at 8, 16 and 32 pieces, with extension domains of at most
  7.8k, 6.1k and 5.2k rows (`~/+Simulations/fac_892/slip_coarse_lib`).

## Open

- 3-D: the cost of a band-wide extension per piece against one LU per rank, and
  whether `_segment_block`, which cuts along the leading principal direction only,
  should cut a fault surface into patches rather than strips.
- Whether the correction should become the default once the 3-D numbers are in.
  In 2-D one uncut block per rank is still the faster solve (`segment_sweep/README.md`).
- #894 (an interface law's Newton loop leaves the finest smoother on the iterate-0
  operator): the correction is set up from the smoother's own operator, so it
  inherits that staleness and does not add to it.
