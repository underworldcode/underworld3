# Multigrid hierarchies for split-node faults

Technical note, 2026-10-13. Exploration, not yet in the library. Drivers:
`scripts/sessions/fault_mg/`.

## The question

A split-node fault lives on the finest mesh. The geometric-multigrid levels below it
come from the add-on hierarchy: the static background mesh, which is never cut, and
the add-on levels refined from it toward the fault. How the fault is represented on
those coarser levels decides whether the velocity solve stays mesh-independent as the
fault is refined.

## Alternatives

| | finest | add-on levels below it | background levels |
|---|---|---|---|
| **A** | split | welded, Galerkin | welded, Galerkin |
| **B2** | split | welded, TI weak band, rediscretised | Galerkin from the TI levels |
| **C** | split | split, Galerkin | welded, Galerkin |
| **D** | split | split, Galerkin | broad TI weak zone, rediscretised |

A is the current add-on behaviour (`mesh.adapt` then `add_fault`).

TI band: a Gaussian in the distance to the trace, half-width equal to the level's
near-fault cell size, director along the fault normal, $\eta_1 = \eta^{1-f}\eta_w^{f}$
with $\eta_w = 0.1\,\eta$ for a frictionless fault and $\eta_w = \eta_f w$ (the
compliance of the interface) for a viscous one.

Split levels (C, D) need two things the current hierarchy does not provide.

**A side-aware transfer.** For each fine node, one fine cell containing it is taken, and
that cell's centroid is located in the coarse mesh. A centroid is never on the fault, so
the coarse cell is on the node's own side. The coarse cell's Lagrange interpolant is
then evaluated at the node (dual-basis identity $W = BM^{-1}$, as in
`custom_mg.nested_refine_pair_prolongation`), so a node on the slit interpolates from
its own side. The node-cloud builders cannot do this: they triangulate the coarse
nodes, and coincident slit nodes make the transfer degenerate. The levels are not
nested near the trace, because each cut pulls vertices onto it, so a fine node can lie
slightly outside its located coarse cell. The evaluation point is therefore clamped
into the cell (its barycentric coordinates clipped and renormalised). Without the
clamp, extrapolated weights reached 5.4 near a tip, two coarse unknowns were reached
only through them, and the Galerkin operator had a near-zero diagonal that made SOR
return NaN. With it, every weight is at most 1.

**A per-level pair frame.** Each split level carries its own pair rotation (mean and
jump), and its opening (normal-jump) unknowns are removed:
$\hat P_l = Q_{l+1} P_l Q_l^{T}$, restricted to admissible unknowns. Coarse unknowns
that no fine unknown reaches at all (a few near the tips) are dropped too; they are null
modes of the Galerkin coarse operator.

## Test problem

- Box $[0,1]^2$ with simple shear on the walls; incompressible Stokes, $\eta = 1$.
- A straight fault, $(0.30, 0.45)$ to $(0.70, 0.55)$, frictionless or viscous ($\eta_f = 1$).
- Background: `UnstructuredSimplexBox(cellSize=0.1, refinement=1)` (levels $h = 0.1, 0.05$).
- Add-on level $j$: `base.adapt` to near-fault $h = 0.05/2^j$, built independently from the
  background for each $j$, so the levels are conforming but not nested; split by
  `add_fault` where the alternative asks.
- $k$ add-on levels in total, the finest at $h = 0.05/2^k$.
- Rotated fieldsplit solve: velocity block custom FMG, pressure Schur 1/μ mass.
- The measure is velocity-block iterations per solve.

## Results

Frictionless fault:

| $k$ | near-fault $h$ | A | B2 | C | D |
|---|---|---|---|---|---|
| 1 | 0.025 | 9 | 10 | 10 | 14 |
| 2 | 0.0125 | 13 | 14 | 7 | 9 |
| 3 | 0.00625 | 18 | 22 | 7 | 6 |
| 4 | 0.0031 | 23 | 30 | 6 | 6 |
| 5 | 0.0016 | 32 | 46 | 5 | 5 |

Viscous fault:

| $k$ | A | B2 | C | D |
|---|---|---|---|---|
| 1 | 9 | 9 | 9 | 17 |
| 2 | 12 | 19 | 7 | 9 |
| 3 | 17 | 53 | 6 | 8 |
| 4 | 22 | 200 (cap) | 6 | 7 |
| 5 | 30 | 200 (cap; stopped at 1 h) | 5 | 6 |

The peak slip agrees to five figures across all alternatives at each $k$, so only the
solver path changes.

Coulomb fault ($\mu = 0.6$, $\sigma_n = 0.8$; four Newton steps in every case),
velocity iterations per Newton step:

| $k$ | A | C |
|---|---|---|
| 1 | 5, 15, 17, 17 | 5, 15, 17, 17 |
| 3 | 9, 21, 24, 24 | 4, 14, 15, 16 |
| 5 | 15, 38, 45, 50 | 3, 13, 15, 15 |

Supporting measurements:

- **A single TI level.** One TI level directly below the split finest level, with
  Galerkin below it, is worse than A at every band strength. At $k = 4$ and a band
  half-width of half a cell, A takes 27; the band at 0.1, 0.03 and 0.01 takes 29, 46
  and 73. With a half-width of a full cell the band stalls.
- **Transfer alone.** C's levels left welded but with the side-aware transfer behave
  like A (35 iterations at $k = 5$). The gain comes from splitting the levels.
- **3-D extruded layer (preliminary).** Only the through-thickness levels split, the
  lateral levels welded: 11 to 16 iterations as $h$ halves, against 14 to 25 for A.

## A moving fault

The same box and fault, translated in 8 steps from $x = 0.3$ to $x = 0.7$. The
background is built once; each step re-adapts the add-on part from it with one
`mesh.adapt` call, splits the finest level and, for C and D, splits each intermediate
level of that same call. The table gives the mean seconds per step. The runs shared a
busy machine, so absolute times are good to a factor of 1.5 to 2; the ratios between
strategies and the iteration counts are not affected.

$k = 5$ (seven levels, near-fault $h = 0.0016$):

| | adapt | split finest | split levels | level frames | background TI | solve | total | velocity its |
|---|---|---|---|---|---|---|---|---|
| A | 3.1 | 1.9 | – | – | – | 137 | 143 | 30–34 |
| C | 2.7 | 1.7 | 2.8 | 1.0 | – | 38 | 46 | 5–6 |
| D | 4.7 | 2.9 | 5.1 | 1.8 | 2.4 | 61 | 79 | 4–5 |
| GAMG | 3.8 | 2.4 | – | – | – | 127 | 134 | 200 (cap) |

At $k = 3$ the totals are 27 (A), 15 (C), 17 (D) and 28 (GAMG) seconds per step. The
solve includes building the transfers: 2 s for A, 3.4 s for C. With the clamped
transfer, C takes 5 to 6 velocity iterations at every position at $k = 3$ and 4 to 8 at
$k = 5$.

The peak slip is identical across A, C and D at every position. Level frames come from
throwaway solvers in the drivers (`build_rotation`); they could be computed from the
pair geometry directly.

## Parallel

A, the existing path, had three defects at np > 1:

1. **Fixed** (`test_1086`). The rotated solve zeroed the opening rows of its fine
   transfer by local position instead of global row.
2. **Open.** `mesh.adapt` keeps a different intermediate level for the multigrid tail in
   parallel than in serial (1358 against 1520 cells; the finest mesh is the same).
3. **Open.** `add_fault` gathers the fault's cell star onto one rank, so the split finest
   level is not co-partitioned with its tail. The cross-partition transfer to it then
   left 618 coarse unknowns with no fine image. Their nearest-node injection duplicated
   512 columns, the Galerkin coarse operators became singular, and PETSc's inode SOR
   returned NaN. With inodes off it converges (34 iterations at np = 2 against 18
   serial), but the coarse operators are still singular.

For C, the plan is to co-locate each add-on level with the finest *before* splitting it,
so that every split level holds its fault star on the same rank, and to build the
side-aware transfer rank-locally on those co-located levels.

## What works best

**C: split every add-on level and leave the background welded.** The iteration count
falls slightly with refinement, for frictionless, viscous and Coulomb faults alike.
It needs nothing rediscretised and no TI parameters. With a moving fault its extra
surgery and transfers cost about 5 s per step against a solve that is three times
cheaper than A's at $k = 5$.

D matches C at depth but is worse on shallow hierarchies, and it adds a rediscretised
background operator to re-assemble at every step and band parameters to choose for
every problem, for no gain. B2 is worse than A, and badly so for the
viscous fault. A smeared band on a coarse level is a continuous shear that the split
fine operator rejects, so the coarse space has to contain the discontinuity itself.

## Not yet done

- Building the add-on levels from one `adapt` call (its NVB generations, each split)
  instead of $k$ independent ones.
- The 3-D layer with the lateral levels split too.
- Fault networks, and faults near a boundary.
- Parallel: see above.
- Moving the transfer and the per-level frames into `custom_mg` as a mesh-owned tail
  built by `add_fault`.
