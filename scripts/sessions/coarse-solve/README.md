# Coarse solve of the rotated free-slip custom FMG (hyperion, 2026-10-24)

Work in progress on `feature/coarse-solve-nullspace` (from
`bugfix/custom-mg-empty-rank`). Goal: a coarse solve that scales and needs no
sizing to np, in place of the dense SVD on one rank.

## Findings so far

- **No null space below the full operator.** The rotated path attaches the
  rigid-rotation mode to the full Stokes operator only. The velocity block
  `A_vv` carries no null space and no near-null space (`probe` in
  `bench_coarse_solve.py`), so the Galerkin coarse operators know nothing of
  the mode, and SVD is the only coarse PC that tolerates their singularity.
- **The mode restricts cleanly.** Least-squares restriction down the custom-P
  hierarchy, (P^T P) m_c = P^T m_f, gives a null vector of every level:
  ||A m|| / (||A||_F ||m||) = 4e-15 one level below the finest, 1.1e-7,
  8.7e-7, and 1.4e-5 on the coarsest (annulus, cell 0.2, four refinements,
  one fault strand, np 4).
- **Plain refined annulus gets GAMG on the rotated path.** Without a custom
  tail (set_custom_fmg or a placement), the rotated path records
  `rotated.velocity_pc: GAMG, reason unavailable` and ignores the mesh's own
  refinement hierarchy. Benchmarks here therefore place a fault strand.
- **The build's fallback is rank-local.** `build_transfers` catches a build
  exception per rank and retries (RBF) or falls back alone, while the other
  ranks are inside the barycentric build's collectives: MPI_ERR_BUFFER or a
  hang (seen at np 8 and np 32 on the faulted annulus; the kaiju signature).
- **Faulted annulus at np 8 (cell 0.8, four refinements)** still diverges
  inside the rotated solve after the empty-rank fixes: one rank skips a matrix
  assembly another performs. Reproducer: `repro_faulted_annulus_np8_test.py`.

## Partial sweep (`sweep-1-swap-ineffective.txt`)

Sweep 1's variant columns are void: the installed coarse PC was SVD in every
variant (checked after the solve). The prototype swapped the PC before the
bundle's options were removed from the database, and a later setFromOptions
restored SVD. Fixed in the next commit (the swap now runs after the cleanup and
keeps its own keys); MUMPS then installs as `lu mumps` and the redundant
variants as `redundant`. Only the SVD rows below are valid.

| np | coarse cells | coarse solve | V-cycles | coarse ms/cycle | linear solve s |
|---|---|---|---|---|---|
| 8 | 144 | svd | 435 | 1.83 | 2.48 |
| 16 | 144 | svd | 420 | 2.33 | 1.92 |
| 32 | 144 | svd | 450 | 2.92 | 2.12 |
| 8 | 612 | svd | 420 | 268.3 | 118.1 |
| 16 | 612 | svd | 435 | 285.9 | 127.4 |

What the SVD rows already show: at 612 coarse cells the coarse solve is 95% of
the linear solve, and it does not fall with more ranks (268 ms/cycle at np 8,
286 at np 16). At 144 cells it is about 35-60% and grows with np (1.8, 2.3,
2.9 ms/cycle at np 8, 16, 32). That matches the kaiju measurement.

## Files

- `bench_coarse_solve.py`, `bench_coarse_run.sh`, `bench_coarse_sweep.sh`: the
  benchmark (paths point at /mnt/data/lmoresi).
- `probe_bugB_annulus.py`, `probe_annulus_pc.py`, `measure_coarse_cells_per_rank.py`:
  probes used for the empty-rank and fallback findings.
- `repro_faulted_annulus_np8_test.py`: the np 8 solve divergence, kept out of
  the suite.
- The prototype is `_prototype_coarse_with_nullspace` in
  `src/underworld3/utilities/rotated_bc.py`, behind `UW_ROTATED_COARSE`
  (unset: no change).

## Sweep 2 (`sweep-2.txt`, 2026-10-25): the variants take effect

Faulted annulus (one strand), free slip on both boundaries, isoviscous, P2/P1,
three refinements above the coarsest level; velocity sub-solve FGMRES to
0.033 x tol, one V-cycle per iteration. "vs SVD" is the linear solve time
against the SVD row at the same np and coarse size.

| coarse cells | np | coarse solve | V-cycles | coarse ms/cycle | linear solve s | vs SVD |
|---|---|---|---|---|---|---|
| 144 | 8 | svd | 435 | 1.85 | 2.49 | 1.00 |
| 144 | 8 | svd_nsp | 450 | 0.36 | 1.81 | 0.73 |
| 144 | 8 | mumps | 450 | 1.18 | 2.24 | 0.90 |
| 144 | 8 | redundant_mumps | 450 | 0.19 | 1.76 | 0.71 |
| 144 | 8 | lu_shift | timed out (rc 124) | | | |
| 144 | 16 | svd | 420 | 2.27 | 1.86 | 1.00 |
| 144 | 16 | svd_nsp | 425 | 0.46 | 1.13 | 0.60 |
| 144 | 16 | mumps | 425 | 1.48 | 1.53 | 0.82 |
| 144 | 16 | redundant_mumps | 425 | 0.22 | 1.02 | 0.55 |
| 144 | 16 | lu_shift | 425 | 0.12 | 0.97 | 0.52 |
| 144 | 32 | svd | 450 | 3.16 | 2.24 | 1.00 |
| 144 | 32 | svd_nsp | 445 | 0.73 | 1.12 | 0.50 |
| 144 | 32 | mumps | 445 | 2.03 | 1.68 | 0.75 |
| 144 | 32 | redundant_mumps | 445 | 0.29 | 0.88 | 0.39 |
| 144 | 32 | lu_shift | 445 | 0.16 | 0.84 | 0.38 |
| 612 | 8 | svd | 420 | 270.19 | 118.70 | 1.00 |
| 612 | 8 | svd_nsp | 425 | 8.42 | 9.33 | 0.08 |
| 612 | 8 | mumps | 425 | 1.49 | 5.98 | 0.05 |
| 612 | 8 | redundant_mumps | 425 | 0.70 | 5.89 | 0.05 |
| 612 | 8 | lu_shift | 425 | 0.54 | 5.53 | 0.05 |
| 612 | 16 | svd | 435 | 287.03 | 127.93 | 1.00 |
| 612 | 16 | svd_nsp | 435 | 9.38 | 7.13 | 0.06 |
| 612 | 16 | mumps | 435 | 1.62 | 3.83 | 0.03 |
| 612 | 16 | redundant_mumps | 435 | 0.74 | 3.35 | 0.03 |
| 612 | 16 | lu_shift | 435 | 0.61 | 3.39 | 0.03 |
| 612 | 32 | svd | 420 | 416.26 | 176.75 | 1.00 |
| 612 | 32 | svd_nsp | 430 | 14.72 | 8.29 | 0.05 |
| 612 | 32 | mumps | 430 | 2.31 | 2.98 | 0.02 |
| 612 | 32 | redundant_mumps | 430 | 0.89 | 2.31 | 0.01 |
| 612 | 32 | lu_shift | 430 | 0.79 | 2.30 | 0.01 |

Read:
- Contraction is unchanged: every variant takes the SVD's V-cycle count
  (420-450). The restricted rotation mode is enough for a sparse direct coarse
  solve to replace the SVD with nothing lost in the cycle.
- The coarse solve stops mattering. At 612 coarse cells a redundant sparse LU is
  300-500x cheaper per cycle than the SVD and the linear solve 20-77x faster,
  and the time now FALLS with np (5.9, 3.4, 2.3 s at np 8, 16, 32) where the
  SVD's rose (119, 128, 177 s). That is the kaiju symptom removed.
- Redundant (gathered) MUMPS beats distributed MUMPS at these sizes: the coarse
  matrix is too small to distribute.
- Open: (1) lu_shift timed out once (np 8, 144 cells; log
  `sweep-2-np8-c144-lu_shift-timeout.log`), so the shift is not robust;
  redundant MUMPS with null-pivot detection ran clean in every case.
  (2) svd_nsp, the SAME SVD with only the null space attached and the PC reset,
  is 30x cheaper than the default SVD per cycle. A factorisation does not get
  cheaper from a null space, so the default path is doing extra work per apply
  (a re-setup, or a non-preonly coarse KSP); to be explained before any default
  changes.

## Correction to sweep 2 (2026-10-27)

Sweep 2's "linear solve" column is not a fair comparison. The default SVD
factors its coarse matrix lazily, inside the first coarse solve, so its
factorisation (110 s at 612 cells, np 8; `MGSetup Level 0`) lands in the
per-cycle and linear-solve times. The prototype variants replaced the coarse PC
AFTER `vel_pc.setUp()` had already built that SVD, so every variant paid the
110 s and then discarded it, and their own factorisation sat in `PCSetUp`,
outside the tabulated solve time. "SVD with the null space attached is 30x
cheaper per cycle" was this accounting, not a property of the SVD: applying
the factored SVD costs about 8 ms per cycle at 612 cells.

Whole-solve wall times from sweep 2 (factorisation included):

| coarse cells | np | svd | svd_nsp | mumps | redundant_mumps | lu_shift |
|---|---|---|---|---|---|---|
| 144 | 8 | 4.36 | 4.29 | 4.14 | 3.66 | (timed out) |
| 144 | 16 | 3.59 | 3.61 | 3.29 | 2.76 | 2.73 |
| 144 | 32 | 3.98 | 3.75 | 3.34 | 2.82 | 2.56 |
| 612 | 8 | 231.1 | 232.3 | 118.2 | 118.7 | 118.4 |
| 612 | 16 | 250.3 | 244.3 | 126.1 | 125.4 | 126.0 |
| 612 | 32 | 346.9 | 347.9 | 173.6 | 174.1 | 172.2 |

Two things follow. (1) The default SVD path factors the coarse SVD TWICE per
solve (110 s in `MGSetup Level 0`, then again in the first coarse solve,
`PCSetUp` count 2): a production inefficiency in its own right. (2) The sparse
variants' totals are one wasted SVD (about 110 s) plus about 8 s of their own;
without the wasted SVD the sparse coarse solve should be about 25-30x cheaper
than the SVD at 612 cells. Sweep 3 measures that directly: the coarse solve is
chosen BEFORE any setup (a `"mumps"` coarse option in the multigrid bundle:
redundant MUMPS LU with null-pivot detection), with and without the restricted
rotation null space attached.

## Sweep 3 (`sweep-3.txt`, 2026-10-27): the coarse solve chosen before setup

The rotated path's switch now selects the coarse option before any setup, so
nothing is factored and thrown away. `mumps`: redundant (gathered) MUMPS LU
with null-pivot detection (ICNTL(24)); `mumps_nsp`: the same with the
restricted rotation mode attached to the coarse operator. Whole-solve wall
time (factorisation included), coarse factorisations and V-cycles:

| coarse cells | np | svd | mumps | mumps_nsp | speed-up (mumps) | V-cycles svd / mumps |
|---|---|---|---|---|---|---|
| 144 | 8 | 4.33 | 2.74 | 2.85 | 1.6x | 435 / 435 |
| 144 | 16 | 3.64 | 1.86 | 2.01 | 2.0x | 420 / 420 |
| 144 | 32 | 3.77 | 1.75 | 1.59 | 2.2x | 450 / 450 |
| 612 | 8 | 231.3 | 8.13 | 8.49 | 28x | 420 / 420 |
| 612 | 16 | 245.3 | 5.07 | 5.13 | 48x | 435 / 435 |
| 612 | 32 | 364.3 | 3.68 | 3.80 | 99x | 420 / 420 |

Coarse factorisation: SVD 0.8-1.1 s at 144 cells and 110-178 s at 612 (one
rank, dense, O(n^3), and slower with np); MUMPS under 0.01 s at both.

Read:
- MUMPS with null-pivot detection replaces the SVD at the same V-cycle count
  everywhere, and its time FALLS with np where the SVD's rises: the coarse
  level no longer needs sizing to the rank count.
- The restricted null space is not needed for MUMPS: ICNTL(24) handles the
  singular coarse operator, and attaching the mode changes nothing measurable.
  (It would be needed for PETSc's own LU, which hits the zero pivot; the
  shifted-LU variant timed out once in sweep 2, so it is not a candidate.)

## Proposed design (for the hub and Louis)

1. Coarse solve on the rotated path when the velocity block carries verified
   rotation modes (where SVD is chosen today): `mumps`, i.e. redundant MUMPS LU
   with ICNTL(24), when PETSc has MUMPS; otherwise SVD, recorded in
   `solver.pc_fallbacks` as a degradation. A default change, so Louis decides
   with this table.
2. Keep the null-space restriction out of the library for now: measured
   unnecessary with MUMPS. Reintroduce only if a coarse solve without null-pivot
   handling is ever wanted.
3. Build failures stay collective (no rank-local fallback in parallel: a
   failure inside a collective stops with a clear message naming level, cells,
   cells per rank and np), as proposed earlier; the empty-rank fixes stay.
4. Agglomeration of small levels onto fewer ranks is no longer needed for the
   coarse solve (the redundant gather already does it for the coarsest level).
   It remains a candidate for the levels just above, if they show the
   few-cells-per-rank failures again.
