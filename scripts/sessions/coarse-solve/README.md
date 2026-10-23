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
