---
title: "Performance Optimization"
---

# Performance Optimization

```{note} Documentation In Progress
Performance tuning strategies and profiling techniques.
```

## Profiling

Tools for identifying bottlenecks:
- Python profilers
- PETSc logging
- Timing decorators

## Optimization Strategies

- Solver performance tuning
- Memory optimization
- Batch operations for better performance
- Parallel scaling

## Choosing MPI ranks for a MatMult-bound Stokes solve (#635)

A large nonlinear Stokes solve is usually **memory-bandwidth bound, not
compute bound**. In a controlled Gadi campaign on a 2-D power-law Barr–Houseman
fault benchmark (P2/P1, cell size 0.01, `n=3`, viscosity contrast 1e5, tolerance
1e-8), roughly **97% of solve time was inside PETSc `MatMult`**, and every
configuration ran the identical numerical workload — four SNES solves, 24 KSP
solves, the same mesh SHA-256.

That is the useful diagnostic: when the solver path is fixed and the runtime still
moves, you are looking at throughput, not convergence.

| queue / launch | ranks | Stokes solve (s) | peak memory | `MatMult` (Mflop/s) |
|---|---:|---:|---:|---:|
| `normal`, default placement | 8 | 4932 | 8.99 GB | 6707 |
| `normal`, explicit core binding | 8 | 5751 | 9.13 GB | 5738 |
| `normal`, binding + no-ML attempt | 8 | 5708 | 8.97 GB | 5782 |
| `normal`, binding + no-ML attempt | 4 | 8964 | 7.31 GB | 3842 |
| `normal`, binding + no-ML attempt | 12 | **3288** | 10.1 GB | **9995** |
| `normalsr` (Sapphire Rapids) | 8 | 3992 | 9.25 GB | 8308 |

What this supports:

- **Runtime tracks `MatMult` throughput**, not nonlinear behaviour. Twelve ranks gave
  the best wall time in this sample (56:02); four ranks cut peak memory to 7.3 GB but
  took 2:30:48.
- **Fewer ranks buy memory, not speed.** If a job is near a memory ceiling, dropping
  ranks is a legitimate trade — but expect the wall time to move roughly inversely.
- **Newer nodes help.** The `normalsr` 8-rank run beat the `normal` 8-rank run, though
  it was still slower than 12 ranks on `normal`.

```{warning}
**Explicit core binding did not help here**, and this campaign cannot say whether it
ever does. `--map-by core --bind-to core` was *slower* than default placement at 8
ranks (5751 s vs 4932 s), but the configurations ran concurrently on different nodes,
and a repeat of the same default-placement case varied by 8.6% on its own (4540 s vs
4932 s). Node-to-node variation is the same size as the effect. A controlled same-node
comparison, or a socket-distributed mapping, is needed before drawing any affinity
conclusion — do not read this table as a recommendation against binding.
```

**Harmless Open MPI noise.** Messages of the form

```text
[LOG_CAT_ML] component basesmuma is not available but requested in hierarchy
[LOG_CAT_ML] ml_discover_hierarchy exited with error
```

are **nonfatal**. Setting `OMPI_MCA_coll=^ml` neither suppressed them nor changed the
timings materially.

**Serialise your BLAS.** Every run above set one thread per rank, which is what you
want when MPI already owns the cores:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

**Still open.** A recommended Gadi rank placement for bandwidth-bound solves, whether
UW3's examples should ship explicit mapping/binding options, and a small repeatable
PETSc timing benchmark to accompany the setup notes — see #635.

## Related Documentation

- [Developer: Performance Guidelines](../developer/guidelines/performance-optimization.md)
