# Diagnostic probes: the parallel add-on (2026-10-17)

Evidence for `docs/developer/design/parallel-addon-distribution-2026-10.md`. These are
diagnostics, kept as they were run, not model scripts: they read `sys.argv` and print
from rank 0 with mpi4py, which the Charter's section 11 rules out for session scripts
that are models. Run with `OMP_NUM_THREADS=1 mpirun -np N python <probe> ...`.

| probe | what it measures |
|---|---|
| `probe_scaling.py fault` | velocity iterations, transfer quality (partition of unity, coordinate reproduction) on the split refined patch box; `plain`, `wall`, `fault_xp` modes for comparison |
| `probe_builders.py` | the two node-cloud builders on the finest pair, with wrong-row dissection (the f21f3d85 defects) |
| `probe_coords_order.py`, `probe_slots.py` | ruled-out hypotheses of the same hunt: node order, native-pair slots |
| `probe_scaling_2d.py` | the 2-D analogue attempt |
| `probe_migration_sf.py` | a per-cell field through `distributeField` on the migration star forest (exact) |
| `probe_sf_exchange.py` | star-forest broadcast and reduce semantics in petsc4py |
| `probe_partition_2d.py [k]` | `partition_addon` invariants on the 2-D adapt-on-top fault |
| `probe_load_3d.py` | per-rank load of the split refined patch box after the partition |
| `probe_partition_cost.py [h] [k] [dim]` | wall time and cProfile of `partition_addon` on a larger adapt child |
| `probe_split_cost.py [dim]` | wall time and cProfile of a fault split |
| `check_topology_equivalence.py` | every function moved onto `_plex_topology` against its previous version (needs the old files in `/tmp/parallel-addon/old/`, `git show dd1d6eea:<file>`) |
| `check_transfer_assembly.py` | the vectorised node-transfer assembly against the per-row builders it replaced |
