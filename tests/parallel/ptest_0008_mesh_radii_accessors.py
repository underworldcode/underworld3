"""MPI smoke test for the parallel-safe mesh radius accessors.

Run via mpi_runner.sh (mpirun -np N python ptest_0008_*.py).
Asserts:
  - min/mean/max return positive, ordered values on every rank
  - every rank sees IDENTICAL min, mean, max (parallel-consistent)
  - the methods are tagged as @uw.collective_operation
"""
import underworld3 as uw

mesh = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5, cellSize=1.0 / 16, qdegree=3)

rmin = mesh.get_min_radius()
rmean = mesh.get_mean_radius()
rmax = mesh.get_max_radius()

assert rmin > 0.0, f"rank {uw.mpi.rank}: min must be positive, got {rmin}"
assert rmin <= rmean <= rmax, (
    f"rank {uw.mpi.rank}: expected min <= mean <= max, got "
    f"{rmin} <= {rmean} <= {rmax}")

# All ranks must agree to within float-round-off (the values come from
# global allreduce / gather, so any disagreement would indicate the
# accessor leaked rank-local state).
all_min = uw.mpi.comm.allgather(rmin)
all_mean = uw.mpi.comm.allgather(rmean)
all_max = uw.mpi.comm.allgather(rmax)

tol = 1.0e-12
assert max(all_min) - min(all_min) < tol, (
    f"min_radius differs across ranks: {all_min}")
assert max(all_mean) - min(all_mean) < tol, (
    f"mean_radius differs across ranks: {all_mean}")
assert max(all_max) - min(all_max) < tol, (
    f"max_radius differs across ranks: {all_max}")

# Decorator must be present so any selective_ranks() misuse fails fast
for name in ("get_min_radius", "get_mean_radius", "get_max_radius"):
    fn = getattr(mesh, name)
    assert getattr(fn, "_is_collective", False), (
        f"rank {uw.mpi.rank}: {name} must be @uw.collective_operation")

if uw.mpi.rank == 0:
    print(f"OK: min={rmin:.6f}  mean={rmean:.6f}  max={rmax:.6f}  "
          f"agree across {uw.mpi.size} ranks", flush=True)
