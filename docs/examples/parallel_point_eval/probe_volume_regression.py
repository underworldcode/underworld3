"""
Regression probe: `uw.function.global_evaluate` on a regular volume
mesh, exercising the same code path as `probe_real_uw3_path.py` but
with `dim == cdim`. Tests both 2-D and 3-D box meshes.

Run:
    python probe_volume_regression.py
    mpirun -n 2 python probe_volume_regression.py
    mpirun -n 4 python probe_volume_regression.py

Pass: every query returns at FE-interp accuracy on every rank.
"""

import numpy as np

import underworld3 as uw


def analytic(c):
    """A field that's nontrivial in every dimension."""
    if c.shape[-1] == 2:
        return 0.3 * c[..., 0] + 0.5 * c[..., 1] ** 2
    return 0.3 * c[..., 0] + 0.5 * c[..., 1] ** 2 + c[..., 2]


def run_one(mesh, label: str) -> bool:
    uw.pprint("-" * 60)
    uw.pprint(f"{label}: mesh.dim={mesh.dim}, mesh.cdim={mesh.cdim}")

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    coords = np.asarray(T.coords)
    T.data[:, 0] = analytic(coords)

    rng = np.random.default_rng(7 + uw.mpi.rank)
    n_local = 30
    coords_arr = np.asarray(mesh.X.coords)
    lo_local = coords_arr.min(axis=0)
    hi_local = coords_arr.max(axis=0)
    from mpi4py import MPI
    lo = np.zeros(mesh.cdim)
    hi = np.zeros(mesh.cdim)
    MPI.COMM_WORLD.Allreduce(lo_local, lo, op=MPI.MIN)
    MPI.COMM_WORLD.Allreduce(hi_local, hi, op=MPI.MAX)
    span = hi - lo
    # Stay clear of the boundary by 5%.
    query = lo + 0.05 * span + 0.9 * span * rng.random(size=(n_local, mesh.cdim))

    expected = analytic(query)
    result = uw.function.global_evaluate(T.sym, query)
    actual = np.asarray(result).reshape(n_local)

    finite = np.isfinite(actual)
    err_max = (
        float(np.abs(actual[finite] - expected[finite]).max())
        if finite.any() else float("nan")
    )

    # FE-interp tolerance: P1 on cellSize=0.1 with a quadratic field.
    # Errors should be O(h²) ~ 0.01.
    tol = 0.05
    ok = finite.all() and err_max < tol

    # Print per-rank diagnostics.
    for r in range(uw.mpi.size):
        uw.mpi.barrier()
        if uw.mpi.rank == r:
            print(
                f"  [rank {r}] {n_local}/{n_local} finite, "
                f"max|err|={err_max:.3e}, ok={ok}",
                flush=True,
            )
    uw.mpi.barrier()
    return ok


def main():
    uw.pprint("=" * 60)
    uw.pprint(
        f"Volume-mesh regression — {uw.mpi.size} rank(s)"
    )
    uw.pprint("=" * 60)

    results = {}

    # 2D box
    mesh2 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.1,
    )
    results["2D box"] = run_one(mesh2, "2D box")

    # 3D box
    mesh3 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.2,
    )
    results["3D box"] = run_one(mesh3, "3D box")

    uw.pprint("-" * 60)
    if all(results.values()):
        uw.pprint("✅ all volume-mesh regressions pass")
    else:
        uw.pprint("❌ regression:")
        for label, ok in results.items():
            uw.pprint(f"   {label}: {'pass' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
