"""
Probe: does `uw.function.global_evaluate` work on a SphericalManifold
(dim=2, cdim=3) in serial and parallel?

The investigation said: the cd-1 wall is pure UW3-side reshapes. The
user pushed back: "if you're using DMSWARM_BASIC then you are reproducing
what we already have and what does not work because of the cdim / dim
issue."

This probe runs the *actual* UW3 path on a real manifold mesh — first
unpatched, then with the proposed 8-line cdim plumbing — and reports
exactly which call fails and what it says. The point is to find the
real obstacle if there is one, or confirm that the 8-line patch is
sufficient.

Run:
    python probe_real_uw3_path.py
    mpirun -n 2 python probe_real_uw3_path.py
"""

import sys
import traceback
import numpy as np

import underworld3 as uw


def banner(msg: str) -> None:
    uw.pprint("=" * 70)
    uw.pprint(msg)
    uw.pprint("=" * 70)


def probe_step(label: str, fn):
    """Run fn() and report; on failure print the exception trace from
    every rank without aborting the script."""
    uw.mpi.barrier()
    uw.pprint(f"\n[{label}] starting...")
    try:
        result = fn()
        uw.pprint(f"[{label}] OK")
        return True, result
    except Exception as exc:  # noqa: BLE001
        # Print on every rank so we see whether failure is collective.
        print(
            f"[rank {uw.mpi.rank}] [{label}] FAILED: {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        sys.stdout.flush()
        return False, None


def main():
    banner(f"Probe — uw.function.global_evaluate on a manifold mesh "
           f"({uw.mpi.size} rank(s))")

    # --- 1. Build a SphericalManifold (dim=2, cdim=3) -----------------
    ok, mesh = probe_step(
        "build SphericalManifold",
        lambda: uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3),
    )
    if not ok:
        return

    uw.pprint(f"  mesh.dim={mesh.dim}, mesh.cdim={mesh.cdim}")
    uw.pprint(f"  local cell count: {mesh.dm.getStratumSize('depth', 2)}")

    # --- 2. Make a scalar MeshVariable and put a known field on it ----
    ok, T = probe_step(
        "create MeshVariable T",
        lambda: uw.discretisation.MeshVariable("T", mesh, 1, degree=1),
    )
    if not ok:
        return

    coords = np.asarray(T.coords)
    uw.pprint(f"  T.coords.shape={coords.shape}")
    # Analytic field: z + 0.3*x — easy to inspect.
    def analytic(c):
        return c[..., 2] + 0.3 * c[..., 0]
    T.data[:, 0] = analytic(coords)

    # --- 3. Build a batch of query coords on the sphere surface --------
    # Per-rank disjoint batch, all genuinely lying on r=1 (otherwise
    # cell-location can't find them).
    rng = np.random.default_rng(7 + uw.mpi.rank)
    n_local = 25
    theta = rng.uniform(0.1, np.pi - 0.1, size=n_local)
    phi = rng.uniform(0, 2 * np.pi, size=n_local)
    query = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    uw.pprint(f"  query.shape={query.shape}  cdim={mesh.cdim}")

    expected = analytic(query)

    # --- 4. Try rank-local uw.function.evaluate first ------------------
    ok, local_result = probe_step(
        "rank-local uw.function.evaluate(T.sym, query)",
        lambda: uw.function.evaluate(T.sym, query),
    )
    if ok:
        local_arr = np.asarray(local_result).reshape(n_local)
        # Many will be off-rank in parallel and either NaN or wrong; on
        # rank-0 in serial they should all match.
        if uw.mpi.size == 1:
            err = float(np.abs(local_arr - expected).max())
            uw.pprint(f"  rank-local serial max err: {err:.3e}")

    # --- 5. The headline test: global_evaluate on cdim coords ---------
    ok, global_result = probe_step(
        "uw.function.global_evaluate(T.sym, query)",
        lambda: uw.function.global_evaluate(T.sym, query),
    )
    if ok:
        global_arr = np.asarray(global_result).reshape(n_local)
        finite = np.isfinite(global_arr)
        n_finite = int(finite.sum())
        if n_finite > 0:
            err = float(np.abs(global_arr[finite] - expected[finite]).max())
        else:
            err = float("nan")
        # Per-point diagnostic on rank 1 to see which points are wrong.
        diffs = np.abs(global_arr - expected)
        worst = np.argsort(diffs)[::-1][:5]
        for r in range(uw.mpi.size):
            uw.mpi.barrier()
            if uw.mpi.rank == r:
                print(
                    f"[rank {r}] global_evaluate: "
                    f"{n_finite}/{n_local} finite, max|err|={err:.3e}",
                    flush=True,
                )
                print(f"[rank {r}] worst 5 indices: {worst}", flush=True)
                for i in worst:
                    print(
                        f"[rank {r}]   idx={i}  coord={query[i]}  "
                        f"expected={expected[i]:.4f}  got={global_arr[i]:.4f}",
                        flush=True,
                    )
    uw.mpi.barrier()


if __name__ == "__main__":
    main()
