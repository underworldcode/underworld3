"""
Probe: SNES_MultiComponent_Projection on a SphericalManifold.

The architectural unlock for SLCN flux history on manifolds. Project
a known cdim=3-component expression onto a manifold MeshVariable and
verify the math comes out right. Block-diagonal projection, no
cross-component coupling — each component is an independent L2
projection (with optional smoothing) of the scalar target.

Target field: F_k(x,y,z) = -k·∇T  where T = z + 0.3·x (so
∇T = (0.3, 0, 1) and we expect F = -k·(0.3, 0, 1) at every DOF for
constant κ). Project onto a 3-component MeshVariable. Compare to
analytic per-component.

Run:
    python probe_multicomponent_projection.py
    mpirun -n 2 python probe_multicomponent_projection.py
"""

import numpy as np
import sympy

import underworld3 as uw


def main():
    uw.pprint("=" * 60)
    uw.pprint(f"SNES_MultiComponent_Projection on SphericalManifold — "
              f"{uw.mpi.size} rank(s)")
    uw.pprint("=" * 60)

    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3)
    uw.pprint(f"mesh.dim={mesh.dim}, mesh.cdim={mesh.cdim}")

    # Target: a (1, cdim) MATRIX mesh variable to hold the projected
    # flux. SNES_MultiComponent_Projection wants this shape.
    cdim = mesh.cdim
    F = uw.discretisation.MeshVariable(
        "Fproj", mesh, (1, cdim), vtype=uw.VarType.MATRIX, degree=2,
    )

    uw.pprint(f"F.coords.shape={F.coords.shape}")

    # Build the projection solver.
    proj = uw.systems.solvers.SNES_MultiComponent_Projection(
        mesh, u_Field=F, n_components=cdim, degree=2,
    )
    proj.smoothing = 0.0  # pure L2 projection

    # Project the constant vector (0.3, 0, 1) — equivalent to -∇T for
    # T = z + 0.3·x in the embedding space.
    target = sympy.Matrix([[sympy.Float(0.3), sympy.Float(0.0), sympy.Float(1.0)]])
    proj.uw_function = target

    uw.pprint("Solving projection...")
    try:
        proj.solve()
    except Exception as exc:  # noqa: BLE001
        uw.pprint(f"[FAIL] solve raised: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return

    # Compare per-component to analytic.
    F_arr = np.asarray(F.data)  # shape (n_dof, cdim)
    n_dof = F_arr.shape[0]
    analytic = np.array([0.3, 0.0, 1.0])

    err_per_comp = np.abs(F_arr - analytic).max(axis=0)
    for r in range(uw.mpi.size):
        uw.mpi.barrier()
        if uw.mpi.rank == r:
            print(
                f"[rank {r}] {n_dof} DOFs  "
                f"per-component max|err| = {err_per_comp.tolist()}",
                flush=True,
            )

    uw.mpi.barrier()
    if uw.mpi.rank == 0:
        if err_per_comp.max() < 1e-6:
            print("\nL2 projection of a constant vector recovered to "
                  "machine precision.")
        else:
            print(f"\nMax error across components: {err_per_comp.max():.3e}")


if __name__ == "__main__":
    main()
