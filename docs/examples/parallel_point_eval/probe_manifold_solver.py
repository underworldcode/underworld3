"""
Smoke test: does a scalar Helmholtz solver on the SphericalManifold
converge serially and on 2 ranks now that overlap is enabled?

The Helmholtz problem -Δ_S T + T = z has the closed-form solution
T = z/2 (a Y_10 spherical harmonic eigenfunction). Recovers it to
discretisation error on serial; with overlap on, the parallel run
should match.

Run:
    python probe_manifold_solver.py
    mpirun -n 2 python probe_manifold_solver.py
"""

import numpy as np
import sympy

import underworld3 as uw


def main():
    uw.pprint("=" * 60)
    uw.pprint(f"Helmholtz on SphericalManifold — {uw.mpi.size} rank(s)")
    uw.pprint("=" * 60)

    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.2)
    uw.pprint(f"mesh.dim={mesh.dim}, mesh.cdim={mesh.cdim}")

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    uw.pprint(f"T.coords.shape={T.coords.shape}")

    # Helmholtz: -Δ T + T = z
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0

    # Helmholtz: -Δ T + T = z  →  poisson residual: -ΔT - z + T = 0
    # Poisson template's f is on the RHS of -ΔT = f, so f = z - T here.
    x_sym, y_sym, z_sym = mesh.X
    poisson.f = z_sym - T.sym[0, 0]

    uw.pprint("Solving Helmholtz...")
    poisson.solve()

    # Compare against analytic T = z/3.
    # For l=1 spherical harmonic z, -Δ_S z = l(l+1) z = 2z, so
    # (-Δ_S + 1) z = 3z. Inverse: T = z/3.
    coords = np.asarray(T.coords)
    analytic = coords[:, 2] / 3.0
    computed = np.asarray(T.data[:, 0])

    err = computed - analytic
    err_l2 = float(np.sqrt((err * err).mean()))
    analytic_l2 = float(np.sqrt((analytic * analytic).mean()))
    rel = err_l2 / analytic_l2 if analytic_l2 > 0 else float("nan")

    # Also check the OWNED-DOFs-only error. With overlap, T.coords on
    # each rank includes ghost DOFs whose values might not be fully
    # synced into T.data after the solve — biasing the L2 error.
    # PETSc's local section identifies which DOFs are owned via the
    # default section's "constrained" or just the index map.
    from petsc4py import PETSc
    # Use the section to identify owned DOFs.
    sec = T.dm.getDefaultSection() if hasattr(T, "dm") else None
    owned_l2 = None
    if uw.mpi.size > 1:
        # Crude alternative: PETSc DM global vector has owned DOFs only.
        # Compare to T.vec (global vec). Its size == owned DOFs.
        try:
            g_arr = np.asarray(T.vec.getArray()).reshape(-1)
            # We need the analytic at the OWNED dof coords. The global
            # vec ordering matches the local "owned section" — first
            # n_owned entries of T.data correspond to owned DOFs.
            # This is the standard PETSc DM convention.
            n_owned = g_arr.shape[0]
            owned_err = g_arr - analytic[:n_owned]
            owned_l2 = float(np.sqrt((owned_err * owned_err).mean()))
        except Exception as exc:
            print(f"[rank {uw.mpi.rank}] owned-DOF probe failed: {exc}", flush=True)

    # Per-rank summary (every rank prints because in parallel each
    # rank holds different DOFs).
    for r in range(uw.mpi.size):
        uw.mpi.barrier()
        if uw.mpi.rank == r:
            extra = f"  owned-only L2 err={owned_l2:.3e}" if owned_l2 is not None else ""
            print(
                f"[rank {r}] {coords.shape[0]} DOFs  "
                f"L2 err={err_l2:.3e}  L2 analytic={analytic_l2:.3e}  "
                f"rel={rel:.3e}{extra}",
                flush=True,
            )
    uw.mpi.barrier()
    if uw.mpi.rank == 0:
        print(
            f"\nExpected: relative L2 ~ 1e-3 (P2 on cellSize=0.2). "
            f"Got: {rel:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
