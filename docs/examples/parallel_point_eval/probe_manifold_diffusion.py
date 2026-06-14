"""
Probe: time-dependent diffusion on a SphericalManifold.

The Laplace–Beltrami diffusion equation on the sphere:

    ∂T/∂t = κ Δ_S T

with no advection and no source. Time-stepped via backward Euler at
each step:

    (I/dt − κ Δ_S) T_new = T_old / dt

which is just a Helmholtz problem with α = 1/dt, source f = T_old / dt.

Uses `Poisson` (a SNES_Scalar subclass) — same solver that already
works on the manifold for steady Helmholtz. NO vector projection /
SL trace-back / SNES_Vector involved.

Initial condition: spherical-harmonic mode Y_10 = z. Analytic decay:
T(t) = z · exp(-l(l+1) κ t) = z · exp(-2 κ t).

Run:
    python probe_manifold_diffusion.py
    mpirun -n 2 python probe_manifold_diffusion.py
"""

import numpy as np
import sympy

import underworld3 as uw


def main():
    uw.pprint("=" * 60)
    uw.pprint(f"Time-dependent diffusion on SphericalManifold — "
              f"{uw.mpi.size} rank(s)")
    uw.pprint("=" * 60)

    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.2)
    uw.pprint(f"mesh.dim={mesh.dim}, mesh.cdim={mesh.cdim}")

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T_old = uw.discretisation.MeshVariable("T_old", mesh, 1, degree=2)

    # Initial condition: T(t=0) = z (the Y_10 spherical harmonic).
    coords = np.asarray(T.coords)
    T.data[:, 0] = coords[:, 2]
    T_old.data[:, 0] = coords[:, 2]

    kappa = 0.5
    dt = 0.05
    n_steps = 8

    # Backward-Euler diffusion:
    #   ∂T/∂t = κ Δ T  →  (T_new - T_old)/dt = κ Δ T_new
    #                  →  -κ Δ T_new + T_new/dt = T_old/dt
    # The Poisson template solves -∇·(κ∇T) = f, so the residual is
    #   -κ Δ T - f = 0. We want -κ Δ T - (T_old - T)/dt = 0, i.e.
    #   f = (T_old - T)/dt = -(T - T_old)/dt. Diffusivity is the κ that
    #   multiplies the Laplacian on the LHS; do NOT include κ in f.
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = kappa
    poisson.f = -(T.sym[0, 0] - T_old.sym[0, 0]) / dt

    uw.pprint(f"\nκ={kappa}, dt={dt}, n_steps={n_steps}")
    uw.pprint(f"Analytic decay factor per step: exp(-2·κ·dt) = "
              f"{np.exp(-2 * kappa * dt):.4f}")

    for step in range(n_steps):
        # Snapshot T_old before solving.
        T_old.data[:, 0] = T.data[:, 0].copy()
        # Solve for T_new.
        poisson.solve()
        # T.data now holds the new T.
        t_now = (step + 1) * dt

        # Project T onto the analytic Y_10 mode = z / norm(z).
        # On the unit sphere ∫ z² dA = 4π/3, so ‖z‖_L2 = √(4π/3).
        # Project: amplitude(t) = ⟨T(t), z⟩ / ‖z‖² (rank-local sum).
        Tn = np.asarray(T.data[:, 0])
        z_at_dof = coords[:, 2]
        amp_num = float((Tn * z_at_dof).sum())
        amp_den = float((z_at_dof * z_at_dof).sum())
        amp = amp_num / amp_den if amp_den != 0 else float("nan")
        analytic_amp = float(np.exp(-2 * kappa * t_now))

        # Crude per-DOF L2 of (T - analytic·z) on this rank.
        residual = Tn - analytic_amp * z_at_dof
        l2_err = float(np.sqrt((residual ** 2).mean()))

        for r in range(uw.mpi.size):
            uw.mpi.barrier()
            if uw.mpi.rank == r:
                print(
                    f"[rank {r}] step {step + 1}/{n_steps}  t={t_now:.3f}  "
                    f"amp(numeric)={amp:.4f}  amp(analytic)={analytic_amp:.4f}  "
                    f"l2(T - analytic)={l2_err:.3e}",
                    flush=True,
                )
        uw.mpi.barrier()


if __name__ == "__main__":
    main()
