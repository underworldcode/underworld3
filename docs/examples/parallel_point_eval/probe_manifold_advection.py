"""
Probe: SLCN advection-diffusion on a SphericalManifold.

The original blocker for this PR — SL trace-back upstream coords go
through `uw.function.global_evaluate`, which had to work on manifold
meshes. Now that it does, this probe asks: can we actually run
`AdvDiffusionSLCN` on a sphere?

Setup:
    Solid-body rotation about z-axis: v = ω × r = (-y, x, 0). This is
    tangent to the unit sphere everywhere and divergence-free, so a
    scalar carried by the flow simply rotates around the pole.

    Initial T: a Gaussian on the equator at (1, 0, 0).

    After integration to t = 2π / ω = 2π (with ω = 1), the field
    should be back to its starting configuration (one full rotation).
    For a few steps just verify the solve runs, the field rotates in
    the expected direction, and mass is approximately conserved.

Run:
    python probe_manifold_advection.py
    mpirun -n 2 python probe_manifold_advection.py
"""

import sys
import traceback

import numpy as np
import sympy

import underworld3 as uw


def main():
    uw.pprint("=" * 60)
    uw.pprint(f"SLCN advection-diffusion on SphericalManifold — "
              f"{uw.mpi.size} rank(s)")
    uw.pprint("=" * 60)

    # ------------------------------------------------------------
    # Build the mesh + fields.
    # ------------------------------------------------------------
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3)
    uw.pprint(f"mesh.dim={mesh.dim}, mesh.cdim={mesh.cdim}, "
              f"_nav_dm={'set' if mesh._nav_dm is not None else 'None'}")

    # Scalar to advect, on the mesh.
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    # Velocity field as a sympy expression. Using a MeshVariable for V
    # is fragile on a manifold because P2 edge-midpoint DOFs sit on
    # the chord (interior of the sphere), not on the surface — the
    # MeshVariable's per-DOF (-y, x, 0) assignment produces meaningless
    # values there, polluting the FE-interpolated V used by the SL
    # trace-back. A sympy expression evaluates exactly at any coord.
    import os
    x_sym, y_sym, z_sym = mesh.X
    if os.environ.get("UW_NO_ADVECTION", "0") == "1":
        V_sym = sympy.Matrix([[sympy.sympify(0)] * mesh.cdim])
    else:
        V_sym = sympy.Matrix([[-y_sym, x_sym, sympy.sympify(0)]])

    # Initial T: Gaussian centred at (1, 0, 0) (equator).
    coords = np.asarray(T.coords)
    sigma = 0.3
    dx = coords[:, 0] - 1.0
    dy = coords[:, 1]
    dz = coords[:, 2]
    T.data[:, 0] = np.exp(-(dx**2 + dy**2 + dz**2) / (2 * sigma**2))

    uw.pprint(f"T.coords.shape={T.coords.shape}, V_sym={V_sym}")

    # ------------------------------------------------------------
    # Solver setup.
    # ------------------------------------------------------------
    try:
        adv = uw.systems.AdvDiffusionSLCN(
            mesh, u_Field=T, V_fn=V_sym, order=1,
        )
        adv.constitutive_model = uw.constitutive_models.DiffusionModel
        adv.constitutive_model.Parameters.diffusivity = 1.0e-4
        adv.f = sympy.Matrix.zeros(1, 1)
    except Exception as exc:  # noqa: BLE001
        uw.pprint(f"[FAIL] solver setup raised: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.stdout.flush()
        return

    # ------------------------------------------------------------
    # Sanity checks before stepping.
    # ------------------------------------------------------------
    T0 = np.asarray(T.data[:, 0]).copy()
    T0_max = float(T0.max())
    T0_int = float(T0.sum())  # crude mass proxy on per-rank-local DOFs

    uw.pprint(f"\nInitial T: max={T0_max:.4f}, sum={T0_int:.4f}")

    # ------------------------------------------------------------
    # Step a few times.
    # ------------------------------------------------------------
    n_steps = 8
    # ω = 1 → full rotation in t = 2π. Use a much smaller per-step
    # displacement so the SL trace-back chord is a small fraction of
    # the cell size. cellSize=0.3 corresponds to ~17° per cell on the
    # unit sphere; 0.05 rad/step (~3°) is well-resolved.
    dt = 0.05

    for step in range(n_steps):
        uw.pprint(f"\n--- step {step + 1}/{n_steps}  dt={dt:.4f} ---")
        try:
            adv.solve(timestep=dt)
        except Exception as exc:  # noqa: BLE001
            uw.pprint(f"[FAIL] solve raised at step {step}: "
                      f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            sys.stdout.flush()
            return

        Tn = np.asarray(T.data[:, 0])
        Tn_max = float(Tn.max())
        Tn_int = float(Tn.sum())
        Tn_min = float(Tn.min())
        # Where is the Gaussian's centre now? (max location in rank-local DOFs.)
        ix = int(np.argmax(Tn))
        peak = coords[ix]
        # Expected angle after step+1 rotations: phi_expected = (step+1)*dt.
        phi_expected = (step + 1) * dt
        # Initial peak was at (1, 0, 0) → after rotation by phi:
        # (cos phi, sin phi, 0).
        peak_expected = np.array([np.cos(phi_expected), np.sin(phi_expected), 0.0])

        for r in range(uw.mpi.size):
            uw.mpi.barrier()
            if uw.mpi.rank == r:
                print(
                    f"  [rank {r}] T: min={Tn_min:.4f}, max={Tn_max:.4f}, "
                    f"sum={Tn_int:.4f}, "
                    f"local-peak at {peak}",
                    flush=True,
                )
        uw.pprint(f"  Expected peak (rotation by {phi_expected:.4f}): {peak_expected}")

    uw.pprint("\nDone.")


if __name__ == "__main__":
    main()
