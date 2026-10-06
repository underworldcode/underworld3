"""Viscoplastic shear box — yield stress validation.

For simple shear with constant strain rate:
  - Below yield: σ_xy = η · γ̇  (viscous)
  - At/above yield: σ_xy = τ_y  (plastic cap)

The transition occurs when η · γ̇ = τ_y, i.e. γ̇_crit = τ_y / η.

We test with a range of driving velocities and check the stress
transition matches the analytical prediction.
"""

import time as timer
import numpy as np
import sympy
import underworld3 as uw


def run_vp_shear(V0, eta, yield_stress):
    """Single-step VP shear solve, return σ_xy at centre."""

    H, W = 1.0, 2.0
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-W/2, -H/2), maxCoords=(W/2, H/2),
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.constitutive_model.Parameters.yield_stress = yield_stress
    stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-6

    stokes.add_dirichlet_bc((V0, 0.0), "Top")
    stokes.add_dirichlet_bc((-V0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-4

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_max_it"] = 50

    # Initialise with uniform shear to avoid singular Jacobian at zero strain rate
    gamma_dot = 2.0 * V0 / H
    v.data[:, 0] = v.coords[:, 1] * gamma_dot
    v.data[:, 1] = 0.0

    stokes.solve(zero_init_guess=False)

    # Read stress via tau
    centre = np.array([[0.0, 0.0]])
    val = uw.function.evaluate(stokes.tau.sym[0, 1], centre)
    sigma_xy = float(val.flatten()[0])

    converged = stokes.snes.getConvergedReason() > 0
    del stokes, mesh
    return sigma_xy, converged


if __name__ == "__main__":
    ETA = 1.0
    TAU_Y = 0.5
    H = 1.0

    # γ̇ = 2V0/H, σ_xy = η·γ̇ in viscous regime, σ_xy = τ_y when η·γ̇ ≥ τ_y
    gamma_dot_crit = TAU_Y / ETA
    print(f"eta={ETA}, tau_y={TAU_Y}, gamma_dot_crit={gamma_dot_crit}")
    print()

    V0_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0])

    print(f"{'V0':>6} {'gamma_dot':>10} {'sigma_xy':>10} {'analytical':>10} {'rel_err':>10} {'status':>8}")
    print("-" * 62)

    for V0 in V0_values:
        gamma_dot = 2.0 * V0 / H
        ana = min(ETA * gamma_dot, TAU_Y)

        t0 = timer.time()
        sigma_xy, converged = run_vp_shear(V0, ETA, TAU_Y)
        dt = timer.time() - t0

        rel_err = abs(sigma_xy - ana) / max(ana, 1e-10)
        status = "OK" if converged else "FAIL"
        print(f"{V0:6.3f} {gamma_dot:10.4f} {sigma_xy:10.6f} {ana:10.6f} {rel_err:10.3e} {status:>8}")

    print()
    print("Done")
