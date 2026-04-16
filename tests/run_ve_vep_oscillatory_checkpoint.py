"""Run VE and VEP oscillatory shear, save time series to .npz for plotting."""

import time as timer
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


def maxwell_oscillatory(t, eta, mu, gamma_dot_0, omega):
    """Full analytical σ_xy for oscillatory Maxwell shear."""
    t_r = eta / mu
    De = omega * t_r
    prefactor = eta * gamma_dot_0 / (1.0 + De**2)
    return prefactor * (np.sin(omega * t) - De * np.cos(omega * t) + De * np.exp(-t / t_r))


def run_oscillatory(order, n_steps, dt, omega, V0, tau_y=None):
    """Run oscillatory shear (VE or VEP depending on tau_y)."""

    ETA, MU, H, W = 1.0, 1.0, 1.0, 2.0
    gamma_dot_0 = 2.0 * V0 / H

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-W/2, -H/2), maxCoords=(W/2, H/2),
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=order)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model.Parameters.shear_modulus = MU
    stokes.constitutive_model.Parameters.dt_elastic = dt

    if tau_y is not None:
        stokes.constitutive_model.Parameters.yield_stress = tau_y
        stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-6

    v.data[:, 0] = v.coords[:, 1] * gamma_dot_0
    v.data[:, 1] = 0.0

    V_bc = expression(r"{V_{bc}}", 0.0, "Time-dependent boundary velocity")
    stokes.add_dirichlet_bc((V_bc, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_bc, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-5
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_max_it"] = 50

    centre = np.array([[0.0, 0.0]])
    times, stress_num = [], []
    time_phys = 0.0

    for step in range(n_steps):
        time_phys += dt
        V_bc.sym = V0 * np.sin(omega * time_phys)

        stokes.solve(zero_init_guess=False, evalf=False)

        val = uw.function.evaluate(stokes.tau.sym[0, 1], centre)
        sigma_xy = float(val.flatten()[0])
        times.append(time_phys)
        stress_num.append(sigma_xy)

        label = "VEP" if tau_y is not None else "VE"
        print(f"  {label} step {step:3d}  t={time_phys:.2f}  sigma={sigma_xy:+.6f}", flush=True)

    del stokes, mesh
    return np.array(times), np.array(stress_num)


if __name__ == "__main__":
    De = 1.0
    ETA, MU = 1.0, 1.0
    t_r = ETA / MU
    omega = De / t_r
    V0 = 0.5
    gamma_dot_0 = 2.0 * V0 / 1.0
    period = 2.0 * np.pi / omega
    TAU_Y = 0.4

    dt_ratio = 0.1
    dt = dt_ratio * t_r
    n_periods = 2
    n_steps = int(n_periods * period / dt)

    ve_amp = ETA * gamma_dot_0 / np.sqrt(1.0 + De**2)

    print(f"De={De}, omega={omega:.3f}, period={period:.3f}")
    print(f"VE amplitude={ve_amp:.4f}, tau_y={TAU_Y}")
    print(f"dt={dt}, n_steps={n_steps}")
    print()

    # VE (no yield)
    print("=== VE (order 1) ===")
    t0 = timer.time()
    t_ve, s_ve = run_oscillatory(1, n_steps, dt, omega, V0, tau_y=None)
    print(f"  Wall: {timer.time()-t0:.0f}s\n")

    # VEP (with yield)
    print("=== VEP (order 1) ===")
    t0 = timer.time()
    t_vep, s_vep = run_oscillatory(1, n_steps, dt, omega, V0, tau_y=TAU_Y)
    print(f"  Wall: {timer.time()-t0:.0f}s\n")

    # Analytical
    t_ana = np.linspace(dt, n_periods * period, 500)
    s_ana = maxwell_oscillatory(t_ana, ETA, MU, gamma_dot_0, omega)

    # Save
    outfile = "output/ve_vep_oscillatory.npz"
    import os
    os.makedirs("output", exist_ok=True)
    np.savez(
        outfile,
        t_ve=t_ve, s_ve=s_ve,
        t_vep=t_vep, s_vep=s_vep,
        t_ana=t_ana, s_ana=s_ana,
        De=De, tau_y=TAU_Y, ve_amp=ve_amp, omega=omega,
    )
    print(f"Saved to {outfile}")
