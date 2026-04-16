"""Run VE and VEP oscillatory shear from consistent initial conditions.

Both cases start from the analytical Maxwell solution at t0=0.01 to avoid
zero-velocity singularity and ensure identical starting stress.

Saves time series to stdout and generates plot directly.
"""

import time as timer
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression
import os


def maxwell_oscillatory(t, eta, mu, gamma_dot_0, omega):
    """Full analytical σ_xy for oscillatory Maxwell shear."""
    t_r = eta / mu
    De = omega * t_r
    prefactor = eta * gamma_dot_0 / (1.0 + De**2)
    return prefactor * (np.sin(omega * t) - De * np.cos(omega * t) + De * np.exp(-t / t_r))


def run_oscillatory(order, n_steps, dt, omega, V0, t0, tau_y=None):
    """Run oscillatory shear from analytical initial condition at t0."""

    ETA, MU, H, W = 1.0, 1.0, 1.0, 2.0
    gamma_dot_0 = 2.0 * V0 / H
    t_r = ETA / MU

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

    # Initialise velocity to match analytical solution at t0
    V_t0 = V0 * np.sin(omega * t0)
    gamma_dot_t0 = 2.0 * V_t0 / H
    v.data[:, 0] = v.coords[:, 1] * gamma_dot_t0
    v.data[:, 1] = 0.0

    # Initialise stress history to analytical at t0
    sigma_t0 = maxwell_oscillatory(t0, ETA, MU, gamma_dot_0, omega)
    # psi_star[0] is (N, dim, dim) sym tensor — set the xy component
    stokes.DFDt.psi_star[0].array[:, 0, 1] = sigma_t0
    stokes.DFDt.psi_star[0].array[:, 1, 0] = sigma_t0
    stokes.DFDt._history_initialised = True

    V_bc = expression(r"{V_{bc}}", V_t0, "Time-dependent boundary velocity")
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
    time_phys = t0

    label = "VEP" if tau_y is not None else "VE"

    for step in range(n_steps):
        time_phys += dt
        V_bc.sym = V0 * np.sin(omega * time_phys)

        stokes.solve(zero_init_guess=False, evalf=False)

        val = uw.function.evaluate(stokes.tau.sym[0, 1], centre)
        sigma_xy = float(val.flatten()[0])
        times.append(time_phys)
        stress_num.append(sigma_xy)

        print(f"  {label} step {step:3d}  t={time_phys:.3f}  sigma={sigma_xy:+.6f}", flush=True)

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
    t0 = 0.01
    n_periods = 2
    n_steps = int(n_periods * period / dt)

    ve_amp = ETA * gamma_dot_0 / np.sqrt(1.0 + De**2)
    sigma_t0 = maxwell_oscillatory(t0, ETA, MU, gamma_dot_0, omega)

    print(f"De={De}, omega={omega:.3f}, period={period:.3f}")
    print(f"VE amplitude={ve_amp:.4f}, tau_y={TAU_Y}")
    print(f"t0={t0}, sigma(t0)={sigma_t0:.6f}")
    print(f"dt={dt}, n_steps={n_steps}")
    print()

    # VE
    print("=== VE (order 1) ===")
    t0w = timer.time()
    t_ve, s_ve = run_oscillatory(1, n_steps, dt, omega, V0, t0, tau_y=None)
    print(f"  Wall: {timer.time()-t0w:.0f}s\n")

    # VEP
    print("=== VEP (order 1) ===")
    t0w = timer.time()
    t_vep, s_vep = run_oscillatory(1, n_steps, dt, omega, V0, t0, tau_y=TAU_Y)
    print(f"  Wall: {timer.time()-t0w:.0f}s\n")

    # Analytical
    t_ana = np.linspace(t0, t_ve[-1], 500)
    s_ana = maxwell_oscillatory(t_ana, ETA, MU, gamma_dot_0, omega)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(t_ana, s_ana, "k-", linewidth=0.8, alpha=0.5, label="VE analytical")
    ax.plot(t_ve, s_ve, "b-o", markersize=3, linewidth=1.2, label="VE numerical (order 1)")
    ax.plot(t_vep, s_vep, "r-s", markersize=3, linewidth=1.2, label=f"VEP numerical ($\\tau_y$={TAU_Y})")

    ax.axhline(TAU_Y, color="r", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(-TAU_Y, color="r", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(0.3, TAU_Y + 0.02, f"$\\tau_y$ = {TAU_Y}", color="r", fontsize=9, alpha=0.6)

    ax.set_xlabel("Time ($t / t_r$)")
    ax.set_ylabel("$\\sigma_{xy}$")
    ax.set_title(f"Oscillatory Maxwell VE vs VEP shear (De = {De})")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, t_ve[-1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/ve_vep_oscillatory.png", dpi=150)
    plt.savefig("output/ve_vep_oscillatory.pdf")
    print("Saved output/ve_vep_oscillatory.png and .pdf")
