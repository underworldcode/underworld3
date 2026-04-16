"""Oscillatory VEP shear box — does the yield stress cap the oscillation?

Same setup as the VE oscillatory test but with a yield stress.
The VE amplitude at steady state is η γ̇₀/√(1+De²).
If τ_y is below this amplitude, the stress should be clipped.
"""

import time as timer
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


def maxwell_oscillatory(t, eta, mu, gamma_dot_0, omega):
    """Full analytical σ_xy for oscillatory Maxwell shear (incl. transient)."""
    t_r = eta / mu
    De = omega * t_r
    prefactor = eta * gamma_dot_0 / (1.0 + De**2)
    return prefactor * (np.sin(omega * t) - De * np.cos(omega * t) + De * np.exp(-t / t_r))


def run_vep_oscillatory(order, n_steps, dt_over_tr, De, tau_y):
    """Run oscillatory VEP shear box."""

    ETA, MU, H, W = 1.0, 1.0, 1.0, 2.0
    t_r = ETA / MU
    omega = De / t_r
    V0 = 0.5
    gamma_dot_0 = 2.0 * V0 / H
    dt = dt_over_tr * t_r

    # VE steady-state amplitude
    ve_amplitude = ETA * gamma_dot_0 / np.sqrt(1 + De**2)

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
    stokes.constitutive_model.Parameters.yield_stress = tau_y
    stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-6

    # Seed with linear shear
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
    times, stress_num, stress_ve = [], [], []
    time_phys = 0.0

    for step in range(n_steps):
        time_phys += dt
        V_t = V0 * np.sin(omega * time_phys)
        V_bc.sym = V_t

        t0s = __import__('time').time()
        stokes.solve(zero_init_guess=False, evalf=False)
        solve_t = __import__('time').time() - t0s

        val = uw.function.evaluate(stokes.tau.sym[0, 1], centre)
        sigma_xy = float(val.flatten()[0])
        ve_stress = maxwell_oscillatory(time_phys, ETA, MU, gamma_dot_0, omega)

        times.append(time_phys)
        stress_num.append(sigma_xy)
        stress_ve.append(ve_stress)
        print(f"  step {step:3d}  t={time_phys:.2f}  solve={solve_t:.1f}s  "
              f"vep={sigma_xy:+.6f}  ve={ve_stress:+.6f}", flush=True)

    del stokes, mesh
    return np.array(times), np.array(stress_num), np.array(stress_ve), ve_amplitude


if __name__ == "__main__":
    De = 1.0
    dt_ratio = 0.1
    n_periods = 2
    t_r = 1.0
    omega = De / t_r
    period = 2 * np.pi / omega
    n_steps = int(n_periods * period / (dt_ratio * t_r))

    ETA, gamma_dot_0 = 1.0, 1.0
    ve_amp = ETA * gamma_dot_0 / np.sqrt(1 + De**2)

    # Set τ_y below the VE amplitude so we see clipping
    TAU_Y = 0.4

    print(f"De={De}, VE amplitude={ve_amp:.4f}, tau_y={TAU_Y}")
    print(f"Expect clipping at ±{TAU_Y}")
    print(f"dt/t_r={dt_ratio}, n_steps={n_steps}")
    print()

    for order in [1]:
        t0 = timer.time()
        t, num, ve, amp = run_vep_oscillatory(order, n_steps, dt_ratio, De, TAU_Y)
        wall = timer.time() - t0

        print(f"Order {order} (wall={wall:.0f}s):")
        print(f"  VE amplitude: {amp:.4f}")
        print(f"  Max |σ_xy|:   {np.max(np.abs(num)):.6f}")
        print(f"  Min σ_xy:     {np.min(num):.6f}")
        print(f"  Max σ_xy:     {np.max(num):.6f}")
        print(f"  Clipped at τ_y={TAU_Y}? {np.max(np.abs(num)) <= TAU_Y + 0.01}")
        print()

        # Print last period
        last_period = t > (n_periods - 1) * period
        if np.any(last_period):
            idx = np.where(last_period)[0]
            sample = idx[::max(1, len(idx)//12)]
            print(f"  Last period samples:")
            for i in sample:
                clipped = " CLIPPED" if abs(num[i]) > TAU_Y - 0.01 else ""
                print(f"    t={t[i]:.3f}  vep={num[i]:+.6f}  ve={ve[i]:+.6f}{clipped}")
        print()
