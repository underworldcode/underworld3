"""Oscillatory VE shear box — time-harmonic analytical validation.

Maxwell material under sinusoidal boundary velocity V(t) = V0 sin(ωt).
Full solution including startup transient:

    σ_xy(t) = η γ̇₀/(1+De²) [sin(ωt) - De cos(ωt) + De exp(-t/t_r)]

where De = ω t_r is the Deborah number.

At steady state (t >> t_r), the transient dies out and the stress
oscillates with amplitude η γ̇₀/√(1+De²) and phase lag arctan(De).
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


def run_oscillatory(order, n_steps, dt_over_tr, De):
    """Run oscillatory VE shear box."""

    ETA, MU, H, W = 1.0, 1.0, 1.0, 2.0
    t_r = ETA / MU
    omega = De / t_r
    V0 = 0.5
    gamma_dot_0 = 2.0 * V0 / H
    dt = dt_over_tr * t_r

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

    # Use a UWexpression for time-dependent BCs — avoids JIT recompilation
    V_bc = expression(r"{V_{bc}}", 0.0, "Time-dependent boundary velocity")

    stokes.add_dirichlet_bc((V_bc, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_bc, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6

    centre = np.array([[0.0, 0.0]])
    times, stress_num, stress_ana = [], [], []
    time_phys = 0.0

    for step in range(n_steps):
        time_phys += dt

        # Update boundary velocity for this timestep
        V_t = V0 * np.sin(omega * time_phys)
        V_bc.sym = V_t

        stokes.solve(zero_init_guess=False, evalf=False)

        val = uw.function.evaluate(stokes.tau.sym[0, 1], centre)
        sigma_xy = float(val.flatten()[0])

        ana = maxwell_oscillatory(time_phys, ETA, MU, gamma_dot_0, omega)
        times.append(time_phys)
        stress_num.append(sigma_xy)
        stress_ana.append(ana)

    del stokes, mesh
    return np.array(times), np.array(stress_num), np.array(stress_ana)


if __name__ == "__main__":
    De = 1.0        # Deborah number (equal viscous and elastic timescales)
    dt_ratio = 0.1   # coarser for speed; finer (0.05) for publication quality
    n_periods = 3
    t_r = 1.0
    omega = De / t_r
    period = 2 * np.pi / omega
    n_steps = int(n_periods * period / (dt_ratio * t_r))

    print(f"De = {De}, omega = {omega:.3f}, period = {period:.3f}")
    print(f"dt/t_r = {dt_ratio}, n_steps = {n_steps}")
    print()

    for order in [1, 2]:
        t0 = timer.time()
        t, num, ana = run_oscillatory(order, n_steps, dt_ratio, De)
        wall = timer.time() - t0

        # Error over the last period (steady state)
        last_period = t > (n_periods - 1) * period
        if np.any(last_period):
            rms = np.sqrt(np.mean((num[last_period] - ana[last_period])**2))
            amp_ana = np.max(np.abs(ana[last_period]))
            rel_rms = rms / amp_ana if amp_ana > 0 else float('nan')
        else:
            rel_rms = float('nan')

        print(f"Order {order}: wall={wall:.0f}s, "
              f"last-period relative RMS = {rel_rms:.4e}")

        # Samples from the last period
        if np.any(last_period):
            idx = np.where(last_period)[0]
            sample = idx[::max(1, len(idx)//10)]
            for i in sample:
                err = abs(num[i] - ana[i])
                print(f"  t={t[i]:.3f}  num={num[i]:+.6f}  ana={ana[i]:+.6f}  err={err:.4e}")
        print()
