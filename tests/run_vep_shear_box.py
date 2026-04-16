"""Viscoelastic-plastic shear box — stress buildup with yield cap.

Maxwell VE material with Drucker-Prager yield stress under constant shear.

Phase 1 (elastic buildup): σ_xy follows the Maxwell curve
Phase 2 (yielded): σ_xy = τ_y (plastic cap)

Transition time: t_yield = -t_r ln(1 - τ_y/(η γ̇))
(only exists if η γ̇ > τ_y)
"""

import time as timer
import numpy as np
import sympy
import underworld3 as uw


def maxwell_stress_xy(t, eta, mu, gamma_dot):
    """Uncapped VE stress (for comparison)."""
    t_r = eta / mu
    return eta * gamma_dot * (1.0 - np.exp(-t / t_r))


def vep_stress_xy(t, eta, mu, gamma_dot, tau_y):
    """VEP stress: Maxwell buildup capped at τ_y."""
    ve = maxwell_stress_xy(t, eta, mu, gamma_dot)
    return np.minimum(ve, tau_y)


def run_vep_shear(order, n_steps, dt_over_tr, tau_y):
    """Run VEP shear box, return (times, num_stress, ana_stress)."""

    ETA, MU, V0, H, W = 1.0, 1.0, 0.5, 1.0, 2.0
    t_r = ETA / MU
    dt = dt_over_tr * t_r
    gamma_dot = 2.0 * V0 / H  # = 1.0

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

    stokes.add_dirichlet_bc((V0, 0.0), "Top")
    stokes.add_dirichlet_bc((-V0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-5

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_max_it"] = 50

    # Seed with linear shear
    v.data[:, 0] = v.coords[:, 1] * gamma_dot
    v.data[:, 1] = 0.0

    centre = np.array([[0.0, 0.0]])
    times, stress_num, stress_ana, stress_ve = [], [], [], []
    time_phys = 0.0

    for step in range(n_steps):
        t0 = timer.time()
        stokes.solve(zero_init_guess=False, evalf=False)
        solve_t = timer.time() - t0
        time_phys += dt

        val = uw.function.evaluate(stokes.tau.sym[0, 1], centre)
        sigma_xy = float(val.flatten()[0])

        ana = vep_stress_xy(time_phys, ETA, MU, gamma_dot, tau_y)
        ve_only = maxwell_stress_xy(time_phys, ETA, MU, gamma_dot)

        times.append(time_phys)
        stress_num.append(sigma_xy)
        stress_ana.append(ana)
        stress_ve.append(ve_only)

    del stokes, mesh
    return np.array(times), np.array(stress_num), np.array(stress_ana), np.array(stress_ve)


if __name__ == "__main__":
    ETA, MU = 1.0, 1.0
    gamma_dot = 1.0  # 2*V0/H
    t_r = ETA / MU

    # Choose τ_y below the viscous steady state (η·γ̇ = 1.0)
    TAU_Y = 0.5
    t_yield = -t_r * np.log(1 - TAU_Y / (ETA * gamma_dot))

    print(f"eta={ETA}, mu={MU}, gamma_dot={gamma_dot}")
    print(f"tau_y={TAU_Y}, viscous_limit={ETA*gamma_dot}")
    print(f"t_yield={t_yield:.4f} (stress reaches tau_y)")
    print()

    for order in [1, 2]:
        t0 = timer.time()
        t, num, ana, ve = run_vep_shear(order, n_steps=30, dt_over_tr=0.1, tau_y=TAU_Y)
        wall = timer.time() - t0

        print(f"Order {order} (wall={wall:.0f}s):")
        for i in range(len(t)):
            marker = " *" if t[i] > t_yield - 0.05 and t[i] < t_yield + 0.15 else ""
            print(f"  t={t[i]:.2f}  num={num[i]:.6f}  ana={ana[i]:.6f}  "
                  f"ve={ve[i]:.6f}  err={abs(num[i]-ana[i]):.4e}{marker}")

        # Error in post-yield regime
        yielded = t > t_yield + 0.2
        if np.any(yielded):
            post_err = np.max(np.abs(num[yielded] - ana[yielded]))
            print(f"  Max post-yield error: {post_err:.4e}")
        print()
