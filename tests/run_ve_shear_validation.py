"""Quick validation script for the VE shear box test.

Run with: pixi run -e amr-dev python tests/run_ve_shear_validation.py
"""

import time as timer
import numpy as np
import sympy
import underworld3 as uw


def maxwell_xy(t, eta, mu, gamma_dot):
    """Analytical σ_xy for Maxwell material under constant shear rate."""
    return eta * gamma_dot * (1.0 - np.exp(-t * mu / eta))


def run(order, n_steps, dt_ratio):
    ETA, MU, V0, H, W = 1.0, 1.0, 0.5, 1.0, 2.0
    t_relax = ETA / MU
    dt = dt_ratio * t_relax
    gamma_dot = 2.0 * V0 / H

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8),
        minCoords=(-W / 2, -H / 2),
        maxCoords=(W / 2, H / 2),
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=order)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model.Parameters.shear_modulus = MU
    stokes.constitutive_model.Parameters.dt_elastic = dt

    stokes.add_dirichlet_bc((V0, 0.0), "Top")
    stokes.add_dirichlet_bc((-V0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6

    Stress = uw.discretisation.MeshVariable(
        "Stress", mesh, (2, 2),
        vtype=uw.VarType.SYM_TENSOR, degree=2, continuous=True,
    )
    work = uw.discretisation.MeshVariable("W", mesh, 1, degree=2)
    sigma_proj = uw.systems.Tensor_Projection(mesh, tensor_Field=Stress, scalar_Field=work)

    times, num, ana = [], [], []
    time_phys = 0.0

    for step in range(n_steps):
        t0 = timer.time()
        stokes.solve(zero_init_guess=False, evalf=False)
        solve_time = timer.time() - t0
        time_phys += dt

        sigma_proj.uw_function = stokes.stress_deviator
        sigma_proj.solve()

        val = uw.function.evaluate(Stress.sym[0, 1], np.array([[0.0, 0.0]]))
        sigma_xy = float(val.flatten()[0])
        ana_val = maxwell_xy(time_phys, ETA, MU, gamma_dot)

        times.append(time_phys)
        num.append(sigma_xy)
        ana.append(ana_val)

        rel_err = abs(sigma_xy - ana_val) / max(ana_val, 1e-10)
        print(f"  step {step:3d}  t={time_phys:.2f}  solve={solve_time:.1f}s  "
              f"sigma_xy={sigma_xy:.6f}  analytical={ana_val:.6f}  rel_err={rel_err:.3e}")

    del stokes, mesh
    return np.array(times), np.array(num), np.array(ana)


if __name__ == "__main__":
    print("=== Order 1, dt/tr=0.1, 20 steps ===")
    t0 = timer.time()
    t, n, a = run(1, 20, 0.1)
    wall = timer.time() - t0
    mask = a > 0.01 * a[-1]
    rms = np.sqrt(np.mean(((n[mask] - a[mask]) / a[mask]) ** 2))
    final_err = abs(n[-1] - a[-1]) / a[-1]
    print(f"  Wall time: {wall:.0f}s")
    print(f"  Final rel error: {final_err:.4e}")
    print(f"  RMS rel error:   {rms:.4e}")
    print()

    print("=== Order 2, dt/tr=0.1, 20 steps ===")
    t0 = timer.time()
    t2, n2, a2 = run(2, 20, 0.1)
    wall2 = timer.time() - t0
    mask2 = a2 > 0.01 * a2[-1]
    rms2 = np.sqrt(np.mean(((n2[mask2] - a2[mask2]) / a2[mask2]) ** 2))
    final_err2 = abs(n2[-1] - a2[-1]) / a2[-1]
    print(f"  Wall time: {wall2:.0f}s")
    print(f"  Final rel error: {final_err2:.4e}")
    print(f"  RMS rel error:   {rms2:.4e}")
    print()

    viscous_limit = 1.0
    print(f"Steady-state check (order 1, t=2.0):")
    print(f"  sigma_xy = {n[-1]:.6f}, viscous limit = {viscous_limit:.6f}, "
          f"rel error = {abs(n[-1] - viscous_limit) / viscous_limit:.4e}")
    print()

    if rms2 < rms:
        print("PASS: Order 2 has smaller RMS error than order 1")
    else:
        print("FAIL: Order 2 should have smaller error than order 1")
