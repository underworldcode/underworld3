"""
Viscoelastic shear box — analytical validation.

A Maxwell viscoelastic material under uniform simple shear has the exact
solution for the shear stress component:

    σ_xy(t) = η · γ̇ · (1 - exp(-t / t_r))

where:
    t_r = η/μ           Maxwell relaxation time
    γ̇  = dv_x/dy       engineering shear rate (velocity gradient)
    ε̇_xy = γ̇/2         tensor strain rate

The steady-state stress is σ_xy = η·γ̇ = 2η·ε̇_xy, the purely viscous limit.

Boundary conditions:
    Top / Bottom:  prescribed horizontal velocity ±V₀, zero vertical
    Left / Right:  free horizontal (outflow), zero vertical velocity

This avoids periodic BCs (untested) and corner singularities from free-slip.
The result is a spatially uniform simple shear — effectively a 1D problem.

The stress is read from psi_star[0] (the projected actual stress stored by
VE_Stokes after each solve), not recomputed from the constitutive formula.
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_3, pytest.mark.tier_b]


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

def maxwell_stress_xy(t, eta, mu, gamma_dot):
    """Exact σ_xy for constant-rate simple shear of a Maxwell material."""
    t_relax = eta / mu
    return eta * gamma_dot * (1.0 - np.exp(-t / t_relax))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_ve_shear(order, n_steps, dt_over_tr):
    """Run a VE shear-box, return (times, numerical_stress, analytical_stress)."""

    ETA = 1.0
    MU = 1.0
    V0 = 0.5
    H = 1.0
    W = 2.0

    t_relax = ETA / MU
    dt = dt_over_tr * t_relax
    gamma_dot = 2.0 * V0 / H

    res = 8
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(2 * res, res),
        minCoords=(-W / 2.0, -H / 2.0),
        maxCoords=(W / 2.0, H / 2.0),
    )

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.VE_Stokes(
        mesh, velocityField=v, pressureField=p, order=order, verbose=False,
    )

    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model.Parameters.shear_modulus = MU
    stokes.constitutive_model.Parameters.dt_elastic = dt
    # bdf_blend auto-detects: 1.0 for pure VE, 0.75 for VEP

    stokes.add_dirichlet_bc((V0, 0.0), "Top")
    stokes.add_dirichlet_bc((-V0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")

    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    times = []
    stress_num = []
    stress_ana = []
    centre = np.array([[0.0, 0.0]])

    time = 0.0
    for step in range(n_steps):
        stokes.solve(timestep=dt, zero_init_guess=False, evalf=False)
        time += dt

        # Read stress from solver.tau — the actual projected stress
        sigma_xy_val = uw.function.evaluate(
            stokes.tau.sym[0, 1], centre
        )
        sigma_xy = float(sigma_xy_val.flatten()[0])

        times.append(time)
        stress_num.append(sigma_xy)
        stress_ana.append(maxwell_stress_xy(time, ETA, MU, gamma_dot))

    del stokes, mesh

    return np.array(times), np.array(stress_num), np.array(stress_ana)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVEShearBox:
    """Viscoelastic shear box with analytical Maxwell solution."""

    def test_order1_converges(self):
        """Order-1 VE stress should track the analytical Maxwell curve."""

        times, num, ana = _run_ve_shear(order=1, n_steps=20, dt_over_tr=0.1)

        rel_error_final = abs(num[-1] - ana[-1]) / abs(ana[-1])
        mask = ana > 0.01 * ana[-1]
        rel_error_rms = np.sqrt(np.mean(((num[mask] - ana[mask]) / ana[mask]) ** 2))

        print(f"Order 1: final rel error = {rel_error_final:.4e}, "
              f"RMS rel error = {rel_error_rms:.4e}")

        assert rel_error_final < 0.05
        assert rel_error_rms < 0.06

    def test_order2_converges(self):
        """Order-2 VE stress should converge more accurately than order-1."""

        times, num, ana = _run_ve_shear(order=2, n_steps=20, dt_over_tr=0.1)

        rel_error_final = abs(num[-1] - ana[-1]) / abs(ana[-1])
        mask = ana > 0.01 * ana[-1]
        rel_error_rms = np.sqrt(np.mean(((num[mask] - ana[mask]) / ana[mask]) ** 2))

        print(f"Order 2: final rel error = {rel_error_final:.4e}, "
              f"RMS rel error = {rel_error_rms:.4e}")

        # Order-2 should be significantly better than order-1
        assert rel_error_final < 0.005
        assert rel_error_rms < 0.02

    def test_order2_better_than_order1(self):
        """Order 2 should have smaller error than order 1 at the same Δt."""

        _, num1, ana1 = _run_ve_shear(order=1, n_steps=20, dt_over_tr=0.1)
        _, num2, ana2 = _run_ve_shear(order=2, n_steps=20, dt_over_tr=0.1)

        mask = ana1 > 0.01 * ana1[-1]
        err1 = abs(num1[-1] - ana1[-1]) / ana1[-1]
        err2 = abs(num2[-1] - ana2[-1]) / ana2[-1]

        print(f"Final errors: Order 1 = {err1:.4e}, Order 2 = {err2:.4e}")

        assert err2 < err1, (
            f"Order 2 error ({err2:.4e}) should be less than order 1 ({err1:.4e})"
        )

    def test_steady_state_approach(self):
        """Stress should monotonically approach the viscous limit η·γ̇."""

        times, num, ana = _run_ve_shear(order=1, n_steps=30, dt_over_tr=0.1)

        final_err = abs(num[-1] - ana[-1]) / ana[-1]

        print(f"Steady state approach: σ_xy = {num[-1]:.6f}, "
              f"analytical = {ana[-1]:.6f}")

        diffs = np.diff(num)
        assert np.all(diffs >= -1.0e-10), "Stress should be monotonically increasing"
        assert final_err < 0.02
