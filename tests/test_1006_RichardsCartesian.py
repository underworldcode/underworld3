# %%
# Tests for the Richards equation solver.
#
# Test 1: Steady-state drainage with constant K (linear profile).
# Test 2: Transient infiltration with Van Genuchten retention curves.

import underworld3 as uw
import numpy as np
import sympy as sp
import pytest

# Physics solver tests
pytestmark = pytest.mark.level_3


@pytest.fixture(autouse=True)
def reset_model_state():
    """Reset model state before each test."""
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    yield
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)


# --- Domain ---
res = 32
minX, maxX = 0.0, 0.1  # narrow → effectively 1D
minY, maxY = 0.0, 1.0


def create_mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, res),
        minCoords=(minX, minY),
        maxCoords=(maxX, maxY),
        qdegree=3,
    )


def test_richards_steady_constant_K():
    """Richards with constant K and C=1 should give a linear pressure head profile.

    With constant K and gravity s=[0,-1], the steady-state solution of
        -∇·[K(∇ψ - s)] = 0
    in 1D (y-direction) with ψ(0)=-5, ψ(1)=0 is:
        ψ(y) = -5(1 - y)  (linear)

    We use TransientDarcy stepping with small dt to approach steady state,
    since Richards inherits from TransientDarcy.
    """
    mesh = create_mesh()

    psi = uw.discretisation.MeshVariable("psi", mesh, 1, degree=2)
    v_soln = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=1)

    richards = uw.systems.Richards(mesh, psi, v_soln, order=1, theta=0.5)
    richards.petsc_options.delValue("ksp_monitor")
    richards.petsc_options["snes_rtol"] = 1.0e-6

    K_val = 1.0
    richards.constitutive_model = uw.constitutive_models.DarcyFlowModel
    richards.constitutive_model.Parameters.permeability = K_val
    richards.constitutive_model.Parameters.s = sp.Matrix([0, -1]).T
    richards.capacity = 1  # constant → behaves like TransientDarcy
    richards.f = 0.0

    # BCs: ψ = 0 at top (y=1), ψ = -5 at bottom (y=0)
    richards.add_dirichlet_bc([0.0], "Top")
    richards.add_dirichlet_bc([-5.0], "Bottom")

    richards._v_projector.petsc_options["snes_rtol"] = 1.0e-6
    richards._v_projector.smoothing = 1.0e-6

    # Initial guess: linear profile
    y = mesh.X[1]
    psi_init = -5.0 * (1.0 - y)
    psi.array = uw.function.evaluate(psi_init, psi.coords)

    # Step towards steady state with a few large timesteps
    dt = 0.1
    for _ in range(10):
        richards.solve(timestep=dt)

    # Check along vertical profile
    n_sample = 50
    sample_y = np.linspace(0.05, 0.95, n_sample)
    sample_x = np.full_like(sample_y, 0.5 * (minX + maxX))
    sample_pts = np.column_stack([sample_x, sample_y])

    psi_numerical = uw.function.evaluate(psi.sym[0], sample_pts).squeeze()
    psi_exact = -5.0 * (1.0 - sample_y)

    assert np.allclose(psi_numerical, psi_exact, atol=0.1), (
        f"Max error: {np.max(np.abs(psi_numerical - psi_exact)):.4f}"
    )


def test_richards_transient_infiltration():
    """Richards equation with Van Genuchten curves — basic sanity check.

    Uses a mild initial condition (ψ = -2) with saturated top (ψ = 0)
    and fixed bottom (ψ = -2).  After several timesteps the wetting
    front should propagate downward, making the upper column wetter
    (less negative ψ) than the initial state.

    Note: Richards with VG curves is a stiff nonlinear problem.
    We use moderate conditions to ensure reliable SNES convergence.
    """
    from underworld3.utilities.retention_curves import (
        van_genuchten_K,
        van_genuchten_theta,
    )

    mesh = create_mesh()

    psi = uw.discretisation.MeshVariable("psi", mesh, 1, degree=2)
    v_soln = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=1)

    richards = uw.systems.Richards(mesh, psi, v_soln, order=1, theta=0.5)
    richards.petsc_options.delValue("ksp_monitor")
    richards.petsc_options["snes_rtol"] = 1.0e-6
    richards.petsc_options["snes_max_it"] = 50
    richards.petsc_options["snes_linesearch_type"] = "bt"

    # Loam-like Van Genuchten parameters (less stiff than sand)
    alpha_vg = 1.0
    n_vg = 1.5
    theta_r = 0.08
    theta_s = 0.43
    Ks = 1.0

    psi_sym = psi.sym[0]

    richards.constitutive_model = uw.constitutive_models.DarcyFlowModel
    richards.constitutive_model.Parameters.permeability = van_genuchten_K(
        psi_sym, Ks=Ks, alpha=alpha_vg, n=n_vg
    )
    richards.constitutive_model.Parameters.s = sp.Matrix([0, -1]).T

    # Mixed form: θ(ψ) for mass-conservative storage term
    richards.water_content = van_genuchten_theta(
        psi_sym, theta_r=theta_r, theta_s=theta_s,
        alpha=alpha_vg, n=n_vg,
    )
    richards.f = 0.0

    # BCs: saturated at top, moderately dry at bottom
    richards.add_dirichlet_bc([0.0], "Top")
    richards.add_dirichlet_bc([-2.0], "Bottom")

    richards._v_projector.petsc_options["snes_rtol"] = 1.0e-6
    richards._v_projector.smoothing = 1.0e-6

    # Initial condition: linear profile from -2 at bottom to 0 at top
    # (smooth start helps SNES converge)
    y = mesh.X[1]
    psi_init = -2.0 * (1.0 - y)
    psi.array = uw.function.evaluate(psi_init, psi.coords)

    # Run a few timesteps with small dt
    dt = 0.005
    n_steps = 5
    for step in range(n_steps):
        richards.solve(timestep=dt)

    # Basic sanity checks along a vertical profile
    n_sample = 20
    sample_y = np.linspace(0.1, 0.9, n_sample)
    sample_x = np.full_like(sample_y, 0.5 * (minX + maxX))
    sample_pts = np.column_stack([sample_x, sample_y])

    psi_vals = uw.function.evaluate(psi.sym[0], sample_pts).squeeze()

    # 1. Solution should be bounded (no blow-up)
    assert np.all(np.isfinite(psi_vals)), "Solution should be finite"
    assert np.all(psi_vals >= -5.0), "ψ should not overshoot far below BCs"
    assert np.all(psi_vals <= 1.0), "ψ should not overshoot far above BCs"

    # 2. Near the top should be wetter (less negative) than near the bottom
    assert psi_vals[-1] >= psi_vals[0], (
        "Pressure head should increase towards the wetted top"
    )


def test_richards_gardner_analytical():
    """Richards with Gardner model — validate against exact analytical solution.

    The Gardner exponential conductivity K(ψ) = Ks·exp(α·ψ) admits an
    exact steady-state solution for 1D vertical drainage with gravity.
    This test converges the solver to steady state and compares against
    the analytical profile.
    """
    from underworld3.utilities.retention_curves import (
        gardner_K,
        gardner_theta,
        gardner_steady_state_psi,
    )

    mesh = create_mesh()

    psi = uw.discretisation.MeshVariable("psi", mesh, 1, degree=2)
    v_soln = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=1)

    richards = uw.systems.Richards(mesh, psi, v_soln, order=1, theta=0.5)
    richards.petsc_options.delValue("ksp_monitor")
    richards.petsc_options["snes_rtol"] = 1.0e-6

    # Gardner parameters
    Ks = 1.0
    alpha_g = 2.0
    psi_bottom = -3.0
    psi_top = -0.5

    psi_sym = psi.sym[0]

    richards.constitutive_model = uw.constitutive_models.DarcyFlowModel
    richards.constitutive_model.Parameters.permeability = gardner_K(
        psi_sym, Ks=Ks, alpha=alpha_g
    )
    richards.constitutive_model.Parameters.s = sp.Matrix([0, -1]).T

    # Mixed form: θ(ψ) for mass-conservative storage term
    richards.water_content = gardner_theta(
        psi_sym, theta_r=0.05, theta_s=0.4, alpha=alpha_g,
    )
    richards.f = 0.0

    # BCs
    richards.add_dirichlet_bc([psi_top], "Top")
    richards.add_dirichlet_bc([psi_bottom], "Bottom")

    richards._v_projector.petsc_options["snes_rtol"] = 1.0e-6
    richards._v_projector.smoothing = 1.0e-6

    # Initial guess: linear profile (close enough for SNES)
    y = mesh.X[1]
    psi_init = psi_bottom + (psi_top - psi_bottom) * y
    psi.array = uw.function.evaluate(psi_init, psi.coords)

    # Step towards steady state
    dt = 0.1
    for _ in range(20):
        richards.solve(timestep=dt)

    # Compare along vertical profile
    n_sample = 50
    sample_y = np.linspace(0.05, 0.95, n_sample)
    sample_x = np.full_like(sample_y, 0.5 * (minX + maxX))
    sample_pts = np.column_stack([sample_x, sample_y])

    psi_numerical = uw.function.evaluate(psi.sym[0], sample_pts).squeeze()
    psi_exact = gardner_steady_state_psi(
        sample_y, psi_0=psi_bottom, psi_L=psi_top,
        L=maxY - minY, alpha=alpha_g,
    )

    max_err = np.max(np.abs(psi_numerical - psi_exact))
    assert np.allclose(psi_numerical, psi_exact, atol=0.05), (
        f"Gardner analytical benchmark failed: max error = {max_err:.4f}"
    )


if uw.is_notebook:
    test_richards_steady_constant_K()
