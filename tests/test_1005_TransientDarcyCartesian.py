# %%
# Test transient Darcy flow solver against analytical diffusion solution.
#
# A 1D vertical column with constant K and constant S_s reduces to
# simple diffusion:  S_s dh/dt = K d²h/dy²
# With step-change BC at the top (h=1) and fixed h=0 at the bottom,
# the analytical solution is an error-function diffusion profile.

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


# Domain parameters
res = 32
minX, maxX = 0.0, 0.1  # narrow to make it effectively 1D
minY, maxY = 0.0, 1.0

# Physical parameters
K_val = 1.0  # hydraulic conductivity
Ss_val = 1.0  # specific storage  →  diffusivity D = K/Ss = 1

t_start = 0.005  # small offset so erf profile is resolved
t_end = 0.02


def create_mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, res),
        minCoords=(minX, minY),
        maxCoords=(maxX, maxY),
        qdegree=3,
    )


# Analytical solution: step-change diffusion in a semi-infinite column
# h(y,t) = erfc(y / (2 sqrt(D t))) where D = K/Ss
# with h(0,t) = 1, h(inf,t) = 0
y_sym, t_sym = sp.symbols("y t", positive=True)
D_val = K_val / Ss_val
h_analytic = sp.erfc(y_sym / (2 * sp.sqrt(D_val * t_sym)))


def test_transient_darcy_diffusion():
    """Transient Darcy with constant K and S_s should match 1D diffusion."""
    mesh = create_mesh()

    h_soln = uw.discretisation.MeshVariable("h", mesh, 1, degree=2)
    v_soln = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=1)

    darcy = uw.systems.TransientDarcy(
        mesh, h_soln, v_soln, order=1, theta=0.5,
    )
    darcy.petsc_options.delValue("ksp_monitor")
    darcy.petsc_options["snes_rtol"] = 1.0e-6

    darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
    darcy.constitutive_model.Parameters.permeability = K_val
    darcy.constitutive_model.Parameters.s = sp.Matrix([0, 0]).T  # no gravity
    darcy.storage = Ss_val
    darcy.f = 0.0

    # BCs: h=1 at bottom (y=0), h=0 at top (y=1)
    darcy.add_dirichlet_bc([1.0], "Bottom")
    darcy.add_dirichlet_bc([0.0], "Top")

    darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-6
    darcy._v_projector.smoothing = 1.0e-6

    # Initial condition: analytical profile at t_start
    h_init = h_analytic.subs(t_sym, t_start)
    h_init_fn = h_init.subs(y_sym, mesh.X[1])
    h_soln.array = uw.function.evaluate(h_init_fn, h_soln.coords)

    # Time-step
    dt = darcy.estimate_dt()
    # Cap dt so we take a few steps
    dt = min(dt, (t_end - t_start) / 4)
    model_time = t_start

    while model_time < t_end:
        if model_time + dt > t_end:
            dt = t_end - model_time
        darcy.solve(timestep=dt)
        model_time += dt

    # Compare along a vertical profile at x = midpoint
    n_sample = 50
    sample_y = np.linspace(0.05, 0.95, n_sample)
    sample_x = np.full_like(sample_y, 0.5 * (minX + maxX))
    sample_pts = np.column_stack([sample_x, sample_y])

    h_numerical = uw.function.evaluate(h_soln.sym[0], sample_pts).squeeze()

    h_exact_fn = h_analytic.subs(t_sym, t_end).subs(y_sym, mesh.X[1])
    h_exact = uw.function.evaluate(h_exact_fn, sample_pts).squeeze()

    assert np.allclose(h_numerical, h_exact, atol=0.1), (
        f"Max error: {np.max(np.abs(h_numerical - h_exact)):.4f}"
    )
