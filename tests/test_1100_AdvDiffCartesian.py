# %%
# This is not a great test. The initial condition is not really representable in the mesh
# so it would fail to match the numerical solution if we did not run the problem at all.

# A better test would be the one that is in the examples - make the steps into error function
# analytic solutions, starting at t>0, and transporting over a meaningful distance.

import underworld3 as uw
import numpy as np
import sympy as sp
import math
import pytest

# Physics solver tests - full solver execution
pytestmark = pytest.mark.level_3

# matplotlib only needed for notebook visualization, not test logic
# Import is deferred to where it's used (inside `if uw.is_notebook:` blocks)


@pytest.fixture(autouse=True)
def reset_model_state():
    """Reset model state before each test to prevent pollution from other tests."""
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    yield
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)


# ### Set up variables of the model

# +
res = 24
nsteps = 1
kappa = 1.0  # diffusive constant

velocity = 1 / res  # /res


t_start = 1e-4
t_end = 2e-4

# ### Set up the mesh
xmin, xmax = 0, 1
ymin, ymax = 0, 1

u_degree = 2


# NOTE: Meshes MUST be created inside fixtures or tests, not at module level!
# Module-level creation happens at import time, before fixtures reset state,
# which can cause "model has no reference quantities" errors if another test
# has polluted global state.

def create_mesh(mesh_type):
    """Factory function to create meshes inside tests for proper isolation."""
    if mesh_type == "mesh0":
        return uw.meshing.StructuredQuadBox(
            elementRes=(int(res), int(res)),
            minCoords=(xmin, ymin),
            maxCoords=(xmax, ymax),
            qdegree=3,
        )
    elif mesh_type == "mesh1":
        return uw.meshing.UnstructuredSimplexBox(
            cellSize=1 / res, regular=False, qdegree=3, refinement=0
        )
    elif mesh_type == "mesh2":
        return uw.meshing.UnstructuredSimplexBox(
            cellSize=1 / res, regular=True, qdegree=3, refinement=0
        )
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")

# -

# ### setup analytical function

# +
u, t, x, x0, x1 = sp.symbols("u, t, x, x0, x1")


U_a_x = (
    sp.erf((x1 - x + (u * t)) / (2 * sp.sqrt(kappa * t)))
    + sp.erf((-x0 + x - (u * t)) / (2 * sp.sqrt(kappa * t)))
) / 2


# %%
@pytest.mark.parametrize("mesh_type", ["mesh0", "mesh1", "mesh2"])
def test_advDiff_boxmesh(mesh_type):
    """Test advection-diffusion with analytical error function solution."""
    # Create mesh INSIDE test function to ensure proper isolation
    mesh = create_mesh(mesh_type)
    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")

    # Create an mesh vars
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=u_degree)

    # #### Create the advDiff solver

    adv_diff = uw.systems.AdvDiffusion(
        mesh,
        u_Field=T,
        V_fn=v,
    )

    # ### Set up properties of the adv_diff solver
    # - Constitutive model (Diffusivity)
    # - Boundary conditions
    # - Internal velocity
    # - Initial temperature distribution

    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = kappa

    ### fix temp of top and bottom walls
    adv_diff.add_dirichlet_bc(0.0, "Left")
    adv_diff.add_dirichlet_bc(0.0, "Right")

    # initialise fields
    # v.array[:, 0, 0] = -1*v.coords[:,1]
    v.array[:, 0, 0] = velocity

    U_start = U_a_x.subs({u: velocity, t: t_start, x: mesh.X[0], x0: 0.4, x1: 0.6})

    T.array = uw.function.evaluate(U_start, T.coords)

    model_time = t_start

    #### Solve
    dt = adv_diff.estimate_dt()

    while model_time < t_end:

        if model_time + dt > t_end:
            dt = t_end - model_time

        ### diffuse through underworld
        adv_diff.solve(timestep=dt)

        model_time += dt

    sample_x = np.arange(0, 1, mesh.get_min_radius())
    sample_y = np.zeros_like(sample_x) + 0.5
    sample_points = np.column_stack([sample_x, sample_y])

    ### compare UW and 1D numerical solution
    T_UW = uw.function.evaluate(T.sym[0], sample_points).squeeze()

    U_end = U_a_x.subs({u: velocity, t: t_end, x: mesh.X[0], x0: 0.4, x1: 0.6})
    T_analytical = uw.function.evaluate(U_end, sample_points).squeeze()

    ### moderate atol due to evaluating onto points
    assert np.allclose(T_UW, T_analytical, atol=0.05)


# %%
# NOTE: Notebook visualization code removed from module level.
# Variables T, mesh, v, etc. are only defined inside test functions,
# so referencing them at module level causes NameError during pytest collection.
# For interactive visualization, run this test file directly as a notebook.

# %%
