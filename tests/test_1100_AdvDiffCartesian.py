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
import matplotlib.pyplot as plt


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


### Quads
meshStructuredQuadBox = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)),
    minCoords=(xmin, ymin),
    maxCoords=(xmax, ymax),
    qdegree=3,
)

unstructured_simplex_box_irregular = uw.meshing.UnstructuredSimplexBox(
    cellSize=1 / res, regular=False, qdegree=3, refinement=0
)

unstructured_simplex_box_regular = uw.meshing.UnstructuredSimplexBox(
    cellSize=1 / res, regular=True, qdegree=3, refinement=0
)
# -

# ### setup analytical function

# +
u, t, x, x0, x1 = sp.symbols("u, t, x, x0, x1")


U_a_x = (
    sp.erf((x1 - x + (u * t)) / (2 * sp.sqrt(kappa * t)))
    + sp.erf((-x0 + x - (u * t)) / (2 * sp.sqrt(kappa * t)))
) / 2


# %%
@pytest.mark.parametrize(
    "mesh",
    [
        meshStructuredQuadBox,
        unstructured_simplex_box_irregular,
        unstructured_simplex_box_regular,
    ],
)
def test_advDiff_boxmesh(mesh):
    """Test advection-diffusion with analytical error function solution."""
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
if uw.is_notebook:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Temperature field
    coords = T.coords
    T_vals = T.data.flatten()
    scatter1 = ax1.scatter(
        coords[:, 0], coords[:, 1], c=T_vals, s=15, cmap="coolwarm", alpha=0.8, vmin=0, vmax=1
    )

    # Add velocity vectors
    vel_coords = v.coords
    vel_data = v.data
    skip = max(1, len(vel_coords) // 50)
    ax1.quiver(
        vel_coords[::skip, 0],
        vel_coords[::skip, 1],
        vel_data[::skip, 0],
        vel_data[::skip, 1],
        alpha=0.5,
        scale=velocity * 50,
    )

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"Advection-Diffusion (t={t_end:.2e}, v={velocity:.3f}, κ={kappa})")
    ax1.set_aspect("equal")
    plt.colorbar(scatter1, ax=ax1, label="Temperature")

    # Plot 2: Profile comparison
    ax2.plot(sample_x, T_analytical, "r-", linewidth=2, label="Analytical")
    ax2.plot(sample_x, T_UW, "bo", markersize=5, alpha=0.6, label="Numerical")

    # Show error
    error = np.abs(T_UW - T_analytical)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(sample_x, error, "g--", alpha=0.5, label="Error")
    ax2_twin.set_ylabel("Absolute Error", color="g")
    ax2_twin.tick_params(axis="y", labelcolor="g")

    ax2.set_xlabel("x coordinate (at y=0.5)")
    ax2.set_ylabel("Temperature")
    ax2.set_title(f"Profile Comparison (max error: {error.max():.3e})")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    del mesh
    del adv_diff


del meshStructuredQuadBox
del unstructured_simplex_box_irregular
del unstructured_simplex_box_regular

# %%
