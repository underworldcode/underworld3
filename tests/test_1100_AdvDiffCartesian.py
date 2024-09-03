# This is not a great test. The initial condition is not really representable in the mesh
# so it would fail to match the numerical solution if we did not run the problem at all.

# A better test would be the one that is in the examples - make the steps into error function
# analytic solutions, starting at t>0, and transporting over a meaningful distance.

import underworld3 as uw
import numpy as np
import math
import pytest


# ### Set up variables of the model

# +
res = 12
nsteps = 1
kappa = 1.0  # diffusive constant

velocity = 1 / res  # /res

### min and max temps
tmin = 0.5  # temp min
tmax = 1.0  # temp max

### Thickness of hot pipe in centre of box
pipe_thickness = 0.4


# ### Set up the mesh

xmin, xmax = 0, 1
ymin, ymax = 0, 1

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


@pytest.mark.parametrize(
    "mesh",
    [
        meshStructuredQuadBox,
        unstructured_simplex_box_irregular,
        unstructured_simplex_box_regular,
    ],
)
def test_advDiff_boxmesh(mesh):
    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")

    # Create an mesh vars
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)

    # #### Create the advDiff solver

    adv_diff = uw.systems.AdvDiffusion(
        mesh,
        u_Field=T,
        V_fn=v,
        solver_name="adv_diff",
    )

    # ### Set up properties of the adv_diff solver
    # - Constitutive model (Diffusivity)
    # - Boundary conditions
    # - Internal velocity
    # - Initial temperature distribution

    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = kappa

    ### fix temp of top and bottom walls
    adv_diff.add_dirichlet_bc(tmin, "Bottom")
    adv_diff.add_dirichlet_bc(tmin, "Top")
    # adv_diff.add_dirichlet_bc(0., "Left")
    # adv_diff.add_dirichlet_bc(0., "Right")

    with mesh.access(v):
        # initialise fields
        # v.data[:,0] = -1*v.coords[:,1]
        v.data[:, 1] = velocity

    with mesh.access(T):
        T.data[...] = tmin

        pipePosition = ((ymax - ymin) - pipe_thickness) / 2.0

        T.data[
            (T.coords[:, 1] >= (T.coords[:, 1].min() + pipePosition))
            & (T.coords[:, 1] <= (T.coords[:, 1].max() - pipePosition))
        ] = tmax

    # ### Create points to sample the UW results
    ### y coords to sample
    sample_y = np.arange(
        mesh.data[:, 1].min(), mesh.data[:, 1].max(), 0.1 * mesh.get_min_radius()
    )  ### Vertical profile

    ### x coords to sample
    sample_x = np.zeros_like(sample_y)  ### LHS of box

    sample_points = np.empty((sample_x.shape[0], 2))
    sample_points[:, 0] = sample_x
    sample_points[:, 1] = sample_y
    # -

    ### get the initial temp profile
    T_orig = uw.function.evaluate(T.sym[0], sample_points)

    #### 1D diffusion function
    #### To compare UW results with a numerical results

    def diffusion_1D(sample_points, T0, diffusivity, vel, time_1D):
        x = sample_points
        T = T0
        k = diffusivity
        time = time_1D

        dx = sample_points[1] - sample_points[0]

        dt_dif = dx**2 / k
        dt_adv = dx / velocity

        dt = 0.5 * min(dt_dif, dt_adv)

        if time > 0:
            """determine number of its"""
            nts = math.ceil(time / dt)

            """ get dt of 1D model """
            final_dt = time / nts

            for i in range(nts):
                qT = -k * np.diff(T) / dx
                dTdt = -np.diff(qT) / dx
                T[1:-1] += dTdt * final_dt

        return T

    model_time = 0.0

    #### Solve
    dt_est = adv_diff.estimate_dt()

    # This should be stable, and soluble by the 1D FD
    dt = 0.001

    ### diffuse through underworld
    adv_diff.solve(timestep=dt)

    model_time += dt

    ### compare UW and 1D numerical solution
    T_UW = uw.function.evalf(T.sym[0], sample_points)

    T_1D_model = diffusion_1D(
        sample_points=sample_points[:, 1],
        T0=T_orig.copy(),
        diffusivity=kappa,
        vel=velocity,
        time_1D=model_time,
    )

    #### 1D numerical advection
    new_y = sample_points[:, 1] + (velocity * model_time)

    ### some issues with the projection of data onto the sample points so high rtol
    assert np.allclose(T_UW, T_1D_model, atol=0.2)

    del mesh
    del adv_diff


del meshStructuredQuadBox
del unstructured_simplex_box_irregular
del unstructured_simplex_box_regular
