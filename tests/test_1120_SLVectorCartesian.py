import underworld3 as uw
import numpy as np
import sympy
import math
import pytest

# ### Test Semi-Lagrangian method in advecting vector fields
# ### Scalar field advection together diffusion tested in a different pytest

# ### Set up variables of the model
# +
res = 16
nsteps = 20
velocity = 1 / res  # /res

# ### mesh coordinates
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

sdev = 0.03
x0 = 0.5 * xmax
y0 = 0.5 * ymax

# ### Set up the mesh
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
def test_SLVec_boxmesh(mesh):
    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")

    # Create mesh vars
    Vdeg = 2
    sl_order = 2

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=Vdeg)
    vect_test = uw.discretisation.MeshVariable("A", mesh, mesh.dim, degree=Vdeg)

    # #### Create the SL object
    DuDt = uw.systems.ddt.SemiLagrangian(
        mesh,
        vect_test.sym,
        v.sym,
        vtype=uw.VarType.VECTOR,
        degree=Vdeg,
        continuous=vect_test.continuous,
        varsymbol=vect_test.symbol,
        verbose=False,
        bcs=None,
        order=sl_order,
        smoothing=0.0,
    )

    # ### Set up:
    # - Velocity field
    # - Initial vector distribution
    #TODO: DELETE remove swarm.access / data, replace with direct array assignment
    # with mesh.access(v):
    #     v.data[:, 1] = velocity
    
    v.array[:, 1, 1] = velocity

    x, y = mesh.X

    # vector components based on 2D Gaussian
    #TODO: DELETE remove swarm.access / data, replace with direct array assignment
    # with mesh.access(vect_test):
    #     gauss_2D = sympy.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sdev**2))
    #     vect_test.data[:, 0] = (
    #         2 * velocity * uw.function.evaluate(gauss_2D, vect_test.coords).squeeze()
    #     )
    #     vect_test.data[:, 1] = (
    #         2 * velocity * uw.function.evaluate(gauss_2D, vect_test.coords).squeeze()
    #     )
    
    gauss_2D = sympy.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sdev**2))
    gauss_vals = 2 * velocity * uw.function.evaluate(gauss_2D, vect_test.coords).squeeze()
    vect_test.array[:, 0, 0] = gauss_vals
    vect_test.array[:, 1, 1] = gauss_vals

    # ### Create points to sample the UW results
    ### y coords to sample
    sample_y = np.arange(
        mesh.data[:, 1].min(), mesh.data[:, 1].max(), 0.1 * mesh.get_min_radius()
    )  ### Vertical profile

    ### x coords to sample
    sample_x = 0.5 * xmax * np.ones_like(sample_y)  ### middle of the box

    sample_points = np.empty((sample_x.shape[0], 2))
    sample_points[:, 0] = sample_x
    sample_points[:, 1] = sample_y
    # -

    ### get the initial vector profile
    vec_prof_init = uw.function.evaluate(vect_test.sym, sample_points).squeeze()

    model_time = 0.0
    dt = 0.001

    for i in range(nsteps):
        DuDt.update_pre_solve(dt, verbose=False, evalf=False)
        #TODO: DELETE remove swarm.access / data, replace with direct array assignment
        # with mesh.access(vect_test):  # update
        #     vect_test.data[...] = DuDt.psi_star[0].data[...]
        
        vect_test.array[...] = DuDt.psi_star[0].array[...]
        model_time += dt

    ### compare UW and 1D numerical solution
    vec_prof_uw = uw.function.evaluate(vect_test.sym, sample_points).squeeze()

    #### expected distance traveled in the vertical direction
    travel_y = velocity * model_time
    y_ana_shift = travel_y + sample_y
    y_peak_loc = y0 + travel_y

    # get points around 5 standard deviations from the expected Gaussian peak location
    cond = (np.around(y_ana_shift, 6) >= y_peak_loc - 5 * sdev) & (
        np.around(y_ana_shift, 6) <= y_peak_loc + 5 * sdev
    )
    anax_to_eval = vec_prof_init[cond, 0]
    anay_to_eval = vec_prof_init[cond, 1]

    cond = (np.around(y_ana_shift, 6) >= y_peak_loc - 5 * sdev) & (
        np.around(y_ana_shift, 6) <= y_peak_loc + 5 * sdev
    )
    vx_to_eval = vec_prof_uw[cond, 0]
    vy_to_eval = vec_prof_uw[cond, 1]

    ### compare - use high atol due to low resolution
    assert np.allclose(vx_to_eval, anax_to_eval, atol=0.01)
    assert np.allclose(vy_to_eval, anay_to_eval, atol=0.01)

    del mesh
    del DuDt


del meshStructuredQuadBox
del unstructured_simplex_box_irregular
del unstructured_simplex_box_regular
