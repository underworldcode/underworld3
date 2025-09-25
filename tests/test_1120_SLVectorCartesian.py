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


# %%
@pytest.mark.parametrize(
    "mesh",
    [
        meshStructuredQuadBox,
        unstructured_simplex_box_irregular,
        unstructured_simplex_box_regular,
    ],
)
def test_SLVec_boxmesh(mesh):
    """Test Semi-Lagrangian vector advection with Gaussian pulse."""
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
    
    v.array[:, 0, 1] = velocity  # Shape is (N, 1, 2) for 2D vectors

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
    vect_test.array[:, 0, 0] = gauss_vals  # x component: array[i, 0, 0]
    vect_test.array[:, 0, 1] = gauss_vals  # y component: array[i, 0, 1]

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
    
    # %%
    if uw.is_notebook:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Initial vector field magnitude
        ax1 = axes[0, 0]
        coords_init = vect_test.coords
        
        # Recreate initial condition for visualization
        gauss_init_vals = 2 * velocity * uw.function.evaluate(gauss_2D, coords_init).squeeze()
        vec_mag_init = np.sqrt(2) * gauss_init_vals  # Both components equal
        
        scatter1 = ax1.scatter(coords_init[:, 0], coords_init[:, 1], 
                              c=vec_mag_init, s=20, cmap="viridis", alpha=0.8)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Initial Vector Field Magnitude")
        ax1.set_aspect("equal")
        plt.colorbar(scatter1, ax=ax1, label="|A|")
        
        # Plot 2: Final vector field magnitude
        ax2 = axes[0, 1]
        vec_data = vect_test.data
        vec_mag = np.sqrt(vec_data[:, 0]**2 + vec_data[:, 1]**2)
        
        scatter2 = ax2.scatter(coords_init[:, 0], coords_init[:, 1], 
                              c=vec_mag, s=20, cmap="viridis", alpha=0.8)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title(f"After Advection (t={model_time:.3f})")
        ax2.set_aspect("equal")
        plt.colorbar(scatter2, ax=ax2, label="|A|")
        
        # Plot 3: Vector component profiles
        ax3 = axes[1, 0]
        ax3.plot(sample_y, vec_prof_init[:, 0], 'b--', label='Initial Vx', linewidth=2)
        ax3.plot(sample_y, vec_prof_init[:, 1], 'r--', label='Initial Vy', linewidth=2)
        ax3.plot(sample_y, vec_prof_uw[:, 0], 'bo', markersize=4, alpha=0.6, label='Final Vx')
        ax3.plot(sample_y, vec_prof_uw[:, 1], 'ro', markersize=4, alpha=0.6, label='Final Vy')
        
        # Mark expected peak location
        ax3.axvline(x=y_peak_loc, color='g', linestyle='--', alpha=0.5, label=f'Expected peak (y={y_peak_loc:.3f})')
        
        ax3.set_xlabel("y coordinate (at x=0.5)")
        ax3.set_ylabel("Vector components")
        ax3.set_title("Vertical Profiles")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error analysis
        ax4 = axes[1, 1]
        error_x = np.abs(vx_to_eval - anax_to_eval)
        error_y = np.abs(vy_to_eval - anay_to_eval)
        
        y_vals = sample_y[cond]
        ax4.plot(y_vals, error_x, 'b-', label='|Error Vx|', linewidth=2)
        ax4.plot(y_vals, error_y, 'r-', label='|Error Vy|', linewidth=2)
        ax4.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='atol=0.01')
        
        ax4.set_xlabel("y coordinate")
        ax4.set_ylabel("Absolute Error")
        ax4.set_title(f"Error Analysis (max Vx: {error_x.max():.4f}, max Vy: {error_y.max():.4f})")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.show()

    del mesh
    del DuDt


del meshStructuredQuadBox
del unstructured_simplex_box_irregular
del unstructured_simplex_box_regular
