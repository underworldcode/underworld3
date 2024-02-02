# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Underworld Groundwater Flow Benchmark 1
#
# See the Underworld2 example by Adam Beall.
#
# Flow driven by gravity and topography. We check the flow for constant permeability and for exponentially decreasing permeability as a function of depth.
#
# *Note*, this benchmark is a bit problematic because the surface shape is not really
# consistent with the sidewall boundary conditions - zero gradients at the vertical boundaries.If we replace the sin(x) term with cos(x) to describe the surface then it works a little better because there is no kink in the surface topography at the walls.
#
# *Note*, there is not an obvious way in pyvista to make the streamlines smaller / shorter / fainter where flow rates are very low so the visualisation is a little misleading right now.
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# %%
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

options = PETSc.Options()

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(4.0, 1.0), cellSize=0.05, qdegree=3
)

p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1, continuous=False)


# Mesh deformation

x, y = mesh.X

h_fn = 1.0 + x * 0.2 / 4 + 0.04 * sympy.cos(2.0 * np.pi * x) * y

new_coords = mesh.data.copy()
new_coords[:, 1] = uw.function.evaluate(h_fn * y, mesh.data, mesh.N)

mesh.deform_mesh(new_coords=new_coords)



# %%
if uw.mpi.size == 1 and uw.is_notebook:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
    )

    pl.show(cpos="xy")

# %%
# Create Poisson object
darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=p_soln, v_Field=v_soln)
darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
darcy.constitutive_model.Parameters.permeability = 1
darcy.petsc_options.delValue("ksp_monitor")

# Set some things

k = sympy.exp(-2.0 * 2.302585 * (h_fn - y))  # powers of 10
darcy.constitutive_model.Parameters.permeability = k

k

darcy.f = 0.0
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T

darcy.add_dirichlet_bc(0.0, "Top")

# Zero pressure gradient at sides / base (implied bc)

darcy._v_projector.smoothing = 0.0



# %%
# Solve time
darcy.petsc_options.setValue("snes_monitor", None)
darcy.solve(verbose=False)

# %%
if uw.mpi.size == 1 and uw.is_notebook:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["dP"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym[0] - (h_fn - y))
    pvmesh.point_data["K"] = vis.scalar_fn_to_pv_points(pvmesh, k)
    pvmesh.point_data["S"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.log(v_soln.sym.dot(v_soln.sym)))

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # point sources at cell centres

    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points[::3])

    pvstream = pvmesh.streamlines_from_source(
                                                point_cloud,
                                                vectors="V",
                                                integrator_type=45,
                                                integration_direction="both",
                                                max_steps=1000,
                                                max_time=0.2,
                                                initial_step_length=0.001,
                                                max_step_length=0.01,
                                            )

    pl = pv.Plotter()

    pl.add_mesh(
                pvmesh,
                cmap="coolwarm",
                edge_color="Black",
                show_edges=True,
                scalars="P",
                use_transparency=False,
                opacity=1.0,
            )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.5, opacity=0.5)
    pl.add_mesh(pvstream, line_width=1.0)
    pl.show(cpos="xy")

#
# ## Metrics

_, _, _, max_p, _, _, _ = p_soln.stats()


# +

print("Max pressure         :   {:4f}".format(max_p))
# -


