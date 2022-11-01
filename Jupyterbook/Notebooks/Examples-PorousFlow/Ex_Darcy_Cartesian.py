# %% [markdown]
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

# %%
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

options = PETSc.Options()

# %%
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), 
                                         maxCoords=(4.0, 1.0), 
                                         cellSize=0.05, qdegree=3)

p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)


# %%
# Mesh deformation

x,y = mesh.X

h_fn = 1.0 + x * 0.2 / 4 + 0.04 * sympy.cos(2.0 * np.pi * x) * y

new_coords = mesh.data.copy()
new_coords[:,1] = uw.function.evaluate( h_fn * y, mesh.data, mesh.N)

mesh.deform_mesh(new_coords=new_coords)

# %%
if uw.mpi.size == 1 and uw.is_notebook:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False)

    pl.show(cpos="xy")

# %%
# Create Poisson object
darcy = uw.systems.SteadyStateDarcy(mesh, u_Field=p_soln, v_Field=v_soln)
darcy.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
darcy.constitutive_model.Parameters.diffusivity=1
darcy.petsc_options.delValue("ksp_monitor")

# %%
# Set some things

k = sympy.exp(-2.0 * 2.302585 * (h_fn - y))  # powers of 10
darcy.constitutive_model.Parameters.diffusivity=k

darcy.f = 0.0
darcy.s = sympy.Matrix([0, -1]).T

darcy.add_dirichlet_bc(0.0, "Top")

# Zero pressure gradient at sides / base (implied bc)

darcy._v_projector.smoothing = 1.0e-3

# %%
# Solve time
darcy.solve()

# %%
if uw.mpi.size == 1 and uw.is_notebook:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    with mesh.access():
        usol = v_soln.data.copy()

    pvmesh.point_data["P"] = uw.function.evaluate(p_soln.sym[0], mesh.data, mesh.N)
    pvmesh.point_data["dP"] = uw.function.evaluate(p_soln.sym[0] - (h_fn - y), mesh.data, mesh.N)
    pvmesh.point_data["K"] = uw.function.evaluate(k, mesh.data, mesh.N)
    pvmesh.point_data["S"] = uw.function.evaluate(sympy.log(v_soln.sym.dot(v_soln.sym)), mesh.data, mesh.N)

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # point sources at cell centres

    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points[::3])

    v_vectors = np.zeros((mesh.data.shape[0], 3))
    v_vectors[:, 0:2] = uw.function.evaluate(v_soln.fn, mesh.data)
    pvmesh.point_data["V"] = v_vectors

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
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="P", use_transparency=False, opacity=1.0
    )
    
    pl.add_arrows(arrow_loc, arrow_length, mag=0.5, opacity=0.75)

    pl.add_mesh(pvstream, line_width=10.0)

    pl.show(cpos="xy")

# %%
## Metrics

_, _, _, max_p, _, _, _ = p_soln.stats()
_, _, _, max_vh, _, _, _ = mesh.stats(abs(v_soln.fn.dot(mesh.N.i)))
_, _, _, max_vv, _, _, _ = mesh.stats(abs(v_soln.fn.dot(mesh.N.j)))

print("Max horizontal velocity: {:4f}".format(max_vh))
print("Max vertical velocity:   {:4f}".format(max_vv))
print("Max pressure         :   {:4f}".format(max_p))


# %%

# %%
