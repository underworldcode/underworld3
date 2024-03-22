# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Darcy flow (1d) using swarm variable to define permeability
#
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

options = PETSc.Options()

# +
minX, maxX = -1.0, 0.0
minY, maxY = -1.0, 0.0

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize=0.05, qdegree=3
)

# mesh = uw.meshing.StructuredQuadBox(elementRes=(20,20),
#                                       minCoords=(minX,minY),
#                                       maxCoords=(maxX,maxY),)


p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=3)
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)

# x and y coordinates
x = mesh.N.x
y = mesh.N.y

# +

if uw.mpi.size == 1:
    # plot the mesh
    
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
# -

# Create Darcy Solver
darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=p_soln, v_Field=v_soln)
darcy.petsc_options.delValue("ksp_monitor")
darcy.petsc_options[
    "snes_rtol"
] = 1.0e-6  # Needs to be smaller than the contrast in properties
darcy.constitutive_model = uw.constitutive_models.DiffusionModel
darcy.constitutive_model.Parameters.diffusivity = 1


# +
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_continuous=False)
# k = uw.swarm.IndexSwarmVariable("k", swarm, indices=2)

swarm.populate(fill_param=2)

# +
# Groundwater pressure boundary condition on the bottom wall

max_pressure = 0.5
initialPressure = -1.0 * y * max_pressure

# +
# set up two materials
interfaceY = -0.26


from sympy import Piecewise, ceiling, Abs

k1 = 1.0
k2 = 1.0e-4

# # The piecewise version
# kFunc = Piecewise((k1, y >= interfaceY), (k2, y < interfaceY), (1.0, True))

# darcy.constitutive_model.material_properties = darcy.constitutive_model.Parameters(diffusivity=kFunc)
# -

with swarm.access(material):
    material.data[swarm.data[:, 1] >= interfaceY] = 0
    material.data[swarm.data[:, 1] < interfaceY] = 1

# +
mat_k = np.array([k1, k2])

kFn = mat_k[0] * material.sym[0] + mat_k[1] * material.sym[1]

darcy.constitutive_model.Parameters.diffusivity = kFn

# +
# A smooth version
# kFunc = k2 + (k1-k2) * (0.5 + 0.5 * sympy.tanh(100.0*(y-interfaceY)))

darcy.f = 0.0
darcy.s = sympy.Matrix([0, -1]).T

# set up boundary conditions
darcy.add_dirichlet_bc(0.0, "Top")
darcy.add_dirichlet_bc(-1.0 * minY * max_pressure, "Bottom")

# Zero pressure gradient at sides / base (implied bc)

darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-6
darcy._v_projector.smoothing = 1.0e-6
darcy._v_projector.add_dirichlet_bc(0.0, "Left", 0)
darcy._v_projector.add_dirichlet_bc(0.0, "Right", 0)
# -
# Solve time
darcy.solve()

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["K"] = vis.scalar_fn_to_pv_points(pvmesh, kFn)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # point sources at cell centres
    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud0 = pv.PolyData(points[::3])


    pvstream = pvmesh.streamlines_from_source(
                                                point_cloud0,
                                                vectors="V",
                                                integrator_type=45,
                                                integration_direction="both",
                                                max_steps=1000,
                                                max_time=0.25,
                                                initial_step_length=0.001,
                                                max_step_length=0.01,
                                            )

    points = vis.swarm_to_pv_cloud(swarm)
    point_cloud1 = pv.PolyData(points)
    point_cloud1.point_data["K"] = vis.scalar_fn_to_pv_points(point_cloud1, kFn)

    with swarm.access():
        point_cloud1.point_data["M"] = material.data.copy()
    

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_mesh(
        point_cloud1,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="M",
        use_transparency=False,
        opacity=0.95,
    )

    pl.add_mesh(pvstream, line_width=10.0)

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.005, opacity=0.75)

    pl.show(cpos="xy")


# +
# set up interpolation coordinates
ycoords = np.linspace(minY + 0.001 * (maxY - minY), maxY - 0.001 * (maxY - minY), 100)
xcoords = np.full_like(ycoords, -1)
xy_coords = np.column_stack([xcoords, ycoords])

pressure_interp = uw.function.evaluate(p_soln.sym[0], xy_coords)


# +
La = -1.0 * interfaceY
Lb = 1.0 + interfaceY
dP = max_pressure

S = 1
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic = np.piecewise(
    ycoords,
    [ycoords >= -La, ycoords < -La],
    [
        lambda ycoords: -Pa * ycoords / La,
        lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb,
    ],
)

S = 0
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic_noG = np.piecewise(
    ycoords,
    [ycoords >= -La, ycoords < -La],
    [
        lambda ycoords: -Pa * ycoords / La,
        lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb,
    ],
)

# +
import matplotlib.pyplot as plt

# %matplotlib inline

fig = plt.figure()
ax1 = fig.add_subplot(111, xlabel="Pressure", ylabel="Depth")
ax1.plot(pressure_interp, ycoords, linewidth=3, label="Numerical solution")
ax1.plot(
    pressure_analytic, ycoords, linewidth=3, linestyle="--", label="Analytic solution"
)
ax1.plot(
    pressure_analytic_noG,
    ycoords,
    linewidth=3,
    linestyle="--",
    label="Analytic (no gravity)",
)
ax1.grid("on")
ax1.legend()
# -


