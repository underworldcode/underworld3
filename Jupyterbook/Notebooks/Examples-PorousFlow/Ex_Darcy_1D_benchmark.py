# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# # Darcy flow (1d) using xy coordinates to define permeability distribution

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


# x and y coordinates
x = mesh.N.x
y = mesh.N.y

# +
# Create Darcy Solver
darcy = uw.systems.SteadyStateDarcy(mesh)

p_soln = darcy.Unknowns.u
v_soln = darcy.v

darcy.petsc_options[
    "snes_rtol"
] = 1.0e-6  # Needs to be smaller than the contrast in properties

darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
darcy.constitutive_model.Parameters.permeability = 1
# -


p_soln_0 = p_soln.clone("P_no_g", r"{p_\textrm{no g}}")
v_soln_0 = v_soln.clone("V_no_g", r"{v_\textrm{no g}}")

# +
# Groundwater pressure boundary condition on the bottom wall

max_pressure = 0.5

# +
# set up two materials

interfaceY = -0.26

from sympy import Piecewise, ceiling, Abs

k1 = 1.0
k2 = 1.0e-4

# The piecewise version
kFunc = Piecewise((k1, y >= interfaceY), (k2, y < interfaceY), (1.0, True))

# A smooth version

darcy.constitutive_model.Parameters.permeability = kFunc
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, 0]).T
darcy.f = 0.0

# set up boundary conditions
darcy.add_dirichlet_bc(0.0, "Top")
darcy.add_dirichlet_bc(-1.0 * minY * max_pressure, "Bottom")

# -
# Solve time
darcy.solve()
with mesh.access(p_soln_0, v_soln_0):
    p_soln_0.data[...] = p_soln.data[...]
    v_soln_0.data[...] = v_soln.data[...]


# +
# now switch on gravity

darcy.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T
darcy.solve()

# -



if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["K"] = vis.scalar_fn_to_pv_points(pvmesh, kFunc)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

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
                                                max_time=0.1,
                                                initial_step_length=0.001,
                                                max_step_length=0.01,
                                            )

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

    pl.add_mesh(pvstream, line_width=1.0)

    # pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.005, opacity=0.75)

    pl.show(cpos="xy")


# +
# set up interpolation coordinates
ycoords = np.linspace(minY + 0.001 * (maxY - minY), maxY - 0.001 * (maxY - minY), 100)
xcoords = np.full_like(ycoords, -1)
xy_coords = np.column_stack([xcoords, ycoords])

pressure_interp = uw.function.evalf(p_soln.sym[0], xy_coords)
pressure_interp_0 = uw.function.evalf(p_soln_0.sym[0], xy_coords)


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
ax1.plot(pressure_interp_0, ycoords, linewidth=3, label="Numerical solution (no G)")
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
darcy.view()


darcy.darcy_flux




