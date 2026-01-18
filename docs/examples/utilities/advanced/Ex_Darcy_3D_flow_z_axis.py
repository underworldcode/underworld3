# %% [markdown]
"""
# 🎓 Darcy 3D flow z axis

**PHYSICS:** utilities  
**DIFFICULTY:** advanced  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] editable=true slideshow={"slide_type": ""}
# # Darcy flow 3D (Vertical direction)

# + editable=true slideshow={"slide_type": ""}
# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy
from sympy import Piecewise, ceiling, Abs
import matplotlib.pyplot as plt
# %matplotlib inline

options = PETSc.Options()

# + editable=true slideshow={"slide_type": ""}
# vis tools
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

# +
# Create mesh
minX, maxX = -1.0, 0.0
minY, maxY = -1.0, 0.0
minZ, maxZ = -1.0, 0.0

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(minX, minY, minZ), 
                                         maxCoords=(maxX, maxY, maxZ), 
                                         cellSize=1/10, qdegree=2)

# x and y coordinates
x, y, z = mesh.CoordinateSystem.N

# + editable=true slideshow={"slide_type": ""}
# Create Darcy Solver
darcy = uw.systems.SteadyStateDarcy(mesh)

p_soln = darcy.Unknowns.u
v_soln = darcy.v

# Needs to be smaller than the contrast in properties
darcy.petsc_options["snes_rtol"] = 1.0e-6  
darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
# -


p_soln_0 = p_soln.clone("P_no_g", r"{p_\textrm{(no g)}}")
v_soln_0 = v_soln.clone("V_no_g", r"{v_\textrm{(no g)}}")


# + editable=true slideshow={"slide_type": ""}
def plot_P_V(_mesh, _p_soln, _v_soln):
    '''
    Plot pressure and velcity streamlines
    '''
    pvmesh = vis.mesh_to_pv_mesh(_mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, _p_soln.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, _v_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(_v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, _v_soln.sym)

    # point sources at cell centres
    points = np.zeros((_mesh._centroids.shape[0], 3))
    points[:, 0] = _mesh._centroids[:, 0]
    points[:, 1] = _mesh._centroids[:, 1]
    points[:, 2] = _mesh._centroids[:, 2]
    point_cloud = pv.PolyData(points[::3])

    pvstream = pvmesh.streamlines_from_source(point_cloud, vectors="V", integrator_type=45, 
                                              integration_direction="both", max_steps=1000,
                                              max_time=0.1, initial_step_length=0.001, 
                                              max_step_length=0.01)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="P",
                use_transparency=False, opacity=1.0, clim=[0, 0.5])
    pl.add_mesh(pvstream, line_width=1.0)
    # pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.005, opacity=0.75)

    pl.show(cpos="yz")


# + editable=true slideshow={"slide_type": ""}
# set up two materials
interfaceZ = -0.25

k1 = 1.0
k2 = 1.0e-4

# Groundwater pressure boundary condition on the bottom wall
max_pressure = 0.5

# The piecewise version
kFunc = Piecewise((k1, z >= interfaceZ), (k2, z < interfaceZ), (1.0, True))

# A smooth version

darcy.constitutive_model.Parameters.permeability = kFunc
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, 0, 0]).T
darcy.f = 0.0

# set up boundary conditions
darcy.add_dirichlet_bc(0.0, "Top")
darcy.add_dirichlet_bc(-1.0 * minZ * max_pressure, "Bottom")
# + editable=true slideshow={"slide_type": ""}
# darcy solve without gravity
darcy.solve()


# + editable=true slideshow={"slide_type": ""}
# saving output
mesh.petsc_save_checkpoint(index=0, meshVars=[p_soln, v_soln], outputPath='./output/darcy_3d_no_g')

# + editable=true slideshow={"slide_type": ""}
# plotting soln without gravity
plot_P_V(mesh, p_soln, v_soln)

# + editable=true slideshow={"slide_type": ""}
# copying solution
# TODO: Consider uw.synchronised_array_update() for multi-variable assignment
p_soln_0.data[...] = p_soln.data[...]
v_soln_0.data[...] = v_soln.data[...]
# -

# now switch on gravity
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, 0, -1]).T
darcy.solve()
# + editable=true slideshow={"slide_type": ""}
# saving output
mesh.petsc_save_checkpoint(index=0, meshVars=[p_soln, v_soln], outputPath='./output/darcy_3d_g')

# + editable=true slideshow={"slide_type": ""}
# plotting soln with gravity
plot_P_V(mesh, p_soln, v_soln)


# + editable=true slideshow={"slide_type": ""}
# set up interpolation coordinates
zcoords = np.linspace(minZ + 0.001 * (maxZ - minZ), maxZ - 0.001 * (maxZ - minZ), 100)
xcoords = np.full_like(zcoords, -0.5)
ycoords = np.full_like(zcoords, -0.5)
xyz_coords = np.column_stack([xcoords, ycoords, zcoords])

pressure_interp = uw.function.evaluate(p_soln.sym[0], xyz_coords)
pressure_interp_0 = uw.function.evaluate(p_soln_0.sym[0], xyz_coords)


# +
La = -1.0 * interfaceZ
Lb = 1.0 + interfaceZ
dP = max_pressure

S = 1
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic = np.piecewise(
    zcoords,
    [zcoords >= -La, zcoords < -La],
    [
        lambda zcoords: -Pa * zcoords / La,
        lambda zcoords: Pa + (dP - Pa) * (-zcoords - La) / Lb,
    ])

S = 0
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic_noG = np.piecewise(
    zcoords,
    [zcoords >= -La, zcoords < -La],
    [
        lambda zcoords: -Pa * zcoords / La,
        lambda zcoords: Pa + (dP - Pa) * (-zcoords - La) / Lb,
    ])

# + editable=true slideshow={"slide_type": ""}
# plotting numerical and analytical solution
fig = plt.figure()
ax1 = fig.add_subplot(111, xlabel="Pressure", ylabel="Depth")
ax1.plot(pressure_interp, zcoords, linewidth=3, label="Numerical solution")
ax1.plot(pressure_interp_0, zcoords, linewidth=3, label="Numerical solution (no G)")
ax1.plot(pressure_analytic, zcoords, linewidth=3, linestyle="--", label="Analytic solution")
ax1.plot(pressure_analytic_noG, zcoords, linewidth=3, linestyle="--", label="Analytic (no gravity)")
ax1.grid("on")
ax1.legend()
# + editable=true slideshow={"slide_type": ""}

