# %% [markdown]
"""
# 🎓 NS Couette PostProc

**PHYSICS:** fluid_mechanics  
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
# %% [markdown]
# # Post-processing for Couette flow
# 

# %%
import os

import petsc4py
import underworld3 as uw

import nest_asyncio
nest_asyncio.apply()

import sympy
import numpy as np
import argparse
import pickle

# %%
resolution = 16
refinement = 0
save_every = 50
Cmax       = 1         # target Courant number

order = 1 # solver order

mesh_type = "Pirr" # or Preg, Pirr, Quad
qdeg = 3
Vdeg = 2
Pdeg = Vdeg - 1
Pcont = False

# %%
outdir      = "/Users/jgra0019/Documents/codes/uw3-dev/Navier-Stokes-benchmark/output-Couette/Couette-res16-order1-Pirr-np8"
expt_name   = "Couette-res16-order1-Pirr_run0"
idx = 50

# %%
width   = 4.
height  = 1.
vel     = 1.

fluid_rho   = 1.
kin_visc    = 1.
dyn_visc    = fluid_rho * kin_visc

# %%
minX, maxX = -0.5 * width, 0.5 * width
minY, maxY = -0.5 * height, 0.5 * height

uw.pprint("min X, max X:", minX, maxX)
    print("min Y, max Y:", minY, maxY)
    print("kinematic viscosity: ", kin_visc)
    print("fluid density: ", fluid_rho)
    print("dynamic viscosity: ", kin_visc * fluid_rho)

# %%
# cell size calculation
if mesh_type == "Preg":
    meshbox = uw.meshing.UnstructuredSimplexBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = 1 / resolution, qdegree = qdeg, regular = True)
elif mesh_type == "Pirr":
    meshbox = uw.meshing.UnstructuredSimplexBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = 1 / resolution, qdegree = qdeg, regular = False)
elif mesh_type == "Quad":
    meshbox = uw.meshing.StructuredQuadBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), elementRes = (width * resolution, height * resolution), qdegree = qdeg, regular = False)

# %%
meshbox.dm.view()

# %%
v_soln      = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=Vdeg)
p_soln      = uw.discretisation.MeshVariable("P", meshbox, 1, degree=Pdeg, continuous = Pcont)

v_ana       = uw.discretisation.MeshVariable("Ua", meshbox, meshbox.dim, degree=Vdeg)
shear_force = uw.discretisation.MeshVariable("S", meshbox, 1, degree=Vdeg)

# %%
v_soln.read_timestep(data_filename = expt_name, data_name = "U", index = idx, outputPath = outdir)
p_soln.read_timestep(data_filename = expt_name, data_name = "P", index = idx, outputPath = outdir)

# %%

if uw.mpi.size == 1 and uw.is_notebook:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v_soln.sym.dot(v_soln.sym)))
    pvmesh.point_data["P"]   = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    # point sources at cell centres for streamlines

    points = np.zeros((meshbox._centroids.shape[0], 3))
    points[:, 0] = meshbox._centroids[:, 0]
    points[:, 1] = meshbox._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    # pvstream = pvmesh.streamlines_from_source(
    #     point_cloud, vectors="Vvec", integration_direction="forward", max_steps=10, 
    # )

    pl.add_mesh(
        pvmesh,
        cmap="cividis",
        edge_color="Black",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=1,
        show_scalar_bar=True)

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], 
                  mag=5e-2, opacity=1, 
                  show_scalar_bar=False)
    
    #pl.add_mesh(pvstream, show_scalar_bar=False)

    pl.camera.SetPosition(0.75, 0.2, 1.5)
    pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)


    #pl.camera_position = "xz"
    # pl.screenshot(
    #     filename="{}.png".format(filename),
    #     window_size=(2560, 1280),
    #     return_img=False,
    # )

    # pl.clear()
    pl.show()



# %%
# set the timestep 
# for now, set it to be constant                           
# # target Courant number
delta_x = meshbox.get_min_radius()
max_vel = vel

delta_t = Cmax*delta_x/max_vel

uw.pprint(f"Min radius: {delta_x}")
    print("Timestep used:", delta_t)

# %%
# sample the velocity along the middle
x_samp = 1

if uw.mpi.size == 1:
    y = np.linspace(minY, maxY, 30)
    x = x_samp * np.ones_like(y)

    xy_samp = np.vstack([x, y]).T
    xy_samp

    vx_samp = uw.function.evaluate(v_soln.sym[0], xy_samp)
    vx_theo = 0.5 * vel * (1 + y / maxY)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi = 150)
ax.plot(vx_theo, y, "-", lw = 1, label = "Analytical")
ax.plot(vx_samp, y, "o", ms = 3, label = "Numerical")
ax.set_ylabel("Y")
ax.set_xlabel(r"$V_x$")
ax.set_aspect("equal")
ax.set_title(f"After {idx} timesteps")
ax.legend()

# %%
# calculate the norm of the velocity field 
import math 

# calculate the L2 norm of velocity using integral 
x, y = meshbox.N.x, meshbox.N.y

vx_theo_expr = 0.5 * vel * (1 + y / maxY)

# calculate vx over the entire mesh variable
with meshbox.access(v_ana):
    v_ana.data[:, 0] = uw.function.evaluate(vx_theo_expr, v_ana.coords)

# create the mask function
vel_mask_fn = sympy.Piecewise((1.0, (x >= x_samp) & (x <= (x_samp + 0.2))), 
                              (0, True))

v_diff     = v_soln.sym - v_ana.sym
v_diff_mag = v_diff.dot(v_diff)

v_ana_mag  = v_ana.sym.dot(v_ana.sym) 

v_diff_mag_integ = math.sqrt(uw.maths.Integral(meshbox, vel_mask_fn * v_diff_mag).evaluate())
v_ana_mag_integ = math.sqrt(uw.maths.Integral(meshbox, vel_mask_fn * v_ana_mag).evaluate())
v_norm = v_diff_mag_integ / v_ana_mag_integ

uw.pprint(f"Normalized velocity L2 error: {v_norm}")

# %%
# calculate the shear force per unit area acting on the wall
x, y = meshbox.N.x, meshbox.N.y

shear_force_expr = dyn_visc * (sympy.diff(v_soln.sym[0], y) + sympy.diff(v_soln.sym[1], x))
shear_force_calc = uw.systems.Projection(meshbox, shear_force)
shear_force_calc.uw_function = shear_force_expr
shear_force_calc.smoothing = 0.0
shear_force_calc.petsc_options.delValue("ksp_monitor")

shear_force_calc.solve()

num_val = uw.function.evaluate(shear_force.sym, np.array([[1, 0.5]]))
uw.pprint(f"Theoretical value: {dyn_visc * vel / height}")
    print(f"Numerical value: {num_val}")


