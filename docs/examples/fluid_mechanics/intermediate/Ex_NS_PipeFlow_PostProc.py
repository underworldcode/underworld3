# %% [markdown]
"""
# 🔬 NS PipeFlow PostProc

**PHYSICS:** fluid_mechanics  
**DIFFICULTY:** intermediate  
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
# # Post-processing for Navier-Stokes pipe flow 
# 

# %%
import os

import petsc4py
import underworld3 as uw
import sympy

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import argparse
import pickle

# %%
resolution = 32
refinement = 0
dt_ns = 0.1

order = 1 # solver order
tol = 1e-10

use_dim = True # True if using dimensionalised values; False otherwise
case_num = 1

mesh_type1 = "Pirr"
mesh_type = "Pirr" # or Preg, Pirr, Quad
qdeg = 3
Vdeg = 3
Pdeg = Vdeg - 2
Pcont = False

if case_num == 1:
    if dt_ns == 0.1:
        maxsteps = 10
        #maxsteps = 1
        idx      = 1
    elif dt_ns == 0.01:
        maxsteps = 10
        idx      = 1
    elif dt_ns == 0.001:
        maxsteps = 20
        idx      = 5
    elif dt_ns == 0.00125:
        maxsteps = 20
        idx      = 4
    elif dt_ns == 0.0025:
        maxsteps = 20
        idx      = 2
    elif dt_ns == 0.005:
        maxsteps = 20
        idx      = 1
    elif dt_ns == 0.00625:
        maxsteps = 16
        idx      = 1
    elif dt_ns == 0.0125:
        maxsteps = 8
        idx      = 1
    elif dt_ns == 0.025:
        maxsteps = 4
        idx      = 1
    elif dt_ns == 0.05:
        maxsteps = 2
        idx      = 1

print(f"Additional run-time: {maxsteps * dt_ns * (idx)}")

idx = idx - 1 

p_text = "P" if Pcont else "dP"

# %%
#outdir = f"/Users/jgra0019/Documents/codes/uw3-dev/Navier-Stokes-benchmark/output-Pois/higher-order-elem-{mesh_type}-P{Vdeg}{p_text}{Pdeg}/Pois-HR-{mesh_type}-res{resolution}-order{order}-dt{dt_ns}/"
outdir = f"/Users/jgra0019/Documents/codes/uw3-dev/Navier-Stokes-benchmark/output-Pois/higher-order-elem-{mesh_type}-P{Vdeg}{p_text}{Pdeg}-dt{dt_ns}-tstep10/Pois-HR-{mesh_type}-res{resolution}-order{order}-dt{dt_ns}/"
expt_name = f"Pois-res{resolution}-order{order}-{mesh_type}-case{case_num}_run{idx}"

# %%
# dimensionalized values of problem parameters
# from reference

# velocity - m/s
# fluid_rho - kg/m^3
# dynamic viscosity - Pa.s
# maybe add case_num = 0 where everything is 1
if case_num == 1:       # Re = 10
    vel_dim         = 0.034
    fluid_rho_dim   = 910
    dyn_visc_dim    = 0.3094
elif case_num == 2:     # Re = 100
    vel_dim         = 0.34
    fluid_rho_dim   = 910
    dyn_visc_dim    = 0.3094
elif case_num == 3:     # Re = 1000
    vel_dim         = 3.4
    fluid_rho_dim   = 910
    dyn_visc_dim    = 0.3094
elif case_num == 4:     # Re = 10
    vel_dim         = 1.
    fluid_rho_dim   = 100
    dyn_visc_dim    = 1
elif case_num == 5:     # Re = 100
    vel_dim         = 1
    fluid_rho_dim   = 100
    dyn_visc_dim    = 0.1
elif case_num == 6:     # Re = 1000
    vel_dim         = 1
    fluid_rho_dim   = 100
    dyn_visc_dim    = 0.01

height_dim  = 2 * 0.05          # meters
if case_num in [3, 6]:          # Re = 1000
    width_dim   = 10 * height_dim    # meters
else:
    width_dim   = 8 * height_dim    # meters

kin_visc_dim  = dyn_visc_dim / fluid_rho_dim
Re_num        = fluid_rho_dim * vel_dim * height_dim / dyn_visc_dim
if uw.mpi.rank == 0:
    print(f"Reynold's number: {Re_num}")

# %%
if use_dim:
    height  = height_dim
    width   = width_dim

    vel     = vel_dim

    fluid_rho   = fluid_rho_dim
    kin_visc    = kin_visc_dim
    dyn_visc    = dyn_visc_dim
else:
    pass # perform non-dimensionalization here

# %%
minX, maxX = -0.5 * width, 0.5 * width
minY, maxY = -0.5 * height, 0.5 * height

if uw.mpi.rank == 0:
    print("min X, max X:", minX, maxX)
    print("min Y, max Y:", minY, maxY)
    print("kinematic viscosity: ", kin_visc)
    print("fluid density: ", fluid_rho)
    print("dynamic viscosity: ", kin_visc * fluid_rho)

# %%
# cell size calculation
if mesh_type == "Preg":
    meshbox = uw.meshing.UnstructuredSimplexBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = height / resolution, qdegree = qdeg, regular = True)
elif mesh_type == "Pirr":
    meshbox = uw.meshing.UnstructuredSimplexBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = height / resolution, qdegree = qdeg, regular = False)
elif mesh_type == "Quad":
    meshbox = uw.meshing.StructuredQuadBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), elementRes = ((width/height) * resolution, resolution), qdegree = qdeg, regular = False)

# %%
meshbox.dm.view()

# %%
if uw.mpi.size == 1 and uw.is_notebook:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    pl = pv.Plotter(window_size=(1000, 750))

    # point sources at cell centres for streamlines

    points = np.zeros((meshbox._centroids.shape[0], 3))
    points[:, 0] = meshbox._centroids[:, 0]
    points[:, 1] = meshbox._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pl.add_mesh(pvmesh,
                edge_color="Black",
                show_edges=True,
                show_scalar_bar=False)

    pl.show()

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=Vdeg)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=Pdeg, continuous = Pcont)

v_ana = uw.discretisation.MeshVariable("Va", meshbox, meshbox.dim, degree=Vdeg)

# %%
print(f"Reading: {outdir}")
print(f"File: {expt_name}")

v_soln.read_timestep(data_filename = expt_name, data_name = "U", index = maxsteps, outputPath = outdir)
p_soln.read_timestep(data_filename = expt_name, data_name = "P", index = maxsteps, outputPath = outdir)

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

Cmax = 1                                # target Courant number
delta_x = meshbox.get_min_radius()
max_vel = vel


# %%
# sample the velocity in the hydrodynamically fully developed region
if case_num == 3:
    x_sample = 0.47 # depends on where hydrodynamically fully developed region is
else:
    x_sample = 0.1 # depends on where hydrodynamically fully developed region is

y_samp = np.linspace(minY, maxY, 100)
x_samp = x_sample * np.ones_like(y_samp)

xy_samp = np.vstack([x_samp, y_samp]).T

vx_samp = uw.function.evaluate(v_soln.sym[0], xy_samp)
vx_theo = 1.5 * vel * (1 - (y_samp / ( maxY))**2)

# calculate the normalized L2 error
num = np.sqrt(((vx_samp - vx_theo)**2).sum())
denom = np.sqrt((vx_theo**2).sum())

l2_err = num/denom
print("L2 norm using sampled points")
print(f"Reynolds number: {Re_num}")
print(f"Normalized L2 error: {l2_err}")
print(f"Normalized L2 error in %: {100*l2_err}")

# %%
# calculate norm using integral

import math 

# calculate the L2 norm of velocity using integral 
x, y = meshbox.N.x, meshbox.N.y

vx_theo_expr = 1.5 * vel * (1 - (y / ( maxY))**2)

# calculate vx over the entire mesh variable
with meshbox.access(v_ana):
    v_ana.data[:, 0] = uw.function.evaluate(vx_theo_expr, v_ana.coords)

# create the mask function
vel_mask_fn = sympy.Piecewise((1.0, (x >= x_sample) & (x <= (x_sample + 0.2))), 
                              (0, True))

v_diff     = v_soln.sym - v_ana.sym
v_diff_mag = v_diff.dot(v_diff)

v_ana_mag  = v_ana.sym.dot(v_ana.sym) 

v_diff_mag_integ = math.sqrt(uw.maths.Integral(meshbox, vel_mask_fn * v_diff_mag).evaluate())
v_ana_mag_integ = math.sqrt(uw.maths.Integral(meshbox, vel_mask_fn * v_ana_mag).evaluate())
v_norm = v_diff_mag_integ / v_ana_mag_integ


# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi = 150)
ax.plot(vx_theo, y_samp, "-", lw = 1, label = "Analytical")
ax.plot(vx_samp, y_samp, "o", ms = 3, label = "Numerical")
ax.set_ylabel(r"$Y$ [m]")
ax.set_xlabel(r"$V_x$ [m/s]")
#ax.set_aspect("equal")

ax.legend()

# %%
# sample the pressure over the mid-line
y_sample = 0 # depends on where hydrodynamically fully developed region is
if case_num == 3:
    x_sample_min = 0.35
    x_sample_max = 0.4
else:
    x_sample_min = 0.1
    x_sample_max = 0.30


pres_x_samp = np.linspace(x_sample_min, x_sample_max, 50)
pres_y_samp = y_sample * np.ones_like(pres_x_samp)

pres_xy_samp = np.vstack([pres_x_samp, pres_y_samp]).T
p_samp_nn = np.zeros(pres_xy_samp.shape[0])

p_samp = uw.function.evaluate(p_soln.sym[0], pres_xy_samp)

# get nearest neighbor sampling
with meshbox.access(p_soln):
    for i in range(len(p_samp_nn)):

        dist = np.sqrt((pres_xy_samp[i, 0] - p_soln.coords[:, 0])**2 + (pres_xy_samp[i, 1] - p_soln.coords[:, 1])**2)
        idx = np.where(dist == dist.min())[0][0]
        #print(p_soln.data[idx,0])
        p_samp_nn[i] = p_soln.data[idx, 0]

linear_fit = np.polyfit(pres_x_samp, p_samp, deg = 1)
linear_fit_nn = np.polyfit(pres_x_samp, p_samp_nn, deg = 1)

v_avg = 0.5 * (1.5 * vel_dim)
pres_drop = 8 * dyn_visc_dim * v_avg * (x_sample_max - x_sample_min)/ (maxY**2)
theo_grad2 = pres_drop / (x_sample_min - x_sample_max)


# %%
linear_fit

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi = 150)
ax.plot(pres_x_samp, p_samp, "-o", ms = 3, lw = 1, label = "Numerical")
ax.plot(pres_x_samp, p_samp_nn, "-o", ms = 3, lw = 1, label = "Nearest neighbor")
ax.set_ylabel(r"$P$ [Pa]")
ax.set_xlabel(r"$X$ [m]")
ax.legend()


# %%
print("Summary of results:")
Courant = max_vel * dt_ns / delta_x

print(f"Min radius: {delta_x}")
print(f"Timestep used: {dt_ns}")
print(f"Courant number: {Courant}")
print(f"Reynolds number: {Re_num}")
print(f"Normalized L2 error: {l2_err}")
print(f"Normalized L2 error using integral: {v_norm}")
print(f"Theoretical pressure gradient: {theo_grad2}")
print(f"Error pressure gradient (evaluate): {np.abs(linear_fit[0] - theo_grad2)/np.abs(theo_grad2)}")
print(f"Error pressure gradient (nearest-neighbor): {np.abs(linear_fit_nn[0] - theo_grad2)/np.abs(theo_grad2)}")


