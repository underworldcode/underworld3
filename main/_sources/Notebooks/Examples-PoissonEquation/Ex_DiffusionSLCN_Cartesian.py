# ---
# jupyter:
#   jupytext:
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

# # Linear diffusion of a hot pipe
#
# - Using the adv_diff solver.
# - No advection as the velocity field is not updated (and set to 0).
# - Benchmark comparison between 1D numerical solution and 2D UW model.
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI

import math

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt


# %%
sys = PETSc.Sys()
sys.pushErrorHandler("traceback")


# %%
# Set the resolution.
res = 32

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

pipe_thickness = 0.4  ###

# %%
k0 = 1e-6  ### m2/s (diffusivity)
l0 = 1e5  ### 100 km in m (length of box)
time_scale = l0**2 / k0  ### s
time_scale_Myr = time_scale / (60 * 60 * 24 * 365.25 * 1e6)

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / res, regular=True)

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax)
)


# Create adv_diff object

# Set some things
k = 1.0

tmin = 0.5
tmax = 1.0

# Create an adv
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=T,
    V_fn=v,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k

# %%
### fix temp of top and bottom walls
adv_diff.add_dirichlet_bc(0.5, "Bottom", 0)
adv_diff.add_dirichlet_bc(0.5, "Top", 0)


# %%
maxY = mesh.data[:, 1].max()
minY = mesh.data[:, 1].min()

with mesh.access(T):
    T.data[...] = tmin

    pipePosition = ((maxY - minY) - pipe_thickness) / 2.0

    T.data[
        (mesh.data[:, 1] >= (mesh.data[:, 1].min() + pipePosition))
        & (mesh.data[:, 1] <= (mesh.data[:, 1].max() - pipePosition))
    ] = tmax


# %%
def plot_fig():
    if uw.mpi.size == 1:
        
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(mesh)
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T.sym)
        
        velocity_points = vis.meshVariable_to_pv_cloud(v)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)

        pl = pv.Plotter(window_size=(750, 750))

        pl.add_mesh(pvmesh, "Black", "wireframe")

        # pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.95,
        )

        # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S",
        #               use_transparency=False, opacity=0.5)

        # pl.add_mesh(
        #     point_cloud,
        #     cmap="coolwarm",
        #     edge_color="Black",
        #     show_edges=False,
        #     scalars="M",
        #     use_transparency=False,
        #     opacity=0.95,
        # )

        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=5.0, opacity=0.5)
        # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

        # pl.add_points(pdata)

        pl.show(cpos="xy")

        # return vsol


plot_fig()

# ## Vertical profile across the centre of the box

### y coords to sample
sample_y = np.arange(
    mesh.data[:, 1].min(), mesh.data[:, 1].max(), mesh.get_min_radius()
)  ### Vertical profile

### x coords to sample
# sample_x = np.repeat(mesh.data[:,0].min(), sample_y.shape[0]) ### LHS wall
sample_x = np.zeros_like(sample_y)  ### centre of the box

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y

t0 = uw.function.evaluate(adv_diff.u.fn, sample_points)

# %%
### estimate the timestep based on diffusion only
dt = mesh.get_min_radius() ** 2 / k  ### dt = length squared / diffusivity
# print(f'dt: {dt*time_scale_Myr} Myr')
print(f"dt: {dt*time_scale_Myr}")


# %%
def diffusion_1D(sample_points, tempProfile, k, model_dt):
    x = sample_points
    T = tempProfile

    dx = sample_points[1] - sample_points[0]

    dt = 0.5 * (dx**2 / k)

    """ max time of model """
    total_time = model_dt

    """ get min of 1D and 2D model """
    time_1DModel = min(model_dt, dt)

    """ determine number of its """
    nts = math.ceil(total_time / time_1DModel)

    """ get dt of 1D model """
    final_dt = total_time / nts

    for i in range(nts):
        qT = -k * np.diff(T) / dx
        dTdt = -np.diff(qT) / dx
        T[1:-1] += dTdt * final_dt

    return T


### get the initial temp profile
tempData = uw.function.evaluate(adv_diff.u.fn, sample_points)

# +

step = 0
time = 0.0
# -

# %%
nsteps = 1 # 21

# +

if uw.mpi.size == 1:
    """create figure to show the temp diffuses"""
    plt.figure(figsize=(9, 3))
    plt.plot(t0, sample_points[:, 1], ls=":")
# -


while step < nsteps:
    ### print some stuff
    if uw.mpi.rank == 0:
        # print(f"Step: {str(step).rjust(3)}, time: {time*time_scale_Myr:6.2f} [MYr]")
        print(f"Step: {str(step).rjust(3)}, time: {time:6.5f}")

    ### 1D profile from underworld
    t1 = uw.function.evalf(adv_diff.u.sym[0], sample_points)

    if uw.mpi.size == 1 and step % 10 == 0:
        """compare 1D and 2D models"""
        plt.figure()
        ### profile from UW
        plt.plot(t1, sample_points[:, 1], ls="-", c="red", label="2D linear model")
        ### numerical solution
        plt.plot(tempData, sample_points[:, 1], ls=":", c="k", label="1D linear model")
        plt.legend()
        plt.show()

    ### 1D diffusion
    tempData = diffusion_1D(
        sample_points=sample_points[:, 1], tempProfile=tempData, k=k, model_dt=dt
    )

    ### diffuse through underworld
    adv_diff.solve(timestep=dt)

    step += 1
    time += dt
