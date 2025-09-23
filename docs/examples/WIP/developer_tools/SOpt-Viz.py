# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Structural Optimisation Visualiser
#
# Set up a Stokes flow with obstructions, solve and then try to recover obstructions
#

# %%
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
from underworld3 import timing

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import sympy


# %%
width = 4.0
height = 1.0

# %%
## Equivalent mesh 

openmesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), maxCoords=(width, 1.0), qdegree=3, cellSize=0.033)

v_phi =  uw.discretisation.MeshVariable("V_phi",openmesh, openmesh.dim, degree=2, varsymbol="V_\phi")
v_solno =  uw.discretisation.MeshVariable("Vo",openmesh, openmesh.dim, degree=2)
v_soln1 = uw.discretisation.MeshVariable("V1", openmesh, openmesh.dim, degree=2)
u_soln1 = uw.discretisation.MeshVariable("U1", openmesh, openmesh.dim, degree=2)

obstruction = uw.discretisation.MeshVariable("Beta", openmesh, 1, degree=2, continuous=True, varsymbol=r"\beta")
d_obstruction = uw.discretisation.MeshVariable("dBeta", openmesh, 1, degree=2, continuous=True, varsymbol=r"d\beta")
obstruction_function = (1 + sympy.tanh(10*obstruction.sym[0]))/2


# Set solve options here (or remove default values
# stokes.petsc_options.getAll()


# %%
v_solno.read_timestep("TargetSolution", "V0", 0, outputPath=".", verbose=True)

# %%
# Animate

start = 50
end = 52

expt = "SOpt_6"

import pyvista as pv
import underworld3.visualisation as vis

pvmesh = vis.mesh_to_pv_mesh(openmesh)
velocity_points = vis.meshVariable_to_pv_cloud(v_soln1)
pl = pv.Plotter(window_size=(1000, 750))

for i in range(start, end, 1):

    print(f"Step {i}")

    v_soln1.read_timestep(
        expt,
        data_name="V1",
        outputPath="output",
        index=i)

    v_phi.read_timestep(
        expt,
        data_name="V_phi",
        outputPath="output",
        index=i)

    obstruction.read_timestep(
        expt,
        data_name="Beta",
        outputPath="output",
        index=i)
    
    d_obstruction.read_timestep(
        expt,
        data_name="dBeta",
        outputPath="output",
        index=i)

    pvmesh.point_data["Beta"] = vis.scalar_fn_to_pv_points(pvmesh, obstruction_function)
    pvmesh.point_data["dBeta"] = vis.scalar_fn_to_pv_points(pvmesh, d_obstruction.sym)
    pvmesh.point_data["Vo"] = vis.vector_fn_to_pv_points(pvmesh, v_solno.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln1.sym)
    pvmesh.point_data["Vphi"] = vis.vector_fn_to_pv_points(pvmesh, v_phi.sym)

    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln1.sym)
    velocity_points.point_data["Vo"] = vis.vector_fn_to_pv_points(velocity_points, v_solno.sym)
    velocity_points.point_data["Vphi"] = vis.vector_fn_to_pv_points(velocity_points, v_phi.sym)

    beta_mesh = vis.meshVariable_to_pv_mesh_object(obstruction, alpha=0.05)
    beta_mesh.point_data["Beta"] = vis.scalar_fn_to_pv_points(beta_mesh, obstruction.sym)
    beta_mesh.point_data["dBeta"] = vis.scalar_fn_to_pv_points(beta_mesh, d_obstruction.sym)


    dV = v_soln1.sym - v_solno.sym
    dVmag = sympy.sqrt(dV.dot(dV))

    beta_mesh.point_data["dVmag"] = vis.scalar_fn_to_pv_points(beta_mesh, dVmag)

    Vomag = np.sqrt((velocity_points.point_data["Vo"][:,0]**2 + velocity_points.point_data["Vo"][:,1]**2).mean() )
    Vphimag = np.sqrt((velocity_points.point_data["Vphi"][:,0]**2 + velocity_points.point_data["Vphi"][:,1]**2).mean() )


    # point sources at cell centres
    skip=7
    points = np.zeros((openmesh._centroids[::skip].shape[0], 3))
    points[:, 0] = openmesh._centroids[::skip, 0]
    points[:, 1] = openmesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)


    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="Vo", integration_direction="forward", 
        surface_streamlines=True, max_steps=100)

    pvstream2 = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", 
        surface_streamlines=True, max_steps=100)


    pl.clear()

    # print(f"Vphi: {Vphimag}; Vo: {Vomag}") 


    pl.add_arrows(velocity_points.points, 
                  velocity_points.point_data["Vphi"], 
                  mag=0.01/Vphimag , opacity=0.25, 
                  color="Red",
                  show_scalar_bar=False)

    pl.add_arrows(velocity_points.points, 
                  velocity_points.point_data["V"], 
                  mag=0.01/Vomag, opacity=0.5, 
                  color="Green",
                  show_scalar_bar=False)
    
    pl.add_arrows(velocity_points.points, 
                  velocity_points.point_data["Vo"], 
                  mag=0.01/Vomag, opacity=0.5, 
                  color="Blue",
                  show_scalar_bar=False)

    pl.add_mesh(
        beta_mesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="dBeta",
        use_transparency=False,
        opacity=0.5,
        clim=[-25, 25],
        show_scalar_bar=False)

    pl.add_mesh(
        beta_mesh.copy(),
        cmap="Grays",
        edge_color="Black",
        show_edges=False,
        scalars="Beta",
        use_transparency=False,
        opacity=.75,
        # clim=[-1, 1],
        show_scalar_bar=False)

    
    pl.add_mesh(pvstream, 
                color="Blue",
                show_scalar_bar=False, opacity=0.25)

    pl.add_mesh(pvstream2, 
                color="Green",
                show_scalar_bar=False, opacity=0.25)

    pl.camera.position = (2.0, 0.5, 3)
    pl.camera.focal_point=(2.0,0.5,0.0)

    imagefile ="images/" + f"{expt}_{i}.png"
    
    pl.screenshot(filename=imagefile,
                  window_size=(3000, 1000),
                  return_img=False)
    
    # pl.show(jupyter_backend="html")
    


# %% [markdown]
# ## Visualisation
#
# We can check the progress of the shape optimisation. Here the target arrows / streamlines are in blue and the current
# solution is in green. The dark blobs are the obsructions that the inversion has reconstructed. The red arrows are the 
# interface pseudo-velocities.

# %%
pl.show(jupyter_backend="html")

# %%
# ls -trl output/SOpt_5*
