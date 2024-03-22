#!/usr/bin/env python
# coding: utf-8
# %%
# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# %%
import petsc4py
from petsc4py import PETSc
import underworld3 as uw
from underworld3.swarm import SwarmPICLayout
import numpy as np


# %%
n_els = 2
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), 
    cellSize=1 / n_els, qdegree=2, refinement=0
)
mesh.dm.view()


# %%
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)


# %%
swarm = uw.swarm.Swarm(mesh=mesh)
swarm.populate_petsc(
    fill_param=2,
    layout = SwarmPICLayout.GAUSS
)


# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)

    
    # with mesh.access():
    #     usol = v_soln.data.copy()

    # with mesh.access():
    #     pvmesh.point_data["Vmag"] = uw.function.evalf(
    #         sympy.sqrt(v_soln.sym.dot(v_soln.sym)), mesh.data
    #     )
    #     pvmesh.point_data["P"] = uw.function.evalf(p_soln.fn, mesh.data)


    # v_vectors = np.zeros((mesh.data.shape[0], 3))
    # v_vectors[:, 0] = uw.function.evalf(v_soln[0].sym, mesh.data)
    # v_vectors[:, 1] = uw.function.evalf(v_soln[1].sym, mesh.data)
    # pvmesh.point_data["V"] = v_vectors

    # arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    # arrow_loc[:, 0:2] = v_soln.coords[...]

    # arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    # arrow_length[:, 0:2] = usol[...]

    # point sources at cell centres
    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    spoints = vis.swarm_to_pv_cloud(swarm)
    spoint_cloud = pv.PolyData(spoints)

    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_arrows(arrow_loc, arrow_length, mag=0.025 / U0, opacity=0.75)

    pl.add_points(spoint_cloud,color="Black",
                  render_points_as_spheres=False,
                  point_size=5, opacity=0.66
                )

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
    # pl.add_mesh(pvstream)

    # pl.remove_scalar_bar("mag")

    pl.show()


# %%
with swarm.access():
    print(swarm.data.shape)

# %%
14*6

# %%
mesh._centroids.shape

# %%
