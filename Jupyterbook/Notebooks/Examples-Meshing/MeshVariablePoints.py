#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Creating a mesh (and checking the labels)
#
# This example is for the notch-localization test of Spiegelman et al. For which they supply a geometry file which gmsh can use to construct meshes at various resolutions. NOTE: we are just demonstrating the mesh here, not the solver configuration / benchmarking.
#
# The `.geo` file is provided and we show how to make this into a `.msh` file and
# how to read that into a `uw.discretisation.Mesh` object. The `.geo` file has header parameters to control the mesh refinement, and we provide a coarse version and the original version.
#
# After that, there is some cell data which we can assign to a data structure on the elements (such as a swarm).

# %%

import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

from underworld3.cython import petsc_discretisation


# %%
mesh1 = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.1)
mesh1.dm.view()


# %%
# This always seems to fail in parallel

dC0 = uw.discretisation.MeshVariable(r"dC_0", mesh1, 1, degree=0, continuous=False)
dC1 = uw.discretisation.MeshVariable(r"dC_1", mesh1, 1, degree=1, continuous=False)
dC2 = uw.discretisation.MeshVariable(r"dC_2", mesh1, 1, degree=2, continuous=False)
C1 = uw.discretisation.MeshVariable(r"C_1", mesh1, 1, degree=1, continuous=True)
C2 = uw.discretisation.MeshVariable(r"C_2", mesh1, 1, degree=2, continuous=True)


# %% [markdown]
# This is how we extract cell data from the mesh. We can map it to the swarm data structure

# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh1.vtk("tmp_notch.vtk")
    pvmesh = pv.read("tmp_notch.vtk")

    pl = pv.Plotter()

    var = dC1
    points = np.zeros((var.coords.shape[0], 3))
    points[:, 0] = var.coords[:, 0]
    points[:, 1] = var.coords[:, 1]
    point_cloud = pv.PolyData(points)

    var = dC2
    points = np.zeros((var.coords.shape[0], 3))
    points[:, 0] = var.coords[:, 0]
    points[:, 1] = var.coords[:, 1]
    point_cloud2 = pv.PolyData(points)

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
        opacity=0.5,
    )
    pl.add_points(point_cloud, color="Red", render_points_as_spheres=True, point_size=5, opacity=0.66)
    pl.add_points(point_cloud2, color="Blue", render_points_as_spheres=True, point_size=5, opacity=0.66)

    pl.show(cpos="xy")
