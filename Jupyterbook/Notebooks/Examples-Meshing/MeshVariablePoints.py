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

# + [markdown] magic_args="[markdown]"
# # Creating a mesh (and checking the labels)
#
# This example is for the notch-localization test of Spiegelman et al. For which they supply a geometry file which gmsh can use to construct meshes at various resolutions. NOTE: we are just demonstrating the mesh here, not the solver configuration / benchmarking.
#
# The `.geo` file is provided and we show how to make this into a `.msh` file and
# how to read that into a `uw.discretisation.Mesh` object. The `.geo` file has header parameters to control the mesh refinement, and we provide a coarse version and the original version.
#
# After that, there is some cell data which we can assign to a data structure on the elements (such as a swarm).
# -

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

from underworld3.cython import petsc_discretisation


# %%
mesh1 = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.1)
mesh1.dm.view()


# +
# %%
# petsc_discretisation.petsc_dm_get_periodicity(mesh1.dm)
# -

# %%
petsc_discretisation.petsc_dm_set_periodicity(
    mesh1.dm, (0.5, 3.14159, 0.0), (0.0, 0.0, 0.0), (1.0, 6.28, 0.0)
)

# %%
coodm = mesh1.dm.getCoordinateDM()
coodm.view()
mesh1.dm.localizeCoordinates()

# %%
mesh1.dm.view()

# +
# %%
# petsc_discretisation.petsc_dm_get_periodicity(mesh1.dm)
# -

# %%
dC0 = uw.discretisation.MeshVariable(r"dC_0", mesh1, 1, degree=0, continuous=False)
dC1 = uw.discretisation.MeshVariable(r"dC_1", mesh1, 1, degree=1, continuous=False)
dC2 = uw.discretisation.MeshVariable(r"dC_2", mesh1, 1, degree=2, continuous=False)
C1 = uw.discretisation.MeshVariable(r"C_1", mesh1, 1, degree=1, continuous=True)
C2 = uw.discretisation.MeshVariable(r"C_2", mesh1, 1, degree=2, continuous=True)


# + [markdown] magic_args="[markdown]"
# This is how we extract cell data from the mesh. We can map it to the swarm data structure
# -

# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    points = vis.meshVariable_to_pv_cloud(dC1)
    point_cloud = pv.PolyData(points)

    points = vis.meshVariable_to_pv_cloud(dC2)
    point_cloud2 = pv.PolyData(points)

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
        opacity=0.5,
    )
    pl.add_points(
        point_cloud,
        color="Red",
        render_points_as_spheres=True,
        point_size=5,
        opacity=0.66,
    )
    pl.add_points(
        point_cloud2,
        color="Blue",
        render_points_as_spheres=True,
        point_size=5,
        opacity=0.66,
    )

    pl.show(cpos="xy")


