# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

res = 0.2
r_o = 2
r_int = 1.8
r_i = 1

free_slip_upper = True

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None

import os

os.environ["SYMPY_USE_CACHE"] = "no"


# %%
meshball = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                              radiusInternal=r_int, 
                                              radiusInner=r_i, 
                                              cellSize_Inner=res,
                                              cellSize_Internal=res*0.5,
                                              cellSize_Outer=res,
                                              qdegree = 3,
                                              degree=1,
                                              filename="tmp_fixedstarsMesh.msh")
meshball.dm.view()

# %%
normal_vector = uw.discretisation.MeshVariable(r"\mathbf{n}", meshball, 2, degree=2)

# %%
projection = uw.systems.Vector_Projection(meshball, normal_vector)
projection.uw_function = sympy.Matrix([[0,0]])
projection.smoothing = 1.0e-3

GammaNorm = meshball.Gamma.dot(meshball.CoordinateSystem.unit_e_0) / sympy.sqrt(meshball.Gamma.dot(meshball.Gamma))

projection.add_natural_bc(meshball.Gamma * GammaNorm, "Upper")
projection.add_natural_bc(meshball.Gamma * GammaNorm, "Lower")
projection.add_natural_bc(meshball.Gamma * GammaNorm, "Internal")

projection.solve(verbose=True, debug=True)

with meshball.access(normal_vector):
    normal_vector.data[:,:] /= np.sqrt(normal_vector.data[:,0]**2 + normal_vector.data[:,1]**2).reshape(-1,1)

# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["N"] = vis.vector_fn_to_pv_points(pvmesh, normal_vector.sym)
    pvmesh.point_data["R"] = vis.vector_fn_to_pv_points(pvmesh, meshball.CoordinateSystem.unit_e_0)

    evaluation_points = vis.meshVariable_to_pv_cloud(normal_vector)
    evaluation_points.point_data["N"] = vis.vector_fn_to_pv_points(evaluation_points, normal_vector.sym)
    evaluation_points.point_data["R"] = vis.vector_fn_to_pv_points(evaluation_points, -meshball.CoordinateSystem.unit_e_0)
    evaluation_points.point_data["dR"] = evaluation_points.point_data["N"] - evaluation_points.point_data["R"]

    # pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_cont.sym)
    # pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_init)
    # pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False
    )

    # pl.add_arrows(evaluation_points.points, evaluation_points.point_data["N"], mag=0.1)
    # pl.add_arrows(evaluation_points.points, evaluation_points.point_data["R"], mag=0.1)
    pl.add_arrows(evaluation_points.points, evaluation_points.point_data["dR"], mag=10)
    # pl.add_mesh(pvstream, opacity=0.3, show_scalar_bar=False)


    pl.show(cpos="xy")

# %%
# ls -trl

# %%

# %%
