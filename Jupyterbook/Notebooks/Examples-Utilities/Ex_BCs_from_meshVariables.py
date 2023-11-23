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

# # Cylindrical 2D Diffusion

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Poisson
import numpy as np

options = PETSc.Options()

options["dm_plex_check_all"] = None
options["poisson_ksp_rtol"] = 1.0e-3
# options["poisson_ksp_monitor_short"] = None
# options["poisson_nes_type"]  = "fas"
options["poisson_snes_converged_reason"] = None
options["poisson_snes_monitor_short"] = None
# options["poisson_snes_view"]=None
options["poisson_snes_rtol"] = 1.0e-3

import os

os.environ["SYMPY_USE_CACHE"] = "no"

# -

# Set some things
k = 1.0
h = 5.0
t_i = 2.0
t_o = 1.0
r_i = 0.5
r_o = 1.0

# +
dim = 2
radius_inner = 0.1
radius_outer = 1.0

import pygmsh

# Generate local mesh.
with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_max = 0.1
    if dim == 2:
        ndimspherefunc = geom.add_disk
    else:
        ndimspherefunc = geom.add_ball
    ball_outer = ndimspherefunc(
        [
            0.0,
        ]
        * dim,
        radius_outer,
    )

    if radius_inner > 0.0:
        ball_inner = ndimspherefunc(
            [
                0.0,
            ]
            * dim,
            radius_inner,
        )
        geom.boolean_difference(ball_outer, ball_inner)

    pygmesh0 = geom.generate_mesh()

# -

import meshio

mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.0, cellSize=0.1)

mesh.dm.view()

t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)

# +
# check the mesh if in a notebook / serial

if PETSc.Comm.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    clipped_stack = pvmesh.clip(
        origin=(0.0, 0.0, 0.0), normal=(-1, -1, 0), invert=False
    )

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Blue', 'wireframe' )
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
    )
    pl.show()


# +
# Create Poisson object

poisson = uw.systems.Poisson(
    mesh, u_Field=t_soln, solver_name="poisson", verbose=True
)

poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.k = k
poisson.f = 0.0

bcs_var = uw.discretisation.MeshVariable("bcs", mesh, 1)

# +
import sympy

abs_r = sympy.sqrt(mesh.rvec.dot(mesh.rvec))
bc = sympy.cos(2.0 * mesh.N.y)

with mesh.access(bcs_var):
    bcs_var.data[:, 0] = uw.function.evaluate(bc, bcs_var.coords)

poisson.add_dirichlet_bc(bcs_var.sym[0], "Upper", components=0)
poisson.add_dirichlet_bc(-1.0, "Centre", components=0)
# -

poisson.petsc_options.getAll()

# +
# poisson._setup_terms()
# -

poisson.solve()

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)

    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, mesh.data)

    # clipped_stack = pvmesh.clip(origin=(0.001,0.0,0.0), normal=(1, 0, 0), invert=False)

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Blue', 'wireframe' )
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
    )
    pl.show(cpos="xy")
# -
mesh.petsc_save_checkpoint(index=0, meshVars=[t_soln], outputPath='./output/')


# +
## We should try the non linear version of this next ...
