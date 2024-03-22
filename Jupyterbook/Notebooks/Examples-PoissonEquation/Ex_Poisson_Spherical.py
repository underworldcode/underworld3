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

# # Steady state diffusion in a hollow sphere

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import pygmsh, meshio

# +
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Poisson
import numpy as np
import os

# Define the problem size
#      1 - ultra low res for automatic checking
#      2 - low res problem to play with this notebook
#      3 - medium resolution (be prepared to wait)
#      4 - highest resolution (benchmark case from Spiegelman et al)

problem_size = 1

# For testing and automatic generation of notebook output,
# over-ride the problem size if the UW_TESTING_LEVEL is set

uw_testing_level = os.environ.get("UW_TESTING_LEVEL")
if uw_testing_level:
    try:
        problem_size = int(uw_testing_level)
    except ValueError:
        # Accept the default value
        pass

# -


# Set some things
k = 1.0
f = 0.0
t_i = 2.0
t_o = 1.0
r_i = 0.5
r_o = 1.0

# %%
from underworld3.meshing import Annulus

# %%
# first do 2D
if problem_size <= 1:
    cell_size = 0.05
elif problem_size == 2:
    cell_size = 0.02
elif problem_size == 3:
    cell_size = 0.01
elif problem_size >= 4:
    cell_size = 0.0033

mesh = Annulus(radiusInner=r_i, radiusOuter=r_o, cellSize=cell_size)

t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# Create Poisson object
poisson = Poisson(mesh, u_Field=t_soln)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1

poisson.f = f

poisson.petsc_options["snes_rtol"] = 1.0e-6
poisson.petsc_options.delValue("ksp_monitor")
poisson.petsc_options.delValue("ksp_rtol")


# %%
mesh.dm.view()

# %%
import sympy

poisson.add_dirichlet_bc(t_i, "Lower", 0)
poisson.add_dirichlet_bc(t_o, "Upper", 0)

# %%
poisson.solve()

# +
# poisson.snes.view()
# -

# %%
# Check. Construct simple solution for above config.
import math

A = (t_i - t_o) / (sympy.log(r_i) - math.log(r_o))
B = t_o - A * sympy.log(r_o)
sol = A * sympy.log(sympy.sqrt(mesh.N.x**2 + mesh.N.y**2)) + B

with mesh.access():
    mesh_analytic_soln = uw.function.evaluate(sol, mesh.data, mesh.N)
    mesh_numerical_soln = uw.function.evaluate(t_soln.fn, mesh.data, mesh.N)

import numpy as np

if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.01):
    raise RuntimeError("Unexpected values encountered.")

# %%
poisson.constitutive_model.Parameters.diffusivity = 1.0 + 0.1 * poisson.u.fn**1.5
poisson.f = 0.01 * poisson.u.sym[0] ** 0.5
poisson.solve(zero_init_guess=False)

# Validate

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["T"] = mesh_analytic_soln
    pvmesh.point_data["T2"] = mesh_numerical_soln
    pvmesh.point_data["DT"] = mesh_analytic_soln - mesh_numerical_soln

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="DT",
        use_transparency=False,
        opacity=0.5,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")

# +
# %%

expt_name = "Poisson-Annulus"
outdir = "output"
os.makedirs(f"{outdir}", exist_ok=True)


mesh.write_timestep(
    expt_name, meshUpdates=True, meshVars=[t_soln], outputPath=outdir, index=0
)


# savefile = "output/poisson_disc.h5"
# mesh.save(savefile)
# poisson.u.save(savefile)
# mesh.generate_xdmf(savefile)
# -

# %%
from underworld3.meshing import SphericalShell
from underworld3.meshing import SegmentedSphere

# +
# %%
# now do 3D

problem_size = 1

if problem_size <= 1:
    cell_size = 0.3
elif problem_size == 1:
    cell_size = 0.15
elif problem_size == 2:
    cell_size = 0.05
elif problem_size == 3:
    cell_size = 0.02

mesh_3d = SphericalShell(
    radiusInner=r_i,
    radiusOuter=r_o,
    cellSize=cell_size,
    refinement=1,
)

# mesh_3d = SegmentedSphere(radiusInner=r_i,
#                          radiusOuter=r_o,
#                          cellSize=cell_size
#                         )


t_soln_3d = uw.discretisation.MeshVariable("T", mesh_3d, 1, degree=2)
# -

mesh_3d.dm.view()

# Create Poisson object
poisson = Poisson(mesh_3d, u_Field=t_soln_3d)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = k
poisson.f = f

poisson.petsc_options["snes_rtol"] = 1.0e-6
poisson.petsc_options.delValue("ksp_rtol")

import sympy

poisson.add_dirichlet_bc(t_i, "Lower", 0)
poisson.add_dirichlet_bc(t_o, "Upper", 0)

# Solve time
poisson.solve()

# Check. Construct simple solution for above config.

A = (t_i - t_o) / (1 / r_i - 1 / r_o)
B = t_o - A / r_o
sol = A / (sympy.sqrt(mesh_3d.N.x**2 + mesh_3d.N.y**2 + mesh_3d.N.z**2)) + B

with mesh.access():
    mesh_analytic_soln = uw.function.evaluate(sol, mesh_3d.data, mesh_3d.N)
    mesh_numerical_soln = uw.function.evaluate(t_soln_3d.fn, mesh_3d.data, mesh_3d.N)

import numpy as np

if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.1):
    raise RuntimeError("Unexpected values encountered.")

# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    
    pvmesh = vis.mesh_to_pv_mesh(mesh_3d)
    pvmesh.point_data["T"] = mesh_analytic_soln
    pvmesh.point_data["T2"] = mesh_numerical_soln
    pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"]
    
    clipped = pvmesh.clip(origin=(0.001, 0.0, 0.0), normal=(1, 0, 0), invert=True)

    pl = pv.Plotter()

    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="T2",
        use_transparency=False,
        opacity=1.0,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")

# +
# %%
expt_name = "Poisson-Sphere"
outdir = "output"
os.makedirs(f"{outdir}", exist_ok=True)

mesh_3d.write_timestep(
    expt_name, meshUpdates=True, meshVars=[t_soln_3d], outputPath=outdir, index=0
)

# -


