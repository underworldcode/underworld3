# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Stokes Benchmark SolCx
#
#
# options = PETSc.Options()
# options["help"] = None


# %%
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
import numpy as np

# %%
n_els = 16
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 20, qdegree=2
)


# %%
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
stokes.constitutive_model.Parameters.viscosity = 1


# %%
# Set some things
import sympy
from sympy import Piecewise

x, y = mesh.CoordinateSystem.X

res = 1 / n_els
hw = 1000 / res
surface_fn = sympy.exp(-((y - 1.0) ** 2) * hw)
base_fn = sympy.exp(-(y**2) * hw)
right_fn = sympy.exp(-((x - 1.0) ** 2) * hw)
left_fn = sympy.exp(-(x**2) * hw)

eta_0 = 1.0
x_c = 0.5
f_0 = 1.0


stokes.penalty = 0.0
stokes.bodyforce = sympy.Matrix(
    [
        0,
        Piecewise(
            (f_0, x > x_c),
            (0.0, True),
        ),
    ]
)

# +
# stokes.bodyforce[0] -= 1.0e6 * v.sym[0] * (left_fn + right_fn)
# stokes.bodyforce[1] -= 1.0e6 * v.sym[1] * (surface_fn + base_fn)
# -

stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity

# free slip.
# note with petsc we always need to provide a vector of correct cardinality.
stokes.add_dirichlet_bc(
    (0.0, 0.0), ["Top", "Bottom"], 1
)  # top/bottom: components, function, markers
stokes.add_dirichlet_bc(
    (0.0, 0.0), ["Left", "Right"], 0
)  # left/right: components, function, markers


# We may need to adjust the tolerance if $\Delta \eta$ is large

stokes.petsc_options["snes_rtol"] = 1.0e-6
stokes.petsc_options["ksp_rtol"] = 1.0e-6
stokes.petsc_options["snes_max_it"] = 10

# stokes.petsc_options["snes_monitor"]= None
# stokes.petsc_options["ksp_monitor"] = None


# %%
# Solve time
stokes.solve()

# ### Visualise it !

# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    pvmesh.point_data["P"] = uw.function.evaluate(p.sym[0], mesh.data)
    pvmesh.point_data["V"] = uw.function.evaluate(v.sym.dot(v.sym), mesh.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0] = uw.function.evaluate(stokes.u.sym[0], stokes.u.coords)
    arrow_length[:, 1] = uw.function.evaluate(stokes.u.sym[1], stokes.u.coords)

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="V",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=3)

    pl.show(cpos="xy")


# %% [markdown]
# # SolCx from the same setup

# %%
stokes.bodyforce = sympy.Matrix(
    [0, -sympy.cos(sympy.pi * x) * sympy.sin(2 * sympy.pi * y)]
)
viscosity_fn = sympy.Piecewise(
    (
        1.0e6,
        x > x_c,
    ),
    (1.0, True),
)
stokes.constitutive_model.Parameters.viscosity = viscosity_fn
stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity

# %%
stokes.constitutive_model.Parameters.viscosity

# %%
stokes.solve()

# %%
# OR

# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    pvmesh.point_data["P"] = uw.function.evaluate(p.sym[0], mesh.data)
    pvmesh.point_data["V"] = uw.function.evaluate(v.sym.dot(v.sym), mesh.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0] = uw.function.evaluate(stokes.u.sym[0], stokes.u.coords)
    arrow_length[:, 1] = uw.function.evaluate(stokes.u.sym[1], stokes.u.coords)

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="V",
        use_transparency=False,
        opacity=1.0,
    )

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
    #               use_transparency=False, opacity=1.0)

    pl.add_arrows(arrow_loc, arrow_length, mag=50)

    pl.show(cpos="xy")


# %%
try:
    import underworld as uw2

    solC = uw2.function.analytic.SolC()
    vel_soln_analytic = solC.fn_velocity.evaluate(mesh.data)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    from numpy import linalg as LA

    with mesh.access(v):
        num = function.evaluate(v.fn, mesh.data)  # this appears busted
        if comm.rank == 0:
            print("Diff norm a. = {}".format(LA.norm(v.data - vel_soln_analytic)))
            print("Diff norm b. = {}".format(LA.norm(num - vel_soln_analytic)))
        # if not np.allclose(v.data, vel_soln_analytic, rtol=1):
        #     raise RuntimeError("Solve did not produce expected result.")
    comm.barrier()
except ImportError:
    import warnings

    warnings.warn("Unable to test SolC results as UW2 not available.")

# %%
