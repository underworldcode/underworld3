# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Poisson Equation (simple)
#
# First we show how this works using the generic class and then the minor differences for
# the `Poisson` class
#
# ## Generic scalar solver class


# +
from petsc4py import PETSc

import os
os.environ["UW_TIMING_ENABLE"] = "1"

import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy

from IPython.display import display


# +
mesh1 = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                          cellSize=1.0 / 4, refinement=4)

mesh2 = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), 
                                          cellSize=1.0 / 4, regular=True, refinement=4)
# -

# pick a mesh
mesh = mesh1

phi = uw.discretisation.MeshVariable("Phi", mesh, 1, degree=2, varsymbol=r"\phi" )
scalar = uw.discretisation.MeshVariable("Theta", mesh, 1, degree=1, continuous=False, varsymbol=r"\Theta")

# Create Poisson object

poisson = uw.systems.Poisson(mesh, u_Field=phi, solver_name="diffusion")

# Constitutive law (diffusivity)

poisson.constitutive_model = uw.systems.constitutive_models.DiffusionModel(phi)
poisson.constitutive_model.Parameters.diffusivity = 1


# %%
poisson.constitutive_model.c

# +
# Set some things
poisson.f = 0.0
poisson.add_dirichlet_bc(1.0, "Bottom", component=0)
poisson.add_dirichlet_bc(0.0, "Top", component=0)

poisson.tolerance = 1.0e-6
poisson.petsc_options["snes_type"] = "newtonls"
poisson.petsc_options["ksp_type"] = "fgmres"

poisson.petsc_options['snes_monitor'] = None
poisson.petsc_options['ksp_monitor'] = None
poisson.petsc_options.setValue("pc_type", "mg")
poisson.petsc_options.setValue("pc_mg_type", "multiplicative")
poisson.petsc_options.setValue("pc_mg_type", "kaskade")
# poisson.petsc_options["mg_levels"] = mesh.dm.getRefineLevel()-2
poisson.petsc_options["mg_levels_ksp_type"] = "fgmres"
poisson.petsc_options["mg_levels_ksp_max_it"] = 100
poisson.petsc_options["mg_levels_ksp_converged_maxits"] = None
poisson.petsc_options["mg_coarse_pc_type"] = "svd"

# -

poisson.view()

poisson._setup_pointwise_functions(verbose=True)

poisson._setup_discretisation()

timing.reset()
timing.start()

# %%
# Solve time
poisson.solve()

type(poisson.F1)

# %%
# Check. Construct simple linear function which is solution for
# above config.  Exclude boundaries from mesh data.
import numpy as np

with mesh.access():
    mesh_numerical_soln = uw.function.evalf(poisson.u.fn, mesh.data)
    mesh_analytic_soln = uw.function.evalf(1.0 - mesh.N.y, mesh.data)
    if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.0001):
        print("Unexpected values encountered.")


# Validate

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh = pv.read("ignore_mesh.vtk")

    pvmesh.point_data["T"] = mesh_analytic_soln
    pvmesh.point_data["T2"] = mesh_numerical_soln
    pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"]

    sargs = dict(interactive=True)  # doesn't appear to work :(
    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="DT",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")

# Create some arbitrary function using one of the base scalars x,y[,z] = mesh.X

import sympy

x, y = mesh.X
x0 = y0 = 1 / sympy.sympify(2)
k = sympy.exp(-((x - x0) ** 2 + (y - y0) ** 2))

poisson.constitutive_model.Parameters.diffusivity=k

poisson.constitutive_model.flux

with mesh.access():
    orig_soln = poisson.u.data.copy()

orig_soln_mesh = uw.function.evalf(phi.sym[0], mesh.data)

# %%
poisson.solve(zero_init_guess=True, _force_setup=True)

print(poisson._u.stats())

# Simply confirm results are different

with mesh.access():
    if np.allclose(poisson.u.data, orig_soln, rtol=0.001):
        raise RuntimeError("Values did not change !")


mesh._evaluation_hash = None

# Visual validation

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh2 = pv.read("ignore_mesh.vtk")

    pvmesh2.point_data["T"] = uw.function.evaluate(phi.sym[0], mesh.data)
    pvmesh2.point_data["Te"] = uw.function.evalf(phi.sym[0], mesh.data)
    pvmesh2.point_data["dT"] = pvmesh2.point_data["T"] - pvmesh.point_data["T"]
    pvmesh2.point_data["dTe"] = pvmesh2.point_data["T"] - pvmesh2.point_data["Te"]

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh2,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="dT",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")


# ## Non-linear example


# RHS term

abs_r2 = x**2 + y**2
poisson.f = -16 * abs_r2
poisson.add_dirichlet_bc(abs_r2, "Bottom" , component=0)
poisson.add_dirichlet_bc(abs_r2, "Top", component=0)
poisson.add_dirichlet_bc(abs_r2, "Right", component=0)
poisson.add_dirichlet_bc(abs_r2, "Left", component=0)

display(poisson.f)

# Constitutive law (diffusivity)
# Linear solver first

poisson.constitutive_model = uw.systems.constitutive_models.DiffusionModel(poisson._u)
poisson.constitutive_model.Parameters.diffusivity=1

poisson.solve()


# Non-linear diffusivity

grad_phi = mesh.vector.gradient(phi.sym)
k = 5 + (grad_phi.dot(grad_phi)) / 2
poisson.constitutive_model.Parameters.diffusivity=k
poisson.constitutive_model.c


# %%
poisson._setup_pointwise_functions()
poisson._G3


# Use initial guess from linear solve

poisson.solve(zero_init_guess=False)

# Validate

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh2 = pv.read("ignore_mesh.vtk")

    pvmesh2.point_data["T"] = uw.function.evalf(phi.sym[0], mesh.data)

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh2,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")

# ## Analysis (Gradient recovery)
#
# We'd like to be able to look at the values of diffusivity or the
# heat flux.
#
# These are discontinuous values computed in the element interiors but can
# be projected to a `meshVariable`:
#

# %%
projection = uw.systems.Projection(mesh, scalar)
projection.uw_function = sympy.diff(phi.sym[0], mesh.X[1])
projection.smoothing = 1.0e-4

projection.solve()


with mesh.access():
    print(phi.stats())    
    print(scalar.stats())

# %%
sympy.diff(scalar.sym[0], mesh.X[1])

# Validate

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh2 = pv.read("ignore_mesh.vtk")

    pvmesh2.point_data["K"] = uw.function.evalf(scalar.sym[0], mesh.data)
    pvmesh2.point_data["T"] = uw.function.evalf(phi.sym[0], mesh.data)

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh2,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")

poisson.snes.view()

timing.print_table()



