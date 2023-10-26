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

# # Poisson Equation with flux recovery
#
#
# ## Generic scalar solver class

from petsc4py import PETSc
import underworld3 as uw

import numpy as np
import sympy

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 3.0, qdegree=2
)

mesh.dm.view()


# mesh variables

t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
dTdY = uw.discretisation.MeshVariable(
    r"\partial T/ \partial \mathbf{y}", mesh, 1, degree=2
)
kappa = uw.discretisation.MeshVariable(r"\kappa", mesh, 1, degree=2)
gradT = uw.discretisation.MeshVariable(
    r"\nabla\left[T\right]", mesh, mesh.dim, degree=2
)


# Create Poisson object

gradient = uw.systems.Projection(mesh, dTdY)
delT = mesh.vector.gradient(t_soln.sym)
gradient.uw_function = delT.dot(delT)
gradient.smoothing = 1.0e-3

# These are both SNES Scalar objects

gradT_projector = uw.systems.Vector_Projection(mesh, gradT)
gradT_projector.uw_function = mesh.vector.gradient(t_soln.sym)
# gradT_projector.add_dirichlet_bc((0), ["Left", "Right"], components=(0))

# # the actual solver

poisson = uw.systems.Poisson(mesh, u_Field=t_soln)


# %%
poisson.constitutive_model = uw.constitutive_models.DiffusionModel(t_soln)

# Non-linear diffusivity

delT = mesh.vector.gradient(t_soln.sym)
k = 5 + (delT.dot(delT)) / 2

poisson.constitutive_model.Parameters.diffusivity = k
display(poisson.constitutive_model.c)

# projector for diffusivity (though we can just switch the rhs for the gradient object

# +
diffusivity = uw.systems.Projection(mesh, kappa)
diffusivity.uw_function = sympy.Matrix(
    [poisson.constitutive_model.Parameters.diffusivity]
)

diffusivity.add_dirichlet_bc(k, "Bottom", component=0)
diffusivity.add_dirichlet_bc(k, "Top", component=0)
diffusivity.add_dirichlet_bc(k, "Right", component=0)
diffusivity.add_dirichlet_bc(k, "Left", component=0)

diffusivity.smoothing = 1.0e-6
# -


# %%
poisson.constitutive_model = uw.constitutive_models.DiffusionModel(t_soln)
poisson.constitutive_model.Parameters.diffusivity = k
poisson.constitutive_model.Parameters.diffusivity

# %%
display(gradT_projector.uw_function)
display(diffusivity.uw_function)

# %%
diffusivity.uw_function

# Set some things

x, y = mesh.X

abs_r2 = x**2 + y**2
poisson.f = -16 * abs_r2
poisson.add_dirichlet_bc(abs_r2, "Bottom", component=0)
poisson.add_dirichlet_bc(abs_r2, "Top", component=0)
poisson.add_dirichlet_bc(abs_r2, "Right", component=0)
poisson.add_dirichlet_bc(abs_r2, "Left", component=0)

# +
# %%
# Linear model - starting guess

poisson.constitutive_model.Parameters.diffusivity = 1
poisson.solve(zero_init_guess=True)
# -

# %%
# Solve time
poisson.constitutive_model.Parameters.diffusivity = k
poisson.solve(zero_init_guess=False)

# %%
poisson.constitutive_model

# %%
gradT_projector.solve()

# %%
gradient.uw_function = sympy.diff(t_soln.sym, mesh.N.y)
gradient.solve()

# %%
gradient.uw_function

# %%
diffusivity.solve()

# non-linear smoothing term (probably not needed especially at the boundary)

gradient.uw_function = sympy.diff(t_soln.fn, mesh.N.y)
gradient.solve(_force_setup=True)

# %%
gradT_projector.solve()

# **Check** Construct simple linear function which is solution for
# above config.  Exclude boundaries from mesh data.

import numpy as np

with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(t_soln.sym[0], mesh.data)
    # if not np.allclose(mesh_numerical_soln, -1.0, rtol=0.01):
    #     raise RuntimeError("Unexpected values encountered.")

#
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    with mesh.access():
        pvmesh.point_data["T"] = mesh_numerical_soln
        pvmesh.point_data["dTdY"] = uw.function.evaluate(dTdY.sym[0], mesh.data)
        pvmesh.point_data["dTdY1"] = uw.function.evaluate(gradT.sym[1], mesh.data)
        pvmesh.point_data["dTdX1"] = uw.function.evaluate(gradT.sym[0], mesh.data)
        pvmesh.point_data["kappa"] = uw.function.evaluate(kappa.sym[0], mesh.data)
        pvmesh.point_data["kappa1"] = uw.function.evaluate(
            5 + gradT.sym[0] ** 2 + gradT.sym[1] ** 2, mesh.data
        )

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="dTdX1",
        use_transparency=False,
        opacity=0.5,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")
