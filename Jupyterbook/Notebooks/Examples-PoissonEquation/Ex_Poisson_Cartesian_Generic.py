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

# # Poisson Equation (generic)
#
# First we show how this works using the generic class and then the minor differences for
# the `Poisson` class
#
# ## Generic scalar solver class

import underworld3 as uw
import numpy as np
import sympy

from underworld3.meshing import UnstructuredSimplexBox

mesh = UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), regular=True, cellSize=1.0 / 32
)

t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
t_soln0 = uw.discretisation.MeshVariable("T0", mesh, 1, degree=1)

poisson0 = uw.systems.SNES_Scalar(mesh, u_Field=t_soln0)
poisson0.F0 = 0.0
poisson0.F1 = 1.0 * poisson0._L
poisson0.add_dirichlet_bc(1.0, "Bottom")
poisson0.add_dirichlet_bc(0.0, "Top")

poisson0.solve()

# ## `Poisson` Class
#
# Here is the other way to solve this, using the `Poisson` class which does not much
# more than add a template for the flux term.

# +
# Create Poisson object
poisson = uw.systems.Poisson(mesh, u_Field=t_soln)
poisson.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
poisson.constitutive_model.Parameters.diffusivity = 1.0

poisson.f = 0.0
poisson.add_dirichlet_bc(1.0, "Bottom")
poisson.add_dirichlet_bc(0.0, "Top")
# -

# Solve time
poisson.solve()

poisson.F0
sympy.Matrix((0,))

# +
# Check the flux term
display(poisson._L)

# This is the internal build of the flux term
display(poisson._f1)
# -

poisson._L

poisson.u.sym.jacobian(poisson._L)

poisson._f1.jacobian(poisson._L)

# ## Validation

# +
# Check. Construct simple linear which is solution for
# above config.  Exclude boundaries from mesh data.

import numpy as np

with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(poisson.u.fn, mesh.data)
    mesh_analytic_soln = uw.function.evaluate(1.0 - mesh.N.y, mesh.data)

    if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.001, atol=0.01):
        raise RuntimeError("Unexpected values encountered.")

# +
from mpi4py import MPI

if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("mesh_tmp.vtk")
    pvmesh = pv.read("mesh_tmp.vtk")

    with mesh.access():
        pvmesh.point_data["T"] = mesh_analytic_soln
        pvmesh.point_data["T2"] = mesh_numerical_soln
        pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"]

    pl = pv.Plotter(notebook=True)

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


# -
