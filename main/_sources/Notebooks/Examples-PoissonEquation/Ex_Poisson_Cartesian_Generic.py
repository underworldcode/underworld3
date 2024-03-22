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

# # Poisson Equation (generic)
#
# First we show how this works using the generic class and then the minor differences for
# the `Poisson` class
#
# ## Generic scalar solver class

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
import numpy as np
import sympy

from underworld3.meshing import UnstructuredSimplexBox

mesh = UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), regular=False, cellSize=1.0 / 32
)

t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
t_soln0 = uw.discretisation.MeshVariable("T0", mesh, 1, degree=1)

# +
poisson0 = uw.systems.SNES_Scalar(mesh, u_Field=t_soln0)
poisson0.F0 = 0.0
poisson0.F1 = 1.0 * poisson0.Unknowns.L
poisson0.add_dirichlet_bc(1.0, "Bottom", 0)
poisson0.add_dirichlet_bc(0.0, "Top", 0)

poisson0.constitutive_model = uw.constitutive_models.DiffusionModel
poisson0.constitutive_model.Parameters.diffusivity = 1.0
# -

poisson0.solve()

# ## `Poisson` Class
#
# Here is the other way to solve this, using the `Poisson` class which does not much
# more than add a template for the flux term.

# +
# Create Poisson object
poisson = uw.systems.Poisson(mesh, u_Field=t_soln)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1.0

poisson.f = 0.0
poisson.add_dirichlet_bc(1.0, "Bottom", 0)
poisson.add_dirichlet_bc(0.0, "Top", 0)
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

poisson.Unknowns.L

poisson.u.sym.jacobian(poisson.Unknowns.L)

poisson._f1.jacobian(poisson.Unknowns.L)

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
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["T"] = mesh_analytic_soln
    pvmesh.point_data["T2"] = mesh_numerical_soln
    pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"]

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


# -


