# %% [markdown]
# # Poisson Cartesian
#
# Linear and non-linear diffusion equation

# %%
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy


# %%
mesh1 = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 24)
mesh2 = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 24, regular=True)

# mesh3 = uw.meshing.UnstructuredSimplexBox(
# minCoords=(0.0,0.0,0.0),
# maxCoords=(1.0,1.0,1.0),
# cellSize=1.0/6)

mesh = mesh2

phi = uw.discretisation.MeshVariable(r"\phi", mesh, 1, degree=2)
scalar = uw.discretisation.MeshVariable(r"\Theta", mesh, 1, degree=2)

# %%
# Create Poisson object

poisson = uw.systems.Poisson(mesh, u_Field=phi, solver_name="diffusion")

# Constitutive law (diffusivity)

poisson.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
poisson.constitutive_model.Parameters.diffusivity = 1


# %%
poisson.constitutive_model.c

# %%
# Set some things
poisson.f = 0.0
poisson.add_dirichlet_bc(1.0, "Bottom")
poisson.add_dirichlet_bc(0.0, "Top")

# %%
poisson._setup_terms()

# %%
# Solve time
poisson.solve()

# %%
poisson.constitutive_model.C

# %%
# Check. Construct simple linear which is solution for
# above config.  Exclude boundaries from mesh data.
import numpy as np

with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(poisson.u.fn, mesh.data)
    mesh_analytic_soln = uw.function.evaluate(1.0 - mesh.N.y, mesh.data)
    if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.0001):
        raise RuntimeError("Unexpected values encountered.")

# %%
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
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

# %%
# Create some function using one of the base scalars x,y[,z] = mesh.X

import sympy

x, y = mesh.X
x0 = y0 = 1 / sympy.sympify(2)
k = sympy.exp(-((x - x0) ** 2 + (y - y0) ** 2))

poisson.constitutive_model.Parameters.diffusivity=k

# %%
poisson.constitutive_model.flux(poisson._L)

# %%
with mesh.access():
    orig_soln = poisson.u.data.copy()

orig_soln_mesh = uw.function.evaluate(phi.sym[0], mesh.data)

# %%
poisson.solve(zero_init_guess=True, _force_setup=True)

# %%
# Simply confirm results are different

with mesh.access():
    if np.allclose(poisson.u.data, orig_soln, rtol=0.1):
        raise RuntimeError("Values did not change !")


# %%
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh2 = pv.read("ignore_mesh.vtk")

    pvmesh2.point_data["T"] = uw.function.evaluate(phi.sym[0], mesh.data)
    pvmesh2.point_data["dT"] = pvmesh2.point_data["T"] - pvmesh.point_data["T"]

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


# %%
## Non-linear example


# RHS term

abs_r2 = x**2 + y**2
poisson.f = -16 * abs_r2
poisson.add_dirichlet_bc(abs_r2, ["Bottom", "Top", "Right", "Left"])

display(poisson.f)

# Constitutive law (diffusivity)
# Linear solver first

poisson.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
poisson.constitutive_model.Parameters.diffusivity=1

poisson.solve()


# %%
# Non-linear diffusivity

grad_phi = mesh.vector.gradient(phi.sym)
k = 5 + (grad_phi.dot(grad_phi)) / 2
poisson.constitutive_model.Parameters.diffusivity=k
poisson.constitutive_model.c


# %%
poisson._setup_terms()
poisson._G3


# %%
# Use initial guess from linear solve

poisson.solve(zero_init_guess=False)

# %%
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh2 = pv.read("ignore_mesh.vtk")

    pvmesh2.point_data["T"] = uw.function.evaluate(phi.sym[0], mesh.data)

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

# %% [markdown]
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
projection.uw_function = sympy.diff(phi.sym, mesh.X[1])
projection.smoothing = 1.0e-3

projection.solve()


# %%
sympy.diff(scalar.sym, mesh.X[1])

# %%
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh2 = pv.read("ignore_mesh.vtk")

    pvmesh2.point_data["T"] = uw.function.evaluate(phi.sym[0], mesh.data)
    pvmesh2.point_data["K"] = uw.function.evaluate(scalar.sym[0], mesh.data)

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh2,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="K",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")

# %%
pvmesh2.point_data["K"].max()
