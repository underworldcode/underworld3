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

# # Projection-based function evaluation
#
# Here we Use SNES solvers to project sympy / mesh variable functions and derivatives to nodes. Pointwise / symbolic functions cannot always be evaluated using `uw.function.evaluate` because they contain a mix of mesh variables, derivatives and symbols which may not be defined everywhere.
#
# Our solution is to use a projection of the function to a continuous mesh variable with the SNES machinery performing all of the background operations to determine the values and the optimal fitting.
#
# This approach also allows us to include boundary conditions, smoothing, and constraint terms (e.g. remove a null space) in cases (like piecewise continuous swarm variables) where this is difficult in the original form.
#
# We'll demonstrate this using a swarm variable (scalar / vector), but the same approach is useful for gradient recovery.

import underworld3 as uw
import numpy as np
import sympy

from underworld3.meshing import UnstructuredSimplexBox

meshbox = UnstructuredSimplexBox(minCoords=(0.0, 0.0), 
                                 maxCoords=(1.0, 1.0), 
                                 cellSize=1.0 / 32.0,)

# +
# import meshio
# mesh2 = meshio.read(filename="../Examples-StokesFlow/tmp_ball.vtk")
# meshio.write(filename="tmp_ball.msh", mesh=mesh2)

# +
import sympy

# Some useful coordinate stuff

x, y = meshbox.X
# -

s_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=2)

v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)

s_fn = sympy.cos(5.0 * sympy.pi * x) * sympy.cos(5.0 * sympy.pi * y)
v_fn = sympy.Matrix([sympy.cos(5.0 * sympy.pi * y) ** 2, -sympy.sin(5.0 * sympy.pi * x) ** 2])  # divergence free

meshbox.vector.divergence(v_fn)

# +
swarm = uw.swarm.Swarm(mesh=meshbox)
s_values = uw.swarm.SwarmVariable("Ss", swarm, 1, proxy_degree=3)
v_values = uw.swarm.SwarmVariable("Vs", swarm, meshbox.dim, proxy_degree=3)
# iv_values = uw.swarm.SwarmVariable("Vi", swarm, meshbox.dim, proxy_degree=3)

swarm.populate(fill_param=3)
# -


scalar_projection = uw.systems.Projection(meshbox, s_soln)
scalar_projection.uw_function = s_values.sym
scalar_projection.smoothing = 1.0e-6


# +
vector_projection = uw.systems.Vector_Projection(meshbox, v_soln)
vector_projection.uw_function = v_values.sym
vector_projection.smoothing = 1.0e-6  # see how well it works !
vector_projection.penalty = 1.0e-6

# Velocity boundary conditions (compare left / right walls in the soln !)

vector_projection.add_dirichlet_bc(v_fn, "Left", (0, 1))
vector_projection.add_dirichlet_bc(v_fn, "Right", (0, 1))
vector_projection.add_dirichlet_bc(v_fn, "Top", (0, 1))
vector_projection.add_dirichlet_bc(v_fn, "Bottom", (0, 1))
# -

with swarm.access(s_values, v_values):
    s_values.data[:, 0] = uw.function.evaluate(s_fn, swarm.data, meshbox.N)
    v_values.data[:, 0] = uw.function.evaluate(v_fn[0], swarm.data, meshbox.N)
    v_values.data[:, 1] = uw.function.evaluate(v_fn[1], swarm.data, meshbox.N)


scalar_projection.solve()

# + tags=[]
vector_projection.solve()
# -

scalar_projection.uw_function = meshbox.vector.divergence(v_soln.sym)
scalar_projection.solve()

s_soln.stats()

# +
# check the projection


if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    # pv.start_xvfb()

    meshbox.vtk("mesh_tmp.vtk")
    pvmesh = pv.read("mesh_tmp.vtk")

    with meshbox.access():
        vsol = v_soln.data.copy()

    pvmesh.point_data["S"] = uw.function.evaluate(s_soln.fn, meshbox.data)

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = vsol[...]

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S", use_transparency=False, opacity=0.5
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=1.0e-1, opacity=0.5)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")
# -


