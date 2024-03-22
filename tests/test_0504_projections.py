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

# +
## Set up the mesh(es) etc for tests and examples

mesh = UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / 32.0,
)

x, y = mesh.X

s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
gradient = uw.discretisation.MeshVariable("dTdx", mesh, 1, degree=1)

swarm = uw.swarm.Swarm(mesh=mesh)
s_values = uw.swarm.SwarmVariable("Ss", swarm, 1, proxy_degree=3)
v_values = uw.swarm.SwarmVariable("Vs", swarm, mesh.dim, proxy_degree=3)

swarm.populate(fill_param=3)


# +
def test_scalar_projection():
    # The following test projects scalar values defined on a
    # swarm using a Sympy function to a mesh variable.

    s_fn = sympy.cos(5.0 * sympy.pi * x) * sympy.cos(5.0 * sympy.pi * y)

    # Set the values on the swarm variable
    with swarm.access(s_values):
        s_values.data[:, 0] = uw.function.evalf(s_fn, swarm.data, coord_sys=mesh.N)

    # Prepare projection of swarm values onto the mesh nodes.
    scalar_projection = uw.systems.Projection(mesh, s_soln)
    scalar_projection.uw_function = s_values.sym
    scalar_projection.smoothing = 1.0e-6
    scalar_projection.solve()

    return


def test_vector_projection():
    s_fn_x = sympy.cos(5.0 * sympy.pi * x) * sympy.cos(5.0 * sympy.pi * y)
    s_fn_y = sympy.sin(5.0 * sympy.pi * x) * sympy.sin(5.0 * sympy.pi * y)

    # Set the values on the swarm variable
    with swarm.access(v_values):
        v_values.data[:, 0] = uw.function.evalf(s_fn_x, swarm.data, coord_sys=mesh.N)
        v_values.data[:, 1] = uw.function.evalf(s_fn_y, swarm.data, coord_sys=mesh.N)

    vector_projection = uw.systems.Vector_Projection(mesh, v_soln)
    vector_projection.uw_function = v_values.sym
    vector_projection.smoothing = 1.0e-3

    vector_projection.add_dirichlet_bc(0.0, "Right", 0)
    vector_projection.add_dirichlet_bc(0.0, "Top", 1)
    vector_projection.add_dirichlet_bc(0.0, "Bottom", 1)

    vector_projection.solve()

    return


def test_gradient_recovery():
    fn = sympy.cos(4.0 * sympy.pi * x)

    with mesh.access(s_soln):
        s_soln.data[:, 0] = uw.function.evalf(fn, s_soln.coords[:], coord_sys=mesh.N)

    scalar_projection = uw.systems.Projection(mesh, gradient)
    scalar_projection.uw_function = s_soln.sym.diff(x)[0]
    scalar_projection.solve()


# -
