from re import S
import underworld3 as uw
import numpy as np
import sympy
import pytest

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
p_soln = uw.discretisation.MeshVariable("P0", mesh, 1, degree=0)
p_dc = uw.discretisation.MeshVariable("Pdc", mesh, 1, degree=1, continuous=False)

swarm = uw.swarm.Swarm(mesh=mesh)
s_values_0 = uw.swarm.SwarmVariable("Ss0", swarm, 1, proxy_degree=0)
s_values = uw.swarm.SwarmVariable("Ss1", swarm, 1, proxy_degree=1)
s_values_3 = uw.swarm.SwarmVariable("Ss3", swarm, 1, proxy_degree=3)


swarm.populate(fill_param=3)


def test_integrate_constants():

    calculator = uw.maths.Integral(mesh, fn=1.0)
    value = calculator.evaluate()

    assert abs(value - 1.0) < 0.001

    return


def test_integrate_sympy():

    calculator = uw.maths.Integral(mesh, fn=sympy.cos(x * sympy.pi))
    value = calculator.evaluate()

    assert abs(value) < 0.001

    return


def test_integrate_meshvar():

    with mesh.access(s_soln):
        s_soln.data[:, 0] = np.sin(np.pi * s_soln.coords[:, 0])

    calculator = uw.maths.Integral(mesh, fn=s_soln.sym[0])
    value = calculator.evaluate()

    assert abs(value - 2 / np.pi) < 0.001

    return


def test_integrate_derivative():

    with mesh.access(s_soln):
        s_soln.data[:, 0] = np.sin(np.pi * s_soln.coords[:, 0])

    calculator = uw.maths.Integral(mesh, fn=s_soln.sym.diff(x))
    value = calculator.evaluate()

    assert abs(value) < 0.001

    return

    return


def test_integrate_swarmvar_O1():

    with swarm.access(s_values):
        s_values.data[:, 0] = np.cos(np.pi * swarm.particle_coordinates.data[:, 0])

    calculator = uw.maths.Integral(mesh, fn=s_values.sym[0])
    value = calculator.evaluate()

    assert abs(value) < 0.001

    return


def test_integrate_swarmvar_deriv_O1():

    with swarm.access(s_values):
        s_values.data[:, 0] = np.cos(np.pi * swarm.particle_coordinates.data[:, 1])

    calculator = uw.maths.Integral(mesh, fn=s_values.sym.diff(y))
    value = calculator.evaluate()

    assert abs(value + 2) < 0.001

    return


def test_integrate_swarmvar_O3():

    with swarm.access(s_values_3):
        s_values_3.data[:, 0] = np.cos(np.pi * swarm.particle_coordinates.data[:, 0])

    calculator = uw.maths.Integral(mesh, fn=s_values_3.sym[0])
    value = calculator.evaluate()

    assert abs(value) < 0.001

    return


def test_integrate_swarmvar_deriv_03():

    with swarm.access(s_values_3):
        s_values_3.data[:, 0] = np.cos(np.pi * swarm.particle_coordinates.data[:, 1])

    calculator = uw.maths.Integral(mesh, fn=s_values_3.sym.diff(y))
    value = calculator.evaluate()

    assert abs(value + 2) < 0.001

    return


def test_integrate_swarmvar_O0():

    with swarm.access(s_values_0):
        s_values_0.data[:, 0] = np.cos(np.pi * swarm.particle_coordinates.data[:, 0])

    calculator = uw.maths.Integral(mesh, fn=s_values_0.sym[0])
    value = calculator.evaluate()

    assert abs(value) < 0.001

    return


@pytest.mark.xfail  # since the derivative of the piecewise constant function does not exist
def test_integrate_swarmvar_deriv_00():

    with swarm.access(s_values_0):
        s_values_0.data[:, 0] = np.sin(np.pi * swarm.particle_coordinates.data[:, 1])

    calculator = uw.maths.Integral(mesh, fn=s_values_0.sym.diff(y))
    value = calculator.evaluate()

    assert abs(value + 2) < 0.0001

    return
