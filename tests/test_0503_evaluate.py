import os

# DISABLE SYMPY CACHE, AS IT GETS IN THE WAY FOR IDENTICALLY NAMED VARIABLES.
# NEED TO FIX.

os.environ["SYMPY_USE_CACHE"] = "no"
import underworld3 as uw
import underworld3.function as fn
import numpy as np
import sympy
import pytest


n = 10
x = np.linspace(0.1, 0.9, n)
y = np.linspace(0.2, 0.8, n)
xv, yv = np.meshgrid(x, y, sparse=True)
coords = np.vstack((xv[0, :], yv[:, 0])).T

# Python function which generates a polynomial space spanning function of the required degree.
# For example for degree 2:
# tensor_product(2,x,y) = 1 + x + y + x**2*y + x*y**2 + x**2*y**2


def tensor_product(order, val1, val2):
    sum = 0.0
    order += 1
    for i in range(order):
        for j in range(order):
            sum += val1**i * val2**j
    return sum


def test_non_uw_variable_constant():
    mesh = uw.meshing.StructuredQuadBox()
    result = fn.evaluate(
        sympy.sympify(1.5),
        coords,
        coord_sys=mesh.N,
        verbose=True,
    )
    assert np.allclose(1.5, result, rtol=1e-05, atol=1e-08)

    del mesh


def test_non_uw_variable_constant_evalf():
    mesh = uw.meshing.StructuredQuadBox()
    result = fn.evaluate(
        sympy.sympify(1.5),
        coords,
        coord_sys=mesh.N,
        evalf=True,
        verbose=True,
    )

    assert np.allclose(1.5, result, rtol=1e-05, atol=1e-08)
    del mesh


def test_non_uw_variable_linear():
    mesh = uw.meshing.StructuredQuadBox()
    result = fn.evaluate(mesh.r[0], coords, coord_sys=mesh.N).squeeze()
    assert np.allclose(x, result, rtol=1e-05, atol=1e-08)

    del mesh


def test_non_uw_variable_sine():
    mesh = uw.meshing.StructuredQuadBox()
    result = fn.evaluate(sympy.sin(mesh.r[1]), coords, coord_sys=mesh.N).squeeze()
    assert np.allclose(np.sin(y), result, rtol=1e-05, atol=1e-08)

    del mesh


def test_single_scalar_variable():
    mesh = uw.meshing.StructuredQuadBox()
    var = uw.discretisation.MeshVariable(
        varname="scalar_var_3", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR
    )
    var.array[...] = 1.1

    result = fn.evaluate(var.sym[0], coords, evalf=True)
    assert np.allclose(1.1, result, rtol=1e-05, atol=1e-08)

    del mesh


def test_single_vector_variable():
    mesh = uw.meshing.StructuredQuadBox()
    var = uw.discretisation.MeshVariable(
        varname="vector_var_4", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR
    )
    var.array[...] = (1.1, 1.2)
    result = uw.function.evaluate(var.sym, coords, evalf=True)
    assert np.allclose(np.array(((1.1, 1.2),)), result, rtol=1e-05, atol=1e-08)

    del mesh
