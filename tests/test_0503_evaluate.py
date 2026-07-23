import os

# DISABLE SYMPY CACHE, AS IT GETS IN THE WAY FOR IDENTICALLY NAMED VARIABLES.
# NEED TO FIX.

os.environ["SYMPY_USE_CACHE"] = "no"
import underworld3 as uw
import underworld3.function as fn
import numpy as np
import sympy
import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1


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


@pytest.mark.tier_a
@pytest.mark.parametrize("simplex", [False, True], ids=["quad", "simplex"])
def test_evaluate_on_domain_boundary_faces(simplex):
    """Interpolation at points lying exactly ON the domain boundary must be
    exact, on both the DMLocatePoints path (quad) and the hint-bypass path
    (simplex).

    PETSc's point location drops queries sitting exactly on the domain's
    closed upper faces (half-open cell convention), and a dropped point is
    zero-filled by the evaluator unless the kd-tree cell hint recovers it.
    Losing that recovery on quad meshes silently corrupted semi-Lagrangian
    stress-history trace-backs along the top boundary — the VEP stability
    blow-up of issue #390.
    """
    if simplex:
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5), cellSize=0.25
        )
    else:
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(8, 4), minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5)
        )
    var = uw.discretisation.MeshVariable("bdry_probe", mesh, 1, degree=2)
    var.data[:, 0] = 1.0 + 2.0 * var.coords[:, 0] + 3.0 * var.coords[:, 1]

    xs = np.linspace(-0.9, 0.9, 7)
    ys = np.linspace(-0.4, 0.4, 5)
    pts = np.array(
        [[xi, 0.5] for xi in xs]        # top face (the face PETSc drops)
        + [[xi, -0.5] for xi in xs]     # bottom face
        + [[-1.0, yi] for yi in ys]     # left face
        + [[1.0, yi] for yi in ys]      # right face
        + [[-1.0, -0.5], [1.0, -0.5], [-1.0, 0.5], [1.0, 0.5]]  # corners
    )
    vals = fn.evaluate(var.sym[0], pts).flatten()
    exact = 1.0 + 2.0 * pts[:, 0] + 3.0 * pts[:, 1]
    assert np.allclose(vals, exact, atol=1e-8), (
        f"boundary-face interpolation error: "
        f"max|err|={np.abs(vals - exact).max():.3e} "
        f"worst at {pts[np.argmax(np.abs(vals - exact))]}"
    )

    del mesh
