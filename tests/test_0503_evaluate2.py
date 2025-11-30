import os

# DISABLE SYMPY CACHE, AS IT GETS IN THE WAY FOR IDENTICALLY NAMED VARIABLES.
# NEED TO FIX.

# os.environ["SYMPY_USE_CACHE"] = "no"
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


def tensor_product(order, val1, val2):
    sum = 0.0
    order += 1
    for i in range(order):
        for j in range(order):
            sum += val1**i * val2**j
    return sum


def test_number_vector_mult():
    mesh = uw.meshing.StructuredQuadBox()
    var_vector = uw.discretisation.MeshVariable(
        varname="var_vector_1",
        mesh=mesh,
        num_components=2,
        vtype=uw.VarType.VECTOR,
        varsymbol=r"V",
    )
    var_vector.array[...] = (4.0, 5.0)

    result = uw.function.evaluate(3 * var_vector.sym[0], coords)
    assert np.allclose(np.array(((12.0),)), result, rtol=1e-05, atol=1e-08)
    result = uw.function.evaluate(3 * var_vector.sym[1], coords)
    assert np.allclose(np.array(((15.0),)), result, rtol=1e-05, atol=1e-08)

    del mesh


def test_number_vector_mult_evalf():
    mesh = uw.meshing.StructuredQuadBox()
    var_vector = uw.discretisation.MeshVariable(
        varname="var_vector_1",
        mesh=mesh,
        num_components=2,
        vtype=uw.VarType.VECTOR,
        varsymbol=r"V",
    )
    var_vector.array[...] = (4.0, 5.0)

    result = uw.function.evaluate(3 * var_vector.sym[0], coords, evalf=True)
    assert np.allclose(np.array(((12.0),)), result, rtol=1e-05, atol=1e-08)
    result = uw.function.evaluate(3 * var_vector.sym[1], coords, evalf=True)
    assert np.allclose(np.array(((15.0),)), result, rtol=1e-05, atol=1e-08)

    del mesh


def test_scalar_vector_mult():
    mesh = uw.meshing.StructuredQuadBox()
    var_scalar = uw.discretisation.MeshVariable(
        varname="var_scalar_2", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR
    )
    var_vector = uw.discretisation.MeshVariable(
        varname="var_vector_2", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR
    )
    with uw.synchronised_array_update():
        var_scalar.array[...] = 3.0
        var_vector.array[...] = (4.0, 5.0)

    result = uw.function.evaluate(var_scalar.sym[0] * var_vector.sym[0], coords)
    assert np.allclose(np.array(((12.0),)), result, rtol=1e-05, atol=1e-08)
    result = uw.function.evaluate(var_scalar.sym[0] * var_vector.sym[1], coords)
    assert np.allclose(np.array(((15.0),)), result, rtol=1e-05, atol=1e-08)

    del mesh


def test_vector_dot_product():
    mesh = uw.meshing.StructuredQuadBox()
    var_vector1 = uw.discretisation.MeshVariable(
        varname="var_vector1", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR
    )
    var_vector2 = uw.discretisation.MeshVariable(
        varname="var_vector2", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR
    )
    with uw.synchronised_array_update():
        var_vector1.array[...] = (1.0, 2.0)
        var_vector2.array[...] = (3.0, 4.0)
    result = uw.function.evaluate(var_vector1.sym.dot(var_vector2.sym), coords, evalf=True)
    assert np.allclose(11.0, result, rtol=1e-05, atol=1e-08)

    del mesh


# that test needs to be able to take degree as a parameter...
def test_polynomial_mesh_var_degree():
    mesh = uw.meshing.StructuredQuadBox()
    maxdegree = 10
    vars = []

    # Create required vars of different degree.
    for degree in range(maxdegree + 1):
        vars.append(
            uw.discretisation.MeshVariable(
                varname="var" + str(degree),
                mesh=mesh,
                num_components=1,
                vtype=uw.VarType.SCALAR,
                degree=degree,
            )
        )

    # Set variable data to represent polynomial function.
    for var in vars:
        vcoords = var.coords
        var.array[:, 0, 0] = tensor_product(var.degree, vcoords[:, 0], vcoords[:, 1])

    # Test that interpolated variables reproduce exactly polymial function of associated degree.
    for var in vars:
        result = uw.function.evaluate(var.sym[0], coords)
        assert np.allclose(
            tensor_product(var.degree, coords[:, 0], coords[:, 1]),
            result.squeeze(),
            rtol=1e-05,
            atol=1e-08,
        )


def test_many_many_scalar_mult_var():
    mesh = uw.meshing.StructuredQuadBox()
    # Note that this test fails for n>~15. Something something subdm segfault.
    # Must investigate.
    nn = 15
    vars = []
    for i in range(nn):
        vars.append(
            uw.discretisation.MeshVariable(
                varname=f"var_{i}", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR
            )
        )
    factorial = 1.0
    with uw.synchronised_array_update():
        for i, var in enumerate(vars):
            var.array[...] = float(i)
            factorial *= float(i)
    multexpr = vars[0].fn
    for var in vars[1:]:
        multexpr *= var.fn
    result = uw.function.evaluate(multexpr, coords).squeeze()
    assert np.allclose(factorial, result, rtol=1e-05, atol=1e-08)


# Let's now do the same, but instead do it Sympy wise.
# We don't really need any UW infrastructure for this test, but it's useful
# to stress our `evaluate()` function. It should however simply reduce
# to Sympy's `lambdify` routine.


def test_polynomial_sympy():
    degree = 20
    mesh = uw.meshing.StructuredQuadBox()

    print(tensor_product(degree, coords[:, 0], coords[:, 1]))

    assert np.allclose(
        tensor_product(degree, coords[:, 0], coords[:, 1]),
        uw.function.evaluate(
            tensor_product(degree, mesh.r[0], mesh.r[1]),
            coords,
            coord_sys=mesh.N,
        ).squeeze(),
        rtol=1e-05,
        atol=1e-08,
    )


# Now we'll do something similar but involve UW variables.
# Instead of using the Sympy symbols for (x,y), we'll set the
# coordinate locations on the var data itself.
# For a cartesian mesh, linear elements will suffice.
# We'll also do it twice, once using (xvar,yvar), and
# another time using (xyvar[0], xyvar[1]).


def test_polynomial_mesh_var_sympy():
    mesh = uw.meshing.StructuredQuadBox()
    xvar = uw.discretisation.MeshVariable(
        varname="xvar", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR
    )
    yvar = uw.discretisation.MeshVariable(
        varname="yvar", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR
    )
    xyvar = uw.discretisation.MeshVariable(
        varname="xyvar", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR
    )
    with uw.synchronised_array_update():
        # Note that all the `coords` arrays should actually reduce to an identical array,
        # as all vars have identical degree and layout.
        xvar.array[:, 0, 0] = xvar.coords[:, 0]
        yvar.array[:, 0, 0] = yvar.coords[:, 1]
        xyvar.array[:, 0, :] = xyvar.coords
    degree = 10
    assert np.allclose(
        tensor_product(degree, coords[:, 0], coords[:, 1]),
        uw.function.evaluate(tensor_product(degree, xvar.fn, yvar.fn), coords).squeeze(),
        rtol=1e-05,
        atol=1e-08,
    )
    assert np.allclose(
        tensor_product(degree, coords[:, 0], coords[:, 1]),
        uw.function.evaluate(
            tensor_product(degree, xyvar.fn.dot(mesh.N.i), xyvar.fn.dot(mesh.N.j)),
            coords,
        ).squeeze(),
        rtol=1e-05,
        atol=1e-08,
    )


# NOTE THAT WE NEEDED TO DISABLE MESH HASHING FOR 3D MESH FOR SOME REASON.
# CHECK `DMInterpolationSetUp_UW()` FOR DETAILS.


def test_3d_cross_product():
    # Create a set of evaluation coords in 3d
    n = 10
    x = np.linspace(0.1, 0.9, n)
    y = np.linspace(0.2, 0.8, n)
    z = np.linspace(0.3, 0.7, n)
    xv, yv, zv = np.meshgrid(x, y, z, sparse=True)
    coords = np.vstack((xv[0, :, 0], yv[:, 0, 0], zv[0, 0, :])).T

    # Now mesh and vars etc.
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4,) * 3)
    name = "vector cross product test"
    var_vector1 = uw.discretisation.MeshVariable(
        varname="var_vector1",
        mesh=mesh,
        num_components=3,
        vtype=uw.VarType.VECTOR,
        varsymbol="V_1",
    )
    var_vector2 = uw.discretisation.MeshVariable(
        varname="var_vector2",
        mesh=mesh,
        num_components=3,
        vtype=uw.VarType.VECTOR,
        varsymbol="V_2",
    )

    with uw.synchronised_array_update():
        var_vector1.array[...] = (1.0, 2.0, 3.0)
        var_vector2.array[...] = (4.0, 5.0, 6.0)
    result = uw.function.evaluate(var_vector1.sym.cross(var_vector2.sym), coords)
    assert np.allclose(np.array(((-3, 6, -3),)), result, rtol=1e-05, atol=1e-08)
