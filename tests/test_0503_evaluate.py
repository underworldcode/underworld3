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


@pytest.mark.tier_a
def test_location_capability_measured():
    """The point-location capability is a measured mesh property:
    rectilinear and smoothly-deformed 2D quad meshes are 'exact' (straight
    edges are always planar; convexity == mesh validity), a regular 3D hex
    box is 'exact', and a warped hex box drops to 'continuous'.
    """
    quad = uw.meshing.StructuredQuadBox(
        elementRes=(8, 4), minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5)
    )
    assert quad._location_capability() == "exact"

    X = quad.X.coords.copy()
    defo = X.copy()
    defo[:, 0] += 0.03 * np.sin(np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
    defo[:, 1] += 0.03 * np.cos(2 * np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    quad.deform(defo)
    assert quad._location_capability() == "exact", (
        "deformed 2D quads have straight edges and convex cells - still exact"
    )

    hexbox = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4, 4), minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0)
    )
    assert hexbox._location_capability() == "exact"

    X3 = hexbox.X.coords.copy()
    d3 = X3.copy()
    d3 += 0.04 * np.sin(np.pi * X3[:, [1, 2, 0]]) * np.cos(np.pi * X3[:, [2, 0, 1]])
    hexbox.deform(d3)
    assert hexbox._location_capability() == "continuous", (
        "warped hex faces are non-planar - estimator authoritative for "
        "continuous fields only"
    )

    del quad, hexbox


@pytest.mark.tier_a
def test_evaluate_deformed_quad_boundary_exact():
    """A deformed (non-affine, convex) 2D quad mesh has 'exact' capability:
    boundary-face and interior evaluation of a linear field must be exact
    (isoparametric Q2 reproduces linears on bilinear cells).
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 4), minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5)
    )
    var = uw.discretisation.MeshVariable("dq_probe", mesh, 1, degree=2)
    # boundary node indices captured BEFORE deforming (node identity is
    # stable under a coordinate move)
    idx_bnd = np.nonzero(
        (np.abs(var.coords[:, 1]) > 0.5 - 1e-12)
        | (np.abs(var.coords[:, 0]) > 1.0 - 1e-12)
    )[0]

    X = mesh.X.coords.copy()
    defo = X.copy()
    defo[:, 0] += 0.03 * np.sin(np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
    defo[:, 1] += 0.03 * np.cos(2 * np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    mesh.deform(defo)

    var.data[:, 0] = 1.0 + 2.0 * var.coords[:, 0] + 3.0 * var.coords[:, 1]

    # interior points plus points ON the (deformed) boundary surface
    bnd = var.coords[idx_bnd[:: max(1, len(idx_bnd) // 16)]]
    interior = np.column_stack([
        np.linspace(-0.8, 0.8, 9), np.linspace(-0.35, 0.35, 9)
    ])
    pts = np.vstack([interior, bnd])
    vals = fn.evaluate(var.sym[0], pts).flatten()
    exact = 1.0 + 2.0 * pts[:, 0] + 3.0 * pts[:, 1]
    assert np.allclose(vals, exact, atol=1e-8), (
        f"max|err|={np.abs(vals - exact).max():.3e}"
    )

    del mesh


@pytest.mark.tier_a
def test_evaluate_warped_hex_rbf_fallback_no_silent_values():
    """On a warped hex box ('continuous' capability) a DISCONTINUOUS variable
    must not trust the cell-wall hint: dropped boundary-face points take the
    RBF fallback. The result must be finite (no NaN sentinel escaping, no
    silent zeros) and bounded by the field's data range; a CONTINUOUS
    variable on the same mesh stays on the authoritative path and evaluates
    a linear field exactly (trilinear isoparametric reproduces linears).
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4, 4), minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0)
    )
    X3 = mesh.X.coords.copy()
    d3 = X3.copy()
    d3 += 0.04 * np.sin(np.pi * X3[:, [1, 2, 0]]) * np.cos(np.pi * X3[:, [2, 0, 1]])
    mesh.deform(d3)
    assert mesh._location_capability() == "continuous"

    cvar = uw.discretisation.MeshVariable("wh_cont", mesh, 1, degree=1)
    cvar.data[:, 0] = 1.0 + 2.0 * cvar.coords[:, 0] - 1.5 * cvar.coords[:, 2]
    dvar = uw.discretisation.MeshVariable(
        "wh_disc", mesh, 1, degree=1, continuous=False
    )
    dvar.data[:, 0] = 1.0 + 2.0 * dvar.coords[:, 0] - 1.5 * dvar.coords[:, 2]

    # query points on the deformed top surface (actual boundary node coords)
    zmax_nodes = cvar.coords[
        cvar.coords[:, 2] > np.percentile(cvar.coords[:, 2], 92)
    ][:15]
    interior = np.column_stack([
        np.linspace(0.15, 0.85, 8),
        np.linspace(0.2, 0.8, 8),
        np.linspace(0.25, 0.75, 8),
    ])
    pts = np.vstack([interior, zmax_nodes])

    # continuous: authoritative path, linear field reproduced exactly
    cvals = fn.evaluate(cvar.sym[0], pts).flatten()
    cexact = 1.0 + 2.0 * pts[:, 0] - 1.5 * pts[:, 2]
    assert np.allclose(cvals, cexact, atol=1e-8), (
        f"continuous-field max|err|={np.abs(cvals - cexact).max():.3e}"
    )

    # discontinuous: locate + RBF fallback for dropped points — finite,
    # in-range, and close to the (smooth) nodal data it interpolates
    dvals = fn.evaluate(dvar.sym[0], pts).flatten()
    dexact = 1.0 + 2.0 * pts[:, 0] - 1.5 * pts[:, 2]
    assert np.all(np.isfinite(dvals)), "NaN escaped the RBF fallback"
    lo, hi = dvar.data.min(), dvar.data.max()
    assert np.all(dvals >= lo - 1e-12) and np.all(dvals <= hi + 1e-12), (
        "fallback values outside the field's data range"
    )
    assert np.abs(dvals - dexact).max() < 0.25, (
        f"fallback error unexpectedly large: {np.abs(dvals - dexact).max():.3e}"
    )

    del mesh
