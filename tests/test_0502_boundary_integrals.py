import underworld3 as uw
import numpy as np
import sympy
import pytest

# All tests in this module are quick core tests
pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

from underworld3.meshing import UnstructuredSimplexBox

## Set up the mesh for tests
mesh = UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / 32.0,
)

x, y = mesh.X

# Need at least one mesh variable for PETSc integration (same as volume Integral)
s_soln = uw.discretisation.MeshVariable("T_bd", mesh, 1, degree=2)


def test_bd_integral_constant_top():
    """Integrating 1.0 over the Top boundary of a unit box should give length 1.0."""

    bd_int = uw.maths.BdIntegral(mesh, fn=1.0, boundary="Top")
    value = bd_int.evaluate()

    assert abs(value - 1.0) < 0.001, f"Expected 1.0, got {value}"


def test_bd_integral_constant_bottom():
    """Integrating 1.0 over the Bottom boundary of a unit box should give length 1.0."""

    bd_int = uw.maths.BdIntegral(mesh, fn=1.0, boundary="Bottom")
    value = bd_int.evaluate()

    assert abs(value - 1.0) < 0.001, f"Expected 1.0, got {value}"


def test_bd_integral_perimeter():
    """Sum of integrals over all four boundaries should give perimeter = 4.0."""

    total = 0.0
    for bnd in ["Top", "Bottom", "Left", "Right"]:
        bd_int = uw.maths.BdIntegral(mesh, fn=1.0, boundary=bnd)
        total += bd_int.evaluate()

    assert abs(total - 4.0) < 0.01, f"Expected perimeter 4.0, got {total}"


def test_bd_integral_coordinate_fn():
    """Integrate x along Top boundary (y=1): int_0^1 x dx = 0.5."""

    bd_int = uw.maths.BdIntegral(mesh, fn=x, boundary="Top")
    value = bd_int.evaluate()

    assert abs(value - 0.5) < 0.01, f"Expected 0.5, got {value}"


def test_bd_integral_coordinate_fn_right():
    """Integrate y along Right boundary (x=1): int_0^1 y dy = 0.5."""

    bd_int = uw.maths.BdIntegral(mesh, fn=y, boundary="Right")
    value = bd_int.evaluate()

    assert abs(value - 0.5) < 0.01, f"Expected 0.5, got {value}"


def test_bd_integral_sympy_fn():
    """Integrate cos(pi*x) along Top boundary: int_0^1 cos(pi*x) dx = 0."""

    bd_int = uw.maths.BdIntegral(mesh, fn=sympy.cos(x * sympy.pi), boundary="Top")
    value = bd_int.evaluate()

    assert abs(value) < 0.01, f"Expected ~0, got {value}"


def test_bd_integral_meshvar():
    """Integrate a mesh variable (sin(pi*x)) along Top boundary."""

    s_soln.data[:, 0] = np.sin(np.pi * s_soln.coords[:, 0])

    bd_int = uw.maths.BdIntegral(mesh, fn=s_soln.sym[0], boundary="Top")
    value = bd_int.evaluate()

    # int_0^1 sin(pi*x) dx = 2/pi ≈ 0.6366
    expected = 2.0 / np.pi
    assert abs(value - expected) < 0.01, f"Expected {expected}, got {value}"


def test_bd_integral_normal_vector():
    """Integrand using surface normal: integrate n_y along Top boundary.
    On Top boundary (y=1), the outward normal is (0, 1), so n_y = 1.
    Integral should be 1.0."""

    Gamma = mesh.Gamma  # Surface normal as row matrix
    n_y = Gamma[1]

    bd_int = uw.maths.BdIntegral(mesh, fn=n_y, boundary="Top")
    value = bd_int.evaluate()

    assert abs(value - 1.0) < 0.01, f"Expected 1.0, got {value}"


def test_bd_integral_invalid_boundary():
    """Should raise ValueError for non-existent boundary name."""

    with pytest.raises(ValueError, match="not found"):
        uw.maths.BdIntegral(mesh, fn=1.0, boundary="Nonexistent")


# --- Internal boundary tests ---

from underworld3.meshing import BoxInternalBoundary

mesh_internal = BoxInternalBoundary(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / 32.0,
    zintCoord=0.5,
    simplex=True,
)

x_i, y_i = mesh_internal.X

# Need a mesh variable for PETSc
_v_internal = uw.discretisation.MeshVariable("T_int", mesh_internal, 1, degree=2)


def test_bd_integral_internal_boundary_length():
    """Internal boundary at y=0.5 across a unit box should have length 1.0."""

    bd_int = uw.maths.BdIntegral(mesh_internal, fn=1.0, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value - 1.0) < 0.001, f"Expected 1.0, got {value}"


def test_bd_integral_internal_coordinate_fn():
    """Integrate x along internal boundary at y=0.5: int_0^1 x dx = 0.5."""

    bd_int = uw.maths.BdIntegral(mesh_internal, fn=x_i, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value - 0.5) < 0.01, f"Expected 0.5, got {value}"


def test_bd_integral_internal_does_not_affect_external():
    """External boundaries should still work on the internal-boundary mesh."""

    total = 0.0
    for bnd in ["Top", "Bottom", "Left", "Right"]:
        bd_int = uw.maths.BdIntegral(mesh_internal, fn=1.0, boundary=bnd)
        total += bd_int.evaluate()

    assert abs(total - 4.0) < 0.01, f"Expected perimeter 4.0, got {total}"


# --- Annulus internal boundary tests ---

from underworld3.meshing import AnnulusInternalBoundary

_R_INTERNAL = 1.0

mesh_annulus = AnnulusInternalBoundary(
    radiusOuter=1.5,
    radiusInternal=_R_INTERNAL,
    radiusInner=0.5,
    cellSize=0.1,
)

_v_annulus = uw.discretisation.MeshVariable("T_ann", mesh_annulus, 1, degree=2)


def test_bd_integral_annulus_internal_circumference():
    """Internal boundary at radius 1.0: circumference = 2*pi."""

    bd_int = uw.maths.BdIntegral(mesh_annulus, fn=1.0, boundary="Internal")
    value = bd_int.evaluate()
    expected = 2 * np.pi * _R_INTERNAL

    assert abs(value - expected) < 0.01, f"Expected {expected:.4f}, got {value}"


def test_bd_integral_annulus_outer_circumference():
    """Outer boundary at radius 1.5: circumference = 3*pi."""

    bd_int = uw.maths.BdIntegral(mesh_annulus, fn=1.0, boundary="Upper")
    value = bd_int.evaluate()
    expected = 2 * np.pi * 1.5

    assert abs(value - expected) < 0.01, f"Expected {expected:.4f}, got {value}"
