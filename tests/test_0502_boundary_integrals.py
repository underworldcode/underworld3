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


# --- Internal boundary tests (BoxInternalBoundary) ---
# BoxInternalBoundary has a pre-existing MPI bug (UnboundLocalError in the mesh
# constructor) so these tests are skipped under MPI. They use lazy initialization
# to avoid crashing the entire module if the mesh constructor fails.

from underworld3.meshing import BoxInternalBoundary

_mesh_internal = None
_x_i = None
_y_i = None


def _get_internal_mesh():
    global _mesh_internal, _x_i, _y_i
    if _mesh_internal is None:
        _mesh_internal = BoxInternalBoundary(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=1.0 / 32.0,
            zintCoord=0.5,
            simplex=True,
        )
        _x_i, _y_i = _mesh_internal.X
        uw.discretisation.MeshVariable("T_int", _mesh_internal, 1, degree=2)
    return _mesh_internal, _x_i, _y_i


@pytest.mark.skipif(uw.mpi.size > 1, reason="BoxInternalBoundary has pre-existing MPI bug")
def test_bd_integral_internal_boundary_length():
    """Internal boundary at y=0.5 across a unit box should have length 1.0."""

    mesh_internal, _, _ = _get_internal_mesh()
    bd_int = uw.maths.BdIntegral(mesh_internal, fn=1.0, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value - 1.0) < 0.001, f"Expected 1.0, got {value}"


@pytest.mark.skipif(uw.mpi.size > 1, reason="BoxInternalBoundary has pre-existing MPI bug")
def test_bd_integral_internal_coordinate_fn():
    """Integrate x along internal boundary at y=0.5: int_0^1 x dx = 0.5."""

    mesh_internal, x_i, _ = _get_internal_mesh()
    bd_int = uw.maths.BdIntegral(mesh_internal, fn=x_i, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value - 0.5) < 0.01, f"Expected 0.5, got {value}"


@pytest.mark.skipif(uw.mpi.size > 1, reason="BoxInternalBoundary has pre-existing MPI bug")
def test_bd_integral_internal_normal_ny():
    """Integrate n_y along internal boundary at y=0.5.
    The internal boundary has normals pointing in +y or -y direction,
    so integrating n_y should give +1 or -1 (length 1 boundary)."""

    mesh_internal, _, _ = _get_internal_mesh()
    Gamma = mesh_internal.Gamma
    n_y = Gamma[1]

    bd_int = uw.maths.BdIntegral(mesh_internal, fn=n_y, boundary="Internal")
    value = bd_int.evaluate()

    # Normal orientation is consistent but direction depends on mesh;
    # absolute value should be 1.0
    assert abs(abs(value) - 1.0) < 0.01, f"Expected |n_y integral| = 1.0, got {value}"


@pytest.mark.skipif(uw.mpi.size > 1, reason="BoxInternalBoundary has pre-existing MPI bug")
def test_bd_integral_internal_normal_nx():
    """Integrate n_x along internal boundary at y=0.5.
    The internal boundary is horizontal, so n_x should be ~0."""

    mesh_internal, _, _ = _get_internal_mesh()
    Gamma = mesh_internal.Gamma
    n_x = Gamma[0]

    bd_int = uw.maths.BdIntegral(mesh_internal, fn=n_x, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value) < 0.01, f"Expected ~0, got {value}"


@pytest.mark.skipif(uw.mpi.size > 1, reason="BoxInternalBoundary has pre-existing MPI bug")
def test_bd_integral_internal_normal_weighted():
    """Integrate x * n_y along internal boundary at y=0.5.
    int_0^1 x * n_y dx = n_y * 0.5. Since |n_y| = 1, result should be ~0.5."""

    mesh_internal, x_i, _ = _get_internal_mesh()
    Gamma = mesh_internal.Gamma
    n_y = Gamma[1]

    bd_int = uw.maths.BdIntegral(mesh_internal, fn=x_i * n_y, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(abs(value) - 0.5) < 0.01, f"Expected |value| = 0.5, got {value}"


@pytest.mark.skipif(uw.mpi.size > 1, reason="BoxInternalBoundary has pre-existing MPI bug")
def test_bd_integral_internal_does_not_affect_external():
    """External boundaries should still work on the internal-boundary mesh."""

    mesh_internal, _, _ = _get_internal_mesh()
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


def test_bd_integral_annulus_internal_normal_squared():
    """Normal magnitude squared integrated over internal circle at r=1.0.

    Since n is a unit normal, n_x^2 + n_y^2 = 1 everywhere on the boundary.
    Integrating 1 over the circle gives the circumference 2*pi*R.
    (We use n_x^2 + n_y^2 rather than n dot r_hat because internal boundary
    normals have arbitrary per-facet orientation that may cancel.)"""

    Gamma = mesh_annulus.Gamma
    n_sq = Gamma[0]**2 + Gamma[1]**2

    bd_int = uw.maths.BdIntegral(mesh_annulus, fn=n_sq, boundary="Internal")
    value = bd_int.evaluate()
    expected = 2 * np.pi * _R_INTERNAL

    assert abs(value - expected) < 0.05, f"Expected {expected:.4f}, got {value}"


def test_bd_integral_annulus_internal_normal_tangential():
    """Tangential component of normal integrated over internal circle should be ~0.

    The tangential direction is t = (-y/r, x/r). Since n is radial,
    n.t should integrate to zero."""

    x_a, y_a = mesh_annulus.X
    Gamma = mesh_annulus.Gamma
    r = sympy.sqrt(x_a**2 + y_a**2)
    # n dot t_hat = (-n_x * y + n_y * x) / r
    n_dot_that = (-Gamma[0] * y_a + Gamma[1] * x_a) / r

    bd_int = uw.maths.BdIntegral(mesh_annulus, fn=n_dot_that, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value) < 0.05, f"Expected ~0, got {value}"


def _build_spherical_shell_for_integrals():
    from underworld3.meshing import SphericalShell

    mesh_spherical = SphericalShell(
        radiusOuter=1.0,
        radiusInner=0.5,
        cellSize=1.0 / 4.0,
        degree=1,
        qdegree=2,
    )
    uw.discretisation.MeshVariable("P_spherical_int", mesh_spherical, 1, degree=1, continuous=True)
    return mesh_spherical


def test_spherical_bd_then_integral_does_not_poison_volume_path():
    """Boundary and volume integrals must not collide in the JIT cache."""

    mesh_spherical = _build_spherical_shell_for_integrals()

    boundary_before = float(uw.maths.BdIntegral(mesh_spherical, fn=1.0, boundary="Lower").evaluate())
    volume = float(uw.maths.Integral(mesh_spherical, fn=1.0).evaluate())
    boundary_after = float(uw.maths.BdIntegral(mesh_spherical, fn=1.0, boundary="Lower").evaluate())

    assert boundary_before > 0.0
    assert volume > 0.0
    assert abs(boundary_after - boundary_before) < 1.0e-10


def test_spherical_integral_then_bd_does_not_poison_boundary_path():
    """Volume and boundary integrals must remain order-independent on spherical meshes."""

    mesh_reference = _build_spherical_shell_for_integrals()
    boundary_reference = float(uw.maths.BdIntegral(mesh_reference, fn=1.0, boundary="Lower").evaluate())

    mesh_spherical = _build_spherical_shell_for_integrals()
    volume = float(uw.maths.Integral(mesh_spherical, fn=1.0).evaluate())
    boundary_after = float(uw.maths.BdIntegral(mesh_spherical, fn=1.0, boundary="Lower").evaluate())

    assert volume > 0.0
    assert boundary_reference > 0.0
    assert abs(boundary_after - boundary_reference) < 1.0e-10
