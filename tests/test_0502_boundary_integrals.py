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
# These run in serial and parallel: the BoxInternalBoundary rank>0
# UnboundLocalError (2026-07 audit, BF-13) is fixed, and signed-normal
# integrands written with plain mesh.Gamma are partition-safe (resolved
# to the declared analytic normal — issue #327).

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


def test_bd_integral_internal_boundary_length():
    """Internal boundary at y=0.5 across a unit box should have length 1.0."""

    mesh_internal, _, _ = _get_internal_mesh()
    bd_int = uw.maths.BdIntegral(mesh_internal, fn=1.0, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value - 1.0) < 0.001, f"Expected 1.0, got {value}"


def test_bd_integral_internal_coordinate_fn():
    """Integrate x along internal boundary at y=0.5: int_0^1 x dx = 0.5."""

    mesh_internal, x_i, _ = _get_internal_mesh()
    bd_int = uw.maths.BdIntegral(mesh_internal, fn=x_i, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value - 0.5) < 0.01, f"Expected 0.5, got {value}"


# `mesh.Gamma` is the single user-facing normal symbol on any boundary.
# On an internal boundary the raw petsc_n[] is orientation-ambiguous
# (DMPlex support[0] is partition-dependent at seam facets — issue #327),
# so BdIntegral resolves the Gamma components to the mesh factory's
# declared analytic normal (Mesh._resolve_boundary_normals). The declared
# internal normal points from region Inner to region Outer (+y here).
def test_bd_integral_internal_normal_ny():
    """Integrate n_y along internal boundary at y=0.5 with plain mesh.Gamma.
    The declared internal normal is +y, so the integral is exactly +1
    (length-1 boundary)."""

    mesh_internal, _, _ = _get_internal_mesh()
    n_y = mesh_internal.Gamma[1]

    bd_int = uw.maths.BdIntegral(mesh_internal, fn=n_y, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value - 1.0) < 1e-6, f"Expected +1.0, got {value}"


def test_bd_integral_internal_normal_nx():
    """Integrate n_x along internal boundary at y=0.5.
    The internal boundary is horizontal, so n_x should be ~0."""

    mesh_internal, _, _ = _get_internal_mesh()
    Gamma = mesh_internal.Gamma
    n_x = Gamma[0]

    bd_int = uw.maths.BdIntegral(mesh_internal, fn=n_x, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value) < 1e-6, f"Expected ~0, got {value}"


def test_bd_integral_internal_normal_weighted():
    """Integrate x * n_y along internal boundary at y=0.5 with plain
    mesh.Gamma: int_0^1 x * (+1) dx = +0.5."""

    mesh_internal, x_i, _ = _get_internal_mesh()
    n_y = mesh_internal.Gamma[1]

    bd_int = uw.maths.BdIntegral(mesh_internal, fn=x_i * n_y, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value - 0.5) < 1e-6, f"Expected +0.5, got {value}"


def test_bd_integral_internal_canonical_normal_accessor():
    """The canonical_normal accessor remains available and agrees with the
    normal that mesh.Gamma resolves to on the internal boundary."""

    mesh_internal, _, _ = _get_internal_mesh()
    normal = mesh_internal.canonical_normal("Internal")
    n_y = normal[1]

    bd_int = uw.maths.BdIntegral(mesh_internal, fn=n_y, boundary="Internal")
    value = bd_int.evaluate()

    assert abs(value - 1.0) < 1e-6, f"Expected +1.0, got {value}"


def test_bd_integral_internal_gamma_off_grid_zint():
    """Regression for the failing partition-through-boundary case from #327.

    With ``zintCoord=0.55`` (off-grid), the mpirun -n 2 partition seam runs
    through the internal boundary and one seam facet's raw ``petsc_n[]``
    flips sign: the unresolved integral returned 0.9375 = 1 − 2/32 instead
    of 1.0. With plain ``mesh.Gamma`` now resolved to the declared analytic
    normal, the value is exact regardless of partition."""
    mesh_off = BoxInternalBoundary(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0/32.0, zintCoord=0.55, simplex=True,
    )
    # Need at least one variable so BdIntegral has a section to integrate against
    uw.discretisation.MeshVariable("T_off", mesh_off, 1, degree=2)

    n_y = mesh_off.Gamma[1]
    val = uw.maths.BdIntegral(mesh_off, fn=n_y, boundary="Internal").evaluate()
    assert abs(val - 1.0) < 1e-6, (
        f"mesh.Gamma internal integral should be exactly +1, got {val}")


@pytest.mark.skipif(
    uw.mpi.size > 1,
    reason="mesh.deform crashes at np>1 (issue #360, kd-tree index rebuild)",
)
def test_bd_integral_internal_gamma_stale_after_deform():
    """Deformation invalidates the factory-declared analytic normal.

    The declaration describes the original geometry; after mesh.deform()
    resolving mesh.Gamma on the internal boundary must fail loudly rather
    than integrate a stale normal. Re-assigning mesh.boundary_normals
    re-declares it for the new geometry. The deformation used here
    vanishes on y=0.5 (sin(2*pi*y) = 0), so the internal boundary is
    unmoved and the re-declared +y normal gives exactly +1 again."""

    mesh_d = BoxInternalBoundary(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0/16.0, zintCoord=0.5, simplex=True,
    )
    uw.discretisation.MeshVariable("T_deform", mesh_d, 1, degree=2)
    n_y = mesh_d.Gamma[1]

    coords = np.array(mesh_d.X.coords)
    new_coords = coords.copy()
    new_coords[:, 1] += (
        0.01 * np.sin(np.pi * coords[:, 0]) * np.sin(2.0 * np.pi * coords[:, 1])
    )
    mesh_d.deform(new_coords)

    with pytest.raises(RuntimeError, match="coordinates have changed"):
        uw.maths.BdIntegral(mesh_d, fn=n_y, boundary="Internal").evaluate()

    mesh_d.boundary_normals = mesh_d.boundary_normals
    val = uw.maths.BdIntegral(mesh_d, fn=n_y, boundary="Internal").evaluate()
    assert abs(val - 1.0) < 1e-6, f"Expected +1.0 after re-declaration, got {val}"


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


# --- Spherical shell internal boundary tests ---

from underworld3.meshing import SphericalShellInternalBoundary

_R_SHELL_INNER = 0.55
_R_SHELL_INTERNAL = 0.775
_R_SHELL_OUTER = 1.0
_mesh_spherical_internal = None


def _get_spherical_internal_mesh():
    global _mesh_spherical_internal
    if _mesh_spherical_internal is None:
        _mesh_spherical_internal = SphericalShellInternalBoundary(
            radiusOuter=_R_SHELL_OUTER,
            radiusInternal=_R_SHELL_INTERNAL,
            radiusInner=_R_SHELL_INNER,
            cellSize=0.25,
            degree=1,
            qdegree=2,
        )
        uw.discretisation.MeshVariable(
            "T_spherical_internal", _mesh_spherical_internal, 1, degree=1
        )
    return _mesh_spherical_internal


@pytest.mark.level_2
@pytest.mark.tier_b
def test_bd_integral_spherical_internal_boundary_areas():
    """SphericalShellInternalBoundary preserves Lower/Internal/Upper labels.

    Overrides the module-level level_1/tier_a marks: this builds a full 3D
    gmsh+embed mesh (not a seconds-scale level_1 op), and the embed generator
    is not yet production-soaked for tier_a. See PR #242 review.
    """

    mesh_spherical = _get_spherical_internal_mesh()
    expected_areas = {
        "Lower": 4.0 * np.pi * _R_SHELL_INNER**2,
        "Internal": 4.0 * np.pi * _R_SHELL_INTERNAL**2,
        "Upper": 4.0 * np.pi * _R_SHELL_OUTER**2,
    }

    for boundary, expected in expected_areas.items():
        value = uw.maths.BdIntegral(mesh_spherical, fn=1.0, boundary=boundary).evaluate()
        relative_error = abs(value - expected) / expected
        assert relative_error < 0.06, (
            f"{boundary} area should be close to {expected:.4f}; "
            f"got {value:.4f} (relative error {relative_error:.3f})"
        )


@pytest.mark.level_2
@pytest.mark.tier_b
def test_spherical_internal_boundary_mesh_files_only(tmp_path, monkeypatch):
    """File generation can bypass Mesh construction and preserve labels."""
    from enum import Enum

    import underworld3.meshing.spherical as spherical
    from underworld3.coordinates import CoordinateSystemType

    mesh_file = str(tmp_path / "spherical_internal_mesh_files_only.msh")

    def fail_mesh_construction(*args, **kwargs):
        raise AssertionError(
            "write_mesh_files_only=True must not construct an Underworld Mesh"
        )

    with monkeypatch.context() as patch:
        patch.setattr(spherical, "Mesh", fail_mesh_construction)
        h5_file = spherical.SphericalShellInternalBoundary(
            radiusOuter=_R_SHELL_OUTER,
            radiusInternal=_R_SHELL_INTERNAL,
            radiusInner=_R_SHELL_INNER,
            cellSize=0.25,
            filename=mesh_file,
            write_mesh_files_only=True,
        )

    assert h5_file == f"{mesh_file}.h5"
    assert (tmp_path / "spherical_internal_mesh_files_only.msh").is_file()
    assert (tmp_path / "spherical_internal_mesh_files_only.msh.h5").is_file()

    class Boundaries(Enum):
        Centre = 1
        Lower = 11
        Internal = 12
        Upper = 13
        All_Boundaries = 1001

    reloaded_mesh = uw.discretisation.Mesh(
        h5_file,
        degree=1,
        qdegree=2,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=Boundaries,
    )

    for boundary in ("Lower", "Internal", "Upper"):
        label = reloaded_mesh.dm.getLabel(boundary)
        assert label is not None
        assert label.getNumValues() > 0


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


def test_bd_integral_after_deform_matches_expected_areas():
    """Boundary integrals must remain correct after mesh.deform().

    Serial counterpart of
    tests/parallel/test_0765_internal_boundary_integral_mpi.py::test_deformed_spherical_shell_boundary_area_parallel,
    which only exercises this path under --parallel (MPI rank-ownership
    edge case). Added 2026-06-25 after a regression in that MPI-only test
    (an unmigrated mesh._deform_mesh() call tripped the new
    _assert_coord_mutation_allowed() guard from commit f99c8aa2) went
    unnoticed for over a week because nothing in the serial suite exercised
    deformation + BdIntegral together. This test covers that basic
    correctness path -- not the MPI-specific rank-ownership case, which
    stays in the parallel-only test.
    """

    mesh_spherical = _build_spherical_shell_for_integrals()

    coords = np.asarray(mesh_spherical.X.coords, dtype=np.float64).copy()
    radii = np.linalg.norm(coords, axis=1)
    thickness = 0.5
    t = (radii - 0.5) / thickness
    a = np.log(2.0)
    mapped = (np.exp(a * t) - 1.0) / (np.exp(a) - 1.0)
    new_radii = 0.5 + thickness * mapped
    mesh_spherical.deform(coords * (new_radii / radii)[:, None])

    expected_lower = 4.0 * np.pi * 0.5**2
    expected_upper = 4.0 * np.pi * 1.0**2

    lower = float(uw.maths.BdIntegral(mesh_spherical, fn=1.0, boundary="Lower").evaluate())
    upper = float(uw.maths.BdIntegral(mesh_spherical, fn=1.0, boundary="Upper").evaluate())

    rel_err_lower = abs(lower - expected_lower) / expected_lower
    rel_err_upper = abs(upper - expected_upper) / expected_upper

    assert rel_err_lower < 5.0e-2, (
        f"Deformed lower area rel_err={rel_err_lower:.3e}, value={lower}, expected={expected_lower}"
    )
    assert rel_err_upper < 5.0e-2, (
        f"Deformed upper area rel_err={rel_err_upper:.3e}, value={upper}, expected={expected_upper}"
    )
