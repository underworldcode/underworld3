"""
Test DMFieldEvaluate — FE-exact field and gradient evaluation at arbitrary points.

Verifies that ``uw.function.dmfield_evaluate`` returns machine-precision-exact
values and gradients for FE-representable fields, and that the results can be
used to compute derived quantities (strain rate, viscosity) without a mass-matrix
projection.

.. note::
   DMFieldEvaluate uses PETSc's ``DMFieldEvaluate`` which computes the FE
   interpolant at each query point.  For fields that are in the FE space
   (e.g., a quadratic velocity field on P2 elements), the gradient is exact
   to machine precision at any interior point.
"""
import numpy as np
import pytest
import underworld3 as uw

from underworld3.function import dmfield_evaluate, dmfield_evaluate_clear_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh():
    """Coarse quad mesh — good enough for machine-precision tests."""
    return uw.meshing.StructuredQuadBox(elementRes=(6, 6))


@pytest.fixture(scope="module")
def quad_mesh():
    """Quadratic velocity field (exact on P2) and scalar (exact on P1)."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6))
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
    s = uw.discretisation.MeshVariable("s", mesh, 1, degree=1)
    # v = (x², y²)  — gradient: dvx/dx=2x, dvy/dy=2y, cross terms=0
    v.data[:, 0] = v.coords[:, 0] ** 2
    v.data[:, 1] = v.coords[:, 1] ** 2
    # s = x + y     — gradient: ds/dx=1, ds/dy=1
    s.data[:, 0] = s.coords[:, 0] + s.coords[:, 1]
    return mesh, v, s


@pytest.fixture(scope="module")
def tri_mesh():
    """Unstructured simplex box — non-affine elements."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.15
    )
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
    v.data[:, 0] = v.coords[:, 0] ** 2
    v.data[:, 1] = v.coords[:, 1] ** 2
    return mesh, v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCellVolumes:
    """Cell volumes computed via PETSc computeCellGeometryFVM."""

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_quad_box_volume(self):
        mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))
        v = mesh._cell_volumes
        assert (v > 0).all()
        assert np.allclose(v.sum(), 1.0, rtol=1e-3)

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_simplex_box_volume(self):
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(2, 1), cellSize=0.3
        )
        v = mesh._cell_volumes
        assert (v > 0).all()
        assert np.allclose(v.sum(), 2.0, rtol=1e-3)


class TestDMFieldEvaluate:
    """FE-exact field and gradient evaluation."""

    # ---- Basic field values ------------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_field_values_quad(self, quad_mesh):
        """B (field values) should equal the variable's nodal data."""
        mesh, v, s = quad_mesh
        B_v, _, _ = dmfield_evaluate(v, mesh.X.coords, gradient=False)
        B_s, _, _ = dmfield_evaluate(s, mesh.X.coords, gradient=False)
        # Compare to the known function
        cx, cy = mesh.X.coords[:, 0], mesh.X.coords[:, 1]
        assert np.allclose(B_v[:, 0], cx ** 2, atol=1e-13)
        assert np.allclose(B_v[:, 1], cy ** 2, atol=1e-13)
        assert np.allclose(B_s[:, 0], cx + cy, atol=1e-13)

    # ---- Gradient accuracy -------------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_gradient_scalar(self, quad_mesh):
        """Scalar gradient: s = x + y → ds/dx=1, ds/dy=1."""
        mesh, _, s = quad_mesh
        _, D, _ = dmfield_evaluate(s, mesh.X.coords, gradient=True)
        assert np.allclose(D[:, 0, 0], 1.0, atol=1e-13), "ds/dx"
        assert np.allclose(D[:, 1, 0], 1.0, atol=1e-13), "ds/dy"

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_gradient_vector_identity(self, quad_mesh):
        """Vector gradient: v = (x², y²) → dvx/dx=2x, dvy/dy=2y."""
        mesh, v, _ = quad_mesh
        _, D, _ = dmfield_evaluate(v, mesh.X.coords, gradient=True)
        cx, cy = mesh.X.coords[:, 0], mesh.X.coords[:, 1]
        assert np.allclose(D[:, 0, 0], 2 * cx, atol=1e-12), "dvx/dx"
        assert np.allclose(D[:, 1, 1], 2 * cy, atol=1e-12), "dvy/dy"
        # Cross-derivatives should be close to zero.  At element-boundary
        # nodes the FE basis from one adjacent element contributes tiny FP
        # residuals (~1e-12–1e-14 for double precision).
        assert np.max(np.abs(D[:, 1, 0])) < 1e-12, "dvx/dy"
        assert np.max(np.abs(D[:, 0, 1])) < 1e-12, "dvy/dx"

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_gradient_vector_nontrivial(self, tri_mesh):
        """Non-trivial gradients on unstructured simplex mesh."""
        mesh, v = tri_mesh
        # v = (x², y²) → dvx/dx = 2x, dvy/dy = 2y, cross = 0
        _, D, _ = dmfield_evaluate(v, mesh.X.coords, gradient=True)
        cx, cy = mesh.X.coords[:, 0], mesh.X.coords[:, 1]
        assert np.allclose(D[:, 0, 0], 2 * cx, atol=1e-12), "dvx/dx"
        assert np.allclose(D[:, 1, 1], 2 * cy, atol=1e-12), "dvy/dy"

    # ---- Strain rate from gradient -----------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_strain_rate(self, quad_mesh):
        """Strain rate computed from D matches analytical value."""
        mesh, v, _ = quad_mesh
        _, D, _ = dmfield_evaluate(v, mesh.X.coords, gradient=True)
        # D[k, i, j] = ∂ⱼ/∂xᵢ
        exx = D[:, 0, 0]
        eyy = D[:, 1, 1]
        exy = 0.5 * (D[:, 1, 0] + D[:, 0, 1])
        # For v = (x², y²): ε̇_xx = 2x, ε̇_yy = 2y, ε̇_xy = 0
        cx, cy = mesh.X.coords[:, 0], mesh.X.coords[:, 1]
        assert np.allclose(exx, 2 * cx, atol=1e-12)
        assert np.allclose(eyy, 2 * cy, atol=1e-12)
        assert np.max(np.abs(exy)) < 1e-12
        # Second invariant: √((ε̇_xx² + ε̇_yy²) / 2)
        eII = np.sqrt((exx ** 2 + eyy ** 2) / 2)
        expected = np.sqrt((4 * cx ** 2 + 4 * cy ** 2) / 2)
        assert np.allclose(eII, expected, atol=1e-12)

    # ---- Fill mesh variable directly (no Projection) -----------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_fill_mesh_variable(self, quad_mesh):
        """Fill a degree-1 mesh variable from D without uw.systems.Projection."""
        mesh, v, _ = quad_mesh
        _, D, _ = dmfield_evaluate(v, mesh.X.coords, gradient=True)
        exx = D[:, 0, 0]
        eII_target = uw.discretisation.MeshVariable("eII", mesh, 1, degree=1)
        eII_target.data[:, 0] = exx  # just ε̇_xx
        # Verify by reading back
        B_check, _, _ = dmfield_evaluate(eII_target, mesh.X.coords, gradient=False)
        cx = mesh.X.coords[:, 0]
        assert np.allclose(B_check[:, 0], 2 * cx, atol=1e-12)

    # ---- Hessian (second derivatives) -------------------------------------

    @pytest.mark.level_2
    @pytest.mark.tier_b
    def test_hessian_scalar(self, quad_mesh):
        """Hessian of a linear field is zero."""
        mesh, _, s = quad_mesh
        _, _, H = dmfield_evaluate(s, mesh.X.coords, gradient=True, hessian=True)
        # s = x + y → all second derivatives are zero.
        # Hessian at element-boundary nodes has small FP noise (~1e-11
        # for double precision on quad elements).
        assert H.shape == (mesh.X.coords.shape[0], mesh.dim, mesh.dim, 1)
        assert np.max(np.abs(H)) < 1e-10

    # ---- Caching -----------------------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_cache_reuse(self, quad_mesh):
        """Second call returns identical results (cached DMField reused)."""
        mesh, v, _ = quad_mesh
        B1, D1, _ = dmfield_evaluate(v, mesh.X.coords, gradient=True)
        B2, D2, _ = dmfield_evaluate(v, mesh.X.coords, gradient=True)
        assert np.allclose(B1, B2)
        assert np.allclose(D1, D2)

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_cache_clear(self, quad_mesh):
        """After clearing the cache, results are still correct."""
        mesh, v, _ = quad_mesh
        dmfield_evaluate_clear_cache()
        B, D, _ = dmfield_evaluate(v, mesh.X.coords, gradient=True)
        cx = mesh.X.coords[:, 0]
        assert np.allclose(D[:, 0, 0], 2 * cx, atol=1e-12)

    # ---- No gradient requested ---------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_no_gradient(self, quad_mesh):
        """gradient=False returns D=None."""
        mesh, v, _ = quad_mesh
        B, D, H = dmfield_evaluate(v, mesh.X.coords, gradient=False)
        assert D is None
        assert B is not None
        assert B.shape == (mesh.X.coords.shape[0], mesh.dim)

    # ---- Cleanup -----------------------------------------------------------

    @pytest.mark.level_2
    @pytest.mark.tier_b
    def test_cleanup(self):
        """Repeated create/destroy cycles don't leak or crash."""
        dmfield_evaluate_clear_cache()
        for _ in range(5):
            mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
            v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
            v.data[:, 0] = v.coords[:, 0]
            v.data[:, 1] = v.coords[:, 1]
            B, D, _ = dmfield_evaluate(v, mesh.X.coords, gradient=True)
            assert np.allclose(D[:, 0, 0], 1.0, atol=1e-13)
            dmfield_evaluate_clear_cache()
