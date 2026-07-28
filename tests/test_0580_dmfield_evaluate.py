"""
Test DMFieldEvaluate — FE-exact field and gradient evaluation at arbitrary points.

Verifies that ``uw.function.dmfield_evaluate`` returns machine-precision-exact
values and gradients for FE-representable fields.  Tests the corrected
``(n, nc, dim)`` layout for vector-gradient tensors, NaN semantics for
unlocated points, and parallel-safe collective operation.

.. note::
   DMFieldEvaluate uses PETSc's ``DMFieldEvaluate`` which computes the FE
   interpolant at each query point.  For fields that are in the FE space
   (e.g., a quadratic velocity field on P2 elements), the gradient is exact
   to machine precision at any interior point.
"""
import numpy as np
import pytest
import underworld3 as uw

from underworld3.function import dmfield_evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quad_mesh():
    """Quadratic velocity field (exact on P2) and scalar (exact on P1)."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6))
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
    s = uw.discretisation.MeshVariable("s", mesh, 1, degree=1)
    # v = (x^2, y^2)  -- gradient: dvx/dx=2x, dvy/dy=2y, cross terms=0
    v.data[:, 0] = v.coords[:, 0] ** 2
    v.data[:, 1] = v.coords[:, 1] ** 2
    # s = x + y     -- gradient: ds/dx=1, ds/dy=1
    s.data[:, 0] = s.coords[:, 0] + s.coords[:, 1]
    return mesh, v, s


@pytest.fixture(scope="module")
def tri_mesh():
    """Unstructured simplex box -- non-affine elements."""
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

class TestDMFieldEvaluate:
    """FE-exact field and gradient evaluation."""

    # ---- Basic field values ------------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_field_values_tri(self):
        """B (field values) should equal the variable's nodal data.

        Uses an unstructured simplex mesh — DMLocatePoints locates all
        P2 nodes reliably (0/433 failures on tri vs ~7.6% on quads).
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.15
        )
        v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
        s = uw.discretisation.MeshVariable("s", mesh, 1, degree=1)
        v.data[:, 0] = v.coords[:, 0] ** 2
        v.data[:, 1] = v.coords[:, 1] ** 2
        s.data[:, 0] = s.coords[:, 0] + s.coords[:, 1]
        # Evaluate at each variable's own nodes
        B_v, _, _ = dmfield_evaluate(v, v.coords, gradient=False)
        assert np.allclose(B_v[:, 0], v.coords[:, 0] ** 2, atol=1e-13)
        assert np.allclose(B_v[:, 1], v.coords[:, 1] ** 2, atol=1e-13)
        B_s, _, _ = dmfield_evaluate(s, s.coords, gradient=False)
        assert np.allclose(B_s[:, 0], s.coords[:, 0] + s.coords[:, 1], atol=1e-13)

    # ---- Gradient accuracy -------------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_gradient_scalar(self, quad_mesh):
        """Scalar gradient: s = x + y -> ds/dx=1, ds/dy=1."""
        mesh, _, s = quad_mesh
        _, D, _ = dmfield_evaluate(s, s.coords, gradient=True)
        # D[k, j, i]: scalar nc=1, so D[k, 0, 0] = ds/dx, D[k, 0, 1] = ds/dy
        assert np.allclose(D[:, 0, 0], 1.0, atol=1e-13), "ds/dx"
        assert np.allclose(D[:, 0, 1], 1.0, atol=1e-13), "ds/dy"

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_gradient_vector_identity(self, quad_mesh):
        """Vector gradient: v = (x^2, y^2) -> dvx/dx=2x, dvy/dy=2y.
        
        Evaluates the P2 velocity gradient at a degree-1 variable's nodes
        (the natural target for derivative quantities like strain rate).
        """
        mesh, v, s = quad_mesh
        _, D, _ = dmfield_evaluate(v, s.coords, gradient=True)
        # D[k, j, i] -> D[:, 0, 0] = dvx/dx, D[:, 1, 1] = dvy/dy
        # Cross terms: D[:, 0, 1] = dvx/dy, D[:, 1, 0] = dvy/dx
        cx, cy = s.coords[:, 0], s.coords[:, 1]
        assert np.allclose(D[:, 0, 0], 2 * cx, atol=1e-12), "dvx/dx"
        assert np.allclose(D[:, 1, 1], 2 * cy, atol=1e-12), "dvy/dy"
        assert np.max(np.abs(D[:, 0, 1])) < 1e-12, "dvx/dy"
        assert np.max(np.abs(D[:, 1, 0])) < 1e-12, "dvy/dx"

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_gradient_vector_nontrivial(self):
        """Non-trivial gradients on unstructured simplex mesh.
        
        Evaluates the P2 velocity gradient at a degree-1 variable's nodes.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.15
        )
        v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
        v.data[:, 0] = v.coords[:, 0] ** 2
        v.data[:, 1] = v.coords[:, 1] ** 2
        # Degree-1 variable for evaluating the gradient
        sr = uw.discretisation.MeshVariable("sr", mesh, 1, degree=1)
        _, D, _ = dmfield_evaluate(v, sr.coords, gradient=True)
        cx, cy = sr.coords[:, 0], sr.coords[:, 1]
        assert np.allclose(D[:, 0, 0], 2 * cx, atol=1e-12), "dvx/dx"
        assert np.allclose(D[:, 1, 1], 2 * cy, atol=1e-12), "dvy/dy"

    # ---- Asymmetric layout (CRITICAL-2 guard) -----------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_gradient_asymmetric_layout(self, quad_mesh):
        """v = (y, 0) — dvx/dy=1, all others 0.  Pins D[k, j, i] layout."""
        mesh, _, s = quad_mesh
        v = uw.discretisation.MeshVariable("v_asym", mesh, mesh.dim, degree=1)
        v.data[:, 0] = v.coords[:, 1]   # vx = y
        v.data[:, 1] = 0.0              # vy = 0
        _, D, _ = dmfield_evaluate(v, v.coords, gradient=True)
        # D[k, j, i]: j=component, i=spatial dim
        assert np.allclose(D[:, 0, 1], 1.0, atol=1e-13),  "dvx/dy should be 1"
        assert np.max(np.abs(D[:, 1, 0])) < 1e-13, "dvy/dx should be 0"
        assert np.max(np.abs(D[:, 0, 0])) < 1e-13, "dvx/dx should be 0"
        assert np.max(np.abs(D[:, 1, 1])) < 1e-13, "dvy/dy should be 0"

    # ---- Hessian ----------------------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_hessian_asymmetric_layout(self):
        """w = (x^2, xy, 0) — in P2, pins H[k, j, i, l] layout.
        
        Uses a simplex mesh (DMLocatePoints can locate all P2 nodes) and
        evaluates the Hessian at a degree-1 variable's nodes.
        Field is quadratic, so second derivatives are exact for P2.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.15
        )
        w = uw.discretisation.MeshVariable("w", mesh, 2, degree=2)
        wc_x, wc_y = w.coords[:, 0], w.coords[:, 1]
        w.data[:, 0] = wc_x * wc_x               # w0 = x^2
        w.data[:, 1] = wc_x * wc_y               # w1 = xy
        # Degree-1 variable for evaluating the Hessian
        sr = uw.discretisation.MeshVariable("sr", mesh, 1, degree=1)
        _, _, H = dmfield_evaluate(w, sr.coords, gradient=True, hessian=True)
        # H[k, j, i, l]: partial^2(component j)/partial x_i partial x_l
        # w0 = x^2: H[:, 0, 0, 0] = 2, all others 0
        assert np.allclose(H[:, 0, 0, 0], 2.0, atol=1e-12), "partial^2 w0/partial x^2"
        assert np.max(np.abs(H[:, 0, 0, 1])) < 1e-12, "partial^2 w0/partial x partial y"
        assert np.max(np.abs(H[:, 0, 1, 0])) < 1e-12, "partial^2 w0/partial y partial x"
        assert np.max(np.abs(H[:, 0, 1, 1])) < 1e-12, "partial^2 w0/partial y^2"
        # w1 = xy: H[:, 1, 0, 1] = H[:, 1, 1, 0] = 1, diagonal = 0
        assert np.allclose(H[:, 1, 0, 1], 1.0, atol=1e-12), "partial^2 w1/partial x partial y"
        assert np.allclose(H[:, 1, 1, 0], 1.0, atol=1e-12), "partial^2 w1/partial y partial x"
        assert np.max(np.abs(H[:, 1, 0, 0])) < 1e-12, "partial^2 w1/partial x^2"
        assert np.max(np.abs(H[:, 1, 1, 1])) < 1e-12, "partial^2 w1/partial y^2"

    @pytest.mark.level_2
    @pytest.mark.tier_b
    def test_hessian_scalar(self, quad_mesh):
        """Hessian of a linear field is zero."""
        mesh, _, s = quad_mesh
        _, _, H = dmfield_evaluate(s, s.coords, gradient=True, hessian=True)
        # H[k, j, i, l]: scalar nc=1 -> H.shape == (n, 1, dim, dim)
        assert H.shape == (s.coords.shape[0], 1, mesh.dim, mesh.dim)
        assert np.max(np.abs(H)) < 1e-10

    # ---- Strain rate from gradient -----------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_strain_rate(self, quad_mesh):
        """Strain rate computed from D at a degree-1 variable's nodes."""
        mesh, v, _ = quad_mesh
        # Create a degree-1 variable to represent where we want strain rate
        sr = uw.discretisation.MeshVariable("sr", mesh, 1, degree=1)
        # Evaluate P2 velocity at the degree-1 variable's nodes
        _, D, _ = dmfield_evaluate(v, sr.coords, gradient=True)
        # D[k, j, i] -> strain rate symmetric part
        exx = D[:, 0, 0]
        eyy = D[:, 1, 1]
        exy = 0.5 * (D[:, 0, 1] + D[:, 1, 0])
        # For v = (x^2, y^2): eps_xx = 2x, eps_yy = 2y, eps_xy = 0
        cx, cy = sr.coords[:, 0], sr.coords[:, 1]
        assert np.allclose(exx, 2 * cx, atol=1e-12)
        assert np.allclose(eyy, 2 * cy, atol=1e-12)
        assert np.max(np.abs(exy)) < 1e-12
        # Second invariant: sqrt((eps_xx^2 + eps_yy^2) / 2)
        eII = np.sqrt((exx ** 2 + eyy ** 2) / 2)
        expected = np.sqrt((4 * cx ** 2 + 4 * cy ** 2) / 2)
        assert np.allclose(eII, expected, atol=1e-12)

    # ---- Fill mesh variable directly (no Projection) -----------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_fill_mesh_variable(self):
        """Fill a degree-1 mesh variable from D without uw.systems.Projection.
        
        Uses its own mesh to avoid fixture contamination from tests that
        register extra variables on the shared fixture mesh.
        """
        mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6))
        v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
        eII_target = uw.discretisation.MeshVariable("eII", mesh, 1, degree=1)
        v.data[:, 0] = v.coords[:, 0] ** 2
        v.data[:, 1] = v.coords[:, 1] ** 2
        # Evaluate v at the target variable's nodes (degree-1 = mesh vertices)
        _, D, _ = dmfield_evaluate(v, eII_target.coords, gradient=True)
        eII_target.data[:, 0] = D[:, 0, 0]  # just eps_xx
        B_check, _, _ = dmfield_evaluate(eII_target, eII_target.coords, gradient=False)
        assert np.allclose(B_check[:, 0], 2 * eII_target.coords[:, 0], atol=1e-12)

    # ---- Unlocated points (CRITICAL-1 guard) ------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_unlocated_points_nan(self, quad_mesh):
        """Points outside domain return NaN -- no silent garbage."""
        mesh, v, _ = quad_mesh
        outside = np.array([[2.0, 2.0], [-1.0, -1.0], [0.5, 2.0]])
        B, D, H = dmfield_evaluate(v, outside, gradient=True, hessian=True)
        assert np.all(np.isnan(B)), "B should be NaN for outside points"
        assert np.all(np.isnan(D)), "D should be NaN for outside points"
        assert np.all(np.isnan(H)), "H should be NaN for outside points"

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_upper_face_no_crash(self, quad_mesh):
        """Points near the closed-domain upper face -- no crash, no garbage.
        
        Uses y=1.0-1e-10 instead of exact y=1.0 because PETSc 3.25.3
        ``DMFieldEvaluate_DS`` raises error 62 when ``DMLocatePoints``
        fails to locate points exactly on the boundary.
        """
        mesh, v, _ = quad_mesh
        eps = 1e-10
        upper = np.column_stack([np.linspace(0.0, 1.0, 20), np.full(20, 1.0 - eps)])
        B, _, _ = dmfield_evaluate(v, upper, gradient=False)
        assert np.all(np.isfinite(B)), "Values near upper face should be finite"

    # ---- No-cache determinism ----------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_repeated_calls_consistent(self, quad_mesh):
        """Without cache, repeated calls return identical results.
        
        Evaluates at a degree-1 variable's nodes (gradient target).
        """
        mesh, v, s = quad_mesh
        B1, D1, _ = dmfield_evaluate(v, s.coords, gradient=True)
        B2, D2, _ = dmfield_evaluate(v, s.coords, gradient=True)
        assert np.allclose(B1, B2)
        assert np.allclose(D1, D2)

    # ---- Empty coordinates --------------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_empty_coords(self, quad_mesh):
        """Zero-point evaluation returns correct-shape empty arrays."""
        mesh, v, _ = quad_mesh
        empty = np.empty((0, mesh.dim))
        B, D, H = dmfield_evaluate(v, empty, gradient=True, hessian=True)
        assert B.shape == (0, v.num_components)
        assert D.shape == (0, v.num_components, mesh.dim)
        assert H.shape == (0, v.num_components, mesh.dim, mesh.dim)

    # ---- Non-square shape (nc != dim) --------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_b
    def test_non_square_nc_dim(self):
        """nc=3, dim=2 -> D shape (n, 3, 2) -- catches reshape regression.
        
        Uses ``VarType.MATRIX`` with a ``(3, 1)`` shape tuple so that
        the variable has 3 components in 2D (which would otherwise
        fail vtype auto-inference).
        """
        mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6))
        v3 = uw.discretisation.MeshVariable("v3", mesh, (3, 1), degree=1,
                                             vtype=uw.VarType.MATRIX)
        cx, cy = v3.coords[:, 0], v3.coords[:, 1]
        v3.data[:, 0] = cx
        v3.data[:, 1] = cy
        v3.data[:, 2] = 0.0
        _, D, _ = dmfield_evaluate(v3, v3.coords, gradient=True)
        assert D.shape == (v3.coords.shape[0], 3, 2), \
            f"Expected (n, 3, 2), got {D.shape}"

    # ---- NULL-pointer paths -------------------------------------------------

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_hessian_only_no_gradient(self, quad_mesh):
        """gradient=False, hessian=True -- D=NULL path through PETSc."""
        mesh, v, s = quad_mesh
        B, D, H = dmfield_evaluate(v, s.coords, gradient=False, hessian=True)
        assert D is None
        assert B is not None
        assert H is not None
        assert H.shape == (s.coords.shape[0], v.num_components, mesh.dim, mesh.dim)

    @pytest.mark.level_1
    @pytest.mark.tier_a
    def test_values_only(self, quad_mesh):
        """gradient=False, hessian=False -- D=H=NULL path through PETSc."""
        mesh, v, s = quad_mesh
        B, D, H = dmfield_evaluate(v, s.coords, gradient=False, hessian=False)
        assert D is None
        assert H is None
        assert B is not None

    # ---- Create/destroy stress test -----------------------------------------

    @pytest.mark.level_2
    @pytest.mark.tier_b
    def test_repeated_create_destroy(self):
        """10 create/destroy cycles -- no PETSc leak or crash."""
        for i in range(10):
            mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
            v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
            sr = uw.discretisation.MeshVariable("sr", mesh, 1, degree=1)
            v.data[:, 0] = v.coords[:, 0]
            v.data[:, 1] = v.coords[:, 1]
            # Evaluate gradient at the degree-1 variable's nodes
            B, D, _ = dmfield_evaluate(v, sr.coords, gradient=True)
            assert np.allclose(D[:, 0, 0], 1.0, atol=1e-13)

    # ---- MPI parallel test (CRITICAL-3 guard) ------------------------------

    @pytest.mark.level_2
    @pytest.mark.tier_b
    @pytest.mark.mpi(min_size=2)
    def test_parallel_rank_local(self):
        """Each rank evaluates its own points; zero-point rank doesn't deadlock."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6))
        v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
        v.data[:, 0] = v.coords[:, 0] ** 2
        v.data[:, 1] = v.coords[:, 1] ** 2

        # Create degree-1 variable on ALL ranks (DM is shared across ranks)
        sr = uw.discretisation.MeshVariable("sr", mesh, 1, degree=1)

        rank = uw.mpi.rank
        if rank == 0:
            # Evaluate gradient at the degree-1 variable's nodes
            B, D, _ = dmfield_evaluate(v, sr.coords, gradient=True)
            assert B is not None
            assert D is not None
            cx = sr.coords[:, 0]
            assert np.allclose(D[:, 0, 0], 2 * cx, atol=1e-12)
        else:
            # Rank with zero local points -- must still call (collective)
            empty = np.empty((0, mesh.dim))
            B, D, _ = dmfield_evaluate(v, empty, gradient=True)
            assert B.shape == (0, v.num_components)
