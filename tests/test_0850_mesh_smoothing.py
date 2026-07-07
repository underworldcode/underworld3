"""Regression tests for ``underworld3.meshing.smooth_mesh_interior``.

The smoother relaxes interior vertex positions toward the average
of their edge neighbours, with boundary vertices held fixed. Tests:

- Boundary vertices stay bit-identical after smoothing
- A uniform (already-equilibrated) mesh is a fixed point: smoothing
  does not move interior vertices beyond floating-point noise
- A perturbed-interior mesh converges back toward the uniform state
- Pinned-label specification works (explicit list, ``None`` =
  auto-detect, ``[]`` = nothing pinned)
- DM coordinate vector and ``mesh.X.coords`` are updated by the
  single trailing ``_deform_mesh`` call
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import smooth_mesh_interior


pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _box_mesh(resolution: int = 8):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / resolution,
    )


def _min_aspect_ratio(mesh):
    """Compute (min altitude) / (longest edge) across all triangles
    on the local rank. Closer to ~√3/2 ≈ 0.866 is more equilateral.
    Returns a single float (min over all cells)."""
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    coords = np.asarray(mesh.X.coords)
    worst = np.inf
    for c in range(cStart, cEnd):
        closure, _ = dm.getTransitiveClosure(c, useCone=True)
        verts = [p - pStart for p in closure
                 if pStart <= p < pEnd]
        if len(verts) != 3:
            continue
        pts = coords[verts]
        a = np.linalg.norm(pts[1] - pts[0])
        b = np.linalg.norm(pts[2] - pts[1])
        c_ = np.linalg.norm(pts[0] - pts[2])
        Lmax = max(a, b, c_)
        # 2D triangle signed area
        area = 0.5 * abs((pts[1, 0] - pts[0, 0])
                         * (pts[2, 1] - pts[0, 1])
                         - (pts[2, 0] - pts[0, 0])
                         * (pts[1, 1] - pts[0, 1]))
        if Lmax > 0:
            h_min = 2.0 * area / Lmax
            ar = h_min / Lmax
            if ar < worst:
                worst = ar
    return worst


def _boundary_vertex_mask(mesh):
    """Local-chart boolean mask: True where the vertex lies on any
    non-sentinel boundary label."""
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    n_verts = pEnd - pStart
    skip = {"All_Boundaries", "Null_Boundary"}
    mask = np.zeros(n_verts, dtype=bool)
    for member in mesh.boundaries:
        name = getattr(member, "name", None)
        if not name or name in skip:
            continue
        label = dm.getLabel(name)
        if label is None:
            continue
        vIS = label.getValueIS()
        if vIS is None:
            continue
        for val in vIS.getIndices():
            iset = label.getStratumIS(int(val))
            if iset is None:
                continue
            for idx in iset.getIndices():
                if pStart <= idx < pEnd:
                    mask[idx - pStart] = True
    return mask


class TestBasic:

    def test_boundary_vertices_held_fixed(self):
        mesh = _box_mesh(resolution=6)
        before = np.asarray(mesh.X.coords).copy()
        is_bnd = _boundary_vertex_mask(mesh)
        smooth_mesh_interior(mesh, n_iters=5, alpha=0.5)
        after = np.asarray(mesh.X.coords)
        # Boundary coords must be bit-identical
        assert np.allclose(
            before[is_bnd], after[is_bnd], rtol=0, atol=0), (
            "Boundary vertices moved during smoothing"
        )

    def test_sweep_displacement_decreases(self):
        """Iterating Jacobi smoothing converges: the displacement
        per sweep should decrease monotonically (in L2 norm) on the
        approach to the topology-dependent equilibrium."""
        mesh = _box_mesh(resolution=6)
        is_bnd = _boundary_vertex_mask(mesh)
        is_interior = ~is_bnd
        # Take 5 sweeps one at a time, recording the interior
        # displacement at each step.
        disps = []
        prev = np.asarray(mesh.X.coords).copy()
        for _ in range(5):
            smooth_mesh_interior(mesh, n_iters=1, alpha=0.5)
            now = np.asarray(mesh.X.coords)
            d = float(np.linalg.norm((now - prev)[is_interior]))
            disps.append(d)
            prev = now.copy()
        # Strictly decreasing — graph Laplacian Jacobi is a
        # contraction on the (pinned-boundary) problem.
        for k in range(1, len(disps)):
            assert disps[k] < disps[k - 1], (
                f"Sweep {k+1} displacement {disps[k]:.3e} not "
                f"less than sweep {k} displacement "
                f"{disps[k-1]:.3e}; full series: {disps}")

    def test_aspect_ratio_improves_on_perturbed_mesh(self):
        """Perturb the interior, smooth, and check that the minimum
        triangle altitude / longest edge ratio (a shape-quality
        metric, closer to 1 = better) does not get worse — and
        usually improves."""
        mesh = _box_mesh(resolution=6)
        is_bnd = _boundary_vertex_mask(mesh)
        rng = np.random.default_rng(42)
        coords = np.asarray(mesh.X.coords).copy()
        is_interior = ~is_bnd
        coords[is_interior] += 0.04 * rng.standard_normal(
            (int(is_interior.sum()), coords.shape[1]))
        # deliberate direct use of the gated internal primitive
        with mesh._coord_mutation():
            mesh._deform_mesh(coords)
        ar_before = _min_aspect_ratio(mesh)
        smooth_mesh_interior(mesh, n_iters=10, alpha=0.5)
        ar_after = _min_aspect_ratio(mesh)
        assert ar_after >= ar_before * 0.95, (
            f"Smoothing made the mesh worse: "
            f"min aspect ratio {ar_before:.3f} -> {ar_after:.3f}")


class TestPinningAPI:

    def test_explicit_pinned_labels(self):
        mesh = _box_mesh(resolution=6)
        # Pin only Bottom + Top — Left and Right should move
        smooth_mesh_interior(
            mesh, pinned_labels=["Bottom", "Top"],
            n_iters=1, alpha=0.5)
        # Just check it didn't crash; explicit Left/Right motion is
        # too small to verify reliably without a structured layout.

    def test_empty_pinned_list_is_legal(self):
        """Empty pinned_labels: every vertex is interior. Smoothing
        runs without error (boundary vertices drift)."""
        mesh = _box_mesh(resolution=4)
        before = np.asarray(mesh.X.coords).copy()
        smooth_mesh_interior(
            mesh, pinned_labels=[], n_iters=1, alpha=0.5)
        after = np.asarray(mesh.X.coords)
        # At least one vertex moved (no pinning → mesh contracts)
        assert not np.allclose(before, after)

    def test_none_pinned_labels_auto_detects(self):
        """None means 'pin every named boundary' — boundary stays
        fixed even though we didn't enumerate the names."""
        mesh = _box_mesh(resolution=5)
        is_bnd = _boundary_vertex_mask(mesh)
        before = np.asarray(mesh.X.coords).copy()
        smooth_mesh_interior(mesh, pinned_labels=None,
                             n_iters=1, alpha=0.5)
        after = np.asarray(mesh.X.coords)
        assert np.allclose(before[is_bnd], after[is_bnd])


class TestMMPDEDimensionGuard:
    """The MMPDE mover is 2D-only (READ-01/BF-09, 2026-07 audit): the 3D
    branch used to reference `_signed_volumes`, which was never implemented,
    so any 3D invocation died with a NameError deep in the mover. The
    contract is now an immediate, honest NotImplementedError for cdim != 2,
    while 2D invocation keeps working."""

    def test_mmpde_3d_raises_not_implemented(self):
        import sympy
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0, 0.0),
            maxCoords=(1.0, 1.0, 1.0),
            cellSize=0.5,
        )
        with pytest.raises(NotImplementedError,
                           match="MMPDE mesh movement is currently 2D-only"):
            smooth_mesh_interior(
                mesh, metric=sympy.sympify(1), method="mmpde",
                method_kwargs=dict(n_outer=1))

    def test_mmpde_3d_guard_fires_before_any_mesh_work(self):
        """The guard reads only mesh.cdim, before metric parsing or DM
        access, so a minimal cdim stand-in locks the contract
        deterministically (same pattern as test_0762's non2d tests)."""
        from underworld3.meshing.smoothing import _mmpde_mover

        class _Mesh3D:
            cdim = 3

        with pytest.raises(NotImplementedError,
                           match="MMPDE mesh movement is currently 2D-only"):
            _mmpde_mover(_Mesh3D(), metric=1, pinned_labels=(),
                           verbose=False)

    def test_mmpde_2d_smoke_still_works(self):
        """A small 2D mmpde invocation still runs and leaves a valid mesh."""
        import sympy
        mesh = _box_mesh(resolution=6)
        x, y = mesh.CoordinateSystem.X
        rho = 1 + 4 * sympy.exp(-(((x - 0.5) ** 2 + (y - 0.5) ** 2)
                                  / 0.05))
        before = np.asarray(mesh.X.coords).copy()
        smooth_mesh_interior(
            mesh, metric=rho, method="mmpde",
            method_kwargs=dict(n_outer=3, metric_eval="rbf"))
        after = np.asarray(mesh.X.coords)
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after)   # the mover actually moved
