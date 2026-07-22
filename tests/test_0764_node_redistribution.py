"""Contract tests for the purposeful adapt/redistribute naming (2026-07).

The 2026-07 naming ruling: user-facing names state the CAPABILITY;
algorithm names (NVB, MMPDE) live in internals and docs. Locked here:

* ``uw.meshing.node_redistribution(mesh, metric)`` — the user entry for
  fixed-topology node redistribution — dispatches through the
  mesh-controlled ``Mesh.redistribute_nodes`` method (mesh types control
  how they can be modified);
* supported: 2D simplex (triangle) meshes, warning-free;
* unsupported mesh types (quad/hex, 3D simplex, manifolds) raise an
  honest ``NotImplementedError`` stating what exists;
* ``mesh.adapt(metric, max_levels=...)`` needs no ``engine=`` and
  defaults to the graded NVB engine on 2D meshes.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw


pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _tri_box(cs=1.0 / 6):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cs)


def _bump_metric(mesh):
    x, y = mesh.CoordinateSystem.X
    return 1 + 4 * sympy.exp(-(((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05))


class TestNodeRedistribution:
    def test_moves_2d_simplex_zero_warnings(self):
        import warnings
        mesh = _tri_box()
        rho = _bump_metric(mesh)
        n_verts = np.asarray(mesh.X.coords).shape[0]
        before = np.asarray(mesh.X.coords).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            uw.meshing.node_redistribution(
                mesh, rho, method_kwargs=dict(n_outer=3))
        after = np.asarray(mesh.X.coords)
        assert after.shape[0] == n_verts          # topology preserved
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after)     # nodes actually moved

    def test_free_function_matches_method(self):
        """The free function is a thin dispatch to the mesh-controlled
        method — identical result on twin meshes."""
        m1 = _tri_box()
        m2 = _tri_box()
        uw.meshing.node_redistribution(
            m1, _bump_metric(m1), method_kwargs=dict(n_outer=3))
        m2.redistribute_nodes(
            _bump_metric(m2), method_kwargs=dict(n_outer=3))
        assert np.array_equal(np.asarray(m1.X.coords),
                              np.asarray(m2.X.coords))

    def test_quad_mesh_raises_not_implemented(self):
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
        with pytest.raises(NotImplementedError,
                           match="2D simplex"):
            uw.meshing.node_redistribution(mesh, sympy.sympify(1))

    def test_3d_simplex_raises_not_implemented(self):
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
            cellSize=0.5)
        with pytest.raises(NotImplementedError,
                           match="2D simplex"):
            uw.meshing.node_redistribution(mesh, sympy.sympify(1))

    def test_exported_from_meshing_namespace(self):
        assert "node_redistribution" in uw.meshing.__all__
        assert callable(uw.meshing.node_redistribution)



class TestEngineLessAdapt:
    """mesh.adapt(metric, max_levels=...) works without engine= and
    resolves to the graded NVB engine on a 2D mesh (the algorithm name
    stays an advanced selector)."""

    def _base_and_metric(self):
        # adapt() needs a base built with refinement>=1 (the dm_hierarchy
        # supplies the geometric-MG coarse tail).
        base = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
            cellSize=0.25, refinement=1, qdegree=3)
        # 1/h^2 metric demanding refinement in a central band
        M = uw.discretisation.MeshVariable("Mnr", base, 1, degree=1)
        band = sympy.exp(-(((base.N.x - 0.5) / 0.08) ** 2))
        vals = uw.function.evaluate(
            1.0 / (0.07 + (1.0 / 60 - 0.07) * band) ** 2,
            np.asarray(M.coords))
        M.data[:, 0] = np.asarray(vals).reshape(-1)
        return base, M

    def test_engine_less_adapt_returns_child(self):
        base, M = self._base_and_metric()
        child = base.adapt(M, max_levels=1)
        assert child is not base
        assert child.parent is base

    def test_engine_less_adapt_3d_returns_child(self):
        # The 2026-07-17 "refuse honestly" placeholder is retired: 3D
        # refinement landed with the adaptivity capstone (round 1), so the
        # engine-less call now returns a graded tetrahedral child. Full 3D
        # gates live in test_0840/test_0842.
        base3d = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
            cellSize=0.5, refinement=1,
        )
        M = uw.discretisation.MeshVariable("M3d", base3d, 1, degree=1)
        M.data[:, 0] = 1.0 / 0.12**2      # finer than the refined base (~0.22)
        child = base3d.adapt(M, max_levels=1)
        assert child.parent is base3d
        cs, ce = child.dm.getHeightStratum(0)
        bs, be = base3d.dm.getHeightStratum(0)
        assert ce - cs > be - bs

    def test_redistribute_then_adapt_composition(self, caplog):
        """adapt() after redistribute_nodes refines the MOVED geometry and
        must not touch the parent's static hierarchy: DMClone shares the
        coordinates Vec by reference, so the moved-coordinate carry has to
        install a duplicate (review finding, 2026-07 capstone close-out).

        The same mover run also guards #376: masked in-place updates
        (``data[mask] /= s``) must not fire the canonical PETSc-sync
        callback on the fancy-indexed COPY — in serial that surfaced as
        swallowed 'Callback error ... could not broadcast' log warnings;
        in parallel the partition-dependent mask made the collective sync
        rank-asymmetric and hung the mover. The refinement=1 box is the
        trigger (`Gamma_P1` has one zero-magnitude corner normal, so the
        mask drops one node)."""
        import logging

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
            cellSize=0.4, refinement=1, qdegree=2)
        hier_before = mesh.dm_hierarchy[-1].getCoordinatesLocal().array.copy()

        x, y = mesh.X
        d = sympy.Abs((x - 0.35) * 0.866 + (y - 1.0) * 0.5)
        h = sympy.sqrt(0.08**2 + (1.6 * d) ** 2)
        with caplog.at_level(logging.WARNING,
                             logger="underworld3.utilities.nd_array_callback"):
            mesh.redistribute_nodes(1.0 / h**2, slip_surfaces=True)
        bad = [r.message for r in caplog.records
               if "allback error" in r.message]
        assert not bad, f"swallowed canonical-callback errors: {bad[:3]}"
        moved = mesh.dm.getCoordinatesLocal().array.copy()
        assert not np.array_equal(moved, hier_before)   # mover moved nodes

        def metric(centroids):
            dd = np.abs((np.asarray(centroids)[:, :2]
                         - np.array([0.35, 1.0])) @ np.array([0.866, 0.5]))
            return 1.0 / np.minimum(np.sqrt(0.02**2 + (1.6 * dd) ** 2),
                                    0.4) ** 2

        child = mesh.adapt(metric, max_levels=2)
        # the child's base generation is the MOVED mesh, vertex for vertex
        base_verts = moved.reshape(-1, 2)
        child_verts = np.asarray(child.X.coords)
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(child_verts).query(base_verts)
        assert dist.max() < 1e-12
        # ... and the parent's static hierarchy is untouched
        hier_after = mesh.dm_hierarchy[-1].getCoordinatesLocal().array
        assert np.array_equal(hier_after, hier_before)

    def test_engine_less_adapt_is_nvb(self):
        """The engine-less child matches an explicit engine='nvb' child
        cell-for-cell (and differs from SBR's uniform-patch closure)."""
        base1, M1 = self._base_and_metric()
        c_default = base1.adapt(M1, max_levels=1)
        base2, M2 = self._base_and_metric()
        c_nvb = base2.adapt(M2, max_levels=1, engine="nvb")

        def n_cells(m):
            cS, cE = m.dm.getHeightStratum(0)
            return cE - cS

        assert n_cells(c_default) == n_cells(c_nvb)
        assert np.array_equal(np.asarray(c_default.X.coords),
                              np.asarray(c_nvb.X.coords))
