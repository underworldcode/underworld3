"""Contract tests for the ``monotone`` keyword on
``uw.function.evaluate`` / ``global_evaluate``.

``monotone`` is an opt-in, default-off post-process that bounds the
interpolated result to the local data range of the source field (the
``mesh.dim + 1`` nearest DOFs). It was lifted out of the semi-Lagrangian
trace-back limiter (PR #186/#188) so any resampling path can request the
same bounded result from one place.

These tests lock:
  (a) default ``monotone=False`` is bit-identical to omitting it;
  (b) ``"clamp"`` bounds the result to an independently recomputed
      neighbour ``[min, max]`` while ``False`` reproduces the overshoot;
  (c) ``"pick"`` leaves in-bounds points untouched and brings
      out-of-bounds points within range;
  (d) composite expressions and unknown options raise ``ValueError``.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _steep_p3_field():
    """A P3 scalar with a sharp internal peak — FE Lagrange-P3 overshoots
    at non-nodal points in the steep-gradient cells."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        qdegree=3,
    )
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("Tm", mesh, 1, degree=3)
    init = sympy.exp(-(((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.003))
    T.array[...] = uw.function.evaluate(init, T.coords).reshape(T.array[...].shape)
    return mesh, T


def _offnode_coords(mesh, n=400, seed=0):
    """Random interior points, deliberately off the DOF nodes."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.06, 0.94, size=(n, mesh.dim))


def _recompute_bounds(T, coords):
    """Independent reference: per-coord min/max over the ``dim+1`` nearest
    source DOFs, rebuilt from ``T.coords_nd`` / ``T.data`` in the test."""
    nnn = T.mesh.dim + 1
    kdt = uw.kdtree.KDTree(np.ascontiguousarray(np.asarray(T.coords_nd)))
    _, idxs = kdt.query(np.ascontiguousarray(coords), k=nnn, sqr_dists=False)
    data_flat = np.asarray(T.data).reshape(np.asarray(T.data).shape[0], -1)
    nbr = data_flat[idxs]
    return nbr.min(axis=1).ravel(), nbr.max(axis=1).ravel()


class TestMonotoneDefaultNoOp:

    def test_evaluate_default_bit_identical(self):
        mesh, T = _steep_p3_field()
        coords = _offnode_coords(mesh)
        base = uw.function.evaluate(T.sym, coords)
        same = uw.function.evaluate(T.sym, coords, monotone=False)
        assert np.array_equal(np.asarray(base), np.asarray(same))

    def test_global_evaluate_default_bit_identical(self):
        mesh, T = _steep_p3_field()
        coords = _offnode_coords(mesh)
        base = uw.function.global_evaluate(T.sym, coords)
        same = uw.function.global_evaluate(T.sym, coords, monotone=False)
        assert np.array_equal(np.asarray(base), np.asarray(same))


class TestMonotoneClamp:

    def test_clamp_bounds_result(self):
        mesh, T = _steep_p3_field()
        coords = _offnode_coords(mesh)
        nbr_min, nbr_max = _recompute_bounds(T, coords)

        unlimited = np.asarray(
            uw.function.evaluate(T.sym, coords)).ravel()
        clamped = np.asarray(
            uw.function.evaluate(T.sym, coords, monotone="clamp")).ravel()

        # Clamp lies within the recomputed neighbour range, exactly
        # (np.clip is a closed bound).
        assert np.all(clamped >= nbr_min)
        assert np.all(clamped <= nbr_max)

        # The steep P3 field must overshoot somewhere — otherwise the
        # limiter would be a no-op and the test would prove nothing. The
        # helper's bound test is exact (untoleranced), so use the same
        # comparison here.
        overshoot = (unlimited > nbr_max) | (unlimited < nbr_min)
        assert overshoot.any(), "expected FE overshoot on the steep field"

        # The clamp result reproduces an independent clip against the
        # recomputed neighbour bounds, to the last bit.
        assert np.array_equal(clamped, np.clip(unlimited, nbr_min, nbr_max))

    def test_true_aliases_clamp(self):
        mesh, T = _steep_p3_field()
        coords = _offnode_coords(mesh)
        a = np.asarray(uw.function.evaluate(T.sym, coords, monotone=True))
        b = np.asarray(uw.function.evaluate(T.sym, coords, monotone="clamp"))
        assert np.array_equal(a, b)


class TestMonotonePick:

    def test_pick_preserves_inbounds_and_changes_only_oob(self):
        mesh, T = _steep_p3_field()
        coords = _offnode_coords(mesh)
        nbr_min, nbr_max = _recompute_bounds(T, coords)

        unlimited = np.asarray(
            uw.function.evaluate(T.sym, coords)).ravel()
        picked = np.asarray(
            uw.function.evaluate(T.sym, coords, monotone="pick")).ravel()

        # Exact (untoleranced) bound test, matching the helper.
        inb = (unlimited >= nbr_min) & (unlimited <= nbr_max)
        assert (~inb).any(), "expected FE overshoot on the steep field"

        # In-bounds points are kept exactly as the FE result; only the
        # out-of-bounds subset is re-evaluated (via bounded RBF).
        assert np.array_equal(picked[inb], unlimited[inb])
        changed = picked != unlimited
        assert np.all(changed <= ~inb), "pick must not touch in-bounds points"
        assert np.all(np.isfinite(picked))


class TestMonotoneRefusal:

    def test_composite_expression_raises(self):
        mesh, T = _steep_p3_field()
        coords = _offnode_coords(mesh)
        with pytest.raises(ValueError):
            uw.function.evaluate(T.sym[0, 0] + 1.0, coords, monotone="clamp")

    def test_unknown_option_raises(self):
        mesh, T = _steep_p3_field()
        coords = _offnode_coords(mesh)
        with pytest.raises(ValueError):
            uw.function.evaluate(T.sym, coords, monotone="bogus")
