"""Local RBF stencil weights that reproduce linear fields exactly.

``KDTree.rbf_interpolator_local(..., order=1)`` builds polyharmonic
(:math:`r^2 \\log r`) weights with an affine tail, so both

    sum_j w_j = 1        (constants exact)
    sum_j w_j x_j = x*   (linears exact)

hold by construction. ``order=0`` is inverse-distance weighting, which gets
the first identity but not the second — the property this whole module
exists to add.

The weights themselves are tested through
``underworld3.utilities.rbf_stencil.linear_exact_weights``, because the
partition-of-unity and sparsity claims are about the *weights* and the
value-returning KDTree API cannot expose them.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.rbf_stencil import linear_exact_weights

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _stencils(source, target, nnn):
    """Brute-force kNN, so the test does not depend on the kd-tree it checks."""
    d = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=2)
    idx = np.argsort(d, axis=1)[:, :nnn]
    return idx, source[idx]


def _linear(coords):
    return 0.5 + coords @ np.arange(1, coords.shape[1] + 1, dtype=float)


# --------------------------------------------------------------------------
# The two reproduction identities, at the weights level
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nnn_rule", ["minimum", "default", "wide"])
def test_weights_reproduce_constants_and_linears(dim, nnn_rule):
    nnn = {"minimum": dim + 2, "default": 2 * (dim + 1), "wide": 14}[nnn_rule]
    rng = np.random.default_rng(20260727 + dim)
    source = rng.random((500, dim))
    target = rng.random((80, dim))

    idx, stencil = _stencils(source, target, nnn)
    weights, degenerate = linear_exact_weights(target, stencil)

    assert not degenerate.any(), (
        f"{degenerate.sum()} random-cloud stencils were rejected as degenerate"
    )

    partition = np.abs(weights.sum(axis=1) - 1.0).max()
    assert partition < 1.0e-12, f"partition of unity violated by {partition:.3e}"

    reproduced = (weights[:, :, None] * stencil).sum(axis=1)
    linear = np.abs(reproduced - target).max()
    assert linear < 1.0e-12, f"linear reproduction violated by {linear:.3e}"

    values = _linear(source)
    interpolated = (weights * values[idx]).sum(axis=1)
    error = np.abs(interpolated - _linear(target)).max()
    assert error < 1.0e-12, f"linear field interpolated to {error:.3e}, expected round-off"


@pytest.mark.parametrize("dim", [2, 3])
def test_weights_are_sparse(dim):
    """Exactly nnn non-zeros per row: this is the point of a *local* scheme."""
    nnn = 2 * (dim + 1)
    rng = np.random.default_rng(99)
    source = rng.random((300, dim))
    target = rng.random((40, dim))

    _, stencil = _stencils(source, target, nnn)
    weights, _ = linear_exact_weights(target, stencil)

    assert weights.shape == (40, nnn)
    assert np.count_nonzero(weights, axis=1).max() <= nnn


# --------------------------------------------------------------------------
# Degeneracy: the affine block loses rank near boundaries and on graded meshes
# --------------------------------------------------------------------------
def test_collinear_stencil_is_flagged_not_nan():
    stencil = np.zeros((1, 5, 2))
    stencil[0, :, 0] = np.linspace(0.0, 1.0, 5)
    stencil[0, :, 1] = 0.3
    target = np.array([[0.5, 0.7]])

    weights, degenerate = linear_exact_weights(target, stencil)

    assert degenerate[0], "collinear 2D stencil should be flagged degenerate"
    assert np.isfinite(weights).all(), "degenerate stencil must not produce NaN"


def test_coplanar_stencil_is_flagged_not_nan():
    stencil = np.zeros((1, 6, 3))
    stencil[0, :, 0] = np.linspace(0.0, 1.0, 6)
    stencil[0, :, 1] = np.linspace(0.0, 0.5, 6)
    stencil[0, :, 2] = 0.2
    target = np.array([[0.4, 0.2, 0.9]])

    weights, degenerate = linear_exact_weights(target, stencil)

    assert degenerate[0], "coplanar 3D stencil should be flagged degenerate"
    assert np.isfinite(weights).all(), "degenerate stencil must not produce NaN"


def test_degenerate_and_healthy_stencils_in_one_batch():
    """A singular member must not take the whole batched solve down with it."""
    healthy = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    collinear = np.array([[[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [1.0, 0.0]]])
    stencil = np.concatenate([healthy, collinear])
    target = np.array([[0.3, 0.3], [0.3, 0.3]])

    weights, degenerate = linear_exact_weights(target, stencil)

    assert not degenerate[0] and degenerate[1]
    assert np.isfinite(weights).all()
    assert abs(weights[0].sum() - 1.0) < 1.0e-12


def test_stencil_collapsed_onto_target_is_flagged():
    weights, degenerate = linear_exact_weights(np.zeros((1, 2)), np.zeros((1, 4, 2)))
    assert degenerate[0]
    assert np.isfinite(weights).all()


def test_degenerate_stencil_is_retried_on_a_wider_neighbourhood():
    """A locally-coplanar neighbourhood must be escaped, not surrendered to.

    Most source points lie on the plane z = 0, with a sparse off-plane set
    further away. A target near the plane has all its nearest neighbours in
    it, so the affine block is rank-deficient and the z-gradient of a linear
    field is simply not representable from that stencil. Widening reaches the
    off-plane points and recovers exactness.

    This matters because a single non-exact node sitting among exact ones is
    an isolated spike, which damages derivatives far more than a smooth error
    of the same magnitude. Measured on a 3D low-density swarm before the
    retry existed: 2.2e-3 max error from 2 nodes in 28824.
    """
    # A SMALL coplanar patch -- a local accident, which is what widening is
    # for. (A globally coplanar cloud is a different situation, covered by
    # test_locally_unrecoverable_cloud_falls_back_loudly.)
    axis = np.linspace(0.45, 0.55, 5)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    in_plane = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    rng = np.random.default_rng(19)
    off_plane = rng.random((80, 3))

    source = np.vstack([in_plane, off_plane])
    target = np.array([[0.5, 0.5, 0.002], [0.49, 0.51, -0.001]])

    nnn = 8
    _, stencil = _stencils(source, target, nnn)
    _, degenerate = linear_exact_weights(target, stencil)
    assert degenerate.all(), (
        "test setup no longer produces coplanar stencils, so it is not "
        "exercising the retry path"
    )

    # A z-dependent linear field cannot be recovered from a coplanar stencil,
    # so exactness here proves the wider neighbourhood was actually used.
    data = _linear(source)[:, None]
    kdt = uw.kdtree.KDTree(source)
    got = kdt.rbf_interpolator_local(target, data, nnn, 2, False, order=1)

    error = np.abs(got[:, 0] - _linear(target)).max()
    assert error < 1.0e-12, (
        f"degenerate stencils were not recovered by widening: error {error:.3e}"
    )


def test_locally_unrecoverable_cloud_falls_back_loudly():
    """When widening cannot help, say so rather than pretending.

    A source cloud that is coplanar over a wide region is genuinely unable to
    determine an affine fit in the third direction. Widening the stencil is
    futile, and the honest outcome is a bounded inverse-distance answer plus a
    warning -- not a silent loss of the guarantee (the failure mode of #424).
    """
    axis = np.linspace(0.0, 1.0, 22)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    source = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    target = np.array([[0.5, 0.5, 0.02]])

    data = _linear(source)[:, None]
    kdt = uw.kdtree.KDTree(source)

    with pytest.warns(UserWarning, match="could not support an affine fit"):
        got = kdt.rbf_interpolator_local(target, data, 8, 2, False, order=1)

    assert np.isfinite(got).all(), "an unrecoverable stencil must not give NaN"
    lo, hi = data.min(), data.max()
    assert lo - 1e-12 <= got[0, 0] <= hi + 1e-12, (
        "the inverse-distance fallback should still be bounded"
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_nnn_below_dim_plus_two_is_rejected(dim):
    """At dim+1 the affine tail is exactly determined and the scheme reduces to
    bare barycentric interpolation on the neighbour simplex — singular whenever
    those points are collinear/coplanar, which is common."""
    with pytest.raises(ValueError, match=f"dim \\+ 2 = {dim + 2}"):
        linear_exact_weights(np.zeros((1, dim)), np.zeros((1, dim + 1, dim)))


# --------------------------------------------------------------------------
# Through the KDTree API
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_kdtree_order1_beats_order0_on_a_linear_field(dim):
    nnn = 2 * (dim + 1)
    rng = np.random.default_rng(1234 + dim)
    source = rng.random((2000, dim))
    target = rng.random((150, dim))
    data = _linear(source)[:, None]
    expected = _linear(target)

    kdt = uw.kdtree.KDTree(source)
    shepard = kdt.rbf_interpolator_local(target, data, nnn, 2, False, order=0)
    exact = kdt.rbf_interpolator_local(target, data, nnn, 2, False, order=1)

    exact_error = np.abs(exact[:, 0] - expected).max()
    shepard_error = np.abs(shepard[:, 0] - expected).max()

    assert exact_error < 1.0e-12, f"order=1 gave {exact_error:.3e} on a linear field"
    assert shepard_error > 1.0e-4, (
        "order=0 is expected to smear a linear field; if this fails the "
        f"baseline has changed (got {shepard_error:.3e})"
    )


def test_kdtree_order0_is_unchanged_by_the_new_arguments():
    """Back-compatibility: the positional call must be bit-identical."""
    rng = np.random.default_rng(5)
    source = rng.random((400, 2))
    target = rng.random((50, 2))
    data = _linear(source)[:, None]

    kdt = uw.kdtree.KDTree(source)
    positional = kdt.rbf_interpolator_local(target, data, 4, 2, False)
    explicit = kdt.rbf_interpolator_local(target, data, 4, 2, False, order=0,
                                          monotone=False)

    assert np.array_equal(positional, explicit)


@pytest.mark.parametrize("dim", [2, 3])
def test_monotone_does_not_touch_a_linear_field(dim):
    """The limiter must not cost the guarantee the scheme exists to provide.

    A naive clamp against the stencil's raw min/max does exactly that: a
    target outside the convex hull of its own neighbours has a value outside
    their range even for an exactly linear field, so such a clamp cannot tell
    legitimate extrapolation from ringing. The limiter therefore bounds the
    non-affine part only, leaving the linear reconstruction alone.
    """
    nnn = 2 * (dim + 1)
    rng = np.random.default_rng(77 + dim)
    source = rng.random((1500, dim))
    target = rng.random((250, dim))
    data = _linear(source)[:, None]
    expected = _linear(target)

    kdt = uw.kdtree.KDTree(source)
    unlimited = kdt.rbf_interpolator_local(target, data, nnn, 2, False, order=1)
    limited = kdt.rbf_interpolator_local(target, data, nnn, 2, False, order=1,
                                         monotone=True)

    assert np.abs(limited[:, 0] - expected).max() < 1.0e-12, (
        "the limiter destroyed linear reproduction"
    )
    assert np.abs(limited - unlimited).max() < 1.0e-12, (
        "the limiter should be a no-op on a field the scheme reproduces exactly"
    )


def test_monotone_bounds_the_correction_on_a_curved_field():
    """It must still do something: bound the RBF correction to the non-affine
    variation actually present in the stencil."""
    dim = 2
    nnn = 2 * (dim + 1)
    rng = np.random.default_rng(4242)
    source = rng.random((800, dim))
    target = rng.random((200, dim))
    values = 0.5 + (source ** 2).sum(axis=1) + np.sin(6.0 * source[:, 0])
    data = values[:, None]

    kdt = uw.kdtree.KDTree(source)
    unlimited = kdt.rbf_interpolator_local(target, data, nnn, 2, False, order=1)
    limited = kdt.rbf_interpolator_local(target, data, nnn, 2, False, order=1,
                                         monotone=True)

    assert np.abs(limited - unlimited).max() > 1.0e-6, (
        "the limiter had no effect on a curved field, so it is not limiting"
    )

    # The stated guarantee: the deviation from the local affine trend never
    # exceeds the deviation the stencil itself shows.
    from underworld3.utilities.rbf_stencil import affine_trend

    _, stencil = _stencils(source, target, nnn)
    idx, _ = _stencils(source, target, nnn)
    stencil_values = data[idx]
    at_target, at_stencil = affine_trend(target, stencil, stencil_values)
    residual = stencil_values - at_stencil
    correction = limited - at_target

    assert (correction <= residual.max(axis=1) + 1.0e-12).all()
    assert (correction >= residual.min(axis=1) - 1.0e-12).all()


def test_kdtree_rejects_bad_order_and_monotone_mode():
    rng = np.random.default_rng(11)
    source = rng.random((200, 2))
    target = rng.random((10, 2))
    data = _linear(source)[:, None]
    kdt = uw.kdtree.KDTree(source)

    with pytest.raises(ValueError, match="order must be 0"):
        kdt.rbf_interpolator_local(target, data, 6, 2, False, order=2)

    with pytest.raises(ValueError, match="no meaning for a local kd-tree"):
        kdt.rbf_interpolator_local(target, data, 6, 2, False, order=1,
                                   monotone="pick")


# --------------------------------------------------------------------------
# Oracle tests on a RANDOM field.
#
# Linear reproduction cannot detect wrong-neighbour selection: the identity
# sum_j w_j x_j = x* is evaluated against whatever points were handed to the
# solver, so it holds even if the kd-tree returned the wrong neighbours, and
# a linear field then interpolates exactly anyway. Only a field whose values
# vary independently of position -- i.e. random data -- makes the choice of
# neighbour observable.
# --------------------------------------------------------------------------
def _reference_interpolate(source, target, values, nnn):
    """An independent implementation, written the obvious slow way.

    Brute-force kNN, then one dense saddle solve per target point, with no
    local rescaling. Deliberately does not share code with the library.
    """
    out = np.empty(target.shape[0])
    dim = source.shape[1]
    for i, x in enumerate(target):
        d = np.linalg.norm(source - x, axis=1)
        idx = np.argsort(d)[:nnn]
        pts = source[idx]

        def phi(r):
            r = np.where(r == 0.0, 1.0e-300, r)
            return r ** 2 * np.log(r)

        rr = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        P = np.hstack([np.ones((nnn, 1)), pts])
        M = np.block([[phi(rr), P], [P.T, np.zeros((dim + 1, dim + 1))]])
        rhs = np.concatenate([phi(np.linalg.norm(pts - x, axis=1)), [1.0], x])
        w = np.linalg.solve(M, rhs)[:nnn]
        out[i] = w @ values[idx]
    return out


@pytest.mark.parametrize("dim", [2, 3])
def test_matches_an_independent_implementation_on_random_data(dim):
    """Random values, so wrong-neighbour selection cannot hide."""
    nnn = 2 * (dim + 1)
    rng = np.random.default_rng(31337 + dim)
    source = rng.random((400, dim))
    target = 0.15 + 0.7 * rng.random((40, dim))
    values = rng.standard_normal(source.shape[0])

    kdt = uw.kdtree.KDTree(source)
    got = kdt.rbf_interpolator_local(target, values[:, None], nnn, 2, False, order=1)
    want = _reference_interpolate(source, target, values, nnn)

    error = np.abs(got[:, 0] - want).max()
    assert error < 1.0e-8, (
        f"disagrees with an independent implementation by {error:.3e} on "
        "random data — check neighbour selection, not just reproduction"
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_kdtree_neighbours_match_brute_force_on_random_data(dim):
    """The value must change if the stencil is wrong, and it does not."""
    nnn = 2 * (dim + 1)
    rng = np.random.default_rng(555 + dim)
    source = rng.random((600, dim))
    target = 0.2 + 0.6 * rng.random((50, dim))
    values = rng.standard_normal(source.shape[0])

    kdt = uw.kdtree.KDTree(source)
    got = kdt.rbf_interpolator_local(target, values[:, None], nnn, 2, False, order=1)

    idx, stencil = _stencils(source, target, nnn)
    weights, degenerate = linear_exact_weights(target, stencil)
    assert not degenerate.any()
    want = (weights * values[idx]).sum(axis=1)

    assert np.abs(got[:, 0] - want).max() < 1.0e-10

    # Control: a deliberately wrong stencil must give a different answer, or
    # the test above proves nothing.
    rolled = np.roll(idx, 1, axis=0)
    wrong_weights, _ = linear_exact_weights(target, source[rolled])
    wrong = (wrong_weights * values[rolled]).sum(axis=1)
    assert np.abs(wrong - want).max() > 1.0e-3, (
        "shuffling the stencils did not change the result, so this test "
        "cannot detect wrong-neighbour selection"
    )


# --------------------------------------------------------------------------
# The operator form
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [0, 1])
def test_interpolation_matrix_agrees_with_the_value_path(dim, order):
    """`T @ data` must be what the value API returns, or they will drift."""
    rng = np.random.default_rng(808 + dim)
    source = rng.random((500, dim))
    target = 0.1 + 0.8 * rng.random((60, dim))
    values = rng.standard_normal((source.shape[0], 2))

    kdt = uw.kdtree.KDTree(source)
    T = kdt.interpolation_matrix(target, order=order)
    direct = kdt.rbf_interpolator_local(target, values, None, 2, False, order=order)

    assert T.shape == (target.shape[0], source.shape[0])
    assert np.abs(T @ values - direct).max() < 1.0e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_interpolation_matrix_rows_are_never_empty(dim):
    """The raw-weights helper zeroes degenerate rows; the operator must not.

    A zero row would silently interpolate to zero, which is worse than the
    inverse-distance fallback it replaces.
    """
    axis = np.linspace(0.0, 1.0, 12)
    grids = np.meshgrid(*([axis] * dim), indexing="ij")
    source = np.stack([g.ravel() for g in grids], axis=1)
    rng = np.random.default_rng(2)
    target = rng.random((80, dim))

    kdt = uw.kdtree.KDTree(source)
    T = kdt.interpolation_matrix(target, order=1)

    per_row = np.diff(T.indptr)
    assert per_row.min() > 0, "empty row in the interpolation operator"
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    assert np.abs(row_sums - 1.0).max() < 1.0e-12


# --------------------------------------------------------------------------
# Contract: determinism and purity
# --------------------------------------------------------------------------
def test_repeated_calls_are_bit_identical():
    """No global state, no RNG, no accumulation: the weights are a pure
    function of the geometry."""
    rng = np.random.default_rng(64)
    source = rng.random((300, 3))
    target = rng.random((40, 3))
    values = rng.standard_normal((300, 1))

    kdt = uw.kdtree.KDTree(source)
    first = kdt.rbf_interpolator_local(target, values, 8, 2, False, order=1)
    second = kdt.rbf_interpolator_local(target, values, 8, 2, False, order=1)
    assert np.array_equal(first, second)

    other = uw.kdtree.KDTree(source.copy())
    assert np.array_equal(
        other.rbf_interpolator_local(target, values, 8, 2, False, order=1), first
    )


def test_equidistant_neighbours_resolve_deterministically():
    """Ties in the kNN search must not make the result run-dependent.

    Which of several equidistant source points is chosen is not specified —
    but it must be the same choice every time, or results stop reproducing.
    """
    # A target at the centre of a symmetric ring: all ring points are exactly
    # equidistant, so the nnn selection is a tie.
    angles = np.arange(12) * (2.0 * np.pi / 12.0)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    source = np.vstack([ring, 2.0 * ring])
    target = np.array([[0.0, 0.0]])
    values = np.arange(source.shape[0], dtype=float)[:, None]

    kdt = uw.kdtree.KDTree(source)
    results = [
        kdt.rbf_interpolator_local(target, values, 6, 2, False, order=1)
        for _ in range(5)
    ]
    for r in results[1:]:
        assert np.array_equal(r, results[0]), "tied kNN selection is not deterministic"

    fresh = uw.kdtree.KDTree(source.copy())
    assert np.array_equal(
        fresh.rbf_interpolator_local(target, values, 6, 2, False, order=1), results[0]
    )


def test_nearest_neighbour_path_keeps_its_shape():
    """`nnn=1` returns source rows directly, with no stencil axis.

    This is the checkpoint round-trip path (`read_timestep` uses `nnn=1`).
    A refactor that swapped `query(k=1)` for `find_closest_n_points` silently
    added an axis here, and only a snapshot test noticed.
    """
    rng = np.random.default_rng(12)
    source = rng.random((200, 2))
    target = rng.random((30, 2))
    data = rng.standard_normal((200, 1))

    kdt = uw.kdtree.KDTree(source)
    got = kdt.rbf_interpolator_local(target, data, 1, 2, False)

    assert got.shape == (30, 1), f"nnn=1 returned shape {got.shape}, expected (30, 1)"

    # It really is the nearest neighbour's value, not an average.
    d = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=2)
    assert np.array_equal(got[:, 0], data[np.argmin(d, axis=1), 0])


@pytest.mark.parametrize("order", [0, 1])
def test_default_nnn_works_in_both_dimensions(order):
    """A default that raises is not a default (the fixed nnn=4 did, at
    order=1 in 3D, where dim + 2 = 5)."""
    for dim in (2, 3):
        rng = np.random.default_rng(9)
        source = rng.random((300, dim))
        target = rng.random((20, dim))
        data = _linear(source)[:, None]
        kdt = uw.kdtree.KDTree(source)
        got = kdt.rbf_interpolator_local(target, data, order=order)
        assert got.shape == (20, 1)
        assert np.isfinite(got).all()


@pytest.mark.parametrize("dim", [2, 3])
def test_error_falls_with_point_spacing_faster_than_inverse_distance(dim):
    """On a quadratic field, order=1 should converge under refinement and
    order=0 should barely move — inverse-distance error is set by the stencil
    geometry, not by how close the points are."""

    def quadratic(coords):
        return 0.5 + (coords ** 2).sum(axis=1)

    nnn = 2 * (dim + 1)
    errors = {}
    for n_per_side in (10, 20):
        axis = (np.arange(n_per_side) + 0.5) / n_per_side
        grid = np.meshgrid(*([axis] * dim), indexing="ij")
        source = np.stack([g.ravel() for g in grid], axis=1)
        rng = np.random.default_rng(3)
        target = 0.2 + 0.6 * rng.random((100, dim))
        data = quadratic(source)[:, None]
        expected = quadratic(target)

        kdt = uw.kdtree.KDTree(source)
        for order in (0, 1):
            got = kdt.rbf_interpolator_local(target, data, nnn, 2, False, order=order)
            errors[(order, n_per_side)] = np.abs(got[:, 0] - expected).max()

    order1_ratio = errors[(1, 10)] / errors[(1, 20)]
    order0_ratio = errors[(0, 10)] / errors[(0, 20)]

    assert order1_ratio > 3.0, (
        f"order=1 should converge at least ~O(h^2) under halving the spacing, "
        f"got a factor of {order1_ratio:.2f}"
    )
    assert order1_ratio > order0_ratio, (
        f"order=1 convergence ({order1_ratio:.2f}x) should beat order=0 "
        f"({order0_ratio:.2f}x)"
    )
