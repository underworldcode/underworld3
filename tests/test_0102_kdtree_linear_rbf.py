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
