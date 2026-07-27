"""Local RBF-FD stencil weights that reproduce linear fields exactly.

Inverse-distance (Shepard) weights are positive and sum to one, so they
reproduce a *constant* exactly but not a linear field: any field with a
gradient is smeared, and the error does not vanish as the stencil tightens.

The weights built here do reproduce linears. For a target point
:math:`x^*` and its :math:`m` nearest source points :math:`x_j`, solve the
small saddle-point system

.. math::

    \\begin{bmatrix} A & P \\\\ P^{T} & 0 \\end{bmatrix}
    \\begin{bmatrix} w \\\\ \\lambda \\end{bmatrix}
    = \\begin{bmatrix} \\varphi(|x^* - x_j|) \\\\ 1,\\; x^* \\end{bmatrix}

with :math:`A_{ij} = \\varphi(|x_i - x_j|)`, the polyharmonic (thin-plate)
kernel :math:`\\varphi(r) = r^2 \\log r`, and the affine tail
:math:`P = [1,\\; x_j]`. The lower block *is* the reproduction constraint
:math:`P^{T} w = [1,\\; x^*]`, so

.. math::

    \\sum_j w_j = 1, \\qquad \\sum_j w_j x_j = x^*

hold to round-off by construction — constants and linears are exact. The
weights depend only on geometry, so one solve serves every data component.

Cost is one dense :math:`(m + d + 1)^3` solve per target point, and the
result is sparse: :math:`m` non-zeros per row.

This module is deliberately pure NumPy with no PETSc dependency; it is the
numeric core behind ``KDTree.rbf_interpolator_local(..., order=1)``.
"""

import numpy as np

# The kernel is not scale-invariant: under r -> s.r it gains an r^2 log(s)
# term, which is quadratic and therefore NOT absorbed by the affine tail. So
# every stencil is solved in a local frame (target at the origin, farthest
# neighbour at unit radius); conditioning is then independent of the absolute
# stencil size, and both tolerances below are dimensionless.
_RANK_TOL = 1.0e-8       # smallest/largest singular value of the affine block
_REPRODUCTION_TOL = 1.0e-6  # residual of the two reproduction identities

# Peak working set per chunk of the stacked saddle matrices. Sized for memory,
# not speed: the full (N, m+d+1, m+d+1) stack for a million targets would be
# gigabytes.
_CHUNK_BYTES = 64 * 1024 * 1024


def _polyharmonic(r):
    """:math:`\\varphi(r) = r^2 \\log r`, with :math:`\\varphi(0) = 0`."""
    # The clip only keeps log() finite at coincident points; r**2 drives the
    # product to zero there regardless of the value substituted.
    safe = np.where(r > 0.0, r, 1.0)
    return np.where(r > 0.0, r ** 2 * np.log(safe), 0.0)


def _solve_stencils(matrices, rhs):
    """Batched saddle-point solve, tolerant of an exactly singular member.

    ``np.linalg.solve`` raises for the whole batch if any one matrix is
    singular, and gives no way to tell which. The pre-screen in
    :func:`linear_exact_weights` removes the common case; this catches the
    residue by falling back to the pseudo-inverse for that chunk, which never
    raises. The post-solve reproduction check then rejects anything the
    pseudo-inverse merely papered over.
    """
    try:
        return np.linalg.solve(matrices, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return (np.linalg.pinv(matrices) @ rhs[..., None])[..., 0]


def linear_exact_weights(target_coords, neighbour_coords):
    """Stencil weights that reproduce constant and linear fields exactly.

    Parameters
    ----------
    target_coords : numpy.ndarray
        Points to interpolate to, shape ``(n_targets, dim)``.
    neighbour_coords : numpy.ndarray
        Source points of each target's stencil, shape
        ``(n_targets, nnn, dim)``. Normally the ``nnn`` nearest neighbours
        from a kd-tree query.

    Returns
    -------
    weights : numpy.ndarray
        Shape ``(n_targets, nnn)``. Interpolate with
        ``(weights[:, :, None] * data[stencil]).sum(axis=1)``.
    degenerate : numpy.ndarray
        Boolean, shape ``(n_targets,)``. True where the stencil could not
        support an affine fit — collinear neighbours in 2D, coplanar in 3D,
        or a stencil that collapsed onto its target. **Rows flagged here are
        returned as zeros**; the caller must substitute its own fallback
        (Shepard weights over the same neighbours are the natural choice).

    Notes
    -----
    ``nnn`` must be at least ``dim + 2``. At exactly ``dim + 1`` the affine
    block is square, the constraint alone determines the weights, and the
    scheme degenerates to bare barycentric interpolation on the neighbour
    simplex — which is singular whenever those points are collinear or
    coplanar, a common situation near boundaries and in graded meshes.
    """
    target_coords = np.ascontiguousarray(target_coords, dtype=np.float64)
    neighbour_coords = np.ascontiguousarray(neighbour_coords, dtype=np.float64)

    n_targets, nnn, dim = neighbour_coords.shape
    if target_coords.shape != (n_targets, dim):
        raise ValueError(
            f"target_coords has shape {target_coords.shape}, expected "
            f"({n_targets}, {dim}) to match neighbour_coords "
            f"{neighbour_coords.shape}."
        )
    if nnn < dim + 2:
        raise ValueError(
            f"A linear-exact stencil needs at least dim + 2 = {dim + 2} "
            f"neighbours, got nnn = {nnn}. At dim + 1 the affine tail is "
            "exactly determined and the scheme reduces to barycentric "
            "interpolation on the neighbour simplex, which is singular for "
            "collinear (2D) or coplanar (3D) neighbours."
        )

    weights = np.zeros((n_targets, nnn), dtype=np.float64)
    degenerate = np.zeros(n_targets, dtype=bool)

    size = nnn + dim + 1
    rows_per_chunk = max(1, _CHUNK_BYTES // (size * size * 8))

    for start in range(0, n_targets, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_targets)
        _weights_for_chunk(
            target_coords[start:stop],
            neighbour_coords[start:stop],
            weights[start:stop],
            degenerate[start:stop],
        )

    return weights, degenerate


def _weights_for_chunk(targets, neighbours, weights_out, degenerate_out):
    """Fill one chunk of ``weights``/``degenerate`` in place."""
    n_chunk, nnn, dim = neighbours.shape

    # Local frame: target at the origin, farthest neighbour at unit radius.
    offsets = neighbours - targets[:, None, :]
    radius = np.linalg.norm(offsets, axis=2).max(axis=1)
    collapsed = radius <= 0.0            # every neighbour sits on the target
    scale = np.where(collapsed, 1.0, radius)
    y = offsets / scale[:, None, None]

    # Affine block, and its rank as the primary degeneracy test.
    P = np.concatenate([np.ones((n_chunk, nnn, 1)), y], axis=2)
    singular_values = np.linalg.svd(P, compute_uv=False)
    rank_deficient = singular_values[:, -1] <= _RANK_TOL * singular_values[:, 0]

    healthy = ~(collapsed | rank_deficient)
    degenerate_out[...] = ~healthy
    if not healthy.any():
        return

    y_h = y[healthy]
    P_h = P[healthy]
    n_h = y_h.shape[0]
    size = nnn + dim + 1

    pair_distance = np.linalg.norm(y_h[:, :, None, :] - y_h[:, None, :, :], axis=3)

    matrices = np.zeros((n_h, size, size), dtype=np.float64)
    matrices[:, :nnn, :nnn] = _polyharmonic(pair_distance)
    matrices[:, :nnn, nnn:] = P_h
    matrices[:, nnn:, :nnn] = np.transpose(P_h, (0, 2, 1))

    # RHS: kernel from the target (the origin) to each neighbour, then the
    # polynomial basis evaluated at the origin, [1, 0, ..., 0].
    rhs = np.zeros((n_h, size), dtype=np.float64)
    rhs[:, :nnn] = _polyharmonic(np.linalg.norm(y_h, axis=2))
    rhs[:, nnn] = 1.0

    w = _solve_stencils(matrices, rhs)[:, :nnn]

    # Validate the answer rather than guessing a condition number: these two
    # identities are what the whole construction is for.
    partition_error = np.abs(w.sum(axis=1) - 1.0)
    linear_error = np.linalg.norm(
        np.einsum("nm,nmd->nd", w, y_h), axis=1
    )
    reproduced = (
        np.isfinite(w).all(axis=1)
        & (partition_error <= _REPRODUCTION_TOL)
        & (linear_error <= _REPRODUCTION_TOL)
    )

    accepted = np.zeros(n_chunk, dtype=bool)
    accepted[healthy] = reproduced
    weights_out[accepted] = w[reproduced]
    degenerate_out[...] = ~accepted
