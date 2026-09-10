r"""The cell-restricted Voronoi share: which particles an integration point speaks for.

A particle method has to answer one question before it can assemble anything:
what value does the quadrature rule read at each of its points? Taking the
nearest particle answers it sharply but throws most of the swarm away — at ten
particles per cell and six rule points, better than half of them never reach
the assembly at all. That is fine for a *label*, where sub-sampling costs
nothing but sharpness, and wrong for a *history*, where every particle carries
state that was earned by being advected.

The share is the middle course, and it is what the classic Voronoi
particle-in-cell integration was reaching for. Each particle is assigned to the
nearest integration point **of its own cell**, so the rule points partition the
cell between them and every particle lands in exactly one part. An integration
point then reads the mean over the sub-region it represents. No particle is
discarded, nothing is smeared across a cell boundary, and the whole thing is a
gather with no tree: the owning cell is already known, and the choice within a
cell is over :math:`N_q` candidates.

The restriction to the particle's own cell is not a detail. Without it a
particle just across a cell wall is often nearer to a rule point on the far
side, and a material interface leaks into the neighbouring cell — which
defeats the reason for sampling at the integration points at all.

See Also
--------
underworld3.utilities.cell_polynomial_projection : the fitted alternative,
    which gives a differentiable field per cell rather than a value per point.
"""

import numpy as np

__all__ = ["share_assignment", "share_average"]


def share_assignment(integration_points, coords, cells, chunk_bytes=64 << 20):
    """Assign each particle to the nearest integration point of its own cell.

    Parameters
    ----------
    integration_points : ndarray, shape (ncells, Nq, cdim)
        The physical rule points, in local cell order — the layout
        :attr:`IntegrationPointVariable.integration_points` returns.
    coords : ndarray, shape (Np, cdim)
        Particle coordinates, non-dimensional (the same frame as
        ``integration_points``).
    cells : ndarray, shape (Np,)
        Local owning cell per particle; a negative entry means "not on this
        rank's mesh" and is left unassigned.
    chunk_bytes : int
        Working-set bound for the distance evaluation. The gather is
        ``(chunk, Nq, cdim)``, so this caps memory rather than particle count.

    Returns
    -------
    flat : ndarray, shape (Np,), int64
        Index into the flattened ``(ncells * Nq,)`` point ordering, or ``-1``
        for a particle whose cell is negative.
    """
    integration_points = np.asarray(integration_points, dtype=float)
    coords = np.asarray(coords, dtype=float)
    cells = np.asarray(cells).reshape(-1)

    ncells, Nq, cdim = integration_points.shape
    flat = np.full(coords.shape[0], -1, dtype=np.int64)
    live = np.flatnonzero((cells >= 0) & (cells < ncells))
    if live.size == 0 or ncells == 0:
        return flat

    # (chunk, Nq, cdim) doubles per pass
    step = max(1, int(chunk_bytes // max(Nq * cdim * 8, 1)))
    for start in range(0, live.size, step):
        sel = live[start : start + step]
        cc = cells[sel]
        d2 = ((integration_points[cc] - coords[sel][:, None, :]) ** 2).sum(axis=-1)
        flat[sel] = cc * Nq + d2.argmin(axis=1)

    return flat


def share_average(flat, values, npoints):
    """Mean of each integration point's share, and how many particles it holds.

    Parameters
    ----------
    flat : ndarray, shape (Np,)
        The assignment from :func:`share_assignment`; ``-1`` entries are
        ignored.
    values : ndarray, shape (Np, ncomp)
        Particle values.
    npoints : int
        Total number of integration points on this rank (``ncells * Nq``).

    Returns
    -------
    means : ndarray, shape (npoints, ncomp)
        The share mean; zero where a point holds no particles.
    counts : ndarray, shape (npoints,)
        Particles per integration point. A zero here is the caller's problem
        to fill — a rule point in an empty or badly-sampled cell.
    """
    flat = np.asarray(flat).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(flat.shape[0], -1)
    ncomp = values.shape[1]

    ok = flat >= 0
    idx = flat[ok]
    counts = np.bincount(idx, minlength=npoints)[:npoints]

    means = np.zeros((npoints, ncomp), dtype=float)
    if idx.size:
        vals = values[ok]
        for k in range(ncomp):
            means[:, k] = np.bincount(idx, weights=vals[:, k], minlength=npoints)[:npoints]
        nz = counts > 0
        means[nz] /= counts[nz, None]

    return means, counts
