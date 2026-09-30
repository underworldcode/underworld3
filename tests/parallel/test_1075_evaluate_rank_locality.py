#!/usr/bin/env python3
"""`evaluate` is rank-local, and must say so when that costs the caller (#606).

`uw.function.evaluate` answers from the calling rank's portion of the mesh. In
serial that is the whole mesh. In parallel, a query point owned by another rank
cannot be located here, so the locator extrapolates and returns a plausible
number that is simply wrong — and two ranks asked the identical question return
different answers.

Measured on a P2 field holding `x^2 + 2 y^2`, which P2 represents exactly, so any
discrepancy is location rather than approximation. Querying every rank's own DOF
coordinates allgathered — every point a mesh node — gave a maximum error of 1.48
at np=2 and 2.59 at np=4 on a field whose range is [0, 3], while the same query
restricted to each rank's own coordinates was exact to 1e-16.

The extrapolation mask is an exact detector of those points (69 flagged / 69
wrong at np=2, 120 / 120 at np=4, with no wrong-but-unflagged and no
flagged-but-fine), which is what makes warning on it sound.
"""
import warnings

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _field():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=3)
    field = uw.discretisation.MeshVariable("f1075", mesh, 1, degree=2)
    coords = np.asarray(field.coords, dtype=float)
    field.data[:, 0] = coords[:, 0] ** 2 + 2.0 * coords[:, 1] ** 2
    return field, coords


def _evaluate(field, points):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = np.asarray(
            uw.function.evaluate(field.sym[0], points), dtype=float).ravel()
    warned = [w for w in caught if "rank-local" in str(w.message)]
    truth = points[:, 0] ** 2 + 2.0 * points[:, 1] ** 2
    return float(np.abs(values - truth).max()), bool(warned)


def test_own_coordinates_are_exact_and_silent():
    """The rank-local contract honoured: no warning, and no error to warn about.

    This is the negative control. A warning that fired here would be noise on
    every correct parallel use of `evaluate`.
    """
    field, coords = _field()
    error, warned = _evaluate(field, coords)
    assert error < 1.0e-12, f"own coordinates should be exact, got {error:.2e}"
    assert not warned, "warned about a query that was entirely rank-local"


def test_unowned_coordinates_warn():
    """A query spanning ranks must not answer silently.

    In serial there is nothing to warn about — every point is owned — so the
    assertion is conditioned on rank count rather than skipped, which keeps the
    exactness check alive at np=1.
    """
    field, coords = _field()
    every = np.vstack([g for g in uw.mpi.comm.allgather(coords) if len(g)])
    error, warned = _evaluate(field, every)

    if uw.mpi.size == 1:
        assert error < 1.0e-12, f"serial must be exact, got {error:.2e}"
        assert not warned, "warned in serial, where every point is owned"
        return

    assert warned, (
        f"evaluate returned a max error of {error:.2e} for points this rank does "
        "not own, and said nothing")
