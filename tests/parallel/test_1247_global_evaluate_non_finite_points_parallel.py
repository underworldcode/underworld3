"""global_evaluate refuses non-finite query points on every rank.

A NaN coordinate passes the far-outside refusal (#786), since a comparison with
NaN is false, and reaches the swarm migration inside ``global_evaluate``. There
the ranks part. At np 2, every rank passing the same points gave a different
error on each rank, and with the NaN on the last rank and no points on the
others (a plotting helper with ``viz_rank`` last) the first rank raised an
``IndexError`` while the last waited in the migration's barrier. Points that
are not finite are now refused before any collective, with the same message on
every rank, counted over all ranks: a rank whose own points are finite raises
too. An infinite coordinate is refused the same way, also when
``check_extrapolated=True`` turns the far-outside refusal off.

    python scripts/mpi_supervisor.py --silence 120 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1247_global_evaluate_non_finite_points_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(120, method="thread")]


def _field():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.2)
    T = uw.discretisation.MeshVariable("T1247", mesh, 1, degree=2)
    T.array[:, 0, 0] = np.asarray(T.coords)[:, 0]
    return T


# The sample line of the parallel plotting helpers' check (H1's probe on #1246).
POINTS = np.column_stack((np.linspace(0.1, 0.9, 41), np.linspace(0.2, 0.3, 41)))


def _with_bad_row(value, row=5):
    points = POINTS.copy()
    points[row, 1] = value
    return points


def _messages(call):
    """Run ``call`` on every rank and gather what each raised."""
    message = ""
    try:
        call()
    except Exception as error:  # recorded: a different error on each rank is the defect
        message = f"{type(error).__name__}: {error}"
    return uw.mpi.comm.allgather(message)


def _refused_everywhere(messages, count, total):
    assert all(m == messages[0] for m in messages), messages
    assert messages[0].startswith("ValueError: "), messages
    assert f"{count} of {total} query points have non-finite coordinates" in messages[0], \
        messages


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_the_same_non_finite_points_on_every_rank_are_refused_on_every_rank(value):
    T = _field()
    messages = _messages(lambda: uw.function.global_evaluate(T.sym, _with_bad_row(value)))
    _refused_everywhere(messages, uw.mpi.size, uw.mpi.size * len(POINTS))


@pytest.mark.parametrize("holder", ["first", "last"])
@pytest.mark.parametrize("others", ["empty", "finite"])
def test_a_nan_point_on_one_rank_is_refused_on_every_rank(holder, others):
    """One rank holds one NaN row; the others hold no points, or finite points
    only. With the NaN on the last rank and the others empty, as a plotting
    helper with ``viz_rank`` last arranges, development hung."""
    T = _field()
    nan_rank = 0 if holder == "first" else uw.mpi.size - 1
    if uw.mpi.rank == nan_rank:
        points = _with_bad_row(np.nan)
    elif others == "empty":
        points = np.zeros((0, 2))
    else:
        points = POINTS
    total = uw.mpi.comm.allreduce(len(points))
    messages = _messages(lambda: uw.function.global_evaluate(T.sym, points))
    _refused_everywhere(messages, 1, total)


def test_an_infinite_point_is_refused_also_with_check_extrapolated():
    T = _field()
    messages = _messages(lambda: uw.function.global_evaluate(
        T.sym, _with_bad_row(np.inf), check_extrapolated=True))
    _refused_everywhere(messages, uw.mpi.size, uw.mpi.size * len(POINTS))


def test_finite_points_are_evaluated_as_before():
    """The exact field x at finite points, unchanged by the refusal."""
    T = _field()
    values = uw.function.global_evaluate(T.sym, POINTS)
    np.testing.assert_allclose(np.asarray(values).ravel(), POINTS[:, 0], rtol=1e-12,
                               atol=1e-12)
