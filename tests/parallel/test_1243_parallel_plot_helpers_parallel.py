"""The parallel plotting helpers work at more than one rank (visualisation/parallel.py).

Each helper evaluates on the plotting rank (``viz_rank``) and has every other
rank join the same ``global_evaluate`` with no points. Before this fix, the other
ranks built their empty point array from ``field.mesh.dim``: a sympy expression
has no ``mesh``, so they raised and never joined, and the plotting rank waited
inside ``global_evaluate`` (a hang at np 2 and 3). A MeshVariable has a mesh, but
``global_evaluate`` does not accept one, so every rank raised.

The baseline is exact: ``T`` is a degree-1 variable holding the linear field
``x + 2 y``, so its interpolant at any point inside the mesh is ``x + 2 y`` to
round-off. The helpers are checked with ``T`` itself and with sympy expressions
built from it, for ``viz_rank`` 0 and the last rank.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.visualisation import parallel as vp

pytestmark = [pytest.mark.level_1, pytest.mark.mpi(min_size=2),
              pytest.mark.timeout(120, method="thread")]


def _linear_field():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.2)
    T = uw.discretisation.MeshVariable("T_plot", mesh, 1, degree=1)
    T.array[:, 0, 0] = (np.asarray(T.coords)[:, 0] + 2.0 * np.asarray(T.coords)[:, 1])
    return mesh, T


def _exact(points, scale=1.0):
    return scale * (points[:, 0] + 2.0 * points[:, 1])


def _recording(monkeypatch):
    """Record what global_evaluate returns on this rank."""
    calls = []
    evaluate = uw.function.global_evaluate

    def record(expr, points, *args, **kwargs):
        values = evaluate(expr, points, *args, **kwargs)
        calls.append((np.asarray(points), np.asarray(values)))
        return values

    monkeypatch.setattr(uw.function, "global_evaluate", record)
    return calls


def _viz_ranks():
    return [0, uw.mpi.size - 1]


POINTS = vp.create_line_sample((0.1, 0.2), (0.9, 0.3), num_points=17)


def _check(calls, viz_rank, scales):
    """On the plotting rank, one evaluation per field at POINTS equal to the exact
    field; on the others, one empty evaluation per field."""
    assert len(calls) == len(scales)
    for (points, values), scale in zip(calls, scales):
        if uw.mpi.rank == viz_rank:
            np.testing.assert_array_equal(points, POINTS)
            np.testing.assert_allclose(values.ravel(), _exact(POINTS, scale),
                                       rtol=1e-12, atol=1e-12)
        else:
            assert points.shape == (0, 2)
            assert values.size == 0


@pytest.mark.tier_b
@pytest.mark.parametrize("which", ["variable", "sym", "component"])
@pytest.mark.parametrize("viz", [0, -1])
def test_line_plot(monkeypatch, which, viz):
    mesh, T = _linear_field()
    viz_rank = _viz_ranks()[viz]
    field = {"variable": T, "sym": T.sym, "component": T.sym[0]}[which]
    calls = _recording(monkeypatch)
    vp.parallel_line_plot(field, POINTS, viz_rank=viz_rank)
    plt.close("all")
    _check(calls, viz_rank, [1.0])


@pytest.mark.tier_b
@pytest.mark.parametrize("viz", [0, -1])
def test_scatter_plot(monkeypatch, viz):
    mesh, T = _linear_field()
    viz_rank = _viz_ranks()[viz]
    calls = _recording(monkeypatch)
    vp.parallel_scatter_plot(T, sympy.Integer(2) * T.sym[0], POINTS, viz_rank=viz_rank)
    plt.close("all")
    _check(calls, viz_rank, [1.0, 2.0])


@pytest.mark.tier_b
@pytest.mark.parametrize("viz", [0, -1])
def test_profile_comparison(monkeypatch, viz):
    mesh, T = _linear_field()
    viz_rank = _viz_ranks()[viz]
    calls = _recording(monkeypatch)
    vp.parallel_profile_comparison([T.sym[0], T, sympy.Integer(3) * T.sym[0]], POINTS,
                                   viz_rank=viz_rank)
    plt.close("all")
    _check(calls, viz_rank, [1.0, 1.0, 3.0])


@pytest.mark.tier_b
@pytest.mark.parametrize("viz", [0, -1])
def test_custom_plot(monkeypatch, viz):
    mesh, T = _linear_field()
    viz_rank = _viz_ranks()[viz]
    received = {}

    def plot_function(**data):
        received.update(data)

    calls = _recording(monkeypatch)
    vp.parallel_custom_plot(plot_function, [(T, POINTS, "t"),
                                            (sympy.Integer(2) * T.sym[0], POINTS, "twice")],
                            viz_rank=viz_rank)
    _check(calls, viz_rank, [1.0, 2.0])
    if uw.mpi.rank == viz_rank:
        np.testing.assert_allclose(received["t_data"].ravel(), _exact(POINTS), rtol=1e-12,
                                   atol=1e-12)
        np.testing.assert_allclose(received["twice_data"].ravel(), _exact(POINTS, 2.0),
                                   rtol=1e-12, atol=1e-12)
    else:
        assert received == {}


def _points_only_on(viz_rank):
    """POINTS on the plotting rank, None elsewhere: the other ranks' points are
    not read."""
    return POINTS if uw.mpi.rank == viz_rank else None


@pytest.mark.tier_b
@pytest.mark.parametrize("viz", [0, -1])
def test_only_the_plotting_rank_needs_the_points(monkeypatch, viz):
    """The other ranks pass None for the sample points; every helper completes
    with the plotting rank's values exact (before: they raised on None and the
    plotting rank waited in global_evaluate)."""
    mesh, T = _linear_field()
    viz_rank = _viz_ranks()[viz]
    points = _points_only_on(viz_rank)
    calls = _recording(monkeypatch)
    vp.parallel_line_plot(T, points, viz_rank=viz_rank)
    vp.parallel_scatter_plot(T, sympy.Integer(2) * T.sym[0], points, viz_rank=viz_rank)
    vp.parallel_profile_comparison([T.sym[0]], points, viz_rank=viz_rank)
    vp.parallel_custom_plot(lambda **data: None, [(T, points, "t")], viz_rank=viz_rank)
    plt.close("all")
    _check(calls, viz_rank, [1.0, 1.0, 2.0, 1.0, 1.0])


@pytest.mark.tier_b
@pytest.mark.parametrize("viz", [0, -1])
def test_unusable_points_on_the_plotting_rank_are_refused_on_every_rank(viz):
    """None on the plotting rank is refused with the same message on every rank,
    before global_evaluate, instead of a hang."""
    mesh, T = _linear_field()
    viz_rank = _viz_ranks()[viz]
    with pytest.raises(ValueError, match=r"sample points on the plotting rank"):
        vp.parallel_line_plot(T, None, viz_rank=viz_rank)
