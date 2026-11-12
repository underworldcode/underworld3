"""A geometry failure on one rank refuses the circular pressure lift on every rank.

The lift's cut-cell quadrature runs on each rank's own cells and can refuse them
(a cap that leaves its triangle, a cap area that is not reproduced). The ranks
whose cells are fine go on to collective sums, so a refusal raised on one rank
alone leaves the others waiting: a hang where an error was meant.
"""

import pytest
import underworld3 as uw

from underworld3.utilities import _cylindrical_pressure

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.mpi(min_size=2)]


def test_a_refusal_on_one_rank_is_raised_on_every_rank(monkeypatch):
    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusInner=1.22, radiusInternal=2.0, radiusOuter=2.22,
        cellSize_Inner=1 / 8, cellSize_Internal=1 / 8, cellSize_Outer=1 / 8, qdegree=3)
    v = uw.discretisation.MeshVariable("LiftRefusalV", mesh, 2, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("LiftRefusalP", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

    load_moments = _cylindrical_pressure._load_moments

    def refuse_on_rank_zero(*args, **kwargs):
        if uw.mpi.rank == 0:
            raise ValueError("Circular cap leaves its adjacent affine triangle.")
        return load_moments(*args, **kwargs)

    monkeypatch.setattr(_cylindrical_pressure, "_load_moments", refuse_on_rank_zero)

    refused = False
    try:
        stokes.cylindrical_pressure_lifting(
            interface_radius=2.0, outer_radius=2.22, wavenumber=2, amplitude=1.0,
            quadrature_points=4)
    except (ValueError, RuntimeError) as err:
        refused = "Circular cap leaves" in str(err)
    assert uw.mpi.comm.allgather(refused) == [True] * uw.mpi.size
