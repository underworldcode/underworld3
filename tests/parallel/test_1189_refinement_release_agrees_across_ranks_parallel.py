"""Every rank gives the same verdict on a wall refinement left off its surface.

On this cubed sphere the refinement callback would invert cells at a few vertices
of the inner wall, which stay where refinement placed them (#949), so the mesh
releases the sphere it would have declared for ``Lower``. The held-back vertices
sit in one corner of the mesh: a rank whose partition holds none of them must
release the wall all the same, or a method reading ``mesh.analytic_surface`` would
proceed on that rank and refuse on another.
"""
import warnings

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing.smoothing.graph import _pinned_mask

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread"),
              pytest.mark.skipif(uw.mpi.size == 1,
                                 reason="the ranks' verdicts are compared with each other")]

ON_SURFACE = 64 * np.finfo(float).eps


def test_a_wall_held_back_on_one_rank_is_released_on_every_rank():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mesh = uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.547, numElements=3,
                                      simplex=True, refinement=1)

    def refused(label):
        try:
            mesh.analytic_surface(label)
        except ValueError:
            return True
        return False

    X = np.asarray(mesh.X.coords, dtype=float)
    lower = np.linalg.norm(X[_pinned_mask(mesh.dm, ("Lower",))], axis=1)
    holds_one = bool(np.any(np.abs(lower - 0.547) > ON_SURFACE * 0.547))

    holders = uw.mpi.comm.allgather(holds_one)
    lower_refused = uw.mpi.comm.allgather(refused("Lower"))
    upper_refused = uw.mpi.comm.allgather(refused("Upper"))

    assert any(holders), (
        "on this host's triangulation refinement held back no Lower vertex, so the "
        "test has nothing to release: choose another fixture")
    assert all(lower_refused), f"Lower refused by rank: {lower_refused}, holders: {holders}"
    assert not any(upper_refused), f"Upper refused by rank: {upper_refused}"
    if all(holders):
        # Measured on Linux (MPICH), 2026-11-14: both ranks hold one at two ranks; one
        # rank of four holds none. scripts/test.sh runs this file at four ranks too.
        pytest.skip(f"every one of the {uw.mpi.size} ranks holds a held-back vertex, so "
                    "this rank count does not show a rank learning of it from another")
