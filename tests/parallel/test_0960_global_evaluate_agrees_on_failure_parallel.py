"""global_evaluate agrees on a local failure before its next collective (#960 item 3).

The local evaluation inside ``global_evaluate_nd`` runs between collectives (the
swarm migrations, the containment round's Allreduces). A raise on one rank alone
left the other ranks waiting in the next of them. Every rank now raises.

The failure is forced after the local evaluation's own collectives have run, on
rank 1 only, as a rank-local limiter or kernel error would.

    python scripts/mpi_supervisor.py --silence 120 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0960_global_evaluate_agrees_on_failure_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(120)]


def test_a_failure_on_one_rank_raises_on_every_rank(monkeypatch):
    from underworld3.function import _function

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8)
    T = uw.discretisation.MeshVariable("T0960", mesh, 1, degree=1)
    T.data[:, 0] = np.asarray(T.coords)[:, 0]
    pts = np.random.default_rng(1).uniform(0.05, 0.95, size=(200, 2))

    original = _function.evaluate_nd
    fired = []

    def failing(*args, **kwargs):
        value = original(*args, **kwargs)
        if not fired and uw.mpi.rank == 1:
            fired.append(1)
            raise ValueError("forced local failure on rank 1")
        return value

    monkeypatch.setattr(_function, "evaluate_nd", failing)
    message = ""
    try:
        uw.function.global_evaluate(T.sym, pts)
    except (ValueError, RuntimeError) as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)
    assert messages[1] == "forced local failure on rank 1", messages
    assert all(m for m in messages), messages
    assert all("#960" in m for i, m in enumerate(messages) if i != 1), messages
