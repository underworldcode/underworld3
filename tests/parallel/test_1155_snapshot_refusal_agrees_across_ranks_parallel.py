"""A restore that names a dropped solver's history is refused on every rank (#1041).

The mesh's solver register holds its solvers weakly, so a solver the user drops is
collected, with its history manager. A snapshot that captured that history cannot be
restored, and says so. When a reference to the dropped solver survives on only some
ranks, those ranks found the history and restored it while the others raised, and the
job hung in the next collective (hyperion, np 2 and 4, PR #1050 review round 2). The
verdict is now agreed before anything is written: every rank raises, and the model is
left as it was.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1155_snapshot_refusal_agrees_across_ranks_parallel.py
"""
import gc

import numpy as np
import pytest

import underworld3 as uw
from underworld3.checkpoint.snapshot import SnapshotInvalidatedError

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread"),
              pytest.mark.skipif(uw.mpi.size == 1,
                                 reason="one rank holding a reference needs two ranks; "
                                        "test_1154 covers the serial contract")]


@pytest.mark.parametrize("holder", ["first", "last"])
def test_a_reference_on_one_rank_does_not_split_the_verdict(tmp_path, holder):
    uw.reset_default_model()
    orchestration_model = uw.get_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6), minCoords=(0.0, 0.0),
                                        maxCoords=(1.0, 1.0), qdegree=3)
    v = uw.discretisation.MeshVariable("V1155", mesh, mesh.dim, degree=1)
    T = uw.discretisation.MeshVariable("T1155", mesh, 1, degree=2)
    v.array[:, 0, 0] = 1.0
    T.array[:, 0, 0] = 1.0
    adv_diff = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v)
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = 0.01
    adv_diff.add_dirichlet_bc(0.0, "Left")
    adv_diff.solve(timestep=0.01)

    token = orchestration_model.save_state()
    # one path for the collective write: tmp_path differs per rank
    path = uw.mpi.comm.bcast(str(tmp_path / "held.snap.h5"), root=0)
    orchestration_model.save_state(file=path)

    holding_rank = 0 if holder == "first" else uw.mpi.size - 1
    kept = adv_diff if uw.mpi.rank == holding_rank else None
    del adv_diff
    gc.collect()
    T.array[:, 0, 0] = 7.0                    # what a refused restore must leave alone
    before = np.array(T.array, copy=True)

    # before the verdict was agreed: the holding rank restored, the others raised, and
    # the next collective hung
    with pytest.raises(SnapshotInvalidatedError, match="collected"):
        orchestration_model.load_state(token)
    assert np.array_equal(np.asarray(T.array), before)
    with pytest.raises(ValueError, match="collected"):
        orchestration_model.load_state(path)
    assert np.array_equal(np.asarray(T.array), before)

    # every rank reaches the next collective
    assert uw.mpi.comm.allreduce(1) == uw.mpi.size
    del kept
    uw.reset_default_model()
