"""A transfer builder that fails on one rank is retried on every rank (#886).

``build_transfers`` retries a failed barycentric build with the RBF builder, and
degrades to the solver's default preconditioner when that fails too. The verdict
was each rank's own: a rank whose builder raised retried alone while the others
went on into the transfer's assembly, and the ranks met in different collectives
(on kaiju at np 32 and 52, after #902's assembly failure, the run hung).

The barycentric builder is rank-local work, so its verdict can be agreed right
after it: a failure on any rank is raised on every rank, and the retry is taken
together. Here the builder raises on rank 1 only. Contract: every rank records
the RBF rescue, geometric multigrid is installed, and the solve converges. A
thread timeout turns the old hang into a failure.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1108_custom_mg_builder_failure_collective_parallel.py
"""
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread")]


def test_a_builder_failure_on_one_rank_is_retried_on_every_rank(monkeypatch):
    barycentric = custom_mg._BUILDERS["barycentric"]

    def fails_on_rank_one(coarse_coords, fine_coords):
        if uw.mpi.rank == 1:
            raise RuntimeError("synthetic barycentric failure on rank 1 (test probe)")
        return barycentric(coarse_coords, fine_coords, repair_orphans=False)

    monkeypatch.setitem(custom_mg._BUILDERS, "barycentric", fails_on_rank_one)
    box = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, refinement=2,
        regular=True, qdegree=3)
    T = uw.discretisation.MeshVariable("T1108", box, 1, degree=2)
    p = uw.systems.Poisson(box, T)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1.0
    p.add_dirichlet_bc(0.0, "Bottom")
    p.add_dirichlet_bc(1.0, "Top")
    p.preconditioner = "fmg"
    with pytest.warns(UserWarning, match="retrying with the 'rbf' builder"):
        p.solve()

    assert p.snes.getConvergedReason() > 0
    assert p.snes.getKSP().getPC().getType() == "mg"
    rec = p.pc_fallbacks["custom_mg.transfer_builder"]
    assert rec["reason"] == "build_failed" and "rank 1" in rec["detail"], rec
    assert "custom_mg.build" not in p.pc_fallbacks
