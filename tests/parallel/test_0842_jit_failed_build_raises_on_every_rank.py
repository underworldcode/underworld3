"""A JIT build that fails on rank 0 raises on every rank, with the compiler's
error, instead of hanging the job (#842).

Under MPI with the disk cache on, only rank 0 compiles; the other ranks wait at
a barrier and then load what it published. A rank-0 build failure used to raise
on rank 0 alone, before the barrier, and every other rank waited there until a
wall-clock limit killed the job. The error also omitted the compiler's output
unless ``verbose`` was set.

The build is made to fail with an option no compiler accepts, through
``UW3_JIT_CFLAGS``. The test carries a thread-method timeout, so a regression
ends the waiting rank with a stack dump instead of hanging the run.
"""
import tempfile

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(240, method="thread")]

_BAD_FLAG = "-fno-such-option-842"


@pytest.mark.skipif(uw.mpi.size < 2, reason="needs at least two ranks")
def test_a_failed_rank_zero_build_raises_on_every_rank(monkeypatch):
    # A cache directory of our own: empty, so the build cannot be found on disk,
    # and shared by the ranks, so the rank-0-compiles path is the one taken.
    cache_dir = uw.mpi.comm.bcast(
        tempfile.mkdtemp(prefix="uw_jit_842_") if uw.mpi.rank == 0 else None
    )
    monkeypatch.setenv("UW_JIT_CACHE_DIR", cache_dir)
    monkeypatch.setenv("UW3_JIT_CFLAGS", _BAD_FLAG)

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25
    )
    T = uw.discretisation.MeshVariable("T_842", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    x, y = mesh.X
    # A source term no other test uses, so no in-memory module can stand in.
    poisson.f = x**7 * y**5 + sympy.sin(3 * x * y) ** 3
    poisson.add_dirichlet_bc(0.0, "Bottom")

    with pytest.raises(RuntimeError) as raised:
        poisson.solve()

    # Every rank got here, and every rank carries the compiler's own complaint.
    assert "such-option-842" in str(raised.value)
    assert uw.mpi.comm.allgather(True) == [True] * uw.mpi.size
