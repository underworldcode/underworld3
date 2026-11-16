"""A JIT build that fails raises on every rank, with the compiler's output (#842).

Under MPI with the disk cache on, rank 0 compiles and the other ranks wait for it
at a barrier. A build that failed raised on rank 0 alone, and the other ranks
waited at the barrier until a wall-clock limit (rank 0 raised in 1.4 s, rank 1 was
still waiting at 240 s). The error also named only the build directory, not what
the compiler said. Now every rank raises, and the message carries the tail of the
build's output.

The build is made to fail with a compiler flag gcc and clang both reject, on an
expression no earlier build has cached. Two more paths are agreed the same way: a
rank that cannot see rank 0's published module compiles its own, and that build
may fail there alone; and rank 0 may build and then fail to store the module.

    python scripts/mpi_supervisor.py --silence 120 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0842_jit_build_failure_collective_parallel.py
"""
import os
import uuid

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(240)]

BAD_FLAG = "-fthis-flag-does-not-exist-0842"


def _poisson(T):
    mesh = T.mesh
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    x, y = mesh.X
    poisson.f = sympy.sin(sympy.pi * x) * sympy.cos(3 * sympy.pi * y)
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.petsc_options.delValue("ksp_monitor")
    return poisson


def _fresh_field():
    tag = uw.mpi.comm.bcast(uuid.uuid4().hex[:8], root=0)
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4)
    return uw.discretisation.MeshVariable(f"T0842_{tag}", mesh, 1, degree=1)


def _messages_of(poisson):
    message = ""
    try:
        poisson.solve()
    except RuntimeError as error:
        message = str(error)
    return uw.mpi.comm.allgather(message)


def _shared_tmp(tmp_path_factory):
    return uw.mpi.comm.bcast(str(tmp_path_factory.mktemp("jit0842")) if uw.mpi.rank == 0
                             else None, root=0)


def test_a_failed_build_raises_on_every_rank_with_the_compilers_output(monkeypatch):
    poisson = _poisson(_fresh_field())
    monkeypatch.setenv("UW3_JIT_CFLAGS", BAD_FLAG)
    messages = _messages_of(poisson)
    assert all(messages), f"some rank did not raise: {messages}"
    assert all("this-flag-does-not-exist" in m for m in messages), messages[0][-2000:]


def test_a_failed_fallback_build_on_one_rank_raises_on_every_rank(monkeypatch,
                                                                    tmp_path_factory):
    """Rank 0's cache holds the module and the other ranks' caches are empty, so
    they compile their own copies; the last rank's build fails. That rank raised
    alone and rank 0 went on, to hang in the solve's first collective.

    The failure is simulated on that rank: its compiler flags were fixed when the
    code was generated, and the build takes PETSc's compiler whatever CC says."""
    if uw.mpi.size < 2:
        pytest.skip("needs a rank that cannot see rank 0's module")
    from underworld3.utilities import _jitextension

    root = _shared_tmp(tmp_path_factory)
    T = _fresh_field()
    monkeypatch.setenv("UW_JIT_CACHE_DIR", os.path.join(root, "warm"))
    _poisson(T).solve()                     # every rank healthy: warms one cache
    _jitextension._ext_dict.clear()         # forget the in-memory copies

    if uw.mpi.rank > 0:
        monkeypatch.setenv("UW_JIT_CACHE_DIR", os.path.join(root, f"empty{uw.mpi.rank}"))
    if uw.mpi.rank == uw.mpi.size - 1:
        def broken(*args, **kwargs):
            raise RuntimeError("simulated build failure 0842")
        monkeypatch.setattr(_jitextension, "compile_and_load", broken)
    messages = _messages_of(_poisson(T))
    assert all(messages), f"some rank did not raise: {messages}"
    assert all(f"rank(s) [{uw.mpi.size - 1}]" in m for m in messages), messages
    assert all("simulated build failure 0842" in m for m in messages), messages


def test_a_failed_cache_write_says_so_on_every_rank(monkeypatch, tmp_path_factory):
    """Rank 0 builds into a cache it cannot write. Every rank raises, and the
    message says the build succeeded and the cache write failed."""
    if uw.mpi.size < 2:
        pytest.skip("rank 0 publishes to the cache only at np > 1")
    root = _shared_tmp(tmp_path_factory)
    locked = os.path.join(root, "locked")
    if uw.mpi.rank == 0:
        os.makedirs(locked)
        os.chmod(locked, 0o555)
    uw.mpi.comm.Barrier()
    if os.access(locked, os.W_OK):
        pytest.skip("the directory is writable anyway (running as root?)")
    monkeypatch.setenv("UW_JIT_CACHE_DIR", locked)
    try:
        messages = _messages_of(_poisson(_fresh_field()))
    finally:
        uw.mpi.comm.Barrier()
        if uw.mpi.rank == 0:
            os.chmod(locked, 0o755)
    assert all(messages), f"some rank did not raise: {messages}"
    assert all("storing it in the JIT cache failed" in m for m in messages), messages
