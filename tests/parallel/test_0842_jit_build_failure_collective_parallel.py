"""A JIT build that fails raises on every rank, with the compiler's output (#842).

Under MPI with the disk cache on, rank 0 compiles and the other ranks wait for it
at a barrier. A build that failed raised on rank 0 alone, and the other ranks
waited at the barrier until a wall-clock limit (rank 0 raised in 1.4 s, rank 1 was
still waiting at 240 s). The error also named only the build directory, not what
the compiler said. Now every rank raises, and the message carries the tail of the
build's output.

The build is made to fail with a compiler flag gcc and clang both reject, on an
expression no earlier build has cached.

    python scripts/mpi_supervisor.py --silence 120 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0842_jit_build_failure_collective_parallel.py
"""
import os
import uuid

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(240)]

BAD_FLAG = "-fthis-flag-does-not-exist-0842"


def test_a_failed_build_raises_on_every_rank_with_the_compilers_output(monkeypatch):
    tag = uw.mpi.comm.bcast(uuid.uuid4().hex[:8], root=0)
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4)
    T = uw.discretisation.MeshVariable(f"T0842_{tag}", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    x, y = mesh.X
    poisson.f = sympy.sin(sympy.pi * x) * sympy.cos(3 * sympy.pi * y)
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.petsc_options.delValue("ksp_monitor")

    monkeypatch.setenv("UW3_JIT_CFLAGS", BAD_FLAG)
    message = ""
    try:
        poisson.solve()
    except RuntimeError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)
    assert all(messages), f"some rank did not raise: {messages}"
    assert all("this-flag-does-not-exist" in m for m in messages), messages[0][-2000:]
