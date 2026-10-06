#!/usr/bin/env python3
"""The cold-start verdict that arms the automatic Picard warm-up must be the same on
every rank.

`_solution_is_trivially_zero()` decides whether a solve that was told to warm-start is
really a cold start, and the verdict arms a JIT rewire and an extra `snes.solve` — both
collective. It read `u.vec.norm()`, and `u.vec` is the variable's LOCAL vector, so the
norm was per rank. Measured at np=2 (review of PR #794): a nonlinear Poisson solve with
an initial guess non-zero in one corner only — rank 0 saw a zero norm and took the
warm-up, rank 1 did not; rank 1 failed with MPI_ERR_BUFFER in the Jacobian assembly and
rank 0 spun until killed.

The verdicts are gathered and compared on every rank BEFORE the solve, so a regression
fails the test on all ranks together instead of hanging the job.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_trivially_zero_is_rank_uniform_and_the_solve_completes():
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1)
    u = uw.discretisation.MeshVariable("u1081", mesh, 1, degree=1)
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    pois.constitutive_model.Parameters.diffusivity = 1.0 + u.sym[0] ** 2
    pois.f = 1.0
    for b in ("Left", "Right", "Top", "Bottom"):
        pois.add_dirichlet_bc(0.0, b)
    coords = u.coords
    corner = (coords[:, 0] < 0.15) & (coords[:, 1] < 0.15)
    with uw.synchronised_array_update():
        u.array[:, 0, 0] = np.where(corner, 0.1, 0.0)

    verdicts = uw.mpi.comm.allgather(bool(pois._solution_is_trivially_zero()))
    assert verdicts == [False] * uw.mpi.size, verdicts

    pois.solve(zero_init_guess=False)
    assert pois.snes.getConvergedReason() > 0
