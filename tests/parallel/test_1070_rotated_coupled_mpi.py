"""Parallel (MPI) runs of the rotated-constraint coupled-field tests of test_1070, and
a custom multigrid whose independent coarse level leaves a rank with no cell.

The rotated driver's split 0 (velocity + coupled fields) and each rigid-rotation
mode's coupled part are built with collectives (createSubDM, a KSP on the coupled
block); run at odd rank counts too.

    mpirun -n 3 python -m pytest --with-mpi tests/parallel/test_1070_rotated_coupled_mpi.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import test_1070_stokes_coupled_fields as serial  # noqa: E402

import underworld3 as uw  # noqa: E402

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(900),
              pytest.mark.level_2, pytest.mark.tier_b]


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("fmg", [False, True])
def test_parallel_rotated_freeslip_carries_a_coupled_field(fmg):
    serial.test_rotated_freeslip_carries_a_coupled_field(fmg)


@pytest.mark.mpi(min_size=2)
def test_parallel_rotated_freeslip_carries_two_coupled_fields():
    serial.test_rotated_freeslip_carries_two_coupled_fields()


@pytest.mark.mpi(min_size=2)
def test_parallel_cosserat_rotation_mode():
    serial.test_cosserat_on_a_rotating_annulus_completes_the_rotation_mode()


@pytest.mark.mpi(min_size=2)
def test_parallel_custom_fmg_with_a_rank_that_owns_no_coarse_cell():
    # A two-cell independent coarse level leaves at least one rank with no cell of
    # its own: its fine nodes can only be located across partitions. It used to
    # divide by zero (components per node from no nodes) and then fail to
    # triangulate no points.
    fine = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, refinement=1,
        qdegree=3)
    coarse = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.5, qdegree=3)
    cells = coarse.dm.getHeightStratum(0)
    owns_none = uw.mpi.comm.allreduce(int(cells[1] - cells[0] == 0), op=uw.MPI.MAX)
    if not owns_none:
        pytest.skip("every rank owns a coarse cell at this rank count")
    stokes = uw.systems.Stokes(fine)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.set_custom_fmg([coarse] + fine._coarse_level_meshes()[:-1], field_id=0)
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
