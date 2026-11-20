"""The split-fault stress-history guard (#871) at np >= 2: every rank takes the
same decision, refuses together, and the default grid history solves.

Run:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1107_stress_history_fault_guard_parallel.py
"""
import os
import sys

import numpy as np
import pytest

import underworld3 as uw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_1107_stress_history_fault_guard import faulted_box, ve_stokes  # noqa: E402

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(300)]


@pytest.mark.mpi(min_size=2)
def test_every_rank_defaults_to_the_grid_and_solves():
    uw.reset_default_model()
    stokes = ve_stokes(faulted_box())
    names = uw.mpi.comm.allgather(type(stokes.Unknowns.DFDt).__name__)
    assert set(names) == {"EulerianSUPG"}
    stokes.solve(timestep=0.1)
    local = np.asarray(stokes.Unknowns.DFDt.psi_star[0].data)
    peak = uw.mpi.comm.allreduce(float(np.abs(local).max()) if local.size else 0.0, op=uw.MPI.MAX)
    assert np.isfinite(peak) and peak > 1.0e-3


@pytest.mark.mpi(min_size=2)
def test_every_rank_refuses_a_tracing_history():
    uw.reset_default_model()
    refused = 0
    try:
        ve_stokes(faulted_box(), "semi_lagrangian")
    except ValueError as err:
        refused = int("#871" in str(err))
    assert uw.mpi.comm.allreduce(refused, op=uw.MPI.MIN) == 1


@pytest.mark.mpi(min_size=2)
def test_every_rank_refuses_when_only_one_rank_holds_the_fault_record():
    """The decision is a MAX over ranks: a rank whose fault record is empty
    still refuses with the rest."""
    uw.reset_default_model()
    mesh = faulted_box()
    if uw.mpi.rank != 0:
        mesh._fault_point_pairs = {}
    refused = 0
    try:
        ve_stokes(mesh, "semi_lagrangian", fault_bc=False)
    except ValueError as err:
        refused = int("#871" in str(err))
    assert uw.mpi.comm.allreduce(refused, op=uw.MPI.MIN) == 1
