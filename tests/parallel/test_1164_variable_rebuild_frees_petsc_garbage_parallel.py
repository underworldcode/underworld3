"""Adding a variable to a fixed mesh in parallel does not grow petsc4py's
deferred-destroy list (#868).

Each new MeshVariable rebuilds the mesh's DM and every variable's vectors. In
parallel petsc4py does not destroy a collected object on a communicator larger
than one: it defers it to a list only ``PETSc.garbage_cleanup`` empties. On
development the list grew with every variable (2, 82 and 310 entries after 1,
6 and 12 P2 variables at np 2), so a run on one fixed mesh that keeps creating
variables never released the old DMs and vectors.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1164_variable_rebuild_frees_petsc_garbage_parallel.py
"""
import ctypes
import gc
import os
import re
import sys
import tempfile

import pytest
from petsc4py import PETSc

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.timeout(600)]


def _deferred(comm):
    """Entries on petsc4py's deferred-destroy list, read from garbage_view."""
    gc.collect()
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    with tempfile.TemporaryFile() as f:
        saved = os.dup(1)
        os.dup2(f.fileno(), 1)
        try:
            PETSc.garbage_view(comm)
            libc.fflush(None)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        text = f.read().decode(errors="replace")
    return sum(int(n) for n in re.findall(r"Total entries:\s*(\d+)", text))


@pytest.mark.tier_b
def test_new_variables_on_a_fixed_mesh_leave_no_growing_deferred_list():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 16, qdegree=2)
    held = []
    counts = []
    # Handles a user may hold across the rebuilds: the DM before them, and the
    # first variable's vector before the second variable replaces it. The
    # cleanup must leave both usable (#492): an eager destroy would zero them.
    first = uw.discretisation.MeshVariable("G1164_first", mesh, 1, degree=2)
    first.array[...] = 3.0
    old_dm = mesh.dm
    old_vec = first.vec
    for k in range(12):
        held.append(uw.discretisation.MeshVariable(f"G1164_{k}", mesh, 1, degree=2))
        held[-1].array[...] = float(k)
        counts.append(uw.mpi.comm.allreduce(_deferred(mesh.dm.comm), op=uw.MPI.MAX))
    # development: 2, 10, 22, 38, 58, 82, 110, 142, 178, 218, 262, 310 -- quadratic
    # growth. A bounded list stays at the few objects one rebuild leaves behind.
    assert max(counts[2:]) <= counts[1] + 10, counts

    # Nothing still in use was destroyed: every variable kept its values through
    # the rebuilds, and a solve on the twelfth runs on the current DM.
    import numpy as np
    for k, var in enumerate(held):
        values = np.asarray(var.array)
        local = float(np.abs(values - k).max()) if values.size else 0.0
        assert uw.mpi.comm.allreduce(local, op=uw.MPI.MAX) == 0.0
    _poisson_solves_on(mesh, held[-1])

    assert old_dm.handle != 0 and old_dm.getChart()[1] > 0
    assert old_vec.handle != 0 and old_vec.getLocalSize() >= 0
    assert uw.mpi.comm.allreduce(float(abs(np.asarray(first.array) - 3.0).max())
                                 if np.asarray(first.array).size else 0.0, op=uw.MPI.MAX) == 0.0


def _poisson_solves_on(mesh, T):
    """A Poisson solve on ``T`` after the rebuilds: T = y on the unit box with
    T = 0 on Bottom and T = 1 on Top and no source, exact in P2."""
    import numpy as np
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.tolerance = 1e-10
    poisson.petsc_options.delValue("ksp_monitor")
    poisson.solve(zero_init_guess=True)
    assert poisson.snes.getConvergedReason() > 0
    X = np.asarray(T.coords)
    error = float(np.abs(np.asarray(T.array)[:, 0, 0] - X[:, 1]).max()) if len(X) else 0.0
    assert uw.mpi.comm.allreduce(error, op=uw.MPI.MAX) < 1e-8
