"""Re-solving a rotated free-slip model whose body force is reassigned frees what each rebuild drops, in parallel (#1007).

A reassigned expression re-JITs the solver and rebuilds its DM, and ``dm.handle`` is in
the rotated workspace's geometry signature, so every such solve rebuilds the workspace:
the rotation, the operator and the custom-FMG prolongation. The old workspace's
prolongation list and the builders' intermediate matrices were dropped, not destroyed.
In parallel petsc4py defers each destroy to a collective ``PETSc.garbage_cleanup``,
which only a new mesh ran (test_0078), so each rebuild kept them: measured on
development 15f13d0c at np 2, peak RSS rose 805, 819, 832, 847, 860, 874 MB over six
solves, about 7 MB per rank per solve. A solve that built a new workspace now empties
the deferred list at its end.

The test counts that list's entries itself, through ``PETSc.garbage_view``, rather
than reading the resident set: a test that runs earlier in the same process can free
enough heap to absorb the leak before the resident set moves (about 33 MB after
test_1139 or test_1097, hyperion 2026-11-08), which made a resident-set test pass or
fail by where CI's batches of six files fell. The deferred list is this process's own
and does not depend on what ran before: on development it fails alone, in its CI
batch and after either of those, and with the fix it passes in all four.

Run with at least two ranks.
"""
import ctypes
import gc
import os
import re
import sys
import tempfile

import pytest
import sympy
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900)]

WARMUP, MEASURED = 2, 5


def _deferred_entries(comm):
    """The number of entries in petsc4py's deferred-destroy list, the objects whose
    Python wrappers are gone and whose destroy waits for a collective cleanup, read
    from PETSc.garbage_view: rank 0 prints every rank's "Rank r:: Total entries: N"
    line (a synchronised print), and the other ranks print nothing. PETSc prints
    through C's stdout, so file descriptor 1 is redirected around the call and the C
    streams flushed. Counting entries, not bytes, so names and object ids that print
    wider cannot read as growth. COLLECTIVE."""
    gc.collect()
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    with tempfile.TemporaryFile() as captured:
        saved = os.dup(1)
        try:
            os.dup2(captured.fileno(), 1)
            PETSc.garbage_view(comm)
            libc.fflush(None)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        captured.seek(0)
        text = captured.read().decode(errors="replace")
    return sum(int(n) for n in re.findall(r"Total entries:\s*(\d+)", text))


def test_a_rebuild_leaves_no_deferred_destroys_behind():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2,
                              refinement=2, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y, x)
    unit_r = sympy.Matrix([[x / r, y / r]])
    v = uw.discretisation.MeshVariable("V1144", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1144", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.add_rotated_freeslip_bc(0.0, "Upper")
    stokes.add_rotated_freeslip_bc(0.0, "Lower")
    custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0)
    stokes.tolerance = 1.0e-6
    comm = MPI.COMM_WORLD

    def solve(k):
        stokes.bodyforce = (-1) ** k * sympy.cos(2 * th) * unit_r   # rebuilds the DM
        stokes.solve(zero_init_guess=True)
        info = stokes._rotated_freeslip_info
        assert info["converged"], info["rnorm_history"]
        if k:
            # the path under test: the workspace was rebuilt, not reused
            assert info["rotation_reused"] is False, k

    for k in range(WARMUP):
        solve(k)
    entries = []
    for k in range(WARMUP, WARMUP + MEASURED):
        solve(k)
        entries.append(_deferred_entries(stokes.dm.comm))
    # every rebuild's drops are freed at the end of its solve, so the list does not grow;
    # unfreed, it grows by the old workspace's objects at every rebuild (on development
    # 06c574e8 at np 2, rank 0's summary grew by 5412 bytes a rebuild, hyperion)
    growth = comm.allreduce(entries[-1] - entries[0], op=MPI.MAX)
    assert growth <= 0, (f"the deferred-destroy list grew by {growth} entries over "
                         f"{MEASURED - 1} rebuilds: {entries}")
