"""A reinstall of a custom-P hierarchy leaves nothing on petsc4py's deferred list (#1020).

In parallel petsc4py does not destroy a PETSc object when Python collects its wrapper:
the destroy is collective, so the object goes on a deferred list that only a collective
``PETSc.garbage_cleanup`` empties, and nothing on the standard solve paths ran one. Every
reinstall of a custom-P hierarchy kept its predecessor's transfers and level objects:
on this fixture at np 2, +6 entries per reinstall on Poisson and +28 on the Stokes
velocity block (hyperion, development a6eebb61). The
solve that installed now empties the list at its end, as the rotated path does (#1007).

Run with at least two ranks.
"""
import ctypes
import gc
import os
import re
import sys
import tempfile

import numpy as np
import pytest
import sympy
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600)]

REINSTALLS = 4


def _deferred_entries(comm):
    """The entries on petsc4py's deferred-destroy list, max over the ranks, read from
    ``PETSc.garbage_view``, which prints them."""
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
    n = sum(int(k) for k in re.findall(r"Total entries:\s*(\d+)", text))
    return comm.tompi4py().allreduce(n, op=MPI.MAX)


def _solver(kind):
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 8, regular=False, qdegree=2,
                                             refinement=1)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = base.add_fault(("Flt", np.array([[0.3, 0.45], [0.7, 0.55]])))
    _x, y = mesh.X
    if kind == "stokes":
        v = uw.discretisation.MeshVariable("V1149p", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable("P1149p", mesh, 1, degree=1)
        s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        s.constitutive_model = uw.constitutive_models.ViscousFlowModel
        s.constitutive_model.Parameters.shear_viscosity_0 = 1
        for wall in ("Bottom", "Top", "Left", "Right"):
            s.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
        s.petsc_use_pressure_nullspace = True
        field_id = 0
    else:
        s = uw.systems.Poisson(mesh)
        s.constitutive_model = uw.constitutive_models.DiffusionModel
        s.f = 1
        for wall in ("Bottom", "Top", "Left", "Right"):
            s.add_dirichlet_bc(0, wall)
        field_id = None
    s.tolerance = 1.0e-6
    return s, levels, field_id


@pytest.mark.parametrize("kind", ["poisson", "stokes"])
def test_a_reinstall_leaves_nothing_deferred(kind):
    s, levels, field_id = _solver(kind)
    custom_mg.set_custom_fmg(s, levels, field_id=field_id)
    s.solve()
    counts = []
    for _ in range(REINSTALLS):
        custom_mg.set_custom_fmg(s, levels, field_id=field_id)     # one reinstall
        s.solve()
        assert s.snes.getConvergedReason() > 0
        counts.append(_deferred_entries(s.dm.comm))
    # development: +6 (Poisson) and +28 (Stokes) entries per reinstall
    assert counts[-1] <= counts[0], counts
