"""Dropped meshes and solvers leave nothing in PETSc's global options database (#950).

Every solver writes its PETSc options under a prefix unique to it (``Solver_N_``, with
``private_Solver_N_`` copies for its sub-objects), and every mesh wrote three
``meshproj_N_`` keys for its coordinate element. Nothing removed them, so a loop that
rebuilds a mesh and a solver grew the database without bound: 17 keys a cycle for a
Poisson solve on development eddab707.

- Building, solving and dropping a mesh with its Poisson solver, again and again, leaves
  the database the same size after the first cycle.
- A solver that is still alive keeps every one of its options while another is dropped.
- After ``uw_object._reset()`` a new solver can take a dropped one's prefix; the dropped
  one's finalizer then leaves the new solver's options alone.
- A finalizer that runs after PETSc is finalised (the mesh and its solvers form a
  reference cycle the collector can reach late) returns instead of touching the options
  database, which ended the process (rc 1 on macOS, a segfault on Linux).

A solver lives as long as its mesh: the mesh holds it for its rebuild hook, so its
options go when both are dropped.
"""
import gc
import os
import subprocess
import sys

import pytest
from petsc4py import PETSc

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _poisson_on_a_new_mesh(i):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    T = uw.discretisation.MeshVariable(f"T0079_{i}", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.solve()
    return poisson


def test_rebuilding_a_mesh_and_solver_does_not_grow_the_options():
    _poisson_on_a_new_mesh(0)
    gc.collect()
    before = len(PETSc.Options().getAll())
    for i in range(1, 4):
        _poisson_on_a_new_mesh(i)
        gc.collect()
    assert len(PETSc.Options().getAll()) == before


def _options_of(prefix):
    return {k: v for k, v in PETSc.Options().getAll().items()
            if k.startswith(prefix) or k.startswith("private_" + prefix)}


def test_a_live_solver_keeps_its_options_when_another_is_dropped():
    kept = _poisson_on_a_new_mesh(9)
    kept.petsc_options["ksp_rtol"] = 1.0e-7
    before = _options_of(kept.petsc_options_prefix)
    _poisson_on_a_new_mesh(10)
    gc.collect()
    assert before and _options_of(kept.petsc_options_prefix) == before


def test_a_reused_prefix_keeps_the_new_solvers_options():
    from underworld3.utilities._api_tools import uw_object

    saved_count = uw_object._obj_count
    try:
        uw_object._reset()
        first = _poisson_on_a_new_mesh(20)
        prefix = first.petsc_options_prefix
        uw_object._reset()
        second = _poisson_on_a_new_mesh(21)
        assert second.petsc_options_prefix == prefix     # the premise: a reused prefix
        second.petsc_options["ksp_rtol"] = 1.0e-7
        before = _options_of(prefix)
        del first
        gc.collect()
        assert before and _options_of(prefix) == before
    finally:
        uw_object._obj_count = max(saved_count, uw_object._obj_count)


_LATE_FINALIZER = """
import gc
from petsc4py import PETSc
import underworld3 as uw

def build():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    q = uw.systems.Poisson(mesh, u_Field=T)
    q.constitutive_model = uw.constitutive_models.DiffusionModel
    q.petsc_options["ksp_rtol"] = 1e-6

gc.disable()
build()
PETSc._finalize()
gc.enable()
gc.collect()
print("LATEFINALIZER survived", flush=True)
"""


def test_a_finalizer_after_petsc_is_finalised_does_nothing():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMI", "PMIX"))}
    env["OMP_NUM_THREADS"] = "1"
    out = subprocess.run([sys.executable, "-c", _LATE_FINALIZER], env=env,
                         capture_output=True, text=True, timeout=240)
    assert out.returncode == 0 and "LATEFINALIZER survived" in out.stdout, (
        out.returncode, out.stdout[-1000:], out.stderr[-1000:])
