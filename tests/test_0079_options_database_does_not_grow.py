"""Dropped meshes and solvers leave nothing in PETSc's global options database (#950).

Every solver writes its PETSc options under a prefix unique to it (``Solver_N_``, with
``private_Solver_N_`` copies for its sub-objects), and every mesh wrote three
``meshproj_N_`` keys for its coordinate element. Nothing removed them, so a loop that
rebuilds a mesh and a solver grew the database without bound: 17 keys a cycle for a
Poisson solve on development eddab707.

- Building, solving and dropping a mesh with its Poisson solver, again and again, leaves
  the database the same size after the first cycle.
- A solver that is still alive keeps its options.

A solver lives as long as its mesh: the mesh holds it for its rebuild hook, so its
options go when both are dropped.
"""
import gc

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


def test_a_live_solver_keeps_its_options():
    poisson = _poisson_on_a_new_mesh(9)
    poisson.petsc_options["ksp_rtol"] = 1.0e-7
    gc.collect()
    assert f"{poisson.petsc_options_prefix}ksp_rtol" in PETSc.Options().getAll()
