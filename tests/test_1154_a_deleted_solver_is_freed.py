"""A solver the user no longer holds is freed, with its SNES and KSP tree (#1041).

Each solver registers itself on its mesh, so that a mesh move or a DM change can
mark it for rebuild, and so that a snapshot can capture its step state. The
register was a plain list, which kept every solver ever built on the mesh alive
for as long as the mesh: about 4.6 MB per dropped solver, and its PETSc options
stayed in the global database. The register now holds its solvers weakly, in
the order they were built.
"""
import gc
import weakref

import numpy as np
import pytest
from petsc4py import PETSc

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _stokes(mesh):
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.bodyforce = (0.0, 1.0)
    stokes.petsc_use_pressure_nullspace = True
    return stokes


def _poisson(mesh, name):
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    return poisson


def test_a_dropped_solver_is_collected_with_its_options():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.25, qdegree=2)
    refs, prefixes = [], []
    for _ in range(3):
        stokes = _stokes(mesh)
        stokes.solve()
        refs.append(weakref.ref(stokes))
        prefixes.append(stokes.petsc_options_prefix)
        del stokes
        gc.collect()
    # development: all three alive, held by mesh._equation_systems_register
    assert [r() for r in refs] == [None, None, None]
    assert len(list(mesh._equation_systems_register)) == 0
    options = PETSc.Options().getAll()
    assert not [key for key in options if any(key.startswith(p) for p in prefixes)]


def test_the_register_still_reaches_the_live_solvers_in_build_order():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.25, qdegree=2)
    first, dropped, last = _poisson(mesh, "T1154a"), _poisson(mesh, "T1154b"), _poisson(mesh, "T1154c")
    first.solve()
    last.solve()
    del dropped
    gc.collect()
    assert list(mesh._equation_systems_register) == [first, last]

    # a move marks every live solver for rebuild, so the next solve reads the
    # moved coordinates (#122)
    assert first.is_setup and last.is_setup
    coords = np.asarray(mesh._coords).copy()
    coords[:, 1] *= 1.1
    mesh.deform(coords)
    assert not first.is_setup and not last.is_setup
