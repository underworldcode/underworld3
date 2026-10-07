"""The JIT build finds PETSc's headers on any PETSc installation.

The generated callback header includes ``<petscsystypes.h>`` (#813, so the callbacks
use PETSc's configured integer width). The generated ``setup.py`` passed no PETSc
include directory, which works only where PETSc's headers are on the compiler's default
path (a conda-forge PETSc inside the environment, as in CI). With a custom PETSc build
every JIT compile failed: "'petscsystypes.h' file not found".
"""
import os

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_the_jit_include_dirs_hold_petscs_headers():
    from underworld3.utilities._jitextension import _petsc_include_dirs

    dirs = _petsc_include_dirs()
    assert any(os.path.exists(os.path.join(d, "petscsystypes.h")) for d in dirs), dirs
    assert any(os.path.exists(os.path.join(d, "petscconf.h")) for d in dirs), dirs


def test_a_jit_solve_compiles_and_converges():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    u = uw.discretisation.MeshVariable("U0021", mesh, 1, degree=1)
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    pois.constitutive_model.Parameters.diffusivity = 1.0
    pois.f = 1.0
    pois.add_dirichlet_bc(0.0, "Bottom")
    pois.solve()
    assert pois.snes.getConvergedReason() > 0
