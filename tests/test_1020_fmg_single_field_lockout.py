#!/usr/bin/env python3
"""Regression: native geometric FMG is locked out for single-field solvers (#276).

Native geometric FMG relies on ``DMCreateInjection`` between refined DMPlex
levels. PETSc can build that for the Stokes velocity sub-block but NOT for a
single-field (scalar/vector) discretisation on a refined DMPlex — it fails at
solve time with err62 ("Could not locate matching functional for injection").

So ``preconditioner="fmg"``/``"auto"`` on a scalar/vector solver must fall back
to GAMG (and solve), never route to native FMG and crash. Geometric MG on such
a solver remains available via ``utilities.custom_mg.set_custom_fmg`` (covered by
test_1016).
"""
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]


@pytest.fixture
def annulus_hierarchy():
    # refinement>=1 => a multi-level dm_hierarchy (the FMG trigger)
    return uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5, cellSize=1.0 / 6.0, refinement=2, qdegree=3
    )


def _poisson(mesh):
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    p = uw.systems.Poisson(mesh, T)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1.0
    p.add_dirichlet_bc(1.0, "Lower")
    p.add_dirichlet_bc(0.0, "Upper")
    return p


def test_scalar_fmg_falls_back_and_solves(annulus_hierarchy):
    """preconditioner='fmg' on a scalar solver falls back to GAMG and solves
    (previously err62 DMCreateInjection)."""
    assert len(annulus_hierarchy.dm_hierarchy) > 1  # hierarchy present
    p = _poisson(annulus_hierarchy)
    p.preconditioner = "fmg"
    p.solve()  # must not raise err62
    assert p.snes.getConvergedReason() > 0
    # resolved to GAMG, NOT native geometric mg
    assert p.petsc_options.getString("pc_type") == "gamg"


def test_scalar_auto_uses_gamg_not_native_fmg(annulus_hierarchy):
    """preconditioner='auto' on a scalar solver with a hierarchy must NOT select
    native FMG (it would crash) — it silently uses GAMG."""
    p = _poisson(annulus_hierarchy)
    # 'auto' is the default; solve and confirm no native-mg pc_type
    p.solve()
    assert p.snes.getConvergedReason() > 0
    assert p.petsc_options.getString("pc_type") != "mg"


def test_cartesian_box_scalar_also_falls_back():
    """The lockout is single-field-wide, not geometry-specific: native single-
    field FMG is fragile across geometry×degree×refinement (it errors 62 even on
    some flat high-degree boxes), so a scalar solver on a flat box also falls
    back to GAMG and solves rather than risk the crash."""
    box = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.4, refinement=1, qdegree=3
    )
    assert len(box.dm_hierarchy) > 1
    T = uw.discretisation.MeshVariable("Tbox", box, 1, degree=3)
    p = uw.systems.Poisson(box, T)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1.0
    p.add_dirichlet_bc(0.0, "Bottom")
    p.add_dirichlet_bc(1.0, "Top")
    p.preconditioner = "fmg"
    p.solve()  # must not raise err62
    assert p.snes.getConvergedReason() > 0
    assert p.petsc_options.getString("pc_type") == "gamg"


def test_stokes_velocity_fmg_still_selected(annulus_hierarchy):
    """The lockout is single-field only: the Stokes velocity sub-block still gets
    native geometric FMG on the same hierarchy."""
    v = uw.discretisation.MeshVariable("v", annulus_hierarchy, annulus_hierarchy.dim, degree=2)
    pp = uw.discretisation.MeshVariable("p", annulus_hierarchy, 1, degree=1)
    stokes = uw.systems.Stokes(annulus_hierarchy, velocityField=v, pressureField=pp)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1.0
    stokes.preconditioner = "fmg"
    stokes._build(False, False, None)
    # the velocity sub-block keeps native geometric multigrid
    assert stokes.petsc_options.getString("fieldsplit_velocity_pc_type") == "mg"
