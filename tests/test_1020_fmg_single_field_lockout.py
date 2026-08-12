#!/usr/bin/env python3
"""Explicit geometric MG on a single-field solver is honoured, not locked out (#478).

Native geometric FMG relies on ``DMCreateInjection`` between refined DMPlex
levels. PETSc can build that for the Stokes velocity sub-block but NOT for a
single-field (scalar/vector) discretisation on a refined DMPlex — it fails at
solve time with err62 ("Could not locate matching functional for injection",
#276). The old gate therefore declined ``preconditioner="fmg"`` to GAMG with a
warning — locking geometric MG out of every scalar/vector solver even though
the robust custom-P route (no injection anywhere) was sitting in
``utilities.custom_mg``.

Now: explicit ``"fmg"`` routes to custom-P transfers over the mesh's own
``dm_hierarchy`` (installed on the LIVE PC at first solve; GAMG stays in the
options DB as the degrade base). ``"auto"`` is deliberately unchanged (a
default flip needs its own validation campaign) — it keeps GAMG and records
the decline in ``pc_fallbacks``.
"""
import sympy
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]


@pytest.fixture
def annulus_hierarchy():
    # refinement>=1 => a multi-level dm_hierarchy; annulus + qdegree=3 is the
    # exact geometry x degree combination that err62'd under native FMG (#276)
    return uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5, cellSize=1.0 / 6.0, refinement=2, qdegree=3
    )


def _poisson(mesh, name="T"):
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=3)
    p = uw.systems.Poisson(mesh, T)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1.0
    p.add_dirichlet_bc(1.0, "Lower")
    p.add_dirichlet_bc(0.0, "Upper")
    return p


def test_scalar_fmg_routes_to_custom_geometric_mg(annulus_hierarchy):
    """preconditioner='fmg' on a scalar solver now RUNS geometric MG (custom-P;
    previously silently declined to GAMG) — read off the LIVE PC, with every
    level of the native hierarchy driven."""
    assert len(annulus_hierarchy.dm_hierarchy) > 1
    p = _poisson(annulus_hierarchy)
    p.preconditioner = "fmg"
    p.solve()  # native routing would err62 here; custom-P must not
    assert p.snes.getConvergedReason() > 0
    pc = p.snes.getKSP().getPC()
    assert pc.getType() == "mg", "explicit fmg is still being declined"
    assert pc.getMGLevels() == len(annulus_hierarchy.dm_hierarchy)
    # the reroute is recorded, and the build did NOT degrade
    assert p.pc_fallbacks["single_field_gate"]["reason"] == "declined"
    assert "custom-P" in p.pc_fallbacks["single_field_gate"]["installed"]
    assert "custom_mg.build" not in p.pc_fallbacks
    assert "custom_mg.transfer_builder" not in p.pc_fallbacks


def test_scalar_fmg_degrades_readably_when_the_build_fails(
        annulus_hierarchy, monkeypatch):
    """Negative control for the unlock: if the transfer build fails, the solve
    must still converge on the GAMG base left in the options DB, and say so."""
    def broken(coarse_coords, fine_coords):
        raise RuntimeError("synthetic builder failure (test probe)")

    monkeypatch.setitem(custom_mg._BUILDERS, "barycentric", broken)
    monkeypatch.setitem(custom_mg._BUILDERS, "rbf", broken)
    p = _poisson(annulus_hierarchy, name="Tdeg")
    p.preconditioner = "fmg"
    with pytest.warns(UserWarning, match="custom-P FMG build failed"):
        p.solve()
    assert p.snes.getConvergedReason() > 0          # stability oracle
    assert p.snes.getKSP().getPC().getType() == "gamg"
    assert p.pc_fallbacks["custom_mg.build"]["reason"] == "build_failed"


def test_cartesian_box_scalar_fmg_also_unlocked():
    """The unlock is single-field-wide, not geometry-specific: the flat
    high-degree box that also err62'd under native FMG runs custom-P too."""
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
    p.solve()
    assert p.snes.getConvergedReason() > 0
    pc = p.snes.getKSP().getPC()
    assert pc.getType() == "mg"
    assert pc.getMGLevels() == len(box.dm_hierarchy)


def test_scalar_auto_still_uses_gamg_and_records_the_decline(annulus_hierarchy):
    """'auto' is deliberately NOT flipped by #478 (that default change needs its
    own np2/np4 + adaptivity validation): it keeps GAMG, and the decline is the
    recorded migration probe."""
    p = _poisson(annulus_hierarchy, name="Tauto")
    p.solve()
    assert p.snes.getConvergedReason() > 0
    assert p.snes.getKSP().getPC().getType() == "gamg"
    assert p.pc_fallbacks["single_field_gate"]["reason"] == "declined"
    assert p.pc_fallbacks["single_field_gate"]["installed"] == "gamg"


def test_stokes_velocity_fmg_still_selected(annulus_hierarchy):
    """The single-field reroute does not touch Stokes: the velocity sub-block
    still gets native geometric FMG on the same hierarchy."""
    v = uw.discretisation.MeshVariable("v", annulus_hierarchy, annulus_hierarchy.dim, degree=2)
    pp = uw.discretisation.MeshVariable("p", annulus_hierarchy, 1, degree=1)
    stokes = uw.systems.Stokes(annulus_hierarchy, velocityField=v, pressureField=pp)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1.0
    stokes.preconditioner = "fmg"
    stokes._build(False, False, None)
    # the velocity sub-block keeps native geometric multigrid
    assert stokes.petsc_options.getString("fieldsplit_velocity_pc_type") == "mg"


def test_options_db_and_record_stay_honest_between_build_and_solve(annulus_hierarchy):
    """Between _build and the first solve the options DB deliberately says gamg
    (the safe degrade base) while the plan is custom-P mg. The #471 gating rule
    plus the record's 'resolved at first solve' phrasing keep the report honest;
    after the solve the DB carries the installed bundle."""
    p = _poisson(annulus_hierarchy, name="Thon")
    p.preconditioner = "fmg"
    p._build()
    assert p.petsc_options.getString("pc_type") == "gamg"      # degrade base
    rec = p.pc_fallbacks["single_field_gate"]
    assert "resolved at first solve" in rec["installed"]
    p.solve()
    assert p.preconditioner_settings["pc_type"] == "mg"        # bundle written
    assert p.snes.getKSP().getPC().getType() == "mg"
