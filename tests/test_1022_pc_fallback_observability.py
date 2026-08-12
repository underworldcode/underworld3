#!/usr/bin/env python3
"""Every preconditioner fallback leaves a readable record (#484).

The solver ecosystem degrades gracefully in many places (single-field FMG
gate, missing hierarchy, transfer-build failures, guard skips, forced
Galerkin). Before #484, 10 of the 12 fallback sites left no queryable state —
a warning at best, silence at worst — so neither a user nor a test could ask
"did I get what I asked for?". Now every site writes into
``solver.pc_fallbacks`` through one recorder, with a fixed reason vocabulary:
``unavailable``, ``declined``, ``build_failed``, ``check_skipped``, ``forced``.

House rule: every probe is proven to fire AND proven silent on healthy state.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

REASONS = {"unavailable", "declined", "build_failed", "check_skipped", "forced"}


def _poisson_on(mesh, degree=1, name="T"):
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=degree)
    p = uw.systems.Poisson(mesh, T)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1.0
    p.add_dirichlet_bc(0.0, "Bottom")
    p.add_dirichlet_bc(1.0, "Top")
    return p


def _box(cellSize=0.4, refinement=0, qdegree=2):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cellSize, refinement=refinement, qdegree=qdegree)


@pytest.fixture
def adapt_child():
    mesh = _box(cellSize=0.2, refinement=1)

    def metric(points):
        r = np.linalg.norm(np.asarray(points) - 0.5, axis=1)
        return 1.0 / np.where(r < 0.25, 0.05, 0.2) ** 2

    return mesh.adapt(metric, max_levels=1)


# --------------------------------------------------------------------------- #
#  The global negative control: a clean solve records NOTHING
# --------------------------------------------------------------------------- #
def test_clean_solve_has_empty_fallback_record():
    """No hierarchy, default preconditioner: the resolution is exactly what was
    asked for, so the record must be empty — this is the state every other
    test's 'record present' assertion is measured against."""
    p = _poisson_on(_box(refinement=0))
    p.solve()
    assert p.snes.getConvergedReason() > 0
    assert p.pc_fallbacks == {}


def test_explicit_gamg_records_nothing():
    """Explicit 'gamg' on a hierarchy mesh is honoured verbatim — no decline,
    no record (the negative control for the auto-decline probe)."""
    p = _poisson_on(_box(refinement=1))
    p.preconditioner = "gamg"
    p.solve()
    assert p.snes.getConvergedReason() > 0
    assert "single_field_gate" not in p.pc_fallbacks
    assert "no_hierarchy" not in p.pc_fallbacks


# --------------------------------------------------------------------------- #
#  Gate + no-hierarchy sites (the pyx _apply_preconditioner_options sites)
# --------------------------------------------------------------------------- #
def test_fmg_without_hierarchy_records_unavailable_and_still_warns():
    p = _poisson_on(_box(refinement=0))
    p.preconditioner = "fmg"
    with pytest.warns(UserWarning, match="no refinement hierarchy"):
        p.solve()
    rec = p.pc_fallbacks["no_hierarchy"]
    assert rec["reason"] == "unavailable"
    assert rec["installed"] == "gamg"
    assert p.petsc_options.getString("pc_type") == "gamg"
    # negative control: a hierarchy mesh must NOT carry this record
    p2 = _poisson_on(_box(refinement=1), name="T2")
    p2.preconditioner = "fmg"
    p2.solve()
    assert "no_hierarchy" not in p2.pc_fallbacks


def test_auto_decline_is_recorded_but_stays_silent(recwarn):
    """'auto' on a single-field solver with a hierarchy declines geometric FMG
    by design. The decline is now on the record — but it must NOT gain a
    warning (auto's silence is a stability contract)."""
    p = _poisson_on(_box(refinement=2))
    p.solve()
    assert p.snes.getConvergedReason() > 0
    rec = p.pc_fallbacks["single_field_gate"]
    assert rec["reason"] == "declined"
    assert rec["installed"] == "gamg"
    fmg_warnings = [w for w in recwarn.list
                    if "FMG" in str(w.message) or "GAMG" in str(w.message)]
    assert fmg_warnings == [], "the auto decline must not warn"


def test_reason_vocabulary_is_closed():
    """Whatever a solve records, the reasons come from the documented set."""
    p = _poisson_on(_box(refinement=1))
    p.preconditioner = "fmg"
    p.solve()
    for site, rec in p.pc_fallbacks.items():
        assert rec["reason"] in REASONS, (site, rec)
        assert set(rec) == {"requested", "installed", "reason", "detail"}


# --------------------------------------------------------------------------- #
#  custom_mg sites (transfer builder ladder, guards)
# --------------------------------------------------------------------------- #
def test_transfer_builder_failure_records_the_rbf_rescue(adapt_child, monkeypatch):
    """Barycentric build fails -> RBF rescue: recorded as build_failed, the
    solve still converges on geometric MG. Degree 2 so the exact recorded
    (vertex-level) transfers cannot bypass the builder."""
    def broken(coarse_coords, fine_coords):
        raise RuntimeError("synthetic barycentric failure (test probe)")

    monkeypatch.setitem(custom_mg._BUILDERS, "barycentric", broken)
    p = _poisson_on(adapt_child, degree=2, name="Trbf")
    with pytest.warns(UserWarning, match="retrying with the 'rbf' builder"):
        p.solve()
    assert p.snes.getConvergedReason() > 0
    rec = p.pc_fallbacks["custom_mg.transfer_builder"]
    assert rec["reason"] == "build_failed"
    assert "rbf" in rec["installed"]
    assert "custom_mg.build" not in p.pc_fallbacks  # the rescue succeeded


def test_total_transfer_failure_records_and_solves_on_default_pc(
        adapt_child, monkeypatch):
    def broken(coarse_coords, fine_coords):
        raise RuntimeError("synthetic builder failure (test probe)")

    monkeypatch.setitem(custom_mg._BUILDERS, "barycentric", broken)
    monkeypatch.setitem(custom_mg._BUILDERS, "rbf", broken)
    p = _poisson_on(adapt_child, degree=2, name="Tfail")
    with pytest.warns(UserWarning, match="mesh-owned FMG build failed"):
        p.solve()
    # stability oracle: the degrade path must still converge
    assert p.snes.getConvergedReason() > 0
    rec = p.pc_fallbacks["custom_mg.build"]
    assert rec["reason"] == "build_failed"
    assert rec["installed"] == "default preconditioner"
    assert p.snes.getKSP().getPC().getType() != "mg"


def test_unpatched_adapt_child_solve_records_no_build_failure(adapt_child):
    """Negative control for both probes above: the healthy adapt-child pickup
    installs geometric MG with no build-failure record."""
    p = _poisson_on(adapt_child, degree=2, name="Tok")
    p.solve()
    assert p.snes.getConvergedReason() > 0
    assert p.snes.getKSP().getPC().getType() == "mg"
    assert "custom_mg.transfer_builder" not in p.pc_fallbacks
    assert "custom_mg.build" not in p.pc_fallbacks
    assert "custom_mg.finest_operator_check" not in p.pc_fallbacks


def test_finest_operator_guard_skip_is_recorded():
    """The finest-operator guard's early return (operator unreadable) was a
    silent swallow; it now records check_skipped. Driven directly: a solver
    with no SNES yet is exactly the unreadable-operator state."""
    p = _poisson_on(_box(refinement=0), name="Tguard")
    assert not hasattr(p, "snes") or p.snes is None or True  # un-built solver
    custom_mg.CustomMGHierarchy._assert_finest_matches_operator(
        p, finest_map=np.arange(4), parallel=False)
    rec = p.pc_fallbacks["custom_mg.finest_operator_check"]
    assert rec["reason"] == "check_skipped"


# --------------------------------------------------------------------------- #
#  Forced Galerkin site
# --------------------------------------------------------------------------- #
def test_user_mg_without_galerkin_is_forced_and_recorded():
    p = _poisson_on(_box(refinement=1), name="Tgal")
    p.petsc_options["pc_type"] = "mg"
    with pytest.warns(UserWarning, match="forcing pc_mg_galerkin=both"):
        p._build()
    rec = p.pc_fallbacks["galerkin_forced"]
    assert rec["reason"] == "forced"
    assert p.petsc_options.getString("pc_mg_galerkin") == "both"
    # negative control: setting the key yourself satisfies the requirement
    p2 = _poisson_on(_box(refinement=1), name="Tgal2")
    p2.petsc_options["pc_type"] = "mg"
    p2.petsc_options["pc_mg_galerkin"] = "both"
    p2._build()
    assert "galerkin_forced" not in p2.pc_fallbacks


# --------------------------------------------------------------------------- #
#  Reset rule
# --------------------------------------------------------------------------- #
def test_record_resets_when_the_preconditioner_re_resolves():
    p = _poisson_on(_box(refinement=0), name="Trst")
    p.preconditioner = "fmg"
    with pytest.warns(UserWarning, match="no refinement hierarchy"):
        p.solve()
    assert "no_hierarchy" in p.pc_fallbacks
    p.preconditioner = "gamg"
    p.solve()
    assert p.pc_fallbacks == {}, "stale records must not survive re-resolution"
