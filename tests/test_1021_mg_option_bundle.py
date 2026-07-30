"""The geometric-multigrid option bundle has ONE owner, and every route reads it.

Three routes reach a multigrid velocity block — native (DMPlex refinement),
custom-P on the standard solve path, and custom-P through the rotated free-slip
path — and they are the same preconditioner, not alternatives: custom-P is
mandatory wherever native cannot go (rotated BCs, ``adapt()`` children). They
drifted apart because the bundle was written in two places (#468): the custom-P
route ran richardson smoothing at an iteration count nobody set, inherited from
whatever had last written that options prefix (3 left over from the GAMG bundle
on the standard path, PETSc's own default of 2 on the rotated path).

These tests read the configuration back off the LIVE PETSc objects after setup,
not out of the options database, so they check what actually runs.

The one legitimate per-route difference is the coarse solve: the
Galerkin-coarsened *rotated* velocity block inherits the rigid-rotation null
space, so redundant/LU hits a zero pivot there and it uses SVD (the #306 fix).
That difference is asserted, not tolerated — if it disappears, the rotated coarse
solve has silently reverted.

Also here: the rotated path must pick up a MESH-OWNED hierarchy (the coarse tail
an ``adapt()`` child carries), which it previously ignored, silently solving on
GAMG (#467).
"""
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg, multigrid_options, rotated_bc

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

RES = 0.15
R_IN, R_OUT = 0.5, 1.0


# --------------------------------------------------------------------------- #
#  The owner itself — no solve needed
# --------------------------------------------------------------------------- #
def test_every_bundle_sets_the_smoother_iteration_count():
    """The drift mechanism: a bundle that omits ``mg_levels_ksp_max_it`` inherits
    it from whatever last wrote that prefix. Every bundle must SET it."""
    bundles = [multigrid_options.gamg_bundle()]
    bundles += [multigrid_options.geometric_mg_bundle(coarse=c)
                for c in multigrid_options.GEOMETRIC_MG_COARSE_SOLVERS]
    for bundle in bundles:
        assert "mg_levels_ksp_max_it" in bundle.settings
        assert "pc_type" in bundle.settings


def test_bundles_clear_each_others_keys():
    """Bundles share an options prefix, so switching between them must leave no
    key behind — every key a bundle does not set, a sibling does, and it must
    appear in that bundle's stale list."""
    bundles = [multigrid_options.gamg_bundle()]
    bundles += [multigrid_options.geometric_mg_bundle(coarse=c)
                for c in multigrid_options.GEOMETRIC_MG_COARSE_SOLVERS]
    owned = set().union(*(set(b.settings) for b in bundles))
    for bundle in bundles:
        assert set(bundle.stale) == owned - set(bundle.settings)


# --------------------------------------------------------------------------- #
#  The three live routes
# --------------------------------------------------------------------------- #
def _stokes(mesh, tag, rotated):
    """A buoyancy-driven annulus Stokes solve: fixed inner boundary, and an outer
    boundary that is either an ordinary essential BC or rotated free-slip."""
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    rhat = sympy.Matrix([[x / r, y / r]])
    v = uw.discretisation.MeshVariable(f"V{tag}", mesh, mesh.dim, degree=2,
                                       continuous=True)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    blob = sympy.exp(-(((x - 0.75) ** 2 + y**2) / 0.05))
    s.bodyforce = sympy.Matrix([[50.0 * blob * x / r, 50.0 * blob * y / r]])
    s.add_essential_bc((0.0, 0.0), "Lower")
    if rotated:
        s.add_rotated_freeslip_bc(0.0, "Upper", normal=rhat)
    else:
        s.add_essential_bc((sympy.oo, 0.0), "Upper")
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-8
    return s


def _mg_config(vel_pc):
    """Read the multigrid configuration off a live velocity sub-PC. The finest
    smoother is representative; level 0 is the coarse solve."""
    assert vel_pc.getType() == "mg"
    nlev = vel_pc.getMGLevels()
    smoother = vel_pc.getMGSmoother(nlev - 1)
    coarse = vel_pc.getMGCoarseSolve()
    return {
        "levels": nlev,
        "mg_type": str(vel_pc.getMGType()),
        "smoother_ksp": smoother.getType(),
        "smoother_pc": smoother.getPC().getType(),
        "smoother_max_it": smoother.getTolerances()[3],
        "smoother_norm": str(smoother.getNormType()),
        "coarse_ksp": coarse.getType(),
        "coarse_pc": coarse.getPC().getType(),
    }


def _annulus(cell_size):
    return uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT,
                              cellSize=cell_size, qdegree=3)


def _native_config():
    """Route A: native geometric FMG on a refined DMPlex hierarchy."""
    s = _stokes(uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT,
                                   cellSize=2 * RES, qdegree=3, refinement=1),
                "nat", rotated=False)
    s.solve()
    return _mg_config(s.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC())


def _custom_standard_config():
    """Route B: custom-P FMG reached through the standard solve path."""
    s = _stokes(_annulus(RES), "std", rotated=False)
    custom_mg.set_custom_fmg(s, [_annulus(2 * RES)], field_id=0)
    s.solve()
    return _mg_config(s.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC())


def _custom_rotated_config(monkeypatch):
    """Route C: custom-P FMG reached through the rotated free-slip path.

    The rotated KSP is self-contained and is destroyed at the end of the solve,
    so the configuration is captured from inside the solve — there is no live PC
    left to interrogate afterwards.
    """
    s = _stokes(_annulus(RES), "rot", rotated=True)
    custom_mg.set_custom_fmg(s, [_annulus(2 * RES)], field_id=0)

    real_solve = rotated_bc._solve_rotated_iterative
    captured = {}

    def capture(solver, Ahat, bhat, Q, Qt, normal_rows, **kw):
        result = real_solve(solver, Ahat, bhat, Q, Qt, normal_rows, **kw)
        captured.setdefault("config", _mg_config(
            result[2]["pc"].getFieldSplitSubKSP()[0].getPC()))
        return result

    monkeypatch.setattr(rotated_bc, "_solve_rotated_iterative", capture)
    s.solve()
    assert s._rotated_freeslip_info["velocity_pc"] == "custom-FMG"
    return captured["config"]


def test_all_three_routes_share_one_bundle(monkeypatch):
    """Native, standard custom-P and rotated custom-P must smooth identically —
    same Krylov smoother, same preconditioner, same iteration count, same cycle.
    Only the coarse solve legitimately differs."""
    native = _native_config()
    standard = _custom_standard_config()
    rotated = _custom_rotated_config(monkeypatch)

    shared = ("mg_type", "smoother_ksp", "smoother_pc", "smoother_max_it",
              "smoother_norm", "coarse_ksp")
    for key in shared:
        assert native[key] == standard[key] == rotated[key], (
            f"route drift on {key!r}: native={native[key]!r} "
            f"standard={standard[key]!r} rotated={rotated[key]!r}")

    # The measured bundle, not just self-consistency: gmres/sor at four
    # iterations with no norm computation (see multigrid_options for why).
    assert native["smoother_ksp"] == "gmres"
    assert native["smoother_pc"] == "sor"
    assert native["smoother_max_it"] == 4

    # The ONE deliberate per-route difference (#306): the rotated coarse operator
    # inherits the rigid-rotation null space, where redundant/LU hits a zero pivot.
    assert native["coarse_pc"] == "redundant"
    assert standard["coarse_pc"] == "redundant"
    assert rotated["coarse_pc"] == "svd"


def test_rotated_picks_up_a_mesh_owned_hierarchy():
    """#467: a coarse tail owned by the MESH — what ``adapt()`` leaves on a
    refinement child — must drive geometric MG under rotated free-slip. It used
    to be ignored, and the solve fell back to GAMG silently.

    The tail is attached by hand rather than by running ``adapt()``: it is the
    same attribute the child carries, and what is under test is whether the
    rotated dispatch consults it.
    """
    mesh = _annulus(RES)
    mesh._custom_mg_coarse_meshes = [_annulus(2 * RES)]
    mesh._custom_mg_builder = "barycentric"

    s = _stokes(mesh, "own", rotated=True)
    s.solve()

    assert s._rotated_freeslip_info["velocity_pc"] == "custom-FMG"
    assert s._rotated_freeslip_info["velocity_pc_type"] == "mg"


def _velocity_mg_config(s):
    """(smoother ksp, smoother max_it, coarse pc) off the live velocity sub-PC."""
    pc = s.snes.getKSP().getPC()
    pc.setUp()                                    # MUST precede getFieldSplitSubKSP
    vpc = pc.getFieldSplitSubKSP()[0].getPC()
    assert vpc.getType() == "mg" and vpc.getMGLevels() > 1, (
        f"expected geometric MG on the velocity block, got {vpc.getType()}")
    smoother = vpc.getMGSmoother(vpc.getMGLevels() - 1)
    return (smoother.getType(), smoother.getTolerances()[3],
            vpc.getMGCoarseSolve().getPC().getType())


@pytest.mark.parametrize("preconditioner", [None, "fmg"])
def test_user_set_bundle_keys_are_honoured(preconditioner):
    """A user who sets a bundle key must get it, and must keep the managed values
    for the keys they left alone — per key, not all-or-nothing.

    The bundle used to be applied wholesale on every rebuild. The only escape was
    the ``_pc_user_override`` latch, which keys on ``pc_type`` ALONE, so setting
    ``mg_levels_ksp_max_it`` was silently discarded unless the user also set
    ``pc_type`` to the value it already had. Explicit ``preconditioner="fmg"`` was
    worse — it skips that latch entirely, so the clearer the request the less
    control it carried, which is why both modes are parametrised here.
    """
    mesh = uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT,
                              cellSize=2 * RES, qdegree=3, refinement=1)
    s = _stokes(mesh, f"u{'f' if preconditioner else 'a'}", rotated=False)
    if preconditioner is not None:
        s.preconditioner = preconditioner
    # set TWO of the bundle's keys and leave the rest to the framework
    s.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 6
    s.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    s.solve()
    s.solve()                                     # a rebuild must not clobber it

    smoother, max_it, coarse = _velocity_mg_config(s)
    assert max_it == 6, "user-set smoother iteration count was overwritten"
    assert coarse == "svd", "user-set coarse solver was overwritten"
    # ...and the key the user did NOT set still carries the measured default
    assert smoother == "gmres", (
        "respecting one bundle key must not abandon the rest of the bundle")


def test_unset_bundle_keys_keep_the_managed_defaults():
    """The control for the test above: with nothing set, the whole managed bundle
    applies. Without this, 'user-set keys are honoured' could pass trivially by
    never applying the bundle at all — which is exactly how the first attempt at
    this failed (the routes silently reverted to GAMG)."""
    mesh = uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT,
                              cellSize=2 * RES, qdegree=3, refinement=1)
    s = _stokes(mesh, "dflt", rotated=False)
    s.solve()
    s.solve()
    assert _velocity_mg_config(s) == ("gmres", 4, "redundant")


def _strategy_mg_config(strategy):
    mesh = uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT,
                              cellSize=2 * RES, qdegree=3, refinement=1)
    s = _stokes(mesh, f"s{strategy or 'none'}"[:6], rotated=False)
    if strategy is not None:
        s.strategy = strategy
    s.solve()
    s.solve()
    return _velocity_mg_config(s)


def test_strategy_selects_a_real_smoother_variant():
    """``solver.strategy`` must actually change the smoother.

    ``"fast"`` and ``"robust"`` were accepted-and-inert for a long time: validated
    on input, then configured identically to ``"default"``. A property that checks
    your value and then ignores it is the same defect class as #477/#478 — invisible,
    because the solve still converges. They now select a measured variant:
    ``"robust"`` is gmres/4, ``"fast"`` is richardson/3.
    """
    assert _strategy_mg_config("fast")[:2] == ("richardson", 3)
    assert _strategy_mg_config("robust")[:2] == ("gmres", 4)


def test_default_strategy_does_not_change_behaviour():
    """The control: ``"default"`` must reproduce the framework default exactly, so
    filling the strategy axis moves nobody's results."""
    assert _strategy_mg_config("default") == _strategy_mg_config(None)


def test_a_user_key_still_beats_the_strategy():
    """The two layers compose in the right order: an explicit option the user wrote
    outranks the strategy's choice of variant."""
    mesh = uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT,
                              cellSize=2 * RES, qdegree=3, refinement=1)
    s = _stokes(mesh, "sxu", rotated=False)
    s.strategy = "fast"                                   # would ask for richardson/3
    s.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
    s.solve()
    s.solve()
    smoother, max_it, _ = _velocity_mg_config(s)
    assert smoother == "chebyshev", "the user's smoother lost to the strategy"
    assert max_it == 3, "the strategy should still own the keys the user left alone"


def test_strategy_reports_what_it_resolved_to():
    """``solver.strategy`` reports the preconditioner it configured, so "what am I
    running?" does not require knowing which nine PETSc keys to look up (#484).

    It must still BE the string: comparisons, formatting and serialisation of a
    strategy name are unchanged.
    """
    mesh = uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT,
                              cellSize=2 * RES, qdegree=3, refinement=1)
    s = _stokes(mesh, "rep", rotated=False)

    # before any solve: must NOT present the __init__ defaults as resolved
    assert s.strategy == "default"
    assert f"{s.strategy}" == "default"
    assert "not resolved yet" in repr(s.strategy)

    s.solve()
    assert s.strategy == "default"                    # still the plain string
    summary = repr(s.strategy)
    assert "geometric multigrid (2 levels)" in summary
    assert "gmres" in summary and "sor" in summary
    assert "not resolved yet" not in summary

    # and the machine-readable form agrees with it
    assert s.preconditioner_settings["mg_levels_ksp_type"] == "gmres"
    assert s.preconditioner_settings["mg_levels_ksp_max_it"] == "4"


def test_strategy_report_names_a_user_override():
    """A key the user took over must be called out, not silently folded into the
    summary — otherwise the report reintroduces the ambiguity it exists to remove."""
    mesh = uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT,
                              cellSize=2 * RES, qdegree=3, refinement=1)
    s = _stokes(mesh, "rov", rotated=False)
    s.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 6
    s.solve()
    summary = repr(s.strategy)
    assert "overridden by the user" in summary
    assert "mg_levels_ksp_max_it=6" in summary
    assert s.preconditioner_settings["mg_levels_ksp_max_it"] == "6"


def test_rotated_fmg_survives_repeated_newton_increments():
    """The rotated path applies its bundle under a per-solve options prefix and
    then drops the keys again, so the global database stays bounded under
    time-stepping. That is only safe if nothing re-reads them: a later
    ``setFromOptions`` on the velocity sub-PC would find ``pc_type`` gone and
    silently abandon the multigrid. Several Newton increments, each re-solving
    on a refreshed operator, must all keep ``pc=mg``."""
    mesh = _annulus(RES)
    s = _stokes(mesh, "nl", rotated=True)

    # power-law viscosity: a genuinely nonlinear solve, several increments
    x, y = mesh.X
    v = s.Unknowns.u
    grad = sympy.Matrix([[v.sym[0].diff(x), v.sym[0].diff(y)],
                         [v.sym[1].diff(x), v.sym[1].diff(y)]])
    edot = 0.5 * (grad + grad.T)
    eII = sympy.sqrt(0.5 * (edot[0, 0] ** 2 + edot[1, 1] ** 2)
                     + edot[0, 1] ** 2 + 1.0e-12)
    s.constitutive_model.Parameters.shear_viscosity_0 = eII ** (1.0 / 3.0 - 1.0)
    s.consistent_jacobian = True
    s.tolerance = 1.0e-7
    custom_mg.set_custom_fmg(s, [_annulus(2 * RES)], field_id=0)
    s.solve()

    info = s._rotated_freeslip_info
    assert len(info["ksp_its"]) > 1, "not a multi-increment solve — test is vacuous"
    assert info["velocity_pc"] == "custom-FMG"
    assert info["velocity_pc_type"] == "mg"


def test_rotated_without_a_hierarchy_falls_back_to_gamg():
    """The negative control for the test above: no hierarchy anywhere still
    reports GAMG, so ``custom-FMG`` above is a real pickup and not a label that
    is always set."""
    s = _stokes(_annulus(RES), "none", rotated=True)
    s.solve()
    assert s._rotated_freeslip_info["velocity_pc"] == "GAMG"
