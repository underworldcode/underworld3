"""How a solve went, not only that it ran.

The ``solve`` event goes into the transcript before the solve, because the
order the operators ran in is what the transcript exists to preserve. The
outcome is known only afterwards, and lands on that same event.

Without it a transcript says a run solved three hundred times and nothing
about the fifty that diverged — a record of a run rather than an account of
it. Warnings are the other half: "the velocity block fell back to gamg"
changes what the numbers mean.
"""

import warnings

import sympy

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _fresh_model():
    import underworld3 as uw

    uw.reset_default_model()
    return uw, uw.get_default_model()


def _poisson(uw, mesh, name):
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=2)
    solver = uw.systems.Poisson(mesh, u_Field=T)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = 1.0
    solver.f = 1.0
    solver.add_dirichlet_bc(0.0, "Top")
    solver.add_dirichlet_bc(0.0, "Bottom")
    solver.petsc_options.delValue("ksp_monitor")
    return solver


@pytest.fixture(scope="module")
def mesh():
    import underworld3 as uw

    return uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )


def test_a_converged_solve_records_how_it_converged(mesh):
    uw, model = _fresh_model()
    solver = _poisson(uw, mesh, "T_ok")
    model.tracker.time, model.tracker.step = 0.0, 0

    with model.step(0.1):
        solver.solve()

    solve = next(e for e in model.transcript[0].events if e["kind"] == "solve")
    assert solve["converged"] is True
    assert solve["reason"].startswith("CONVERGED"), solve["reason"]
    assert solve["nl_its"] >= 1
    assert solve["ksp_its"] >= 1
    assert solve["fnorm"] >= 0.0


def test_a_diverged_solve_is_recorded_as_diverged(mesh):
    """The case the record exists for: the solve returns, the script carries
    on, and only the transcript knows the answer is not converged."""
    uw, model = _fresh_model()
    solver = _poisson(uw, mesh, "T_bad")
    # One Krylov iteration against a tolerance it cannot reach.
    solver.petsc_options["ksp_max_it"] = 1
    solver.petsc_options["ksp_rtol"] = 1.0e-30
    solver.petsc_options["snes_max_it"] = 1
    model.tracker.time, model.tracker.step = 0.0, 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with model.step(0.1):
            solver.solve()

    solve = next(e for e in model.transcript[0].events if e["kind"] == "solve")
    assert solve["converged"] is False, solve
    assert solve["reason"].startswith("DIVERGED"), solve["reason"]


def test_the_text_transcript_says_diverged_in_the_outcome_column(mesh, tmp_path):
    """A reader watching the log must see it without parsing JSON."""
    uw, model = _fresh_model()
    solver = _poisson(uw, mesh, "T_text")
    solver.petsc_options["ksp_max_it"] = 1
    solver.petsc_options["ksp_rtol"] = 1.0e-30
    solver.petsc_options["snes_max_it"] = 1
    model.transcript_file = tmp_path / "transcript.log"
    model.tracker.time, model.tracker.step = 0.0, 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with model.step(0.1):
            solver.solve()

    text = (tmp_path / "transcript.log").read_text()
    assert "DIVERGED" in text, text
    assert "!!" in text, "the failing solve is not named under the step"


def test_a_warning_inside_a_step_is_recorded_and_still_shown(mesh):
    """The transcript gets a copy of the warning, not the only copy: the shim
    delegates, so the user's filters and pytest's capture keep working."""
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0

    with pytest.warns(UserWarning, match="keep this"):
        with model.step(0.1):
            warnings.warn("keep this one", UserWarning)

    noted = [e for e in model.transcript[0].events if e["kind"] == "warning"]
    assert len(noted) == 1, noted
    assert noted[0]["name"] == "UserWarning"
    assert "keep this one" in noted[0]["message"]
    assert ":" in noted[0]["where"], "the warning does not say where it came from"


def test_the_warning_hook_is_removed_when_the_step_ends(mesh):
    """Including when the step is abandoned — a shim left installed would
    keep writing into a step that is no longer open."""
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0

    before = warnings.showwarning
    with model.step(0.1):
        assert warnings.showwarning is not before
    assert warnings.showwarning is before

    with pytest.raises(RuntimeError):
        with model.step(0.1):
            raise RuntimeError("abandoned")
    assert warnings.showwarning is before


def test_a_warning_outside_a_step_is_not_recorded(mesh):
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0

    with pytest.warns(UserWarning):
        warnings.warn("before any step", UserWarning)

    with model.step(0.1):
        pass

    assert [e for e in model.transcript[0].events if e["kind"] == "warning"] == []


# ---------------------------------------------------------------------------
# One solve, one event (found in review)
# ---------------------------------------------------------------------------


def test_a_parameter_change_before_a_solve_does_not_add_a_second_solve_event(mesh):
    """``_update_constants`` runs on a Parameter change as well as before a
    solve. Only the solve may write a solve event; a run that changes a
    parameter every step must not read as two solves per step."""
    uw, model = _fresh_model()
    solver = _poisson(uw, mesh, "T_param")
    kappa = uw.expression(r"\kappa_p", 1.0, "diffusivity")
    solver.constitutive_model.Parameters.diffusivity = kappa
    model.tracker.time, model.tracker.step = 0.0, 0

    with model.step(0.1):
        solver.solve()
    with model.step(0.1):
        kappa.sym = sympy.Float(2.0)          # a coefficient change, not a solve
        solver.solve()

    for entry in model.transcript:
        solves = [e for e in entry.events if e["kind"] == "solve"]
        assert len(solves) == 1, [e["name"] for e in entry.events]
        assert "converged" in solves[0], "the one event carries the outcome"


def test_a_continuation_solve_is_one_event_with_its_outcome(mesh):
    """The Picard->Newton continuation toggles alpha through
    ``_update_constants`` mid-solve; found writing four events per solve."""
    import sympy as sp

    uw, model = _fresh_model()
    V = uw.discretisation.MeshVariable("V_cont_ev", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable("P_cont_ev", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0 / (1 + stokes.Unknowns.Einv2)
    stokes.bodyforce = sp.Matrix([0.0, -1.0])
    for b in ("Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), b)
    for b in ("Left", "Right"):
        stokes.add_dirichlet_bc((0.0, sp.oo), b)
    stokes.petsc_options.delValue("ksp_monitor")
    stokes.consistent_jacobian = "continuation"
    model.tracker.time, model.tracker.step = 0.0, 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with model.step(0.1):
            stokes.solve()

    solves = [e for e in model.transcript[0].events if e["kind"] == "solve"]
    assert len(solves) == 1, [e["name"] for e in model.transcript[0].events]
    assert solves[0]["converged"] is True


def test_the_warning_hook_is_left_alone_if_the_block_replaced_it(mesh):
    """A hook installed inside the block is the block's business."""
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0
    before = warnings.showwarning

    def mine(*args, **kwargs):
        pass

    with model.step(0.1):
        warnings.showwarning = mine
    assert warnings.showwarning is mine
    warnings.showwarning = before


def test_an_old_style_four_argument_hook_still_receives_the_warning(mesh):
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0
    before = warnings.showwarning
    got = []

    def old_style(message, category, filename, lineno):
        got.append(str(message))

    warnings.showwarning = old_style
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with model.step(0.1):
                warnings.warn("four args", UserWarning)
    finally:
        warnings.showwarning = before
    assert got == ["four args"]
    assert [e for e in model.transcript[0].events if e["kind"] == "warning"][0]["rank"] == 0
