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
