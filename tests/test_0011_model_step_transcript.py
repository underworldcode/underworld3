"""The step transcript — ``with model.step(dt):``.

One timestep as a transaction. Three guarantees, one test each:

  * the clock reads the END of the interval inside the block, because an
    implicit scheme centres its residual there;
  * the advance commits only on clean exit, so an abandoned step leaves the
    clock alone;
  * everything the block did is recorded, in order.

Documented in ``docs/developer/guides/HOW-TO-WRITE-UW3-SCRIPTS.md``.
"""

import numpy as np
import pytest
import sympy

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


def test_the_clock_reads_the_end_of_the_interval_inside_the_block():
    """An implicit residual is centred at t + dt, so that is where a
    time-dependent coefficient must be evaluated. Committing only on exit
    would evaluate every one of them a step late."""
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 2.0, 7

    with model.step(0.25) as step:
        assert step.t0 == pytest.approx(2.0)
        assert step.t1 == pytest.approx(2.25)
        assert model.tracker.time == pytest.approx(2.25)

    assert model.tracker.time == pytest.approx(2.25)
    assert model.tracker.step == 8


def test_an_abandoned_step_does_not_commit():
    """A step rejected on a Courant check, or one that raises, must leave the
    clock where it was — the caller should not have to unwind a counter."""
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 1.0, 3

    with pytest.raises(RuntimeError, match="Courant"):
        with model.step(0.5):
            raise RuntimeError("Courant too large")

    assert model.tracker.time == pytest.approx(1.0)
    assert model.tracker.step == 3
    assert model.transcript == []
    assert model.open_step is None


def test_the_transcript_records_what_ran_and_in_what_order():
    """The point of the record: it answers what a step actually did, without
    the script being instrumented."""
    uw, model = _fresh_model()
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    first = _poisson(uw, mesh, "T_one")
    second = _poisson(uw, mesh, "T_two")
    model.tracker.time, model.tracker.step = 0.0, 0

    with model.step(0.1, label="a step"):
        first.solve()
        second.solve()

    assert len(model.transcript) == 1
    entry = model.transcript[0]
    assert entry.label == "a step"
    assert entry.completed
    names = [e["name"] for e in entry.events if e["kind"] == "solve"]
    assert names == ["SNES_Poisson(T_one)", "SNES_Poisson(T_two)"], names
    # named by what they solve, so the record is auditable
    assert "T_one" in repr(entry)


def test_steps_do_not_nest():
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0
    with model.step(0.1):
        with pytest.raises(RuntimeError, match="already open"):
            with model.step(0.1):
                pass


def test_a_script_without_steps_is_unaffected():
    """Opening a step is optional; the recording hooks are no-ops without one."""
    uw, model = _fresh_model()
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    solver = _poisson(uw, mesh, "T_free")
    solver.solve()
    assert model.open_step is None
    assert model.transcript == []
    assert np.abs(np.asarray(solver.u.data)).max() > 1.0e-3


def test_the_transcript_is_bounded():
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0
    model.transcript_limit = 3
    for _ in range(7):
        with model.step(0.1):
            pass
    assert len(model.transcript) == 3
    assert [e.index for e in model.transcript] == [4, 5, 6]


# ---------------------------------------------------------------------------
# Recording: a step keeps the state it started from, so the run can be replayed
# ---------------------------------------------------------------------------


def _advdiff(uw, mesh):
    import sympy

    T = uw.discretisation.MeshVariable("T_record", mesh, 1, degree=2)
    V = uw.discretisation.MeshVariable("V_record", mesh, 2, degree=2)
    x, y = mesh.X
    V.array[:, 0, :] = np.asarray(
        uw.function.evaluate(sympy.Matrix([[-(y - 0.5), (x - 0.5)]]), V.coords)
    ).reshape(-1, 2)
    T.array[:, 0, 0] = np.asarray(
        uw.function.evaluate(sympy.exp(-(((x - 0.3) ** 2 + (y - 0.5) ** 2) / 0.02)), T.coords)
    ).ravel()
    solver = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=V.sym)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = 1.0e-4
    solver.petsc_options.delValue("ksp_monitor")
    return solver, T


def test_recording_is_off_by_default():
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0
    with model.step(0.1):
        pass
    assert model.transcript[0].restorable is False
    assert model.restore_points == []


def test_rewind_undoes_a_step_exactly():
    """Fields, history and clock all come back, and the step is undone."""
    uw, model = _fresh_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    solver, T = _advdiff(uw, mesh)
    model.tracker.time, model.tracker.step = 0.0, 0
    model.record_every = 1

    for _ in range(2):
        with model.step(0.02):
            solver.solve(timestep=0.02)

    at_two = np.array(T.array)
    assert model.tracker.step == 2

    # take a third step, then undo it
    with model.step(0.02):
        solver.solve(timestep=0.02)
    assert model.tracker.step == 3
    assert not np.allclose(np.array(T.array), at_two)

    model.rewind()

    assert np.array_equal(np.array(T.array), at_two), "fields did not come back"
    assert model.tracker.step == 2, "the clock did not come back"
    assert model.tracker.time == pytest.approx(0.04)
    assert len(model.transcript) == 2, "the transcript still claims the undone step"


def test_replaying_a_rewound_step_reproduces_it():
    """The property replay debugging rests on: the same step, taken twice from
    the same state, gives the same answer."""
    uw, model = _fresh_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    solver, T = _advdiff(uw, mesh)
    model.tracker.time, model.tracker.step = 0.0, 0
    model.record_every = 1

    with model.step(0.02):
        solver.solve(timestep=0.02)
    first = np.array(T.array)

    model.rewind()
    with model.step(0.02):
        solver.solve(timestep=0.02)
    second = np.array(T.array)

    assert np.array_equal(first, second), (
        "replaying a step from its own snapshot did not reproduce it"
    )


def test_the_record_is_bounded_but_the_transcript_survives():
    """Old steps lose their snapshot and keep their record, so the account of
    what happened outlives the state."""
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0
    model.record_every = 1
    model.record_limit = 2

    for _ in range(5):
        with model.step(0.1):
            pass

    assert len(model.transcript) == 5
    assert [e.index for e in model.restore_points] == [3, 4]


def test_rewind_without_a_record_says_what_to_do():
    uw, model = _fresh_model()
    model.tracker.time, model.tracker.step = 0.0, 0
    with model.step(0.1):
        pass
    with pytest.raises(RuntimeError, match="record_every"):
        model.rewind()


# ---------------------------------------------------------------------------
# Recording, not judging
# ---------------------------------------------------------------------------


def test_a_history_that_advances_twice_is_recorded_twice():
    """Two solves inside one step take the physical step twice. The solve
    counter and the timestep history look identical to a single step, so the
    transcript is the only place it is visible.

    The step does NOT judge that. Whether two shifts in a bar are a mistake or
    legitimate sub-cycling is a reading of the transcript, made by a later pass
    that can look across bars — not a rule asserted inside the loop, which can
    only see one bar and has to guess."""
    uw, model = _fresh_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    solver, T = _advdiff(uw, mesh)
    model.tracker.time, model.tracker.step = 0.0, 0

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with model.step(0.02):
            solver.solve(timestep=0.02)
            solver.solve(timestep=0.02)   # the same step, taken twice

    entry = model.transcript[0]
    shifts = [e for e in entry.events if e["kind"] == "history_shift"]
    assert len(shifts) == 2, "the transcript must hold both shifts, in order"
    assert [e["kind"] for e in entry.events] == [
        "solve", "history_shift", "solve", "history_shift"
    ]


def test_one_solve_per_step_is_quiet():
    """The ordinary loop records one shift and says nothing about it."""
    uw, model = _fresh_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    solver, T = _advdiff(uw, mesh)
    model.tracker.time, model.tracker.step = 0.0, 0

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(3):
            with model.step(0.02):
                solver.solve(timestep=0.02)

    assert len(model.transcript) == 3


def test_the_transcript_shows_the_history_that_moved():
    """The record names which history advanced, not just that a solve ran."""
    uw, model = _fresh_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    solver, T = _advdiff(uw, mesh)
    model.tracker.time, model.tracker.step = 0.0, 0

    with model.step(0.02):
        solver.solve(timestep=0.02)

    kinds = [e["kind"] for e in model.transcript[0].events]
    assert "solve" in kinds and "history_shift" in kinds
    shift = next(e for e in model.transcript[0].events if e["kind"] == "history_shift")
    assert shift["dt"] == pytest.approx(0.02)
    assert "T_record" in shift["name"], shift["name"]
