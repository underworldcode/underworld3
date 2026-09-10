"""The journal, written down.

``model.journal`` is what a run can still undo: bounded, in memory, gone with
the process. ``model.journal_file`` is what the run did: one JSON object per
line, appended and flushed as each step closes.

The two differ deliberately, and the differences are what the tests below pin.
An abandoned step appears in the file and not in memory — it is the part of a
run's history that is otherwise invisible. A step aged out by ``journal_limit``
leaves memory and stays in the file. And one object per line means a run that
is killed keeps everything up to the moment it died.
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import json


def _model(tmp_path, units=False, name="run.journal.jsonl", fmt=None):
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    if units:
        model.set_reference_quantities(
            domain_depth=uw.quantity(500, "km"),
            material_viscosity=uw.quantity(1e21, "Pa*s"),
            lithostatic_pressure=uw.quantity(3300 * 9.81 * 500e3, "Pa"),
        )
    path = tmp_path / name
    model.journal_file = str(path)
    if fmt is not None:
        model.journal_format = fmt
    model.tracker.time = uw.quantity(0.0, "Myr") if units else 0.0
    model.tracker.step = 0
    return uw, model, path


def test_a_completed_step_is_one_line(tmp_path):
    uw, model, path = _model(tmp_path)

    with model.step(0.25, label="convect"):
        pass

    lines = path.read_text().splitlines()
    assert len(lines) == 2, "expected a run header and one step"

    header, step = (json.loads(line) for line in lines)
    assert header["kind"] == "run"

    wall = step.pop("wall")
    assert wall >= 0.0, "a step should record how long it took"
    assert step == {
        "kind": "step",
        "index": 0,
        "label": "convect",
        "t0": 0.0,
        "t1": 0.25,
        "dt": 0.25,
        "completed": True,
        "restorable": False,
        "events": [],
    }


def test_an_abandoned_step_is_in_the_file_and_not_in_memory(tmp_path):
    uw, model, path = _model(tmp_path)

    with model.step(0.25, label="fine"):
        pass
    with pytest.raises(RuntimeError):
        with model.step(9.0, label="too big"):
            raise RuntimeError("courant")

    assert [e.label for e in model.journal] == ["fine"]

    steps = [json.loads(line) for line in path.read_text().splitlines()][1:]
    assert [s["label"] for s in steps] == ["fine", "too big"]
    assert [s["completed"] for s in steps] == [True, False]

    # The clock did not move for the abandoned step, so its t0 is the previous
    # step's t1 and nothing after it is shifted.
    assert steps[1]["t0"] == pytest.approx(0.25)
    assert model.tracker.time == pytest.approx(0.25)


def test_an_abandoned_step_does_not_retain_its_snapshot(tmp_path):
    """It is unreachable — the journal never holds it — so it must not be kept."""
    uw, model, path = _model(tmp_path)
    model.record_every = 1

    captured = {}
    with pytest.raises(RuntimeError):
        with model.step(0.25):
            captured["open"] = model.open_step
            raise RuntimeError("nope")

    assert captured["open"].snapshot is None
    steps = [json.loads(line) for line in path.read_text().splitlines()][1:]
    assert steps[0]["restorable"] is False


def test_a_step_aged_out_of_memory_stays_in_the_file(tmp_path):
    uw, model, path = _model(tmp_path)
    model.journal_limit = 2

    for _ in range(5):
        with model.step(0.1, label="convect"):
            pass

    assert len(model.journal) == 2
    steps = [json.loads(line) for line in path.read_text().splitlines()][1:]
    assert len(steps) == 5
    assert [s["index"] for s in steps] == [0, 1, 2, 3, 4]


def test_events_are_recorded_in_order(tmp_path):
    uw, model, path = _model(tmp_path)

    with model.step(0.1):
        model._record_step_event("solve", "SNES_Stokes(v)")
        model._record_step_event("history_shift", "EulerianSUPG(T)", dt=0.1)
        model._record_step_event("solve", "SNES_AdvectionDiffusion(T)")

    step = json.loads(path.read_text().splitlines()[-1])
    assert [(e["kind"], e["name"]) for e in step["events"]] == [
        ("solve", "SNES_Stokes(v)"),
        ("history_shift", "EulerianSUPG(T)"),
        ("solve", "SNES_AdvectionDiffusion(T)"),
    ]
    assert step["events"][1]["dt"] == pytest.approx(0.1)


def test_dimensional_values_survive_the_round_trip(tmp_path):
    uw, model, path = _model(tmp_path, units=True)

    dt = uw.quantity(1.5, "Myr")
    with model.step(dt, label="sink"):
        pass

    header, step = (json.loads(line) for line in path.read_text().splitlines())
    assert header["scales"]["length"]["units"] == "meter"
    assert header["scales"]["length"]["magnitude"] == pytest.approx(500e3, rel=1e-9)
    assert step["dt"] == {"magnitude": pytest.approx(1.5), "units": "megayear"}
    assert step["t1"] == {"magnitude": pytest.approx(1.5), "units": "megayear"}


def test_clear_journal_opens_a_new_run_in_the_same_file(tmp_path):
    """An inversion runs the forward model many times; one file, many runs."""
    uw, model, path = _model(tmp_path)

    for run in range(3):
        model.clear_journal()
        model.tracker.time = 0.0
        model.tracker.step = 0
        for _ in range(run + 1):
            with model.step(0.1, label=f"run{run}"):
                pass

    runs = uw.read_journal(path)
    # The first header is written when journal_file is set; clear_journal adds
    # one per run, so the leading empty section is expected.
    populated = [r for r in runs if r["steps"]]
    assert [len(r["steps"]) for r in populated] == [1, 2, 3]
    assert [r["steps"][0]["label"] for r in populated] == ["run0", "run1", "run2"]


def test_a_truncated_final_line_does_not_lose_the_rest(tmp_path):
    """A run killed mid-write: everything before the partial line is intact."""
    uw, model, path = _model(tmp_path)

    for _ in range(3):
        with model.step(0.1, label="convect"):
            pass

    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"kind": "step", "index": 3, "lab')

    runs = uw.read_journal(path)
    assert len(runs) == 1
    assert [s["index"] for s in runs[0]["steps"]] == [0, 1, 2]


def test_logging_is_off_by_default(tmp_path):
    uw, model, path = _model(tmp_path)
    model.journal_file = None

    assert model.journal_file is None
    before = path.read_text()
    with model.step(0.1):
        pass
    assert path.read_text() == before, "writing continued after logging was off"
    assert len(model.journal) == 1, "the in-memory journal must be unaffected"


# ---------------------------------------------------------------------------
# The text format — what a run is watched through
# ---------------------------------------------------------------------------


def test_the_default_format_is_text_and_the_suffix_chooses_json(tmp_path):
    uw, model, path = _model(tmp_path, name="run.log")
    assert model.journal_format == "text"

    model.journal_file = str(tmp_path / "run.jsonl")
    assert model.journal_format == "jsonl"

    model.journal_format = "text"
    assert model.journal_format == "text", "an explicit format must win"


def test_text_log_is_one_aligned_line_per_step(tmp_path):
    uw, model, path = _model(tmp_path, units=True, name="run.log")

    dt = uw.quantity(0.5, "Myr")
    for _ in range(3):
        with model.step(dt, label="convect"):
            model._record_step_event("solve", "SNES_Stokes(v)")

    lines = path.read_text().splitlines()
    comments = [l for l in lines if l.startswith("#")]
    rows = [l for l in lines if l.strip() and not l.startswith("#")]

    assert any("underworld3 step log" in c for c in comments)
    assert any("scales:" in c for c in comments)
    assert any("t/Myr" in c and "dt/Myr" in c for c in comments), (
        "the column header must name the unit the time column is in"
    )
    assert len(rows) == 3
    for index, row in enumerate(rows):
        assert row.split()[0] == str(index)
        assert "solve:SNES_Stokes(v)" in row
        assert "ok" in row


def test_text_log_converts_dt_into_the_clock_unit(tmp_path):
    """A dt in seconds beside a clock in Myr is converted; the table has one unit."""
    uw, model, path = _model(tmp_path, units=True, name="run.log")

    dt = uw.quantity(0.5, "Myr").to("s")          # same interval, other unit
    with model.step(dt, label="convect"):
        pass

    row = [l for l in path.read_text().splitlines()
           if l.strip() and not l.startswith("#")][0]
    fields = row.split()
    assert float(fields[1]) == pytest.approx(0.5, rel=1e-6), fields
    assert float(fields[2]) == pytest.approx(0.5, rel=1e-6), (
        f"dt was not converted into the clock's unit: {fields}"
    )


def test_a_backtrack_is_in_the_log(tmp_path):
    """A log that shows step 2, then step 2 again, must say what happened."""
    uw, model, path = _model(tmp_path, name="run.log")
    model.record_every = 1

    for _ in range(3):
        with model.step(0.1, label="convect"):
            pass

    model.rewind()

    lines = path.read_text().splitlines()
    notes = [l for l in lines if l.strip().startswith("--")]
    assert len(notes) == 1, lines
    assert "rewind to the start of step 2" in notes[0]
    assert "1 step(s) undone" in notes[0]


def test_a_bare_restore_is_in_the_log_too(tmp_path):
    """The backstepping idiom is save_state / load_state, not rewind."""
    uw, model, path = _model(tmp_path, name="run.log")

    with model.step(0.1, label="convect"):
        pass
    snap = model.save_state()
    with model.step(0.1, label="convect"):
        pass
    model.load_state(snap)

    notes = [l for l in path.read_text().splitlines() if l.strip().startswith("--")]
    assert len(notes) == 1, notes
    assert "restore from a snapshot" in notes[0]


def test_rewind_logs_one_note_not_two(tmp_path):
    """rewind() restores internally; only its own, more specific, note is written."""
    uw, model, path = _model(tmp_path, name="run.log")
    model.record_every = 1

    with model.step(0.1):
        pass
    model.rewind()

    notes = [l for l in path.read_text().splitlines() if l.strip().startswith("--")]
    assert len(notes) == 1, notes
    assert "rewind" in notes[0]
    assert "restore from" not in notes[0]


def test_backtracks_are_records_in_the_json_format(tmp_path):
    uw, model, path = _model(tmp_path, name="run.jsonl")
    model.record_every = 1

    for _ in range(2):
        with model.step(0.1):
            pass
    model.rewind()

    kinds = [json.loads(line)["kind"] for line in path.read_text().splitlines()]
    assert kinds == ["run", "step", "step", "rewind"]

    rewind = json.loads(path.read_text().splitlines()[-1])
    assert rewind["to_step"] == 1
    assert rewind["steps_undone"] == 1


def test_the_invariant_is_recorded_against_the_step(tmp_path):
    """The complaint belongs in the log, not only in whatever terminal ran it."""
    uw, model, path = _model(tmp_path, name="run.jsonl")

    with pytest.warns(RuntimeWarning, match="history advanced more than once"):
        with model.step(0.1):
            model._record_step_event("history_shift", "EulerianSUPG(T)", dt=0.1)
            model._record_step_event("history_shift", "EulerianSUPG(T)", dt=0.1)

    step = json.loads(path.read_text().splitlines()[-1])
    flags = [e for e in step["events"] if e["kind"] == "invariant"]
    assert len(flags) == 1
    assert "more than once" in flags[0]["name"]
    assert "EulerianSUPG(T) x2" in flags[0]["detail"]
