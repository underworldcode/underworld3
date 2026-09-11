"""The transcript, written down.

``model.transcript`` is what a run can still undo: bounded, in memory, gone with
the process. ``model.transcript_file`` is what the run did, on disk, appended
and flushed as each step closes.

It is ON by default and lands in a stamped directory under ``transcripts/``,
beside a copy of the script that launched it — the last group of tests here
pins that, including the two things that make a default tolerable: nothing is
created for a run that takes no step, and it can be turned off.

The two differ deliberately, and the differences are what the tests below pin.
An abandoned step appears in the file and not in memory — it is the part of a
run's history that is otherwise invisible. A step aged out by ``transcript_limit``
leaves memory and stays in the file. And one object per line means a run that
is killed keeps everything up to the moment it died.
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import json


def _model(tmp_path, units=False, name="run.transcript.jsonl", fmt=None):
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
    model.transcript_file = str(path)
    if fmt is not None:
        model.transcript_format = fmt
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

    assert [e.label for e in model.transcript] == ["fine"]

    steps = [json.loads(line) for line in path.read_text().splitlines()][1:]
    assert [s["label"] for s in steps] == ["fine", "too big"]
    assert [s["completed"] for s in steps] == [True, False]

    # The clock did not move for the abandoned step, so its t0 is the previous
    # step's t1 and nothing after it is shifted.
    assert steps[1]["t0"] == pytest.approx(0.25)
    assert model.tracker.time == pytest.approx(0.25)


def test_an_abandoned_step_does_not_retain_its_snapshot(tmp_path):
    """It is unreachable — the transcript never holds it — so it must not be kept."""
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
    model.transcript_limit = 2

    for _ in range(5):
        with model.step(0.1, label="convect"):
            pass

    assert len(model.transcript) == 2
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


def test_clear_transcript_opens_a_new_run_in_the_same_file(tmp_path):
    """An inversion runs the forward model many times; one file, many runs."""
    uw, model, path = _model(tmp_path)

    for run in range(3):
        model.clear_transcript()
        model.tracker.time = 0.0
        model.tracker.step = 0
        for _ in range(run + 1):
            with model.step(0.1, label=f"run{run}"):
                pass

    runs = uw.read_transcript(path)
    # The first header is written when the first step opens; clear_transcript
    # adds one per run, so an empty leading section is expected.
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

    runs = uw.read_transcript(path)
    assert len(runs) == 1
    assert [s["index"] for s in runs[0]["steps"]] == [0, 1, 2]


def test_turning_it_off_stops_the_writing(tmp_path):
    uw, model, path = _model(tmp_path)
    with model.step(0.1):
        pass
    before = path.read_text()
    assert before, "the first step is what creates the file"

    model.transcript_file = None
    assert model.transcript_file is None
    with model.step(0.1):
        pass

    assert path.read_text() == before, "writing continued after it was turned off"
    assert len(model.transcript) == 2, "the in-memory transcript must be unaffected"


# ---------------------------------------------------------------------------
# The text format — what a run is watched through
# ---------------------------------------------------------------------------


def test_the_default_format_is_text_and_the_suffix_chooses_json(tmp_path):
    uw, model, path = _model(tmp_path, name="run.log")
    assert model.transcript_format == "text"

    model.transcript_file = str(tmp_path / "run.jsonl")
    assert model.transcript_format == "jsonl"

    model.transcript_format = "text"
    assert model.transcript_format == "text", "an explicit format must win"


def test_text_log_is_one_aligned_line_per_step(tmp_path):
    uw, model, path = _model(tmp_path, units=True, name="run.log")

    dt = uw.quantity(0.5, "Myr")
    for _ in range(3):
        with model.step(dt, label="convect"):
            model._record_step_event("solve", "SNES_Stokes(v)")

    lines = path.read_text().splitlines()
    comments = [l for l in lines if l.startswith("#")]
    rows = [l for l in lines if l.strip() and not l.startswith("#")]

    assert any("underworld3 run transcript" in c for c in comments)
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


def test_repeated_events_are_in_the_file_in_order(tmp_path):
    """The transcript records a repeat; it does not judge it.

    Whether two shifts in one bar are a mistake or legitimate sub-cycling is a
    reading of the transcript made later, by a pass that can look across bars.
    The file's job is to hold both, in order, with nothing added."""
    uw, model, path = _model(tmp_path, name="run.jsonl")

    with model.step(0.1):
        model._record_step_event("solve", "SNES_AdvectionDiffusion(T)")
        model._record_step_event("history_shift", "EulerianSUPG(T)", dt=0.1)
        model._record_step_event("solve", "SNES_AdvectionDiffusion(T)")
        model._record_step_event("history_shift", "EulerianSUPG(T)", dt=0.1)

    step = json.loads(path.read_text().splitlines()[-1])
    assert [(e["kind"], e["name"]) for e in step["events"]] == [
        ("solve", "SNES_AdvectionDiffusion(T)"),
        ("history_shift", "EulerianSUPG(T)"),
        ("solve", "SNES_AdvectionDiffusion(T)"),
        ("history_shift", "EulerianSUPG(T)"),
    ], "nothing added, nothing reordered"


# ---------------------------------------------------------------------------
# Where a backtrack landed
# ---------------------------------------------------------------------------


def test_a_repeated_step_index_resolves_to_the_most_recent(tmp_path):
    """After a rewind the same index appears twice; a later backtrack means
    the second one, not the first."""
    uw, model, path = _model(tmp_path, name="run.jsonl")
    model.record_every = 1
    model.tracker.time = 0.0
    model.tracker.step = 0

    for _ in range(3):                      # steps 0, 1, 2
        with model.step(0.1, label="convect"):
            pass
    model.rewind()                          # back to the start of step 2
    for _ in range(2):                      # step 2 again, then 3
        with model.step(0.1, label="redo"):
            pass
    model.rewind()                          # back to the start of step 3

    runs = uw.read_transcript(path)
    steps, notes = runs[-1]["steps"], runs[-1]["notes"]
    assert [s["index"] for s in steps] == [0, 1, 2, 2, 3]

    first, second = notes
    assert first["to_position"] == 2
    # The second rewind targets step 3, which is the LAST row, not an earlier
    # one that happens to share an index.
    assert second["to_step"] == 3
    assert second["to_position"] == 4


def test_a_bare_restore_is_located_by_its_clock(tmp_path):
    """load_state names no step, so the note's recorded time places it."""
    uw, model, path = _model(tmp_path, units=True, name="run.jsonl")
    dt = uw.quantity(0.5, "Myr")

    with model.step(dt, label="a"):
        pass
    snap = model.save_state()
    with model.step(dt, label="b"):
        pass
    model.load_state(snap)

    run = uw.read_transcript(path)[-1]
    note = run["notes"][0]
    assert note["kind"] == "restore"
    assert note["after_position"] == 1
    assert note["to_position"] == 0, (
        "the restore put the clock back to the end of step 0"
    )


def test_a_backtrack_with_nothing_to_point_at_says_so(tmp_path):
    """A restore to a state no recorded step ended at leaves no target."""
    uw, model, path = _model(tmp_path, name="run.jsonl")

    snap = model.save_state()               # before any step: t = 0
    with model.step(0.1, label="a"):
        pass
    with model.step(0.1, label="b"):
        pass
    model.load_state(snap)

    note = uw.read_transcript(path)[-1]["notes"][0]
    assert note["after_position"] == 1
    assert note["to_position"] is None


# ---------------------------------------------------------------------------
# Where a transcript lands when nobody says
# ---------------------------------------------------------------------------


def _auto_model(monkeypatch, tmp_path, setting="on"):
    """A model with the automatic transcript armed, in a scratch directory.

    The automatic path is off under pytest by design — 1800 tests should not
    each leave a directory — so these tests arm it explicitly and point it at
    `tmp_path`.
    """
    import underworld3 as uw

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UW_TRANSCRIPT", setting)
    uw.reset_default_model()
    model = uw.get_default_model()
    model.tracker.time = 0.0
    model.tracker.step = 0
    return uw, model


def test_the_default_lands_in_a_stamped_directory(monkeypatch, tmp_path):
    uw, model = _auto_model(monkeypatch, tmp_path)

    with model.step(0.5, label="convect"):
        model._record_step_event("solve", "SNES_Stokes(v)")

    roots = list(tmp_path.glob("transcripts/*"))
    assert len(roots) == 1, roots
    run_dir = roots[0]
    # A stamp, so the run from this morning is still there after the next one.
    assert run_dir.name[:4].isdigit() and "T" in run_dir.name
    assert (run_dir / "transcript.log").exists()

    text = (run_dir / "transcript.log").read_text()
    assert "underworld3 run transcript" in text
    assert "SNES_Stokes(v)" in text


def test_nothing_is_created_for_a_run_that_takes_no_step(monkeypatch, tmp_path):
    """An import, or a script that only builds a mesh, must leave no trace."""
    uw, model = _auto_model(monkeypatch, tmp_path)

    assert model.transcript_file is not None, "it is armed"
    assert not (tmp_path / "transcripts").exists(), (
        "the directory must not exist before the first step"
    )


def test_two_runs_do_not_overwrite_each_other(monkeypatch, tmp_path):
    uw, model = _auto_model(monkeypatch, tmp_path)
    with model.step(0.5, label="first"):
        pass
    first = model.transcript_file

    uw.reset_default_model()
    second_model = uw.get_default_model()
    second_model.tracker.time = 0.0
    second_model.tracker.step = 0
    second_model._transcript_dir = None          # a later stamp
    import time

    time.sleep(1.1)                              # the stamp is to the second
    with second_model.step(0.5, label="second"):
        pass

    assert second_model.transcript_file != first
    assert "first" in open(first).read()
    assert "second" in open(second_model.transcript_file).read()


def test_the_launch_script_is_kept_beside_the_transcript(monkeypatch, tmp_path):
    """A programmatic launcher cannot be made reproducible by fiat; what CAN be
    done is to write down exactly what was run."""
    import json
    import sys

    script = tmp_path / "my_run.py"
    script.write_text("# the script that launched this\nprint('hello')\n")

    uw, model = _auto_model(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), "-uw_res", "32"])
    model._transcript_dir = None                 # re-resolve with the new argv

    with model.step(0.5):
        pass

    run_dir = list(tmp_path.glob("transcripts/*"))[0]
    assert "my_run" in run_dir.name, run_dir.name
    assert (run_dir / "my_run.py").read_text() == script.read_text()

    manifest = json.loads((run_dir / "launch.json").read_text())
    assert manifest["argv"] == [str(script), "-uw_res", "32"]
    assert manifest["script"] == "my_run.py"
    assert manifest["cwd"] == str(tmp_path)
    assert manifest["mpi_size"] == 1
    assert "underworld3" in manifest and "python" in manifest


def test_an_interactive_run_says_so_rather_than_failing(monkeypatch, tmp_path):
    import json
    import sys

    uw, model = _auto_model(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["-c"])
    model._transcript_dir = None

    with model.step(0.5):
        pass

    run_dir = list(tmp_path.glob("transcripts/*"))[0]
    assert "interactive" in run_dir.name
    manifest = json.loads((run_dir / "launch.json").read_text())
    assert manifest["script"] is None
    assert "script_note" in manifest


def test_an_explicit_path_is_a_file_not_a_place_to_put_things(monkeypatch, tmp_path):
    uw, model = _auto_model(monkeypatch, tmp_path)
    model.transcript_file = "run.log"

    with model.step(0.5):
        pass

    assert (tmp_path / "run.log").exists()
    assert not (tmp_path / "transcripts").exists()
    assert not list(tmp_path.glob("launch.json"))


def test_it_can_be_turned_off(monkeypatch, tmp_path):
    uw, model = _auto_model(monkeypatch, tmp_path)
    model.transcript_file = None

    with model.step(0.5):
        pass

    assert model.transcript_file is None
    assert not (tmp_path / "transcripts").exists()
    assert len(model.transcript) == 1, "the in-memory transcript is unaffected"


def test_the_environment_can_turn_it_off_and_relocate_it(monkeypatch, tmp_path):
    uw, model = _auto_model(monkeypatch, tmp_path, setting="off")
    assert model.transcript_file is None
    with model.step(0.5):
        pass
    assert not (tmp_path / "transcripts").exists()

    uw, model = _auto_model(monkeypatch, tmp_path, setting="somewhere/else")
    with model.step(0.5):
        pass
    assert list(tmp_path.glob("somewhere/else/*/transcript.log"))
    assert not (tmp_path / "transcripts").exists()


def test_it_is_off_under_pytest_by_default(monkeypatch, tmp_path):
    """The reason the tests above have to arm it: 1800 tests, 1800 directories."""
    import underworld3 as uw

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UW_TRANSCRIPT", raising=False)
    uw.reset_default_model()
    model = uw.get_default_model()
    model.tracker.time = 0.0
    model.tracker.step = 0

    assert model.transcript_file is None
    with model.step(0.5):
        pass
    assert not (tmp_path / "transcripts").exists()
