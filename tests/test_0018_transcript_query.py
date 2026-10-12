"""One query object over a run's record.

``uw.Transcript`` reads a transcript back and answers the questions a
debugging session asks: which steps were abandoned and by what, where the
run went back and why, which solves failed or ran capped, what the run's
step patterns were, what a part was solving at a given step, and what
changed between two steps. The digest, a notebook and a tool all read the
same interpretation. The record gains two things here: the exception that
abandoned a step, and the reason and detail a caller gives a rewind.
"""
import json

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _run(tmp_path):
    uw.reset_default_model()
    model = uw.get_default_model()
    path = tmp_path / "run.jsonl"
    model.transcript_file = str(path)
    model.record_every = 1
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 4, qdegree=2)
    u = uw.discretisation.MeshVariable("u", mesh, 1, degree=1)
    kappa = uw.expression(r"\kappa", 1.0, "diffusivity")
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = kappa
    poisson.f = 1.0
    poisson.add_essential_bc(0.0, "Bottom")
    poisson.add_essential_bc(1.0, "Top")
    poisson.petsc_options.delValue("ksp_monitor")

    for _ in range(3):
        with model.step(0.1, label="march"):
            poisson.solve()
            model._record_step_event("history_shift", "T_history")
    # a step rejected by the caller's own test, with the reason recorded
    with pytest.raises(RuntimeError):
        with model.step(0.1, label="too far"):
            poisson.solve()
            raise RuntimeError("displacement 0.18 over the limit 0.10")
    model.rewind(1, reason="displacement over the limit", observed=0.18, threshold=0.10,
                 action="halve dt")
    poisson.constitutive_model.Parameters.diffusivity = 2 * kappa   # the form changes: the part records itself again
    with model.step(0.05, label="retry"):
        poisson.solve()
        model._record_step_event("history_shift", "T_history")
    with model.step(0.05, label="retry"):
        poisson.solve()
        model._record_step_event("history_shift", "T_history")
    return model, path, poisson


def test_the_record_carries_why(tmp_path):
    model, path, poisson = _run(tmp_path)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    abandoned = [l for l in lines if l.get("kind") == "step" and not l.get("completed")]
    assert abandoned and abandoned[0]["abandoned_by"]["type"] == "RuntimeError"
    assert "over the limit" in abandoned[0]["abandoned_by"]["message"]
    rewinds = [l for l in lines if l.get("kind") == "rewind"]
    assert rewinds[0]["reason"] == "displacement over the limit"
    assert rewinds[0]["detail"] == {"observed": 0.18, "threshold": 0.10, "action": "halve dt"}


def test_the_queries_answer_from_the_file(tmp_path):
    model, path, poisson = _run(tmp_path)
    t = uw.Transcript(str(path))
    assert len(t) == 6                                   # 3 marches, 1 abandoned, 2 retries
    assert [s["index"] for s in t.abandoned()] == [3]
    back = t.backtracks()
    assert len(back) == 1 and back[0]["reason"] == "displacement over the limit"
    assert back[0]["to_step"] == 2 and back[0]["after_position"] == 3
    assert t.failed() == [] and t.capped() == []
    assert len(t.solves()) == 6
    assert t.sequence(0) == ["Poisson(u)", "history shift T_history"] or len(t.sequence(0)) == 2
    patterns = t.patterns()
    # marches collapse, the abandoned step stands alone, the retries collapse
    assert [p["count"] for p in patterns] == [3, 1, 2], patterns
    assert patterns[0]["from"] == 0 and patterns[0]["to"] == 2 and patterns[2]["label"] == "retry"
    # after the rewind the run numbered its retries 2 and 3 again: the
    # rejected step 3 and the retry that stands are two attempts
    assert len(t.attempts(3)) == 2 and t.step(3)["completed"] and not t.step(3, attempt=0)["completed"]
    diff = t.compare(t.step(2, attempt=0), t.step(3, attempt=0))
    assert diff["completed"] == (True, False)
    # the rejected step stopped before its history shift
    assert diff["only_in_a"] == ["history shift T_history"] and diff["only_in_b"] == []
    diff = t.compare(0, 2)                     # the march at 0 against the retry that stands at 2
    dt = [v["magnitude"] if isinstance(v, dict) else v for v in diff["dt"]]   # a dimensional run holds {magnitude, units}
    assert dt == [pytest.approx(0.1), pytest.approx(0.05)]


def test_a_part_is_read_at_a_step(tmp_path):
    model, path, poisson = _run(tmp_path)
    t = uw.Transcript(str(path))
    names = t.part_names()
    assert len(names) == 1
    first = t.part(names[0], at_step=0)
    last = t.part(names[0], at_step=5)
    assert first["fingerprint"] != last["fingerprint"], "the flux changed before the retry"
    assert [c[0] for c in t.changes()] == [names[0]]
    assert "F1" in first["forms"]


def test_the_transcript_describes_and_renders(tmp_path, capsys):
    model, path, poisson = _run(tmp_path)
    t = uw.Transcript(str(path))
    d = t.describe()
    assert d["kind"] == "transcript" and d["facts"]["steps"] == 6
    assert d["facts"]["abandoned"] == [3] and d["facts"]["backtracks"] == 1
    assert d["children"] and d["children"][0]["kind"] == "part"
    for fmt in ("markdown", "text", "yaml", "json"):
        assert uw.render(d, fmt).strip()
    t.view(format="text")
    assert "transcript" in capsys.readouterr().out
    # the renderers take the query object as a source
    assert "Poisson" in uw.transcript_key(t, format="text")
    assert uw.transcript_table(t)


def test_a_live_model_is_a_source_too(tmp_path):
    model, path, poisson = _run(tmp_path)
    t = uw.Transcript(model)
    assert t.live
    # the file keeps the abandoned step and the one the rewind undid; the
    # live list holds only what stands
    assert len(t) == 6 and len(model.transcript) == 4
