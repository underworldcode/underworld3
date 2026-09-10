"""A run's log, as a figure.

The log is written to be watched; these renderers turn it into something to
put in a paper or read on a page. The layout decision under test is the one
that makes a long run legible: state the operator sequence ONCE when every
step shares it, and call out only the steps that differ. A hundred identical
rows tell you nothing — a hundred identical rows and one that differs tell you
everything, but only if the identical ones are not in the way.
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import json
import xml.dom.minidom


def _run(steps, notes=None, model="test", scales=None):
    return [{
        "run": {"kind": "run", "model": model, "started": "2026-01-01T00:00:00+00:00",
                "scales": scales or {}},
        "steps": steps,
        "notes": notes or [],
    }]


def _step(index, dt=0.5, unit="megayear", t0=None, completed=True, wall=0.1,
          events=None, label="convect"):
    t0 = index * dt if t0 is None else t0
    quantity = (lambda v: {"magnitude": v, "units": unit}) if unit else (lambda v: v)
    return {
        "kind": "step", "index": index, "label": label,
        "t0": quantity(t0), "t1": quantity(t0 + dt), "dt": quantity(dt),
        "completed": completed, "restorable": True, "wall": wall,
        "events": events if events is not None else [
            {"kind": "solve", "name": "SNES_AdvectionDiffusion_Composed(T)"},
            {"kind": "history_shift", "name": "EulerianSUPG(T)", "dt": dt},
            {"kind": "solve", "name": "SNES_Stokes(v)"},
        ],
    }


def _svg(tmp_path, runs, **kwargs):
    import underworld3 as uw

    out = str(tmp_path / "run.svg")
    uw.journal_diagram(runs, out=out, **kwargs)
    text = open(out, encoding="utf-8").read()
    xml.dom.minidom.parseString(text)          # must be well-formed
    return text


def test_the_figure_is_self_contained_svg(tmp_path):
    text = _svg(tmp_path, _run([_step(i) for i in range(4)]))
    assert text.startswith("<svg")
    assert "xmlns" in text
    # The SVG namespace URI is a required identifier, never fetched; what must
    # not appear is anything the renderer would go and load.
    body = text.replace('xmlns="http://www.w3.org/2000/svg"', "")
    for forbidden in ("<script", "<image", "xlink:href", "url(", "@import",
                      "http://", "https://"):
        assert forbidden not in body, f"the figure is not self-contained: {forbidden}"


def test_nothing_is_drawn_outside_the_canvas(tmp_path):
    """A bar centred on the edge of the lane puts half of itself off the page."""
    import re

    text = _svg(tmp_path, _run([_step(i) for i in range(40)]))
    width = int(re.search(r'width="(\d+)"', text).group(1))
    edges = [float(x) + float(w) for x, w in
             re.findall(r'<rect x="([-\d.]+)"[^>]*width="([-\d.]+)"', text)]
    assert max(edges) <= width + 0.5
    assert min(float(x) for x in re.findall(r'<rect x="([-\d.]+)"', text)) >= -0.5


def test_a_shared_sequence_is_stated_once(tmp_path):
    text = _svg(tmp_path, _run([_step(i) for i in range(12)]))
    assert "Every step:" in text
    assert text.count("AdvectionDiffusion(T)") == 1, (
        "an identical sequence must not be repeated per step"
    )
    assert "Steps that did something else" not in text


def test_a_step_that_differs_is_called_out(tmp_path):
    odd = _step(5, events=[
        {"kind": "solve", "name": "SNES_AdvectionDiffusion_Composed(T)"},
        {"kind": "history_shift", "name": "EulerianSUPG(T)", "dt": 0.5},
        {"kind": "solve", "name": "SNES_Stokes(v)"},
        {"kind": "solve", "name": "SNES_Stokes(v)"},
    ])
    steps = [_step(i) for i in range(5)] + [odd] + [_step(i) for i in range(6, 10)]

    text = _svg(tmp_path, _run(steps))
    assert "9 of 10 steps:" in text
    assert "Steps that did something else" in text
    assert "step 5:" in text


def test_an_abandoned_step_is_marked(tmp_path):
    steps = [_step(0), _step(1, dt=9.0, completed=False), _step(1)]
    text = _svg(tmp_path, _run(steps))
    assert "abandoned" in text
    assert "1 abandoned" in text


def test_a_backtrack_is_drawn(tmp_path):
    steps = [_step(i) for i in range(4)]
    notes = [{"kind": "rewind", "after_position": 3, "to_position": 1,
              "short": "rewind 2", "steps_undone": 2}]
    text = _svg(tmp_path, _run(steps, notes))
    assert "rewind 2" in text
    assert "backtrack(s)" in text
    assert "<path" in text, "the backtrack should be drawn, not only named"


def test_the_invariant_is_reported_on_the_figure(tmp_path):
    flagged = _step(2, events=[
        {"kind": "history_shift", "name": "EulerianSUPG(T)", "dt": 0.5},
        {"kind": "history_shift", "name": "EulerianSUPG(T)", "dt": 0.5},
        {"kind": "invariant", "name": "history advanced more than once",
         "detail": "EulerianSUPG(T) x2"},
    ])
    text = _svg(tmp_path, _run([_step(0), _step(1), flagged]))
    assert "Invariant" in text
    assert "EulerianSUPG(T) x2" in text


def test_a_long_sequence_is_wrapped_not_run_off_the_page(tmp_path):
    import re

    long_step = _step(0, events=[
        {"kind": "solve", "name": f"SNES_VeryLongSolverName_{i}(field_{i})"}
        for i in range(10)
    ])
    text = _svg(tmp_path, _run([long_step]))
    width = int(re.search(r'width="(\d+)"', text).group(1))
    for x, anchor, content in re.findall(
            r'<text x="([-\d.]+)"[^>]*text-anchor="(\w+)"[^>]*>([^<]*)<', text):
        if anchor == "start":
            assert float(x) + len(content) * 7.0 <= width + 40, content


def test_the_time_span_is_on_the_figure(tmp_path):
    """dt is what is plotted; without this the figure never says WHEN."""
    text = _svg(tmp_path, _run([_step(i) for i in range(4)]))
    assert "t = 0 →" in text
    assert "Myr" in text


def test_a_nondimensional_run_still_renders(tmp_path):
    text = _svg(tmp_path, _run([_step(i, unit=None) for i in range(4)]))
    assert "dt" in text
    assert "None" not in text


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------

def test_flowchart_is_one_chain_when_every_step_agrees():
    import underworld3 as uw

    text = uw.journal_flowchart(_run([_step(i) for i in range(6)]))
    assert text.startswith("flowchart LR")
    assert "subgraph" not in text
    assert text.count("-->") == 2
    assert "shift EulerianSUPG(T)" in text


def test_flowchart_separates_the_step_that_differs():
    import underworld3 as uw

    odd = _step(3, events=[{"kind": "solve", "name": "SNES_Stokes(v)"}])
    text = uw.journal_flowchart(_run([_step(0), _step(1), _step(2), odd]))
    assert text.count("subgraph") == 2
    assert 'step 3' in text
    assert 'step 0, 1, 2' in text


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def test_a_live_model_can_be_drawn_without_a_file(tmp_path):
    """The text log cannot be read back, so a live model must be a source."""
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    model.tracker.time = 0.0
    model.tracker.step = 0
    for _ in range(3):
        with model.step(0.25, label="convect"):
            model._record_step_event("solve", "SNES_Stokes(v)")

    out = str(tmp_path / "live.svg")
    uw.journal_diagram(model, out=out)
    text = open(out, encoding="utf-8").read()
    xml.dom.minidom.parseString(text)
    assert "3 step(s) recorded" in text


def test_reading_a_text_log_says_what_to_do_instead(tmp_path):
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    path = tmp_path / "run.log"
    model.journal_file = str(path)
    model.tracker.time = 0.0
    model.tracker.step = 0
    with model.step(0.1):
        pass

    with pytest.raises(ValueError, match="journal_format = 'jsonl'"):
        uw.read_journal(str(path))


def test_a_jsonl_log_round_trips_into_a_figure(tmp_path):
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    path = tmp_path / "run.jsonl"
    model.journal_file = str(path)
    model.record_every = 1
    model.tracker.time = 0.0
    model.tracker.step = 0

    for _ in range(4):
        with model.step(0.25, label="convect"):
            model._record_step_event("solve", "SNES_Stokes(v)")
    model.rewind()

    out = uw.journal_diagram(str(path))
    assert out.endswith(".svg")
    text = open(out, encoding="utf-8").read()
    xml.dom.minidom.parseString(text)
    assert "rewind" in text
