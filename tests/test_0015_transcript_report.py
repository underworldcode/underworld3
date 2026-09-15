"""A run's log, as a figure.

The log is written to be watched; these renderers turn it into something to
put in a paper. Time runs DOWN the page, one row per step, so the figure is
portrait and paginates.

The layout decision under test is the one that makes a long run legible: each
distinct operator sequence gets a LETTER, defined once at the foot. A column
of ``A`` with a single ``B`` in it says at a glance that one step did something
different, where a hundred repeated sequences say nothing and hide the one that
matters.
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
    uw.transcript_diagram(runs, out=out, **kwargs)
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


def test_a_shared_sequence_is_written_once(tmp_path):
    text = _svg(tmp_path, _run([_step(i) for i in range(12)]))
    assert "Operator sequences" in text
    assert text.count("AdvectionDiffusion(T)") == 1, (
        "an identical sequence must not be spelled out per step"
    )
    assert ">12 steps<" in text.replace(" ", " ") or "12 steps" in text


def test_a_step_that_differs_gets_its_own_letter(tmp_path):
    odd = _step(5, events=[
        {"kind": "solve", "name": "SNES_AdvectionDiffusion_Composed(T)"},
        {"kind": "history_shift", "name": "EulerianSUPG(T)", "dt": 0.5},
        {"kind": "solve", "name": "SNES_Stokes(v)"},
        {"kind": "solve", "name": "SNES_Stokes(v)"},
    ])
    steps = [_step(i) for i in range(5)] + [odd] + [_step(i) for i in range(6, 10)]

    text = _svg(tmp_path, _run(steps))
    assert ">A<" in text and ">B<" in text, "two sequences, two letters"
    assert "9 steps" in text and "1 step<" in text
    # The A column is one letter per row, so the legend must be the only place
    # the sequences are spelled out.
    assert text.count("shift EulerianSUPG(T)") == 2


def test_one_sequence_means_one_letter(tmp_path):
    text = _svg(tmp_path, _run([_step(i) for i in range(6)]))
    assert ">B<" not in text


def test_an_abandoned_step_is_marked(tmp_path):
    steps = [_step(0), _step(1, dt=9.0, completed=False), _step(1)]
    text = _svg(tmp_path, _run(steps))
    assert "abandoned" in text
    assert "1 abandoned" in text


def test_a_backtrack_is_drawn(tmp_path):
    steps = [_step(i) for i in range(6)]
    notes = [{"kind": "rewind", "after_position": 5, "to_position": 1,
              "short": "rewind 4", "steps_undone": 4}]
    text = _svg(tmp_path, _run(steps, notes))
    assert "rewind 4" in text
    assert "backtrack(s)" in text
    assert "<path" in text, "the backtrack should be drawn, not only named"


def test_a_repeat_shows_as_its_own_sequence(tmp_path):
    """A bar whose operators ran twice differs from its neighbours, so it gets
    its own letter. The figure reports that and makes no claim about whether it
    is wrong — that reading belongs to a later pass over the transcript."""
    doubled = _step(2, events=[
        {"kind": "solve", "name": "SNES_AdvectionDiffusion_Composed(T)"},
        {"kind": "history_shift", "name": "EulerianSUPG(T)", "dt": 0.5},
        {"kind": "solve", "name": "SNES_AdvectionDiffusion_Composed(T)"},
        {"kind": "history_shift", "name": "EulerianSUPG(T)", "dt": 0.5},
    ])
    text = _svg(tmp_path, _run([_step(0), _step(1), doubled]))
    assert ">A<" in text and ">B<" in text
    assert "Invariant" not in text, (
        "the figure must not assert that a repeat is a mistake"
    )


def test_a_long_sequence_is_wrapped_not_run_off_the_page(tmp_path):
    import re

    long_step = _step(0, events=[
        {"kind": "solve", "name": f"SNES_VeryLongSolverName_{i}(field_{i})"}
        for i in range(10)
    ])
    text = _svg(tmp_path, _run([long_step]))
    width = int(re.search(r'width="(\d+)"', text).group(1))
    for x, size, mono, anchor, content in re.findall(
            r'<text x="([-\d.]+)"[^>]*monospace" font-size="([\d.]+)"()[^>]*'
            r'text-anchor="(\w+)"[^>]*>([^<]*)<', text):
        plain = content.replace("&gt;", ">").replace("&amp;", "&")
        assert float(x) + len(plain) * 0.60 * float(size) <= width - 20, plain


def test_the_page_fits_a_document_column(tmp_path):
    """Time runs down the page, so width is fixed and height grows."""
    import re

    def size(n):
        text = _svg(tmp_path, _run([_step(i) for i in range(n)]))
        return tuple(int(v) for v in
                     re.search(r'width="(\d+)" height="(\d+)"', text).groups())

    small_w, small_h = size(5)
    big_w, big_h = size(40)
    assert small_w == big_w <= 620, "the page must not widen with the run"
    assert big_h > small_h + 30 * 12, "height must grow one row per step"


def test_the_time_span_is_on_the_figure(tmp_path):
    """dt is what is plotted; without this the figure never says WHEN."""
    text = _svg(tmp_path, _run([_step(i) for i in range(4)]))
    assert "t = 0 to" in text
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

    text = uw.transcript_flowchart(_run([_step(i) for i in range(6)]))
    assert text.startswith("flowchart LR")
    assert "subgraph" not in text
    assert text.count("-->") == 2
    assert "shift EulerianSUPG(T)" in text


def test_flowchart_separates_the_step_that_differs():
    import underworld3 as uw

    odd = _step(3, events=[{"kind": "solve", "name": "SNES_Stokes(v)"}])
    text = uw.transcript_flowchart(_run([_step(0), _step(1), _step(2), odd]))
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
    uw.transcript_diagram(model, out=out)
    text = open(out, encoding="utf-8").read()
    xml.dom.minidom.parseString(text)
    assert "3 steps" in text


def test_reading_a_text_log_says_what_to_do_instead(tmp_path):
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    path = tmp_path / "run.log"
    model.transcript_file = str(path)
    model.tracker.time = 0.0
    model.tracker.step = 0
    with model.step(0.1):
        pass

    with pytest.raises(ValueError, match="transcript_format = 'jsonl'"):
        uw.read_transcript(str(path))


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _pdf(tmp_path, runs, name="run.pdf", **kwargs):
    import underworld3 as uw

    out = str(tmp_path / name)
    uw.transcript_diagram(runs, out=out, **kwargs)
    return open(out, "rb").read()


def test_pdf_is_the_default_and_is_a_real_pdf(tmp_path):
    import underworld3 as uw

    out = str(tmp_path / "run")
    written = uw.transcript_diagram(_run([_step(i) for i in range(5)]), out=out)
    assert written == out
    data = open(out, "rb").read()
    assert data.startswith(b"%PDF-1.4")
    assert data.rstrip().endswith(b"%%EOF")
    assert b"/Type /Catalog" in data and b"xref" in data
    assert b"/BaseFont /Helvetica" in data


def test_pdf_paginates_a_long_run(tmp_path):
    steps = [_step(i) for i in range(200)]
    data = _pdf(tmp_path, _run(steps))
    pages = data.count(b"/Type /Page ")
    assert pages >= 4, f"200 steps should not fit on {pages} page(s)"
    assert data.count(b"/Type /Pages") == 1


def test_pdf_offsets_point_at_their_objects(tmp_path):
    """A cross-reference table that lies makes an unopenable file."""
    import re

    data = _pdf(tmp_path, _run([_step(i) for i in range(30)]))
    start = int(re.search(rb"startxref\s+(\d+)", data).group(1))
    assert data[start:start + 4] == b"xref"
    body = data[start:].split(b"trailer")[0].splitlines()
    entries = [line for line in body[2:] if line.strip().endswith(b"n")]
    for index, line in enumerate(entries, start=1):
        offset = int(line.split()[0])
        assert data[offset:offset + len(f"{index} 0 obj")] == \
            f"{index} 0 obj".encode(), f"object {index} is not at its offset"


def test_pdf_is_written_without_a_plotting_library(tmp_path):
    """No dependency is imported to produce the figure."""
    import sys

    for module in ("matplotlib", "cairosvg", "reportlab", "PIL"):
        sys.modules.pop(module, None)
    _pdf(tmp_path, _run([_step(i) for i in range(5)]))
    for module in ("matplotlib", "cairosvg", "reportlab"):
        assert module not in sys.modules, f"{module} was imported to draw"


def test_a_jsonl_log_round_trips_into_a_figure(tmp_path):
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    path = tmp_path / "run.jsonl"
    model.transcript_file = str(path)
    model.record_every = 1
    model.tracker.time = 0.0
    model.tracker.step = 0

    for _ in range(4):
        with model.step(0.25, label="convect"):
            model._record_step_event("solve", "SNES_Stokes(v)")
    model.rewind()

    out = uw.transcript_diagram(str(path))
    assert out.endswith(".pdf")
    assert open(out, "rb").read().startswith(b"%PDF")

    out_svg = uw.transcript_diagram(str(path), out=str(tmp_path / "run.svg"))
    text = open(out_svg, encoding="utf-8").read()
    xml.dom.minidom.parseString(text)
    assert "1 backtrack(s)" in text
    assert "stroke-dasharray" in text, "the backtrack should be drawn"


# ---------------------------------------------------------------------------
# Backtracks read as the path the run took
# ---------------------------------------------------------------------------


def test_a_backtrack_draws_the_undo_and_the_repeat(tmp_path):
    """Back out of the abandoned step, then down to the row that redoes it."""
    # The shape a real run leaves: ... 2 ok, 3 abandoned, back to 2, 2 again.
    steps = [_step(i) for i in range(3)]
    steps.append(_step(3, dt=9.0, completed=False, label="too big"))
    steps.append(_step(2, label="replay"))
    notes = [{"kind": "rewind", "after_position": 3, "to_position": 2,
              "short": "rewind 1", "steps_undone": 1}]

    text = _svg(tmp_path, _run(steps, notes))
    assert "rewind 1" in text
    assert "again" in text, (
        "the row that repeats the step should be joined to the one it repeats"
    )
    # Two arrowheads: one back (up), one forward (down).
    assert text.count("<path") >= 2


def test_no_repeat_arrow_when_the_run_moves_on(tmp_path):
    """A backtrack followed by a DIFFERENT step is not a repeat."""
    steps = [_step(i) for i in range(3)] + [_step(7, label="elsewhere")]
    notes = [{"kind": "rewind", "after_position": 2, "to_position": 1,
              "short": "rewind 1"}]

    text = _svg(tmp_path, _run(steps, notes))
    assert "rewind 1" in text
    assert "again" not in text


def test_notes_that_make_the_same_jump_share_one_arrow(tmp_path):
    """A load_state then a rewind to the same place is one backtrack."""
    steps = [_step(i) for i in range(4)] + [_step(2, label="replay")]
    notes = [
        {"kind": "restore", "after_position": 3, "to_position": 2,
         "short": "restore"},
        {"kind": "rewind", "after_position": 3, "to_position": 2,
         "short": "rewind 1"},
    ]

    text = _svg(tmp_path, _run(steps, notes))
    assert "rewind 1 (+1)" in text, "the group should be labelled once"
    assert "restore" not in text.split("Operator sequences")[0], (
        "the two labels must not both print in the gutter"
    )


def test_gutter_labels_stay_on_the_page(tmp_path):
    import re

    steps = [_step(i) for i in range(4)] + [_step(1)]
    notes = [{"kind": "rewind", "after_position": 3, "to_position": 1,
              "short": "rewind 2 (+1)"}]
    text = _svg(tmp_path, _run(steps, notes))
    for x, anchor, content in re.findall(
            r'<text x="([-\d.]+)"[^>]*text-anchor="(\w+)"[^>]*>([^<]*)<', text):
        if anchor == "end":
            assert float(x) - len(content) * 0.53 * 7.0 >= -1.0, content


def test_pdf_has_no_replacement_characters(tmp_path):
    """Arrows and ellipses must be mapped, not turned into '?'."""
    import zlib
    import re

    steps = [_step(i) for i in range(4)] + [_step(2)]
    notes = [{"kind": "rewind", "after_position": 3, "to_position": 2,
              "short": "rewind 1 (+1)"}]
    data = _pdf(tmp_path, _run(steps, notes))
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S)
    body = b"".join(zlib.decompress(s) for s in streams).decode("latin-1")
    for drawn in re.findall(r"\((.*?)\) Tj", body):
        assert "?" not in drawn, drawn


# ---------------------------------------------------------------------------
# How a solve went, in the figure
# ---------------------------------------------------------------------------

def _solve(name, converged=None, **detail):
    event = {"kind": "solve", "name": name, "part": f"{name}#1"}
    if converged is not None:
        event["converged"] = converged
        event["reason"] = ("CONVERGED_FNORM_RELATIVE" if converged
                           else "DIVERGED_LINEAR_SOLVE")
        event["nl_its"], event["ksp_its"] = 2, 11
        event.update(detail)
    return event


def test_the_figure_marks_each_of_the_three_outcomes(tmp_path):
    """Converged, converged-with-a-block-that-gave-up, and diverged are three
    different things, and the figure has to be able to say which."""
    import underworld3 as uw

    steps = [
        _step(0, events=[_solve("Stokes(v)", True)]),
        _step(1, events=[_solve("Stokes(v)", True, capped={"velocity": 3})]),
        _step(2, events=[_solve("Stokes(v)", False)]),
    ]
    out = str(tmp_path / "transcript.svg")
    uw.utilities.transcript_report.transcript_figure(_run(steps), out=out)
    text = open(out, encoding="utf-8").read()
    xml.dom.minidom.parseString(text)

    # The emoji are what a reader sees; each state must appear at least twice
    # (once in the chart, once in the legend).
    for glyph in ("✅", "⚠", "❌"):
        assert text.count(glyph) >= 2, f"{glyph!r} missing from the figure"


def test_the_pdf_draws_the_outcomes_without_emoji(tmp_path):
    """The PDF is written against the base-14 fonts, which have no emoji, so
    the marks are stroked. The bytes must still be a valid PDF."""
    import underworld3 as uw

    steps = [
        _step(0, events=[_solve("Stokes(v)", True)]),
        _step(1, events=[_solve("Stokes(v)", False)]),
    ]
    out = str(tmp_path / "transcript.pdf")
    uw.utilities.transcript_report.transcript_figure(_run(steps), out=out)
    raw = open(out, "rb").read()
    assert raw.startswith(b"%PDF-")
    assert b"%%EOF" in raw
    assert "❌".encode("utf-8") not in raw


def test_a_capped_block_does_not_hide_inside_a_run_of_clean_steps(tmp_path):
    """The collapse asserts identity. A step whose velocity block gave up did
    not do the same thing as the steps around it, so it must break the run."""
    from underworld3.utilities.transcript_report import transcript_table

    steps = [_step(i, events=[_solve("Stokes(v)", True)]) for i in range(5)]
    steps[2]["events"] = [_solve("Stokes(v)", True, capped={"velocity": 4})]
    text = transcript_table(_run(steps))

    assert "1!" in text, text
    # the bar that differs is printed in full rather than swallowed by a
    # repeat: the clean steps collapse into two runs around it, not one.
    assert text.count("×1 unchanged") == 2, text


def test_a_transcript_without_outcomes_still_renders(tmp_path):
    """Transcripts written before outcomes were recorded carry no verdict, and
    the figure must not invent one."""
    import underworld3 as uw

    steps = [_step(i, events=[_solve("Stokes(v)")]) for i in range(3)]
    out = str(tmp_path / "old.svg")
    uw.utilities.transcript_report.transcript_figure(_run(steps), out=out)
    text = open(out, encoding="utf-8").read()
    xml.dom.minidom.parseString(text)
    # once, in the legend — the chart itself marks nothing.
    assert text.count("✅") == 1, text.count("✅")
