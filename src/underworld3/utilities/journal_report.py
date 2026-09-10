"""Turn a run's step log into a figure.

The log is written to be watched (:attr:`underworld3.Model.journal_file`); this
module turns it into something to put in a paper or read on a page. Both
renderers take the same source — a log file, the list
:func:`underworld3.read_journal` returns, or a live model.

Two views, because a run has two kinds of structure worth drawing:

``journal_diagram``
    What the run DID, as a self-contained SVG. Bars in the order things
    happened, backtracks as arcs over them, and — the point of the layout —
    the operator sequence stated ONCE when every step shares it, with only the
    steps that differ called out. A hundred identical rows tell you nothing; a
    hundred identical rows and one that differs tell you everything, and only
    if the identical ones are not in the way.

``journal_flowchart``
    What ONE step does, as Mermaid, for dropping into documentation. Falls back
    to describing each distinct sequence when a run has more than one.

Neither needs a plotting library: the SVG is written directly, so it has no
dependencies, no rasterisation, and no theme to fight with.
"""

from __future__ import annotations

import html
import json
import math
import os

__all__ = ["journal_diagram", "journal_flowchart"]


# --- palette ---------------------------------------------------------------
# Print-safe and legible in greyscale: the two step colours differ in value as
# well as hue, and the abandoned one carries a hatch so it survives a mono
# photocopier and a colour-blind reader alike.
_INK = "#1c1c1e"
_MUTED = "#6b7280"
_RULE = "#d6d3ce"
_PAPER = "#fdfcfa"
_ACCEPTED = "#4a6fa5"
_ABANDONED = "#b4544a"
_WALL = "#9aa5b1"
_FLAG = "#c2761f"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _as_runs(source):
    """Accept a path, the list ``read_journal`` returns, or a live model."""
    if isinstance(source, (str, os.PathLike)):
        import underworld3 as uw

        return uw.read_journal(str(source))
    if hasattr(source, "journal") and hasattr(source, "tracker"):
        steps = [entry.as_dict() for entry in source.journal]
        return [{"run": source._run_header(), "steps": steps}]
    if isinstance(source, list):
        if source and isinstance(source[0], dict) and "steps" in source[0]:
            return source
        return [{"run": None, "steps": list(source)}]
    raise TypeError(
        f"expected a journal path, the list read_journal returns, or a Model; "
        f"got {type(source).__name__}"
    )


def _pick_run(runs, index):
    populated = [r for r in runs if r.get("steps")]
    if not populated:
        raise ValueError("this journal holds no steps")
    return populated[index]


def _magnitude(value):
    if isinstance(value, dict):
        return float(value.get("magnitude", float("nan")))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _unit(value):
    return value.get("units") if isinstance(value, dict) else None


def _short_unit(unit):
    if unit is None:
        return ""
    return {
        "second": "s", "minute": "min", "hour": "hr", "day": "d", "year": "yr",
        "kiloyear": "kyr", "megayear": "Myr", "gigayear": "Gyr",
        "meter": "m", "kilometer": "km", "kelvin": "K", "kilogram": "kg",
    }.get(str(unit), str(unit))


def _converted(value, unit):
    """``value`` as a bare number in ``unit`` — a figure has one time axis."""
    if not isinstance(value, dict) or unit is None:
        return _magnitude(value)
    if str(value.get("units")) == str(unit):
        return _magnitude(value)
    try:
        import underworld3 as uw

        return float(
            uw.quantity(_magnitude(value), str(value["units"])).to(unit).magnitude
        )
    except Exception:
        return _magnitude(value)


def _short_operator(name):
    """``SNES_AdvectionDiffusion_Composed(T)`` -> ``AdvectionDiffusion(T)``.

    A figure has a width. The prefix says which base class implemented it,
    which is never the thing the reader is checking."""
    text = str(name)
    for prefix in ("SNES_", "uw_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.replace("_Composed", "")


def _signature(step):
    """The operator sequence of a step, as a comparable tuple."""
    return tuple(
        (event["kind"], _short_operator(event["name"]))
        for event in step.get("events", [])
        if event.get("kind") in ("solve", "history_shift")
    )


def _describe(signature):
    parts = []
    for kind, name in signature:
        parts.append(name if kind == "solve" else f"shift {name}")
    return "  →  ".join(parts) or "(nothing)"


def _wrap(signature, budget):
    """The operator sequence as lines that fit ``budget`` characters.

    A step that ran six operators is exactly the step worth showing, and it is
    the one whose description runs off the page. Break between operators, never
    inside one."""
    tokens = [name if kind == "solve" else f"shift {name}"
              for kind, name in signature]
    if not tokens:
        return ["(nothing)"]
    lines, current = [], ""
    for i, token in enumerate(tokens):
        piece = token if not current else f"  →  {token}"
        if current and len(current) + len(piece) > budget:
            lines.append(current + "  →")
            current = token
        else:
            current += piece
    lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------

def _text(x, y, content, size=11, fill=_INK, anchor="start", weight="normal",
          family="'Helvetica Neue', Helvetica, Arial, sans-serif", opacity=1.0):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}" opacity="{opacity:g}">{html.escape(str(content))}</text>'
    )


def _rect(x, y, w, h, fill, stroke="none", rx=1.5, dash=None, opacity=1.0):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.6):.1f}" '
        f'height="{max(h, 0.6):.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}"'
        f'{dash_attr} opacity="{opacity:g}"/>'
    )


def _line(x1, y1, x2, y2, stroke=_RULE, width=1.0, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


# ---------------------------------------------------------------------------
# The diagram
# ---------------------------------------------------------------------------

def journal_diagram(source, out=None, run=-1, title=None, width=920):
    """Render a run's log as a standalone SVG.

    Parameters
    ----------
    source : str, list or Model
        A journal file, the list :func:`underworld3.read_journal` returns, or a
        live model whose ``journal`` is to be drawn.
    out : str, optional
        Where to write. Defaults to the source path with ``.svg``, or
        ``journal.svg``.
    run : int, default -1
        Which run in the file. A file holds one per ``clear_journal()``, so an
        inversion driver leaves many; the last is usually the one you want.
    title : str, optional
        Overrides the heading taken from the run header.
    width : int, default 920
        Figure width in SVG user units (px at 1:1, but it is vector).

    Returns
    -------
    str
        The path written.
    """
    runs = _as_runs(source)
    entry = _pick_run(runs, run)
    header = entry.get("run") or {}
    steps = entry["steps"]
    notes = entry.get("notes", [])

    if out is None:
        if isinstance(source, (str, os.PathLike)):
            out = os.path.splitext(str(source))[0] + ".svg"
        else:
            out = "journal.svg"

    svg = _compose(header, steps, notes, title=title, width=width)
    directory = os.path.dirname(out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return out


def _compose(header, steps, notes, title=None, width=920):
    margin = 44
    inner = width - 2 * margin

    unit = None
    for step in steps:
        unit = _unit(step.get("t1")) or _unit(step.get("dt"))
        if unit:
            break
    short = _short_unit(unit)

    dts = [_converted(step.get("dt"), unit) for step in steps]
    walls = [step.get("wall") or 0.0 for step in steps]
    finite = [d for d in dts if math.isfinite(d) and d > 0]
    dt_max = max(finite) if finite else 1.0
    dt_min = min(finite) if finite else 1.0

    # A rejected step is often tens of times the accepted ones — that IS the
    # reason it was rejected — and on a linear axis it flattens everything
    # else to nothing. Switch to log and say so on the axis rather than
    # quietly clipping the outlier that carries the story.
    log_scale = dt_max / max(dt_min, 1e-300) > 20.0

    def bar_height(value, span):
        if not math.isfinite(value) or value <= 0:
            return 1.0
        if log_scale:
            lo, hi = math.log10(dt_min), math.log10(dt_max)
            frac = 0.12 + 0.88 * ((math.log10(value) - lo) / (hi - lo) if hi > lo else 1.0)
        else:
            frac = value / dt_max
        return max(2.0, frac * span)

    # --- layout ---
    y = margin
    parts = [
        f'<rect x="0" y="0" width="{width}" height="__H__" fill="{_PAPER}"/>'
    ]

    heading = title or f"Run log — {header.get('model', 'model')!r}"
    parts.append(_text(margin, y, heading, size=17, weight="600"))
    y += 17

    subtitle = []
    if header.get("started"):
        subtitle.append(f"started {header['started']}")
    # The model time the run covered. dt is the plotted quantity, so without
    # this the figure never says where in the model's life any of it happened.
    accepted = [s for s in steps if s.get("completed")]
    if accepted:
        t_from = _converted(accepted[0].get("t0"), unit)
        t_to = _converted(accepted[-1].get("t1"), unit)
        if math.isfinite(t_from) and math.isfinite(t_to):
            subtitle.append(
                f"t = {t_from:.4g} → {t_to:.4g}" + (f" {short}" if short else "")
            )
    subtitle.append(f"{len(steps)} step(s) recorded")
    abandoned = [s for s in steps if not s.get("completed")]
    if abandoned:
        subtitle.append(f"{len(abandoned)} abandoned")
    if notes:
        subtitle.append(f"{len(notes)} backtrack(s)")
    parts.append(_text(margin, y, "  ·  ".join(subtitle), size=11, fill=_MUTED))
    y += 15

    scales = header.get("scales") or {}
    if scales:
        text = "  |  ".join(
            f"{name} {value['magnitude']:.4g} {_short_unit(value['units'])}"
            for name, value in scales.items() if isinstance(value, dict)
        )
        parts.append(_text(margin, y, f"scales: {text}", size=10, fill=_MUTED))
        y += 14
    y += 12

    # --- backtrack lane (above the bars, so the arcs read as going back) ---
    arc_lane = 26 if notes else 0
    arc_top = y
    y += arc_lane

    # --- the bars ---
    axis_label = f"dt / {short}" if short else "dt"
    if log_scale:
        axis_label += "  (log)"
    parts.append(_text(margin, y - 4, axis_label, size=10, fill=_MUTED))

    bar_span = 120
    bar_top = y
    baseline = bar_top + bar_span
    n = max(len(steps), 1)
    slot = inner / n
    bar_w = min(max(slot * 0.62, 2.0), 34.0)
    # Inset by half a bar at each end: a centre placed at the very edge of the
    # lane puts half its bar, and all of its label, outside the figure.
    lane = margin + bar_w / 2
    lane_width = inner - bar_w
    slot = lane_width / n if n > 1 else 0.0

    parts.append(_line(margin, baseline, margin + inner, baseline, _RULE, 1.0))

    centres = []
    for i, step in enumerate(steps):
        cx = lane + (slot * i if n > 1 else lane_width / 2)
        centres.append(cx)
        h = bar_height(dts[i], bar_span - 14)
        completed = bool(step.get("completed"))
        fill = _ACCEPTED if completed else _ABANDONED
        parts.append(_rect(cx - bar_w / 2, baseline - h, bar_w, h, fill,
                           opacity=1.0 if completed else 0.30,
                           stroke="none" if completed else _ABANDONED,
                           dash=None if completed else "3 2"))
        if not completed:
            parts.append(_text(cx, baseline - h - 6, "abandoned", size=9,
                               fill=_ABANDONED, anchor="middle"))
        if any(e.get("kind") == "invariant" for e in step.get("events", [])):
            parts.append(_text(cx, baseline - h - 6, "⚠", size=12,
                               fill=_FLAG, anchor="middle"))

    # step index labels, thinned so they never collide
    stride = max(1, int(math.ceil(14.0 / max(slot, 1.0))))
    for i, step in enumerate(steps):
        if i % stride == 0 or not step.get("completed"):
            parts.append(_text(centres[i], baseline + 13, step.get("index", i),
                               size=9, fill=_MUTED, anchor="middle"))
    parts.append(_text(margin, baseline + 13, "step", size=9, fill=_MUTED,
                       anchor="end"))
    y = baseline + 26

    # --- backtrack arcs ---
    for note in notes:
        after = note.get("after_position")
        target = note.get("to_position")
        if after is None or not (0 <= after < len(centres)):
            continue
        x_from = centres[after]
        x_to = centres[target] if target is not None and 0 <= target < len(centres) \
            else margin
        lift = arc_top + 4
        parts.append(
            f'<path d="M {x_from:.1f} {arc_top + arc_lane:.1f} '
            f'C {x_from:.1f} {lift:.1f} {x_to:.1f} {lift:.1f} '
            f'{x_to:.1f} {arc_top + arc_lane:.1f}" fill="none" '
            f'stroke="{_ABANDONED}" stroke-width="1.2" stroke-dasharray="4 2"/>'
        )
        parts.append(
            f'<path d="M {x_to:.1f} {arc_top + arc_lane:.1f} l -3 -5 l 6 0 z" '
            f'fill="{_ABANDONED}"/>'
        )
        parts.append(_text((x_from + x_to) / 2, lift - 3,
                           note.get("short", "backtrack"), size=9,
                           fill=_ABANDONED, anchor="middle"))

    # --- wall clock strip ---
    if any(walls):
        y += 10
        parts.append(_text(margin, y, "wall clock", size=10, fill=_MUTED))
        y += 8
        strip = 26
        wall_max = max(walls) or 1.0
        for i, wall in enumerate(walls):
            h = max(1.0, (wall / wall_max) * strip)
            parts.append(_rect(centres[i] - bar_w / 2, y + strip - h, bar_w, h,
                               _WALL, rx=1.0))
        parts.append(_line(margin, y + strip, margin + inner, y + strip, _RULE))
        slowest = walls.index(wall_max)
        parts.append(_text(margin + inner, y - 6,
                           f"peak {wall_max:.2f} s at step "
                           f"{steps[slowest].get('index', slowest)}",
                           size=9, fill=_MUTED, anchor="end"))
        y += strip + 14

    # --- the operator sequence: once if shared, exceptions called out --------
    y += 14
    parts.append(_line(margin, y, margin + inner, y, _RULE))
    y += 18

    signatures = {}
    for i, step in enumerate(steps):
        signatures.setdefault(_signature(step), []).append(i)
    common = max(signatures.items(), key=lambda kv: len(kv[1]))

    parts.append(_text(margin, y, "Every step:" if len(signatures) == 1
                       else f"{len(common[1])} of {len(steps)} steps:",
                       size=11, weight="600"))
    y += 15
    budget = int((inner - 20) / 6.7)          # monospace at 11px
    for line in _wrap(common[0], budget):
        parts.append(_text(margin + 10, y, line, size=11, fill=_ACCEPTED,
                           family="'SF Mono', Menlo, monospace"))
        y += 15
    y += 3

    exceptions = [(sig, idx) for sig, idx in signatures.items() if sig != common[0]]
    if exceptions:
        y += 6
        parts.append(_text(margin, y, "Steps that did something else:", size=11,
                           weight="600", fill=_FLAG))
        y += 15
        for sig, indices in exceptions:
            named = ", ".join(str(steps[i].get("index", i)) for i in indices[:8])
            if len(indices) > 8:
                named += f", +{len(indices) - 8} more"
            parts.append(_text(margin + 10, y, f"step {named}:", size=10, fill=_MUTED))
            y += 13
            for line in _wrap(sig, int((inner - 30) / 6.7)):
                parts.append(_text(margin + 20, y, line, size=11, fill=_FLAG,
                                   family="'SF Mono', Menlo, monospace"))
                y += 15
            y += 4

    flagged = [
        (step.get("index", i), event["detail"])
        for i, step in enumerate(steps)
        for event in step.get("events", [])
        if event.get("kind") == "invariant"
    ]
    if flagged:
        y += 6
        parts.append(_text(margin, y, "⚠  Invariant:", size=11, weight="600",
                           fill=_FLAG))
        y += 15
        for index, detail in flagged:
            parts.append(_text(margin + 10, y,
                               f"step {index}: history advanced more than once "
                               f"({detail}) — the step was taken twice",
                               size=10, fill=_INK))
            y += 14

    height = int(y + margin)
    body = "\n".join(parts).replace("__H__", str(height))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n{body}\n</svg>\n'
    )


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------

def journal_flowchart(source, run=-1, out=None):
    """The operator flow of a step, as Mermaid, for dropping into documentation.

    Returns the Mermaid source. When every step ran the same sequence — the
    usual case — that is one flowchart. When they did not, each distinct
    sequence becomes its own subgraph, labelled with the steps that took it,
    which is what makes an anomalous step visible rather than averaged away.
    """
    runs = _as_runs(source)
    steps = _pick_run(runs, run)["steps"]

    signatures = {}
    for i, step in enumerate(steps):
        signatures.setdefault(_signature(step), []).append(step.get("index", i))

    lines = ["flowchart LR"]
    for group, (signature, indices) in enumerate(signatures.items()):
        if len(signatures) > 1:
            named = ", ".join(str(i) for i in indices[:6])
            if len(indices) > 6:
                named += f", +{len(indices) - 6}"
            lines.append(f'  subgraph g{group}["step {named}"]')
            lines.append("    direction LR")
        indent = "    " if len(signatures) > 1 else "  "
        if not signature:
            lines.append(f'{indent}n{group}_0["(nothing)"]')
        previous = None
        for i, (kind, name) in enumerate(signature):
            node = f"n{group}_{i}"
            if kind == "history_shift":
                lines.append(f'{indent}{node}[/"shift {name}"/]')
            else:
                lines.append(f'{indent}{node}["{name}"]')
            if previous is not None:
                lines.append(f"{indent}{previous} --> {node}")
            previous = node
        if len(signatures) > 1:
            lines.append("  end")

    text = "\n".join(lines) + "\n"
    if out:
        directory = os.path.dirname(out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return text
