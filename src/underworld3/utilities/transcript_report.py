"""Turn a run's step log into a figure.

The log is written to be watched (:attr:`underworld3.Model.transcript_file`); this
module turns it into something to put in a paper or read on a page.

``transcript_diagram``
    What the run DID, as SVG or PDF. Time runs DOWN the page, one row per step,
    so the figure is portrait, paginates, and drops into a document column.

``transcript_flowchart``
    What ONE step does, as Mermaid, for dropping into documentation.

The layout decision that makes a long run legible: each distinct operator
sequence gets a LETTER, and the letters are defined once at the foot of the
figure. A column of ``A`` with a single ``B`` in it says, at a glance, that one
step did something different — where a hundred repeated sequences say nothing
and hide the one that matters.

Neither renderer needs a plotting library. The SVG and the PDF are both written
directly, so there is no rasterisation, no dependency, and no theme to fight.
"""

from __future__ import annotations

import html
import math
import os
import zlib

__all__ = ["transcript_diagram", "transcript_flowchart",
           "transcript_score", "transcript_score_figure"]


# --- palette ---------------------------------------------------------------
# Print-safe: the two step colours differ in value as well as hue, so they
# survive a greyscale photocopier and a colour-blind reader alike.
_INK = (0.11, 0.11, 0.12)
_MUTED = (0.42, 0.45, 0.50)
_RULE = (0.84, 0.83, 0.81)
_PAPER = (0.992, 0.988, 0.980)
_ACCEPTED = (0.29, 0.44, 0.65)
_ABANDONED = (0.71, 0.33, 0.29)
_WALL = (0.60, 0.65, 0.69)
_FLAG = (0.76, 0.46, 0.12)

# A4 portrait in points, which is also close enough to US Letter that the
# figure sits inside either with margins to spare.
PAGE_W, PAGE_H = 595.0, 842.0


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _as_runs(source):
    """Accept a path, the list ``read_transcript`` returns, or a live model."""
    if isinstance(source, (str, os.PathLike)):
        import underworld3 as uw

        return uw.read_transcript(str(source))
    if hasattr(source, "transcript") and hasattr(source, "tracker"):
        # A live model: the run has not ended, and saying so is different from
        # a file that stops without a terminator, which may have been killed.
        return [{"run": source._run_header(),
                 "steps": [entry.as_dict() for entry in source.transcript],
                 "notes": [], "ended": None, "live": True}]
    if isinstance(source, list):
        if source and isinstance(source[0], dict) and "steps" in source[0]:
            return source
        return [{"run": None, "steps": list(source), "notes": []}]
    raise TypeError(
        f"expected a transcript path, the list read_transcript returns, or a Model; "
        f"got {type(source).__name__}"
    )


def _pick_run(runs, index):
    populated = [r for r in runs if r.get("steps")]
    if not populated:
        raise ValueError("this transcript holds no steps")
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

    The prefix says which base class implemented it, which is never the thing
    the reader is checking."""
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
    return "  ".join(
        name if kind == "solve" else f"shift {name}" for kind, name in signature
    ) or "(nothing)"


def _sequence_text(signature):
    parts = [name if kind == "solve" else f"shift {name}"
             for kind, name in signature]
    return "  >  ".join(parts) or "(nothing)"


def _wrap(text, budget):
    """Break a sequence between operators, never inside one."""
    if len(text) <= budget:
        return [text]
    lines, current = [], ""
    for token in text.split("  >  "):
        piece = token if not current else f"  >  {token}"
        if current and len(current) + len(piece) > budget:
            lines.append(current + "  >")
            current = token
        else:
            current += piece
    lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# A tiny drawing model, so one layout can be written to two formats
# ---------------------------------------------------------------------------

class _Canvas:
    """Ops in a top-left origin, y increasing downward (SVG's convention).

    The PDF writer flips y on the way out; nothing in the layout code has to
    know which format it is being drawn into.
    """

    def __init__(self):
        self.pages = [[]]

    @property
    def ops(self):
        return self.pages[-1]

    def new_page(self):
        self.pages.append([])

    def rect(self, x, y, w, h, fill=None, stroke=None, dash=None, width=1.0):
        self.ops.append(("rect", x, y, max(w, 0.4), max(h, 0.4), fill, stroke,
                         dash, width))

    def line(self, x1, y1, x2, y2, stroke=_RULE, width=0.7, dash=None):
        self.ops.append(("line", x1, y1, x2, y2, stroke, width, dash))

    def curve(self, points, stroke=_ABANDONED, width=1.0, dash=None):
        self.ops.append(("curve", list(points), stroke, width, dash))

    def text(self, x, y, content, size=9.0, fill=_INK, anchor="start",
             bold=False, mono=False):
        self.ops.append(("text", x, y, str(content), size, fill, anchor,
                         bold, mono))

    def note(self, x, y, r, fill, hollow=False):
        """A note head. Hollow when the bar it sits in was abandoned."""
        self.ops.append(("note", x, y, r, fill, bool(hollow)))


_HELV_EM, _COUR_EM = 0.53, 0.60


def _text_width(content, size, mono):
    return len(content) * (_COUR_EM if mono else _HELV_EM) * size


# ---------------------------------------------------------------------------
# Layout — time runs down the page
# ---------------------------------------------------------------------------

_MARGIN = 40.0
_ROW = 14.0
# The left gutter carries the backtrack arrows and their labels. Wide enough
# for "restore + rewind 1" at 7pt, because a label that runs off the page is
# worse than no label.
_GUTTER = 58.0


def _layout(header, steps, notes, title=None, width=PAGE_W, page_height=None):
    """Draw the run onto a canvas. Returns ``(canvas, width, height)``."""
    canvas = _Canvas()
    right = width - _MARGIN

    unit = None
    for step in steps:
        unit = _unit(step.get("t1")) or _unit(step.get("dt"))
        if unit:
            break
    short = _short_unit(unit)

    dts = [_converted(step.get("dt"), unit) for step in steps]
    t1s = [_converted(step.get("t1"), unit) for step in steps]
    walls = [step.get("wall") or 0.0 for step in steps]
    finite = [d for d in dts if math.isfinite(d) and d > 0]
    dt_max = max(finite) if finite else 1.0
    dt_min = min(finite) if finite else 1.0
    wall_max = max(walls) if any(walls) else 1.0

    # A rejected step is often tens of times the accepted ones — that IS why it
    # was rejected — and on a linear axis it flattens everything else to
    # nothing. Switch to log and say so, rather than quietly clipping the bar
    # that carries the story.
    log_scale = dt_max / max(dt_min, 1e-300) > 20.0

    # --- one letter per distinct operator sequence -------------------------
    order, letters = [], {}
    for step in steps:
        signature = _signature(step)
        if signature not in letters:
            letters[signature] = chr(ord("A") + len(order)) if len(order) < 26 \
                else f"#{len(order)}"
            order.append(signature)
    counts = {}
    for step in steps:
        counts[_signature(step)] = counts.get(_signature(step), 0) + 1

    # --- columns -----------------------------------------------------------
    x_gutter = _MARGIN + _GUTTER       # backtrack arrows live to the left
    x_index = x_gutter + 26.0          # step number, right aligned
    x_time = x_index + 54.0            # t, right aligned
    x_dt = x_time + 52.0               # dt, right aligned
    x_letter = x_dt + 16.0             # sequence letter
    x_bar = x_letter + 16.0
    x_wall = right - 34.0
    bar_max = x_wall - x_bar - 12.0

    def bar_length(value):
        if not math.isfinite(value) or value <= 0:
            return 0.6
        if log_scale:
            lo, hi = math.log10(dt_min), math.log10(dt_max)
            frac = 0.06 + 0.94 * ((math.log10(value) - lo) / (hi - lo)
                                  if hi > lo else 1.0)
        else:
            frac = value / dt_max
        return max(1.0, frac * bar_max)

    def draw_header(y, first):
        if first:
            canvas.text(_MARGIN, y + 12,
                        title or f"Run log — {header.get('model', 'model')!r}",
                        size=14, bold=True)
            y += 20
            bits = []
            if header.get("started"):
                bits.append(f"started {header['started']}")
            accepted = [s for s in steps if s.get("completed")]
            if accepted:
                t_from = _converted(accepted[0].get("t0"), unit)
                t_to = _converted(accepted[-1].get("t1"), unit)
                if math.isfinite(t_from) and math.isfinite(t_to):
                    bits.append(f"t = {t_from:.4g} to {t_to:.4g}"
                                + (f" {short}" if short else ""))
            bits.append(f"{len(steps)} steps")
            abandoned = sum(1 for s in steps if not s.get("completed"))
            if abandoned:
                bits.append(f"{abandoned} abandoned")
            if notes:
                bits.append(f"{len(notes)} backtrack(s)")
            canvas.text(_MARGIN, y + 8, "  ·  ".join(bits), size=8.5, fill=_MUTED)
            y += 13
            scales = header.get("scales") or {}
            if scales:
                canvas.text(_MARGIN, y + 8, "scales: " + "   ".join(
                    f"{name} {value['magnitude']:.4g} {_short_unit(value['units'])}"
                    for name, value in scales.items() if isinstance(value, dict)
                ), size=8, fill=_MUTED)
                y += 12
            y += 10
        # column captions
        canvas.text(x_index, y + 8, "step", size=8, fill=_MUTED, anchor="end")
        canvas.text(x_time, y + 8, f"t/{short}" if short else "t", size=8,
                    fill=_MUTED, anchor="end")
        canvas.text(x_dt, y + 8, f"dt/{short}" if short else "dt", size=8,
                    fill=_MUTED, anchor="end")
        canvas.text(x_letter, y + 8, "seq", size=8, fill=_MUTED)
        caption = "dt" + (" (log scale)" if log_scale else "")
        canvas.text(x_bar + 2, y + 8, caption, size=8, fill=_MUTED)
        canvas.text(right, y + 8, "wall", size=8, fill=_MUTED, anchor="end")
        y += 12
        canvas.line(_MARGIN, y, right, y, _RULE, 0.7)
        return y + 4

    # --- rows --------------------------------------------------------------
    y = _MARGIN
    y = draw_header(y, first=True)
    row_y = {}
    row_page = {}

    for i, step in enumerate(steps):
        if page_height is not None and y + _ROW > page_height - _MARGIN - 20:
            canvas.text(_MARGIN, page_height - _MARGIN + 4, "continued", size=7.5,
                        fill=_MUTED)
            canvas.new_page()
            y = _MARGIN
            y = draw_header(y, first=False)

        row_y[i] = y
        row_page[i] = len(canvas.pages) - 1
        completed = bool(step.get("completed"))
        colour = _ACCEPTED if completed else _ABANDONED
        text_colour = _INK if completed else _ABANDONED
        base = y + _ROW - 4

        canvas.text(x_index, base, step.get("index", i), size=8.5,
                    fill=text_colour, anchor="end")
        if math.isfinite(t1s[i]):
            canvas.text(x_time, base, f"{t1s[i]:.4g}", size=8.5,
                        fill=text_colour, anchor="end")
        if math.isfinite(dts[i]):
            canvas.text(x_dt, base, f"{dts[i]:.4g}", size=8.5,
                        fill=text_colour, anchor="end")
        canvas.text(x_letter, base, letters[_signature(step)], size=8.5,
                    fill=colour, bold=True)

        # An abandoned bar is usually the longest on the page — that is why it
        # was abandoned — so it has to leave room for the word that says so.
        room = bar_max - (0.0 if completed else 46.0)
        length = min(bar_length(dts[i]), room)
        canvas.rect(x_bar, y + 2.5, length, _ROW - 6.5,
                    fill=colour if completed else None,
                    stroke=None if completed else _ABANDONED,
                    dash=None if completed else (2.0, 1.5))
        if not completed:
            canvas.text(x_bar + length + 4, base, "abandoned", size=7.5,
                        fill=_ABANDONED)

        if any(walls):
            w = max(0.6, (walls[i] / wall_max) * 30.0)
            canvas.rect(right - w, y + 4.0, w, _ROW - 9.0, fill=_WALL)

        y += _ROW

    canvas.line(_MARGIN, y + 2, right, y + 2, _RULE, 0.7)
    y += 6

    # --- backtracks, in the left gutter ------------------------------------
    # Backtracks go in the left gutter, and they are drawn as the path the run
    # actually took: BACK from the step it bailed out of, to the step whose
    # state it returned to, and then DOWN from that step to the row that redoes
    # it. Two arrows rather than one, because they are two different things —
    # an undo, and the repeat that follows it — and the pair is what makes the
    # repeated step index in the table read as a repeat rather than a typo.
    # They are drawn last but must land on the PAGE THEIR ROWS ARE ON, not on
    # whichever page the cursor happens to have reached.
    # Two calls that make the same jump — a load_state followed by a rewind to
    # the same place — are one backtrack in the run's story and one arrow on
    # the page. Grouping them also stops their labels printing over each other.
    grouped = []
    for note in notes:
        key = (note.get("after_position"), note.get("to_position"))
        if grouped and grouped[-1][0] == key:
            grouped[-1][1].append(note)
        else:
            grouped.append((key, [note]))

    resumed = set()
    for track, ((after, target), members) in enumerate(grouped):
        note = members[0]
        note = dict(note)
        # Label the group by the LAST note in it: that is the call that set
        # where the run ended up, and it is the more specific one (a rewind
        # names how many steps it undid). The others are in the log; a gutter
        # label has room for the net effect, not the sequence of calls.
        note["short"] = members[-1].get("short", members[-1].get("kind", "back"))
        if len(members) > 1:
            note["short"] += f" (+{len(members) - 1})"
        if after is None or after not in row_y:
            continue
        page = row_page[after]
        ops = canvas.pages[page]
        x = x_gutter - 6.0 - 4.0 * (track % 3)
        y_from = row_y[after] + _ROW - 3
        label = note.get("short", "back")

        if target is None or target not in row_y:
            # Nothing recorded to point at: mark where it happened.
            ops.append(("line", x, y_from, x + 4, y_from, _ABANDONED, 0.8, None))
            ops.append(("text", x - 6, y_from, label, 7, _ABANDONED,
                        "end", False, False))
            continue

        if row_page[target] != page:
            # It reached back past a page break; say so where it happened
            # rather than draw a line to a row that is not on this page.
            ops.append(("line", x, y_from, x + 4, y_from, _ABANDONED, 0.8, None))
            ops.append(("line", x, y_from, x, y_from - _ROW * 0.8, _ABANDONED,
                        0.8, (2.0, 1.5)))
            ops.append(("tri", x, y_from - _ROW * 0.8, 2.5, _ABANDONED))
            ops.append(("text", x - 6, y_from, f"{label} \u2191", 7,
                        _ABANDONED, "end", False, False))
            continue

        y_to = row_y[target] + 3
        if y_to <= y_from:
            ops.append(("line", x, y_from, x, y_to, _ABANDONED, 0.8, (2.0, 1.5)))
            ops.append(("line", x, y_from, x + 4, y_from, _ABANDONED, 0.8, None))
            ops.append(("tri", x, y_to, 2.5, _ABANDONED))
            ops.append(("text", x - 6,
                        (y_from + y_to) / 2 + 3 if y_from - y_to >= _ROW * 2.0
                        else y_from + 1, label, 7, _ABANDONED, "end", False,
                        False))

        # ... and the repeat. The row that follows the backtrack is the run
        # picking the step up again; joining the two says so.
        redo = after + 1
        if (redo in row_y and row_page[redo] == page and target not in resumed
                and steps[redo].get("index") == steps[target].get("index")):
            resumed.add(target)
            xr = x - 6.0
            y_top = row_y[target] + _ROW - 3
            y_bottom = row_y[redo] + _ROW / 2
            ops.append(("line", xr, y_top, xr, y_bottom, _ACCEPTED, 0.8, None))
            ops.append(("line", xr, y_top, xr + 3, y_top, _ACCEPTED, 0.8, None))
            ops.append(("tri_down", xr, y_bottom, 2.5, _ACCEPTED))
            ops.append(("text", xr - 4, y_bottom + 3, "again", 7,
                        _ACCEPTED, "end", False, False))

    # --- the legend: what each letter means --------------------------------
    if page_height is not None and y + 30 + 14 * len(order) > page_height - _MARGIN:
        canvas.new_page()
        y = _MARGIN

    y += 10
    canvas.text(_MARGIN, y + 8, "Operator sequences  —  what the seq letter "
                "on each row stands for", size=9.5, bold=True)
    y += 18
    budget = int((right - _MARGIN - 34) / (_COUR_EM * 8.0))
    for signature in order:
        canvas.text(_MARGIN + 2, y + 8, letters[signature], size=9, bold=True,
                    fill=_ACCEPTED if signature == order[0] else _FLAG)
        count = counts[signature]
        canvas.text(right, y + 8, f"{count} step{'' if count == 1 else 's'}",
                    size=8, fill=_MUTED, anchor="end")
        for line in _wrap(_sequence_text(signature), budget):
            canvas.text(_MARGIN + 18, y + 8, line, size=8, mono=True,
                        fill=_ACCEPTED if signature == order[0] else _FLAG)
            y += 11
        y += 5

    height = page_height if page_height is not None else y + _MARGIN
    return canvas, width, height


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

def _hex(colour):
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in colour)


def _svg_ops(ops, width, height):
    out = [f'<rect x="0" y="0" width="{width:.1f}" height="{height:.1f}" '
           f'fill="{_hex(_PAPER)}"/>']
    for op in ops:
        kind = op[0]
        if kind == "rect":
            _, x, y, w, h, fill, stroke, dash, lw = op
            attrs = f'x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="1"'
            attrs += f' fill="{_hex(fill)}"' if fill else ' fill="none"'
            if stroke:
                attrs += f' stroke="{_hex(stroke)}" stroke-width="{lw}"'
                if dash:
                    attrs += f' stroke-dasharray="{dash[0]} {dash[1]}"'
            out.append(f"<rect {attrs}/>")
        elif kind == "line":
            _, x1, y1, x2, y2, stroke, lw, dash = op
            attrs = (f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{_hex(stroke)}" stroke-width="{lw}"')
            if dash:
                attrs += f' stroke-dasharray="{dash[0]} {dash[1]}"'
            out.append(f"<line {attrs}/>")
        elif kind in ("tri", "tri_down"):
            _, x, y, r, colour = op
            dy = r * 1.6 if kind == "tri" else -r * 1.6
            out.append(f'<path d="M {x:.1f} {y:.1f} l {-r:.1f} {dy:.1f} '
                       f'l {r * 2:.1f} 0 z" fill="{_hex(colour)}"/>')
        elif kind == "note":
            _, x, y, r, colour, hollow = op
            if hollow:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
                           f'fill="none" stroke="{_hex(colour)}" '
                           f'stroke-width="1.1" stroke-dasharray="2 1.4"/>')
            else:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
                           f'fill="{_hex(colour)}"/>')
        elif kind == "text":
            _, x, y, content, size, fill, anchor, bold, mono = op
            family = ("'SF Mono', Menlo, monospace" if mono
                      else "'Helvetica Neue', Helvetica, Arial, sans-serif")
            out.append(
                f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
                f'font-size="{size}" fill="{_hex(fill)}" '
                f'text-anchor="{"middle" if anchor == "middle" else anchor}" '
                f'font-weight="{"600" if bold else "normal"}">'
                f'{html.escape(content)}</text>'
            )
    body = "\n".join(out)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">\n'
            f'{body}\n</svg>\n')


# ---------------------------------------------------------------------------
# PDF — written directly, so the figure needs nothing installed to become one
# ---------------------------------------------------------------------------

_PDF_SUBSTITUTIONS = {
    "→": "->", "·": "-", "⚠": "!", "—": "-", "–": "-",
    "…": "...", "↑": "^", "↓": "v", "×": "x",
    "'": "'", "'": "'", """: '"', """: '"', "≥": ">=", "≤": "<=",
}


def _pdf_text(content):
    """WinAnsi-safe, with the escapes PDF strings need."""
    for source, target in _PDF_SUBSTITUTIONS.items():
        content = content.replace(source, target)
    content = content.encode("latin-1", "replace").decode("latin-1")
    return content.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _pdf_page_stream(ops, width, height):
    """One page's content stream. PDF's origin is bottom-left, so y flips."""
    def fy(y):
        return height - y

    out = [f"{_PAPER[0]:.3f} {_PAPER[1]:.3f} {_PAPER[2]:.3f} rg",
           f"0 0 {width:.1f} {height:.1f} re f"]
    for op in ops:
        kind = op[0]
        if kind == "rect":
            _, x, y, w, h, fill, stroke, dash, lw = op
            out.append("q")
            if dash:
                out.append(f"[{dash[0]} {dash[1]}] 0 d")
            if fill:
                out.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
            if stroke:
                out.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG "
                           f"{lw} w")
            out.append(f"{x:.2f} {fy(y + h):.2f} {w:.2f} {h:.2f} re")
            out.append("B" if (fill and stroke) else ("f" if fill else "S"))
            out.append("Q")
        elif kind == "line":
            _, x1, y1, x2, y2, stroke, lw, dash = op
            out.append("q")
            if dash:
                out.append(f"[{dash[0]} {dash[1]}] 0 d")
            out.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG {lw} w")
            out.append(f"{x1:.2f} {fy(y1):.2f} m {x2:.2f} {fy(y2):.2f} l S")
            out.append("Q")
        elif kind in ("tri", "tri_down"):
            _, x, y, r, colour = op
            dy = r * 1.6 if kind == "tri" else -r * 1.6
            out.append(f"q {colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} rg")
            out.append(f"{x:.2f} {fy(y):.2f} m {x - r:.2f} {fy(y + dy):.2f} l "
                       f"{x + r:.2f} {fy(y + dy):.2f} l f Q")
        elif kind == "note":
            # A circle from four Beziers; PDF has no primitive for one.
            _, x, y, r, colour, hollow = op
            k = r * 0.5523
            yy = fy(y)
            out.append("q")
            if hollow:
                out.append(f"{colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} RG "
                           f"1.1 w [2 1.4] 0 d")
            else:
                out.append(f"{colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} rg")
            out.append(f"{x + r:.2f} {yy:.2f} m")
            out.append(f"{x + r:.2f} {yy + k:.2f} {x + k:.2f} {yy + r:.2f} "
                       f"{x:.2f} {yy + r:.2f} c")
            out.append(f"{x - k:.2f} {yy + r:.2f} {x - r:.2f} {yy + k:.2f} "
                       f"{x - r:.2f} {yy:.2f} c")
            out.append(f"{x - r:.2f} {yy - k:.2f} {x - k:.2f} {yy - r:.2f} "
                       f"{x:.2f} {yy - r:.2f} c")
            out.append(f"{x + k:.2f} {yy - r:.2f} {x + r:.2f} {yy - k:.2f} "
                       f"{x + r:.2f} {yy:.2f} c")
            out.append("S Q" if hollow else "f Q")
        elif kind == "text":
            _, x, y, content, size, fill, anchor, bold, mono = op
            font = "/F3" if mono else ("/F2" if bold else "/F1")
            w = _text_width(content, size, mono)
            if anchor == "end":
                x -= w
            elif anchor == "middle":
                x -= w / 2
            out.append(f"BT {font} {size:.1f} Tf "
                       f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg "
                       f"{x:.2f} {fy(y):.2f} Td ({_pdf_text(content)}) Tj ET")
    return "\n".join(out).encode("latin-1", "replace")


def _pdf_document(pages, width, height):
    objects = {}
    n_pages = len(pages)
    font_ids = {"F1": 3, "F2": 4, "F3": 5}
    first_page = 6

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{first_page + 2 * i} 0 R" for i in range(n_pages))
    objects[2] = (f"<< /Type /Pages /Count {n_pages} /Kids [{kids}] >>"
                  ).encode("latin-1")
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    objects[5] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"

    for i, ops in enumerate(pages):
        page_id = first_page + 2 * i
        stream_id = page_id + 1
        resources = ("<< /Font << " + " ".join(
            f"/{name} {oid} 0 R" for name, oid in font_ids.items()) + " >> >>")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.1f} {height:.1f}] "
            f"/Resources {resources} /Contents {stream_id} 0 R >>"
        ).encode("latin-1")
        raw = _pdf_page_stream(ops, width, height)
        packed = zlib.compress(raw)
        objects[stream_id] = (
            f"<< /Length {len(packed)} /Filter /FlateDecode >>\nstream\n"
        ).encode("latin-1") + packed + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for oid in sorted(objects):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode("latin-1") + objects[oid] + b"\nendobj\n"

    xref_at = len(out)
    top = max(objects) + 1
    out += f"xref\n0 {top}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for oid in range(1, top):
        out += f"{offsets.get(oid, 0):010d} 00000 n \n".encode("latin-1")
    out += (f"trailer\n<< /Size {top} /Root 1 0 R >>\nstartxref\n{xref_at}\n"
            f"%%EOF\n").encode("latin-1")
    return bytes(out)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def transcript_diagram(source, out=None, run=-1, title=None, format=None,
                    width=None):
    """Render a run's log as a figure, with time running DOWN the page.

    Parameters
    ----------
    source : str, list or Model
        A ``.jsonl`` transcript file, the list :func:`underworld3.read_transcript`
        returns, or a live model. Not a text log — that format is a report and
        cannot be read back.
    out : str, optional
        Where to write. Defaults to the source path with the format's suffix,
        else ``transcript.pdf``.
    run : int, default -1
        Which run in the file. A file holds one per ``clear_transcript()``.
    title : str, optional
        Overrides the heading taken from the run header.
    format : {"pdf", "svg"}, optional
        Inferred from ``out``'s suffix; PDF by default. PDF paginates onto A4
        portrait; SVG is one continuous page.
    width : float, optional
        Page width in points. Defaults to A4 portrait.

    Returns
    -------
    str
        The path written.
    """
    runs = _as_runs(source)
    entry = _pick_run(runs, run)

    if format is None:
        if out and str(out).lower().endswith(".svg"):
            format = "svg"
        else:
            format = "pdf"
    if format not in ("pdf", "svg"):
        raise ValueError(f"format must be 'pdf' or 'svg', not {format!r}")

    if out is None:
        suffix = ".svg" if format == "svg" else ".pdf"
        out = (os.path.splitext(str(source))[0] + suffix
               if isinstance(source, (str, os.PathLike)) else "transcript" + suffix)

    page_width = width or PAGE_W
    canvas, page_width, height = _layout(
        entry.get("run") or {}, entry["steps"], entry.get("notes", []),
        title=title, width=page_width,
        page_height=PAGE_H if format == "pdf" else None,
    )

    directory = os.path.dirname(out)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if format == "svg":
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(_svg_ops(canvas.pages[0], page_width, height))
    else:
        with open(out, "wb") as handle:
            handle.write(_pdf_document(canvas.pages, page_width, PAGE_H))
    return out


def transcript_flowchart(source, run=-1, out=None):
    """The operator flow of a step, as Mermaid, for dropping into documentation.

    When every step ran the same sequence — the usual case — that is one
    flowchart. When they did not, each distinct sequence becomes its own
    subgraph, labelled with the steps that took it, which is what makes an
    anomalous step visible rather than averaged away.
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


# ---------------------------------------------------------------------------
# The score: parts across the page, bars down it
# ---------------------------------------------------------------------------

def _parts_of(steps):
    """The roster, in a stable order, from what actually played.

    A part that never plays gets no stave — music's rule, and the one that
    keeps a model's registered cast from becoming a page of empty columns.
    Keyed on ``part`` (the instance) rather than ``name`` (the label), so two
    solvers that render alike stay distinct and a change to how labels are
    built does not re-partition an old transcript.
    """
    order, labels = [], {}
    for step in steps:
        for event in step.get("events", []):
            if event.get("kind") not in ("solve", "history_shift"):
                continue
            key = event.get("part") or event.get("name")
            if key not in labels:
                order.append(key)
                labels[key] = _short_operator(event.get("name", key))
            elif event.get("kind") == "history_shift":
                labels[key] = _short_operator(event.get("name", key))
    # Actors first, then the state they carry: a score is laid out by family,
    # not by order of first entry, so a part sits in the same place every run.
    actors = [k for k in order if "#" not in str(k) or not _is_history(k, steps)]
    histories = [k for k in order if k not in actors]
    return [(k, labels[k]) for k in actors + histories]


def _is_history(key, steps):
    for step in steps:
        for event in step.get("events", []):
            if (event.get("part") or event.get("name")) == key:
                return event.get("kind") == "history_shift"
    return False


def _bar_cells(step, parts):
    """One cell per part: the positions at which it played, or a rest."""
    played = []
    for event in step.get("events", []):
        if event.get("kind") in ("solve", "history_shift"):
            played.append(event.get("part") or event.get("name"))
    cells = []
    for key, _ in parts:
        hits = [i + 1 for i, k in enumerate(played) if k == key]
        cells.append(",".join(str(h) for h in hits) if hits else None)
    return cells


def _score_rows(steps, parts, note_at, anchors=()):
    """Bars grouped into rows: a bar, or a run of identical bars collapsed.

    Returns ``(kind, first_index, last_index, cells, count)`` with ``kind``
    either ``"bar"`` or ``"simile"``. The collapse asserts identity — same
    cells, same outcome, same label, nothing recorded between them — which is
    what makes it safe to hide thirteen rows behind one mark.
    """
    anchors = set(anchors)
    rows, i = [], 0
    while i < len(steps):
        cells = _bar_cells(steps[i], parts)
        j = i + 1
        while (j < len(steps)
               and (j - 1) not in note_at
               and j not in anchors
               and _bar_cells(steps[j], parts) == cells
               and steps[j].get("completed") == steps[i].get("completed")
               and steps[j].get("label") == steps[i].get("label")):
            j += 1
        rows.append(("bar", i, i, cells, 1))
        if j - i > 1:
            rows.append(("simile", i + 1, j - 1, cells, j - i - 1))
        i = j
    return rows


def transcript_score_figure(source, out=None, run=-1, title=None,
                            width=PAGE_W, format=None, collapse=True):
    """Draw the score: a stave per part, bars down the page.

    The same reading as :func:`transcript_score`, set as notation rather than
    as text. A filled head is a part that played, with the order it played in;
    a rest is a part that did nothing; a run of identical bars collapses to one
    band carrying music's simile mark and a count.

    Parameters
    ----------
    source : str, list or Model
        A ``.jsonl`` transcript, the list :func:`underworld3.read_transcript`
        returns, or a live model.
    out : str, optional
        Where to write. Defaults to the source path with ``-score`` and the
        format's suffix.
    format : {"pdf", "svg"}, optional
        Inferred from ``out``'s suffix; PDF by default.
    collapse : bool, default True
        Group consecutive steps that did the same thing into one band. The
        band carries the first and last value of anything that changed across
        it, so a growing timestep survives the grouping. Set ``False`` to draw
        every step: a run whose ``dt`` is itself the thing under examination
        is easier to read one row at a time.

    Returns
    -------
    str
        The path written.
    """
    runs = _as_runs(source)
    entry = _pick_run(runs, run)
    steps, notes = entry["steps"], entry.get("notes", [])
    header = entry.get("run") or {}

    if format is None:
        format = "svg" if (out and str(out).lower().endswith(".svg")) else "pdf"
    if out is None:
        suffix = ".svg" if format == "svg" else ".pdf"
        out = (os.path.splitext(str(source))[0] + "-score" + suffix
               if isinstance(source, (str, os.PathLike)) else "score" + suffix)

    canvas, page_w, height = _score_layout(
        header, steps, notes, entry, title=title, width=width,
        collapse=collapse)

    directory = os.path.dirname(out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if format == "svg":
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(_svg_ops(canvas.pages[0], page_w, height))
    else:
        with open(out, "wb") as handle:
            handle.write(_pdf_document(canvas.pages, page_w, height))
    return out


_STAVE = (0.80, 0.79, 0.77)


def _down_arrow(canvas, x, y_from, y_to, colour, width=0.7):
    """A thin continuation arrow: this carries on, from here to there."""
    canvas.line(x, y_from, x, y_to - 2.2, colour, width)
    canvas.ops.append(("tri_down", x, y_to, 2.0, colour))


def _score_layout(header, steps, notes, entry, title=None, width=PAGE_W,
                  collapse=True):
    canvas = _Canvas()
    right = width - _MARGIN

    unit = None
    for step in steps:
        unit = _unit(step.get("t1")) or _unit(step.get("dt"))
        if unit:
            break
    short = _short_unit(unit)

    parts = _parts_of(steps)
    note_at = {}
    for note in notes:
        note_at.setdefault(note.get("after_position"), []).append(note)
    if collapse:
        anchors = {n.get("to_position") for n in notes}
        anchors |= {n.get("after_position") for n in notes}
        anchors |= {p + 1 for p in list(anchors) if p is not None}
        rows = _score_rows(steps, parts, note_at,
                           anchors={a for a in anchors if a is not None})
    else:
        rows = [("bar", i, i, _bar_cells(step, parts), 1)
                for i, step in enumerate(steps)]

    y = _MARGIN
    canvas.text(_MARGIN, y + 12, title or f"Score — {header.get('model', 'model')!r}",
                size=14, bold=True)
    y += 20
    bits = []
    if header.get("started"):
        bits.append(f"started {header['started']}")
    if entry.get("ended"):
        bits.append(f"complete, {entry['ended'].get('steps', len(steps))} bars")
    elif entry.get("live"):
        bits.append(f"in progress, {len(steps)} bars so far")
    else:
        bits.append("no terminator: still running, or interrupted")
    bits.append(f"{len(parts)} parts")
    canvas.text(_MARGIN, y + 8, "  ·  ".join(bits), size=8.5, fill=_MUTED)
    y += 22

    # --- columns ---
    gutter = _MARGIN + 46.0
    x_bar = gutter + 26.0
    x_time = x_bar + 52.0
    x_dt = x_time + 50.0
    lane0 = x_dt + 22.0
    lane_w = max(44.0, (right - lane0) / max(len(parts), 1))

    def lane_x(i):
        return lane0 + lane_w * (i + 0.5)

    canvas.text(x_bar, y + 8, "step", size=8, fill=_MUTED, anchor="end")
    canvas.text(x_time, y + 8, f"t/{short}" if short else "t", size=8,
                fill=_MUTED, anchor="end")
    canvas.text(x_dt, y + 8, f"dt/{short}" if short else "dt", size=8,
                fill=_MUTED, anchor="end")
    for i, (_, label) in enumerate(parts):
        canvas.text(lane_x(i), y + 8, label[:int(lane_w / 4.6)], size=8,
                    fill=_INK, anchor="middle", bold=True)
    y += 13
    canvas.line(_MARGIN, y, right, y, _RULE, 0.8)
    top = y + 3

    # --- rows ---
    row_h, band_h = 16.0, 30.0
    row_y, row_page = {}, {}
    y = top
    for kind, first, last, cells, count in rows:
        h = row_h if kind == "bar" else band_h
        mid = y + h / 2
        if kind == "bar":
            step = steps[first]
            row_y[first] = mid
            completed = bool(step.get("completed"))
            colour = _ACCEPTED if completed else _ABANDONED
            canvas.text(x_bar, mid + 3, step.get("index", first), size=8.5,
                        fill=_INK if completed else _ABANDONED, anchor="end")
            t1 = _converted(step.get("t1"), unit)
            dt = _converted(step.get("dt"), unit)
            canvas.text(x_time, mid + 3, f"{t1:.6g}", size=8.5,
                        fill=_INK if completed else _ABANDONED, anchor="end")
            canvas.text(x_dt, mid + 3, f"{dt:.6g}", size=8.5,
                        fill=_INK if completed else _ABANDONED, anchor="end")
            for i, cell in enumerate(cells):
                cx = lane_x(i)
                if cell is None:
                    # A rest: notated, because a part that did nothing is
                    # different from a part nobody was watching.
                    canvas.rect(cx - 4.5, mid - 1.0, 9.0, 2.0, fill=_MUTED)
                    continue
                orders = cell.split(",")
                span = 11.0 * (len(orders) - 1)
                for k, order in enumerate(orders):
                    nx = cx - span / 2 + 11.0 * k
                    canvas.note(nx, mid, 4.2, colour, hollow=not completed)
                    canvas.text(nx, mid + 2.6, order, size=6.5,
                                fill=_PAPER if completed else _ABANDONED,
                                anchor="middle", bold=True)
            if not completed:
                canvas.text(right, mid + 3, "abandoned", size=7.5,
                            fill=_ABANDONED, anchor="end")
        else:
            # A run of steps that did the same thing. The band shows the first
            # and last value of anything that CHANGED across it, because a
            # timestep that grew by a factor of eight is a diagnostic, and a
            # symbol that only says "repeats" would throw it away.
            canvas.rect(_MARGIN, y, right - _MARGIN, h, fill=(0.965, 0.958, 0.948))
            first_step, last_step = steps[first], steps[last]
            canvas.text(x_bar, y + 9,
                        f"{first_step.get('index', first)}", size=8, fill=_MUTED,
                        anchor="end")
            canvas.text(x_bar, y + h - 3,
                        f"{last_step.get('index', last)}", size=8, fill=_MUTED,
                        anchor="end")
            canvas.text(x_bar + 7, mid + 3, f"×{count}", size=7, fill=_MUTED)
            for column, key in ((x_time, "t1"), (x_dt, "dt")):
                a = _converted(first_step.get(key), unit)
                b = _converted(last_step.get(key), unit)
                canvas.text(column, y + 9, f"{a:.4g}", size=8, fill=_MUTED,
                            anchor="end")
                if abs(b - a) > 1e-12 * max(1.0, abs(a)):
                    canvas.text(column, y + h - 3, f"{b:.4g}", size=8,
                                fill=_MUTED, anchor="end")
                    _down_arrow(canvas, column + 6, y + 11, y + h - 9, _MUTED)
            for i, _ in enumerate(cells):
                _down_arrow(canvas, lane_x(i), y + 4, y + h - 4, _MUTED)
        y += h

    bottom = y
    # staves, drawn behind nothing but spanning the whole block
    for i, _ in enumerate(parts):
        canvas.line(lane_x(i), top, lane_x(i), bottom, _STAVE, 0.7)
    canvas.line(_MARGIN, bottom, right, bottom, _RULE, 0.8)
    y = bottom + 4

    # --- backtracks, in the gutter ---
    grouped = []
    for note in notes:
        key = (note.get("after_position"), note.get("to_position"))
        if grouped and grouped[-1][0] == key:
            grouped[-1][1].append(note)
        else:
            grouped.append((key, [note]))
    for track, ((after, target), members) in enumerate(grouped):
        if after not in row_y or target not in row_y:
            continue
        x = gutter - 6.0 - 4.0 * (track % 3)
        y_from, y_to = row_y[after], row_y[target]
        label = members[-1].get("short", "back")
        if len(members) > 1:
            label += f" (+{len(members) - 1})"
        canvas.line(x, y_from, x, y_to, _ABANDONED, 0.8, dash=(2.0, 1.5))
        canvas.line(x, y_from, x + 5, y_from, _ABANDONED, 0.8)
        canvas.ops.append(("tri", x, y_to + 2.5, 2.5, _ABANDONED))
        canvas.text(x - 5, (y_from + y_to) / 2 + 3, label, size=7,
                    fill=_ABANDONED, anchor="end")
        redo = after + 1
        if redo in row_y and steps[redo].get("index") == steps[target].get("index"):
            xr = x - 6.0
            canvas.line(xr, y_to, xr, row_y[redo], _ACCEPTED, 0.8)
            canvas.line(xr, y_to, xr + 3, y_to, _ACCEPTED, 0.8)
            canvas.ops.append(("tri_down", xr, row_y[redo] - 2.5, 2.5, _ACCEPTED))
            canvas.text(xr - 4, row_y[redo] + 3, "again", size=7,
                        fill=_ACCEPTED, anchor="end")

    # --- legend ---
    y += 14
    canvas.note(_MARGIN + 4, y, 4.2, _ACCEPTED)
    canvas.text(_MARGIN + 16, y + 3,
                "ran; the digit is the order it ran within the step",
                size=8, fill=_INK)
    y += 14
    canvas.note(_MARGIN + 4, y, 4.2, _ABANDONED, hollow=True)
    canvas.text(_MARGIN + 16, y + 3,
                "ran in a step that was then abandoned", size=8, fill=_INK)
    y += 14
    canvas.rect(_MARGIN - 0.5, y - 1.0, 9.0, 2.0, fill=_MUTED)
    canvas.text(_MARGIN + 16, y + 3, "did nothing in this step",
                size=8, fill=_INK)
    y += 14
    _down_arrow(canvas, _MARGIN + 4, y - 5, y + 4, _MUTED)
    canvas.text(_MARGIN + 16, y + 3,
                "the steps between did exactly this, unchanged; first and last "
                "values are shown where they differ", size=8, fill=_INK)
    y += 16

    return canvas, width, y + _MARGIN


def transcript_score(source, run=-1, width=11, collapse=True):
    """The transcript as a score: parts across the page, bars down it.

    Post-hoc by design. The roster is not known until a run has played — a part
    that first enters at bar 300 must still have a stave at bar 1, resting —
    so a score cannot be the thing streamed line-by-line as steps close. It is
    what you read afterwards, or partway through: rendering a transcript that
    is still being written gives the score of what has happened so far, and
    says so.

    Consecutive steps that did the same thing collapse into one line, which
    carries the first and last value of anything that changed across them. The
    grouping asserts that the steps were *identical* in what ran and in what
    order, so nothing can hide behind it and the full record is still
    underneath. Pass ``collapse=False`` to list every step, which is what you
    want when the timestep itself is the thing under examination.

    Parameters
    ----------
    source : str, list or Model
        A ``.jsonl`` transcript, the list :func:`underworld3.read_transcript`
        returns, or a live model.
    run : int, default -1
        Which run in the file.
    width : int, default 11
        Column width for each part.
    collapse : bool, default True
        Group consecutive steps that did the same thing.

    Returns
    -------
    str
    """
    runs = _as_runs(source)
    entry = _pick_run(runs, run)
    steps, notes = entry["steps"], entry.get("notes", [])
    header = entry.get("run") or {}
    ended = entry.get("ended")

    unit = None
    for step in steps:
        unit = _unit(step.get("t1")) or _unit(step.get("dt"))
        if unit:
            break
    short = _short_unit(unit)

    parts = _parts_of(steps)
    note_at = {}
    for note in notes:
        note_at.setdefault(note.get("after_position"), []).append(note)

    out = []
    title = header.get("model")
    out.append(f"score · model {title!r}" if title else "score")
    if header.get("started"):
        out.append(f"started {header['started']}")
    if ended:
        out.append(f"complete — {ended.get('steps', len(steps))} step(s)")
    elif entry.get("live"):
        out.append(f"in progress — {len(steps)} step(s) so far")
    else:
        out.append("no terminator: this run is still going, or it was "
                   "interrupted. What follows is the score of a prefix.")
    out.append("")

    bar_w, t_w = 5, 10
    head = (f"{'step':>{bar_w}} {('t/' + short) if short else 't':>{t_w}} "
            f"{('dt/' + short) if short else 'dt':>{t_w}} │ "
            + " │ ".join(f"{label[:width]:^{width}}" for _, label in parts) + " │")
    out.append(head)
    out.append("─" * len(head))

    def row(step, cells):
        t1 = _converted(step.get("t1"), unit)
        dt = _converted(step.get("dt"), unit)
        body = " │ ".join(f"{(c if c else '·'):^{width}}" for c in cells)
        tail = "" if step.get("completed") else "   ABANDONED"
        return (f"{step.get('index', '?'):>{bar_w}} {t1:>{t_w}.6g} "
                f"{dt:>{t_w}.6g} │ {body} │{tail}")

    i = 0
    while i < len(steps):
        step = steps[i]
        cells = _bar_cells(step, parts)
        # How many bars that follow are IDENTICAL — same cells, same outcome,
        # and nothing happened between them.
        run_len = 0
        j = i + 1
        while (collapse
               and j < len(steps)
               and (j - 1) not in note_at
               and _bar_cells(steps[j], parts) == cells
               and steps[j].get("completed") == step.get("completed")
               and steps[j].get("label") == step.get("label")):
            run_len += 1
            j += 1

        out.append(row(step, cells))
        if run_len:
            last = steps[j - 1]
            t_to = _converted(last.get("t1"), unit)
            dt_from = _converted(step.get("dt"), unit)
            dt_to = _converted(last.get("dt"), unit)
            carry = " │ ".join(f"{'↓':^{width}}" for _ in parts)
            label = f"…{last.get('index', '?')}"
            out.append(f"{label:>{bar_w}} {t_to:>{t_w}.6g} {dt_to:>{t_w}.6g} "
                       f"│ {carry} │   ×{run_len} unchanged"
                       + (f", dt {dt_from:.4g} → {dt_to:.4g}"
                          if abs(dt_to - dt_from) > 1e-12 * max(1.0, abs(dt_from))
                          else ""))
        for k in range(i, j):
            for note in note_at.get(k, []):
                out.append(f"{'':>{bar_w}} {'':>{t_w}} {'':>{t_w}} │ "
                           f"{note.get('message', note.get('kind'))}")
        i = j

    out.append("")
    out.append(f"{len(steps)} step(s), {len(parts)} part(s): "
               + ", ".join(label for _, label in parts))
    out.append("↓  the steps between did exactly this, unchanged")
    out.append("·  this part did nothing in that step")
    out.append("digits are the order the parts ran within the step")
    return "\n".join(out)
