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
import re
import zlib

__all__ = ["transcript_diagram", "transcript_flowchart",
           "transcript_table", "transcript_figure", "transcript_key"]


# --- palette ---------------------------------------------------------------
# Print-safe: the two step colours differ in value as well as hue, so they
# survive a greyscale photocopier and a colour-blind reader alike.
_INK = (0.11, 0.11, 0.12)
_MUTED = (0.42, 0.45, 0.50)
_RULE = (0.84, 0.83, 0.81)
_PAPER = (0.992, 0.988, 0.980)
_ACCEPTED = (0.29, 0.44, 0.65)
_ABANDONED = (0.71, 0.33, 0.29)
# How a solve went. The emoji are what SVG draws; the PDF strokes the same
# three states in these tones, which are the page's palette rather than the
# emoji's own.
_OK = (0.29, 0.50, 0.36)
_CAPPED = (0.80, 0.58, 0.20)
_DIVERGED = (0.71, 0.33, 0.29)
_OUTCOME_EMOJI = {"ok": "\u2705", "capped": "\u26a0\ufe0f", "diverged": "\u274c"}
_OUTCOME_COLOUR = {"ok": _OK, "capped": _CAPPED, "diverged": _DIVERGED}
_TEXT_OUTCOME = {"ok": "", "capped": "!", "diverged": "x"}
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
        path = None
        getter = getattr(source, "transcript_record_path", None)
        if callable(getter):
            path = getter()
        if path and os.path.exists(path):
            # The on-disk record is the complete one — an abandoned step and
            # a rewound one are in the file and not in model.transcript, and
            # a figure that leaves out the step that was rejected is not the
            # figure of that run.
            import underworld3 as uw

            runs = uw.read_transcript(path)
            if runs and runs[-1].get("ended") is None:
                runs[-1]["live"] = True
            return runs
        return [{"run": source._run_header(),
                 "steps": [entry.as_dict() for entry in source.transcript],
                 "parts": list(getattr(source, "_parts", {}).values()),
                 "notes": [], "ended": None, "live": True}]
    if isinstance(source, list):
        if source and isinstance(source[0], dict) and "steps" in source[0]:
            return source
        return [{"run": None, "steps": list(source), "notes": []}]
    raise TypeError(
        f"expected a transcript path, the list read_transcript returns, or a Model; "
        f"got {type(source).__name__}"
    )


_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε",
    "varepsilon": "ε", "eta": "η", "theta": "θ", "kappa": "κ", "upkappa": "κ", "lambda": "λ",
    "uplambda": "λ", "mu": "μ", "nu": "ν", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
    "omega": "ω", "Omega": "Ω", "psi": "ψ", "Psi": "Ψ", "pi": "π", "nabla": "∇", "cdot": "·",
    "times": "×", "partial": "∂", "infty": "∞",
}
_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPT_DIGITS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_FROM_SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_FROM_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _plain_symbol(latex):
    """A LaTeX symbol as readable plain text: ``\\rho_0 \\alpha g`` -> ``ρ₀ α g``,
    ``\\Delta t_{18}`` -> ``Δt₁₈``, ``C^{\\tau}_{u,18}`` -> ``C^τ_u,18``. For a
    page that has no maths renderer; the Markdown key keeps the LaTeX."""
    text = str(latex)
    text = re.sub(r"\\(?:mathrm|mathbf|text|mathtt|left|right)\b", "", text)
    text = re.sub(r"\\([A-Za-z]+)", lambda m: _GREEK.get(m.group(1), m.group(1)), text)
    def sub(m):
        inner = m.group(1)
        return inner.translate(_SUBSCRIPT_DIGITS) if inner.isdigit() else "_" + inner
    def sup(m):
        inner = m.group(1)
        return inner.translate(_SUPERSCRIPT_DIGITS) if inner.isdigit() else "^" + inner
    text = re.sub(r"_\{([^{}]*)\}", sub, text)
    text = re.sub(r"\^\{([^{}]*)\}", sup, text)
    text = re.sub(r"_(\d)", lambda m: m.group(1).translate(_SUBSCRIPT_DIGITS), text)
    text = re.sub(r"\^(\d)", lambda m: m.group(1).translate(_SUPERSCRIPT_DIGITS), text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _plain_unit(units):
    """``meter ** 2 / second`` -> ``m²/s``; ``pascal * second`` -> ``Pa·s``."""
    if not units:
        return ""
    text = str(units)
    for long, short in (("kilogram", "kg"), ("meter", "m"), ("second", "s"),
                        ("kelvin", "K"), ("pascal", "Pa"), ("newton", "N"),
                        ("joule", "J"), ("watt", "W"), ("year", "yr"),
                        ("megayear", "Myr"), ("millimeter", "mm"),
                        ("kilometer", "km"), ("centimeter", "cm")):
        text = re.sub(r"\b" + long + r"\b", short, text)
    text = re.sub(r" \*\* (-?\d+)", lambda m: m.group(1).translate(_SUPERSCRIPT_DIGITS), text)
    return text.replace(" * ", "·").replace(" / ", "/").replace(" ", "")


def _magnitude_and_unit(value_text, units):
    """Split ``'1e-06 [meter ** 2 / second]'`` into ``('1e-06', 'm²/s')``, and
    format a bare number compactly. The recorded value of a quantity prints
    its unit inside the text, so a renderer that appended the unit again
    said it twice."""
    text = "" if value_text is None else str(value_text)
    m = re.match(r"^\s*([-+0-9.eE]+)\s*\[(.*)\]\s*$", text)
    if m:
        number, unit = m.group(1), m.group(2)
        return _compact_number(number), _plain_unit(unit)
    m = re.match(r"^\s*Matrix\(\[\[([-+0-9.eE]+)\]\]\)\s*$", text)
    if m:
        return _compact_number(m.group(1)), _plain_unit(units)
    if re.match(r"^\s*[-+0-9.eE]+\s*$", text):
        return _compact_number(text), _plain_unit(units)
    return text, _plain_unit(units)


def _compact_number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return str(text)
    if value == 0:
        return "0"
    return f"{value:.4g}"


def _latex_value(value_latex, value_text, units):
    """The value for the Markdown key: a quantity's magnitude as LaTeX with
    its unit set upright beside it, a matrix left as it is."""
    number, unit = _magnitude_and_unit(value_text, units)
    if re.match(r"^[-+0-9.eE]+$", number or ""):
        latex = _number_latex(number)
        return latex + (rf"\ \mathrm{{{_plain_unit_latex(unit)}}}" if unit else "")
    return str(value_latex) if value_latex not in (None, "") else str(value_text)


def _number_latex(number):
    text = str(number)
    if "e" in text or "E" in text:
        mant, exp = re.split(r"[eE]", text)
        exp = int(exp)
        mant = mant.rstrip("0").rstrip(".") if "." in mant else mant
        if mant in ("1", "1.0"):
            return f"10^{{{exp}}}"
        return f"{mant} \\times 10^{{{exp}}}"
    return text


def _plain_unit_latex(unit):
    r"""``m²/s`` -> ``m^{2}/s``; ``Pa·s`` -> ``Pa\cdot s``."""
    text = unit.replace("·", r"\cdot ")
    supers = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
    digits = "0123456789-"
    out, run = "", ""
    for ch in text + " ":
        if ch in supers:
            run += digits[supers.index(ch)]
            continue
        if run:
            out += "^{" + run + "}"
            run = ""
        out += ch
    return out.rstrip()



def _flatten_where(where):
    out = []
    for w in where:
        out.append(w)
        out.extend(_flatten_where(w.get("where", [])))
    return out


def _parts_recorded(entry):
    """The ``part`` records of a run, in order, with repeats kept.

    A part is recorded once per run and again whenever its form changes, so
    two records for one part with different fingerprints say the equation
    changed mid-run — which a key must show rather than collapse.
    """
    return [r for r in entry.get("parts", []) if r.get("kind") == "part"]


def _where_lines(where, level, mode):
    """The named expressions inside a form, as ``symbol = value units — what``,
    nested to the depth the record followed them."""
    lines = []
    for w in where:
        symbol = w.get("symbol", "?")
        units = w.get("units")
        what = w.get("description") or ""
        indent = "  " * level
        if mode == "markdown":
            value = _latex_value(w.get("latex"), w.get("value"), units)
            head = f"{indent}- ${symbol}$"
            if value not in (None, ""):
                head += f" $= {value}$"
        else:
            number, unit = _magnitude_and_unit(w.get("value"), units)
            head = f"{indent}  {_plain_symbol(symbol)}"
            if number not in (None, ""):
                head += f" = {number}" + (f" {unit}" if unit else "")
        if what:
            head += f" — {what}"
        lines.append(head)
        lines.extend(_where_lines(w.get("where", []), level + 1, mode))
    return lines


def _key_for_part(record, mode):
    label = _short_operator(record.get("label", record.get("part", "?")))
    solver = record.get("solver", "?")
    unknown = record.get("unknown")
    dim = record.get("dim")
    step = record.get("at_step")
    out = []
    if mode == "markdown":
        out.append(f"### {label}")
        meta = f"`{solver}`"
        if unknown:
            meta += f", unknown `{unknown}`"
        if dim:
            meta += f", {dim}-D"
        if step is not None:
            meta += f"; recorded at step {step}"
        out.append(meta)
        out.append("")
        out.append("Residual $\\int F_0\\,\\phi + F_1 \\cdot \\nabla\\phi = 0$ with")
        for name in ("F0", "F1", "PF0"):
            form = record.get("forms", {}).get(name)
            if not form:
                continue
            sym = form.get("symbol") or name
            what = form.get("description") or ""
            out.append("")
            out.append(f"$${sym} = {form.get('latex', '')}$$")
            if what:
                out.append(f"*{what}*")
            where = _where_lines(form.get("where", []), 0, mode)
            if where:
                out.append("")
                out.append("where")
                out.extend(where)
    else:
        out.append(f"{label}  ({solver}" + (f", unknown {unknown}" if unknown else "")
                   + (f", {dim}-D" if dim else "") + (f"; recorded at step {step}" if step is not None else "") + ")")
        for name in ("F0", "F1", "PF0"):
            form = record.get("forms", {}).get(name)
            if not form:
                continue
            what = form.get("description") or ""
            out.append(f"  {name}: {form.get('text', '')}")
            if what:
                out.append(f"      {what}")
            out.extend(_where_lines(form.get("where", []), 1, mode))
    bcs = record.get("boundary_conditions") or []
    if bcs:
        out.append("")
        out.append("Boundary conditions:" if mode == "markdown" else "  boundary conditions:")
        for bc in bcs:
            kind = bc.get("type", bc.get("mechanism", "?"))
            where = bc.get("boundary", "?")
            number, _u = _magnitude_and_unit(bc.get("text"), None)
            scalar = re.match(r"^[-+0-9.eE]+$", number or "")
            if mode == "markdown":
                value = _number_latex(number) if scalar else bc.get("latex")
            else:
                value = number if scalar else bc.get("text")
            line = f"{kind} on {where}"
            if value:
                line += (f": ${value}$" if mode == "markdown" else f": {value}")
            out.append(("- " if mode == "markdown" else "    ") + line)
    terms = record.get("terms")
    if terms:
        out.append("")
        out.append("Given:" if mode == "markdown" else "  given:")
        for term in terms:
            if mode == "markdown":
                value = _latex_value(term.get("latex"), term.get("text"), None)
            else:
                number, unit = _magnitude_and_unit(term.get("text"), None)
                value = number + (f" {unit}" if unit else "")
            what = term.get("description") or ""
            line = f"`{term.get('name')}`" if mode == "markdown" else f"    {term.get('name')}"
            if value not in (None, ""):
                line += (f" $= {value}$" if mode == "markdown" else f" = {value}")
            if what:
                line += f" — {what}"
            out.append(("- " + line) if mode == "markdown" else line)
    elif record.get("terms_declared") is False:
        out.append("")
        out.append(("*" if mode == "markdown" else "  ") + "this solver does not declare the terms it was given" + ("*" if mode == "markdown" else ""))
    return out


def transcript_key(source, run=-1, out=None, format="markdown"):
    """The key to a run: what each part solved, as it was implemented.

    A run records each solver's residual once, and again if it changes. This
    renders those records as the legend a figure or a note needs: the weak
    form with its templates, the named expressions inside them expanded down
    to the constitutive model — with values, units and descriptions — the
    boundary conditions, and the terms the solver was given.

    Parameters
    ----------
    source : str, list or Model
        A ``.jsonl`` transcript, the list :func:`underworld3.read_transcript`
        returns, or a live model.
    run : int, default -1
        Which run in the file.
    out : str, optional
        Write the key here (``.md`` or ``.txt``) as well as returning it.
    format : {"markdown", "text"}
        Markdown with LaTeX for a note or a notebook; plain text, with the
        forms as SymPy prints them, for a terminal.

    Returns
    -------
    str
    """
    if format not in ("markdown", "text"):
        raise ValueError(f"format must be 'markdown' or 'text', not {format!r}")
    runs = _as_runs(source)
    entry = _pick_run(runs, run)
    header = entry.get("run") or {}
    parts = _parts_recorded(entry)
    lines = []
    title = _run_title(header, fallback="")
    if format == "markdown":
        lines.append(f"## Key — what each part solved" + (f" in {title}" if title else ""))
        if header.get("started"):
            lines.append(f"*run started {header['started']}*")
        lines.append("")
    else:
        lines.append("key — what each part solved" + (f" in {title}" if title else ""))
        if header.get("started"):
            lines.append(f"run started {header['started']}")
        lines.append("")
    if not parts:
        lines.append("no solver recorded its form in this run" if format == "text"
                     else "*No solver recorded its form in this run.*")
    seen = {}
    for record in parts:
        key = record.get("part")
        if key in seen and seen[key] != record.get("fingerprint"):
            note = f"the form of {_short_operator(record.get('label', key))} changed at step {record.get('at_step')}"
            lines.append(("> " if format == "markdown" else "! ") + note)
            lines.append("")
        seen[key] = record.get("fingerprint")
        lines.extend(_key_for_part(record, format))
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    if out:
        directory = os.path.dirname(out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return text


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


_ALIASES = None


def _public_aliases():
    """``{"SNES_AdvectionDiffusion_Composed": "AdvDiffusion", ...}`` — the
    name each solver class is exported under in ``uw.systems``, which is the
    name the user wrote. Built once."""
    global _ALIASES
    if _ALIASES is None:
        aliases = {}
        try:
            import underworld3 as uw

            for attr, obj in vars(uw.systems).items():
                if not isinstance(obj, type) or attr == obj.__name__:
                    continue
                # only a name SHORTER than the class's own, stripped of its
                # base-class prefix — EulerianSUPG is exported as
                # EulerianSUPG_DDt too, and that is not what anyone wrote
                own = re.sub(r"^(SNES_|uw_)", "", obj.__name__).replace("_Composed", "")
                current = aliases.get(obj.__name__)
                if len(attr) < len(own) and (current is None or len(attr) < len(current)):
                    aliases[obj.__name__] = attr
        except Exception:
            # Charter S4 — sanctioned: these aliases only make a report read the
            # way a user wrote their code (Stokes rather than SNES_Stokes).
            # Introspecting uw.systems can fail on a partially-imported module;
            # the fallback is the class's real name, which is correct if less
            # familiar. A report that is slightly more verbose is not a reason
            # to fail.
            pass
        _ALIASES = aliases
    return _ALIASES


def _short_operator(name):
    """``SNES_AdvectionDiffusion_Composed(T)`` -> ``AdvDiffusion(T)``.

    An operator is named in the record by its class, which says which base
    implemented it and is never the thing the reader is checking. The views
    call it what the user called it: the name it is exported under. A class
    with no public alias loses its ``SNES_`` prefix and nothing else.
    """
    text = str(name)
    head, sep, tail = text.partition("(")
    alias = _public_aliases().get(head)
    if alias:
        return alias + sep + tail
    for prefix in ("SNES_", "uw_"):
        if head.startswith(prefix):
            head = head[len(prefix):]
    return head.replace("_Composed", "") + sep + tail


def _run_title(header, fallback="model"):
    """The run's name for a heading: the model's, unless it is the default
    one, in which case the script that ran it — which the launch record
    kept for exactly this."""
    name = (header or {}).get("model")
    script = (header or {}).get("script")
    if (not name or name == "default") and script:
        return str(script).rsplit(".", 1)[0]
    return repr(name) if name else fallback


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

    def math(self, x, y, latex, size=8.0, fill=_INK, fallback=None):
        """A line of mathematics, set from ``latex`` as glyph outlines.

        Returns the width drawn, so a caption can follow it. The outlines
        come from matplotlib's mathtext, which needs no TeX installation;
        without matplotlib, or for LaTeX mathtext cannot set (a matrix), the
        plain-text ``fallback`` is written as text instead.
        """
        segments, box = _mathtext_outline(latex, size)
        if segments is None:
            text = fallback if fallback is not None else _plain_symbol(latex)
            self.text(x, y, text, size=size, fill=fill)
            return _text_width(text, size, False)
        self._outline(x, y, segments, fill)
        return box[2]

    def _outline(self, x, y, segments, fill):
        self.ops.append(("path", [(cmd, *[(x + px, y - py) for px, py in pts])
                                  for cmd, *pts in segments], fill))

    def equation(self, x, y, symbol, latex, size=8.0, fill=_INK, max_width=None):
        """``symbol = latex`` set as mathematics, from ``y`` downwards.

        A matrix on the right-hand side is laid out cell by cell with drawn
        brackets, because mathtext has no matrix environment. The whole
        equation is scaled to ``max_width`` when it would not fit — a long
        residual is set small and stays vector, rather than being cut.
        Returns the height used, or ``None`` when it could not be set.
        """
        cells = _matrix_cells(latex) or [[latex]]
        for attempt in range(3):
            head = _mathtext_outline(symbol + " =", size)
            grid = [[_mathtext_outline(cell, size) for cell in row] for row in cells]
            if head[0] is None or any(g[0] is None for row in grid for g in row):
                return None
            gap_x, gap_y = size * 0.9, size * 0.45
            col_w = [max(grid[r][c][1][2] - grid[r][c][1][0] for r in range(len(grid)))
                     for c in range(len(grid[0]))]
            asc = [max(g[1][3] for g in row) for row in grid]
            desc = [max(-g[1][1], 0.0) for row in grid for g in [max(row, key=lambda g: -g[1][1])]]
            rows_h = [a + d + gap_y for a, d in zip(asc, desc)]
            bracket = size * 0.5 if len(cells) > 1 or len(cells[0]) > 1 else 0.0
            matrix_w = sum(col_w) + gap_x * (len(col_w) - 1) + 2 * bracket + size * 0.6
            total_w = head[1][2] + size * 0.6 + matrix_w
            if max_width is None or total_w <= max_width or attempt == 2:
                break
            if (attempt == 0 and len(cells) == 1 and len(cells[0]) > 1
                    and max_width / total_w < 0.6):
                # A wide row vector would have to shrink past reading; set
                # its components one under another instead, then scale.
                cells = [[cell] for cell in cells[0]]
                continue
            size = max(size * max_width / total_w, 1.0)
        height = sum(rows_h)
        mid = y + height / 2
        self._outline(x, mid + size * 0.35, head[0], fill)
        x0 = x + head[1][2] + size * 0.6
        if bracket:
            for bx, tick in ((x0, 1), (x0 + matrix_w, -1)):
                self.line(bx, y, bx, y + height, fill, 0.6)
                self.line(bx, y, bx + tick * bracket * 0.6, y, fill, 0.6)
                self.line(bx, y + height, bx + tick * bracket * 0.6, y + height,
                          fill, 0.6)
        yy = y
        for r, row in enumerate(grid):
            baseline = yy + gap_y / 2 + asc[r]
            xx = x0 + bracket + size * 0.3
            for c, (segments, box) in enumerate(row):
                cx = xx + (col_w[c] - (box[2] - box[0])) / 2 - box[0]
                self._outline(cx, baseline, segments, fill)
                xx += col_w[c] + gap_x
            yy += rows_h[r]
        return height

    def note(self, x, y, r, fill, hollow=False):
        """A mark that a part ran. Hollow when its step was abandoned."""
        self.ops.append(("note", x, y, r, fill, bool(hollow)))

    def mark(self, x, y, r, state, hollow=False):
        """How a solve went: ``"ok"``, ``"capped"`` or ``"diverged"``.

        Drawn as an emoji in SVG and as a glyph in PDF — the PDF is written
        against the base-14 fonts, which have no emoji, so the same three
        states are stroked by hand there rather than set as text.
        """
        self.ops.append(("mark", x, y, r, state, bool(hollow)))


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
                        title or f"Run log — {_run_title(header)}",
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


_MATHTEXT_REMAP = {"\\upkappa": "\\kappa", "\\uplambda": "\\lambda",
                   "\\upmu": "\\mu", "\\uprho": "\\rho", "\\upeta": "\\eta"}


def _mathtext_outline(latex, size):
    """``latex`` as glyph outlines: ``(segments, width)`` or ``(None, 0)``.

    A segment is ``("M" | "L" | "Q" | "Z", *points)`` with the baseline at
    ``y = 0`` and y UP, as matplotlib gives it; the caller flips it. Quadratic
    curves are kept as quadratics — SVG has them and the PDF writer raises
    them to cubics.
    """
    try:
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        from matplotlib.path import Path
    except Exception:
        return None, 0.0
    text = str(latex)
    for source, target in _MATHTEXT_REMAP.items():
        text = text.replace(source, target)
    # A variable whose name starts with an underscore (the mesh's own
    # ``_h_cell``) reaches sympy's LaTeX as ``{_h_cell}``, which no TeX can
    # set; it becomes ``h_{cell}``.
    text = re.sub(r"\{_([A-Za-z])_([A-Za-z0-9]+)\}", r"\1_{\2}", text)
    text = re.sub(r"\{_([A-Za-z][A-Za-z0-9]*)\}", r"\1", text)
    try:
        path = TextPath((0.0, 0.0), f"${text}$", size=size,
                        prop=FontProperties(family="DejaVu Sans"))
    except Exception:
        return None, 0.0
    segments, i = [], 0
    vertices, codes = path.vertices, path.codes
    while i < len(codes):
        code = codes[i]
        if code == Path.MOVETO:
            segments.append(("M", tuple(vertices[i])))
            i += 1
        elif code == Path.LINETO:
            segments.append(("L", tuple(vertices[i])))
            i += 1
        elif code == Path.CURVE3:
            segments.append(("Q", tuple(vertices[i]), tuple(vertices[i + 1])))
            i += 2
        elif code == Path.CURVE4:
            segments.append(("C", tuple(vertices[i]), tuple(vertices[i + 1]),
                             tuple(vertices[i + 2])))
            i += 3
        else:
            segments.append(("Z",))
            i += 1
    box = path.get_extents()
    return segments, (float(box.x0), float(box.y0), float(box.x1), float(box.y1))


_MATRIX = re.compile(r"^\s*\\left\[\s*\\begin\{matrix\}(.*)\\end\{matrix\}\s*\\right\]\s*$",
                     re.S)


def _matrix_cells(latex):
    """``\\left[\\begin{matrix}a & b\\\\c & d\\end{matrix}\\right]`` as ``[[a, b], [c, d]]``,
    or ``None`` when ``latex`` is not a bare matrix. SymPy writes every
    vector and tensor form this way, and mathtext cannot set the
    environment, so the figure lays the cells out itself."""
    m = _MATRIX.match(latex)
    if not m:
        return None
    return [[cell.strip() for cell in row.split("&")]
            for row in m.group(1).split("\\\\")]


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
        elif kind == "mark":
            _, x, y, r, state, hollow = op
            if hollow:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 2.2:.1f}" '
                           f'fill="none" stroke="{_hex(_ABANDONED)}" '
                           f'stroke-width="1.0" stroke-dasharray="2 1.4"/>')
            glyph = _OUTCOME_EMOJI.get(state, "")
            out.append(
                f'<text x="{x:.1f}" y="{y + r * 0.95:.1f}" '
                f'font-size="{r * 2.1:.1f}" text-anchor="middle">'
                f'{html.escape(glyph)}</text>'
            )
        elif kind == "path":
            _, segments, fill = op
            d = " ".join(
                cmd + " " + " ".join(f"{px:.2f} {py:.2f}" for px, py in pts)
                for cmd, *pts in segments)
            out.append(f'<path d="{d}" fill="{_hex(fill)}" fill-rule="nonzero"/>')
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
    """WinAnsi-safe, with the escapes PDF strings need.

    The base-14 fonts have no Greek and no sub- or superscript digits, so
    the key's symbols are spelled out here — ``κ`` as ``kappa``, ``ρ₀`` as
    ``rho_0`` — rather than printed as ``?``.
    """
    for source, target in _PDF_SUBSTITUTIONS.items():
        content = content.replace(source, target)
    for name, glyph in _GREEK.items():
        if glyph in content and not glyph.isascii():
            content = content.replace(glyph, _PDF_SUBSTITUTIONS.get(glyph, name))
    content = re.sub("[₀₁₂₃₄₅₆₇₈₉]+",
                     lambda m: "_" + m.group(0).translate(_FROM_SUBSCRIPT), content)
    content = re.sub("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+",
                     lambda m: "^" + m.group(0).translate(_FROM_SUPERSCRIPT), content)
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
        elif kind == "mark":
            # Base-14 fonts carry no emoji, so the three states are stroked.
            _, x, y, r, state, hollow = op
            colour = _OUTCOME_COLOUR.get(state, _MUTED)
            k, yy = r * 0.5523, fy(y)
            out.append("q")
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
            out.append("f")
            out.append(f"{_PAPER[0]:.3f} {_PAPER[1]:.3f} {_PAPER[2]:.3f} RG "
                       f"{max(r * 0.30, 0.8):.2f} w 1 J 1 j")
            if state == "ok":
                out.append(f"{x - r * 0.50:.2f} {yy + r * 0.05:.2f} m "
                           f"{x - r * 0.12:.2f} {yy - r * 0.38:.2f} l "
                           f"{x + r * 0.52:.2f} {yy + r * 0.45:.2f} l S")
            elif state == "diverged":
                out.append(f"{x - r * 0.42:.2f} {yy + r * 0.42:.2f} m "
                           f"{x + r * 0.42:.2f} {yy - r * 0.42:.2f} l S")
                out.append(f"{x - r * 0.42:.2f} {yy - r * 0.42:.2f} m "
                           f"{x + r * 0.42:.2f} {yy + r * 0.42:.2f} l S")
            else:
                out.append(f"{x:.2f} {yy + r * 0.52:.2f} m "
                           f"{x:.2f} {yy - r * 0.10:.2f} l S")
                out.append(f"{x:.2f} {yy - r * 0.42:.2f} m "
                           f"{x:.2f} {yy - r * 0.44:.2f} l S")
            out.append("Q")
            if hollow:
                out.append("q")
                out.append(f"{_ABANDONED[0]:.3f} {_ABANDONED[1]:.3f} "
                           f"{_ABANDONED[2]:.3f} RG 1.0 w [2 1.4] 0 d")
                rr = r + 2.2
                kk = rr * 0.5523
                out.append(f"{x + rr:.2f} {yy:.2f} m")
                out.append(f"{x + rr:.2f} {yy + kk:.2f} {x + kk:.2f} {yy + rr:.2f} "
                           f"{x:.2f} {yy + rr:.2f} c")
                out.append(f"{x - kk:.2f} {yy + rr:.2f} {x - rr:.2f} {yy + kk:.2f} "
                           f"{x - rr:.2f} {yy:.2f} c")
                out.append(f"{x - rr:.2f} {yy - kk:.2f} {x - kk:.2f} {yy - rr:.2f} "
                           f"{x:.2f} {yy - rr:.2f} c")
                out.append(f"{x + kk:.2f} {yy - rr:.2f} {x + rr:.2f} {yy - kk:.2f} "
                           f"{x + rr:.2f} {yy:.2f} c")
                out.append("S Q")
        elif kind == "path":
            _, segments, fill = op
            out.append(f"q {fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
            current = (0.0, 0.0)
            for cmd, *pts in segments:
                if cmd == "M":
                    current = pts[0]
                    out.append(f"{pts[0][0]:.2f} {fy(pts[0][1]):.2f} m")
                elif cmd == "L":
                    current = pts[0]
                    out.append(f"{pts[0][0]:.2f} {fy(pts[0][1]):.2f} l")
                elif cmd == "Q":
                    # PDF has no quadratic; raise it to the equal cubic.
                    (qx, qy), (ex, ey) = pts
                    c1 = (current[0] + 2 / 3 * (qx - current[0]),
                          current[1] + 2 / 3 * (qy - current[1]))
                    c2 = (ex + 2 / 3 * (qx - ex), ey + 2 / 3 * (qy - ey))
                    out.append(f"{c1[0]:.2f} {fy(c1[1]):.2f} {c2[0]:.2f} "
                               f"{fy(c2[1]):.2f} {ex:.2f} {fy(ey):.2f} c")
                    current = (ex, ey)
                elif cmd == "C":
                    (ax, ay), (bx, by), (ex, ey) = pts
                    out.append(f"{ax:.2f} {fy(ay):.2f} {bx:.2f} {fy(by):.2f} "
                               f"{ex:.2f} {fy(ey):.2f} c")
                    current = (ex, ey)
                else:
                    out.append("h")
            out.append("f Q")
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
# The transcript as a chart: parts across the page, steps down it
# ---------------------------------------------------------------------------

def _parts_of(steps):
    """The roster, in a stable order, from what actually played.

    A part that never runs gets no column, which keeps a model's registered
    cast from becoming a page of empty ones.
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
    # In the order they first ran. A step is drawn as a bar whose events
    # descend in the order they ran, so with the columns in that same order
    # the usual step reads as a staircase down and to the right, and any step
    # that departs from it — a part run twice, a solve out of turn — breaks
    # the shape.
    return [(k, labels[k]) for k in order]


def _outcome(event):
    """How a solve went: ``"ok"``, ``"capped"``, ``"diverged"``, or ``None``.

    ``None`` covers both a history shift, which has nothing to converge, and a
    transcript written before outcomes were recorded — neither is an outcome
    the figure may invent.

    ``"capped"`` is a solve the SNES called converged while one of its
    fieldsplit blocks ended at its iteration cap. That block did not solve
    (#625), so the answer came back on a preconditioner that was still moving;
    it is neither a clean convergence nor a failure, and reading it as either
    loses the thing worth seeing.
    """
    if event.get("kind") != "solve" or "converged" not in event:
        return None
    if not event.get("converged"):
        return "diverged"
    if event.get("capped") or event.get("deadline_expired"):
        return "capped"
    return "ok"


def _step_cells(step, parts):
    """One cell per part: what it played and how that went, or a rest.

    A cell is a tuple of ``(position, outcome)``, so two steps collapse into a
    repeat only when they ran the same parts in the same order AND those solves
    went the same way. A step where the velocity block gave up does not hide
    inside a run of clean ones.
    """
    played = []
    for event in step.get("events", []):
        if event.get("kind") in ("solve", "history_shift"):
            played.append((event.get("part") or event.get("name"),
                           _outcome(event)))
    cells = []
    for key, _ in parts:
        hits = tuple((i + 1, outcome) for i, (k, outcome) in enumerate(played)
                     if k == key)
        cells.append(hits or None)
    return cells


def _transcript_rows(steps, parts, note_at, anchors=()):
    """Steps grouped into rows: one step, or a run of identical steps collapsed.

    Returns ``(kind, first_index, last_index, cells, count)`` with ``kind``
    either ``"step"`` or ``"repeat"``. The collapse asserts identity — same
    cells, same outcome, same label, nothing recorded between them — which is
    what makes it safe to hide thirteen rows behind one mark.
    """
    anchors = set(anchors)
    rows, i = [], 0
    while i < len(steps):
        cells = _step_cells(steps[i], parts)
        j = i + 1
        while (j < len(steps)
               and (j - 1) not in note_at
               and j not in anchors
               and _step_cells(steps[j], parts) == cells
               and steps[j].get("completed") == steps[i].get("completed")
               and steps[j].get("label") == steps[i].get("label")):
            j += 1
        rows.append(("step", i, i, cells, 1))
        if j - i > 1:
            rows.append(("repeat", i + 1, j - 1, cells, j - i - 1))
        i = j
    return rows


def transcript_figure(source, out=None, run=-1, title=None,
                            width=PAGE_W, format=None, collapse=True, key=False):
    """Draw the transcript: a column per part, a row per step.

    The same reading as :func:`transcript_table`, drawn rather than set in
    text. Each step is a bar: what ran sits one line lower than what ran
    before it, in its own column, joined by a path, so the sequence is the
    shape. A solve's mark says how it went; a dash is a part that did
    nothing; a run of identical steps collapses to one band with a count and
    a downward arrow.

    Parameters
    ----------
    source : str, list or Model
        A ``.jsonl`` transcript, the list :func:`underworld3.read_transcript`
        returns, or a live model.
    out : str, optional
        Where to write. Defaults to the source path with ``-transcript`` and
        the format's suffix.
    format : {"pdf", "svg"}, optional
        Inferred from ``out``'s suffix; PDF by default.
    key : bool, default False
        Append a key: for each part, the named quantities inside its residual
        with their values and units, and its boundary conditions — what a
        reader needs to know which model this is. The full forms are in
        :func:`transcript_key`.
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
        out = (os.path.splitext(str(source))[0] + "-transcript" + suffix
               if isinstance(source, (str, os.PathLike)) else
               "transcript" + suffix)

    canvas, page_w, height = _transcript_layout(
        header, steps, notes, entry, title=title, width=width,
        collapse=collapse, key_records=_parts_recorded(entry) if key else None)

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


def _transcript_layout(header, steps, notes, entry, title=None, width=PAGE_W,
                       collapse=True, *, key_records=None):
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
        rows = _transcript_rows(steps, parts, note_at,
                           anchors={a for a in anchors if a is not None})
    else:
        rows = [("step", i, i, _step_cells(step, parts), 1)
                for i, step in enumerate(steps)]

    y = _MARGIN
    canvas.text(_MARGIN, y + 12, title or f"Transcript — {_run_title(header)}",
                size=14, bold=True)
    y += 20
    bits = []
    if header.get("started"):
        bits.append(f"started {header['started']}")
    if entry.get("ended"):
        bits.append(f"complete, {entry['ended'].get('steps', len(steps))} steps")
    elif entry.get("live"):
        bits.append(f"in progress, {len(steps)} steps so far")
    else:
        bits.append("no terminator: still running, or interrupted")
    bits.append(f"{len(parts)} parts")
    canvas.text(_MARGIN, y + 8, "  ·  ".join(bits), size=8.5, fill=_MUTED)
    y += 22

    # --- columns ---
    gutter = _MARGIN + 46.0
    x_step = gutter + 26.0
    x_time = x_step + 52.0
    x_dt = x_time + 50.0
    lane0 = x_dt + 22.0
    lane_w = max(44.0, (right - lane0) / max(len(parts), 1))

    def lane_x(i):
        return lane0 + lane_w * (i + 0.5)

    canvas.text(x_step, y + 8, "step", size=8, fill=_MUTED, anchor="end")
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
    # A step is a bar. Inside it, time runs down: the first thing that ran
    # sits on the top line, the next one line below, each in its own
    # column, joined by a stepped path. The shape of the path is the
    # sequence — an extra line in a bar is a part that ran twice, and a path
    # that doubles back is a solve out of its usual turn. Nothing has to be
    # read off a digit.
    pad, pitch, band_h = 3.5, 9.0, 30.0
    row_h = 2 * pad + pitch

    def bar_height(cells):
        n = max((order for cell in cells if cell for order, _ in cell),
                default=1)
        return 2 * pad + pitch * n

    row_y, row_page = {}, {}
    y = top
    for kind, first, last, cells, count in rows:
        h = bar_height(cells) if kind == "step" else band_h
        mid = y + h / 2
        if kind == "step":
            step = steps[first]
            row_y[first] = mid
            completed = bool(step.get("completed"))
            colour = _ACCEPTED if completed else _ABANDONED
            line1 = y + pad + pitch / 2   # the bar's first line
            canvas.text(x_step, line1 + 3, step.get("index", first), size=8.5,
                        fill=_INK if completed else _ABANDONED, anchor="end")
            t1 = _converted(step.get("t1"), unit)
            dt = _converted(step.get("dt"), unit)
            canvas.text(x_time, line1 + 3, f"{t1:.6g}", size=8.5,
                        fill=_INK if completed else _ABANDONED, anchor="end")
            canvas.text(x_dt, line1 + 3, f"{dt:.6g}", size=8.5,
                        fill=_INK if completed else _ABANDONED, anchor="end")
            played = []
            for i, cell in enumerate(cells):
                cx = lane_x(i)
                if cell is None:
                    # A rest: notated, because a part that did nothing is
                    # different from a part nobody was watching.
                    canvas.rect(cx - 4.5, mid - 1.0, 9.0, 2.0, fill=_MUTED)
                    continue
                for order, outcome in cell:
                    played.append((order, cx, y + pad + pitch * (order - 0.5),
                                   outcome))
            played.sort()
            # the path first, so the marks sit on it. Each link runs level,
            # drops halfway across, and runs level again: the step is a
            # ticker, and the shape says so.
            for (_, x0, y0, _), (_, x1, y1, _) in zip(played, played[1:]):
                xm = (x0 + x1) / 2
                stroke = _ACCEPTED if completed else _ABANDONED
                canvas.line(x0, y0, xm, y0, stroke, 0.7)
                canvas.line(xm, y0, xm, y1, stroke, 0.7)
                canvas.line(xm, y1, x1, y1, stroke, 0.7)
            for _, nx, ny, outcome in played:
                if outcome is None:
                    # A history shift, or a transcript from before outcomes
                    # were recorded: the plain mark says it ran.
                    canvas.note(nx, ny, 3.4, colour, hollow=not completed)
                else:
                    canvas.mark(nx, ny, 4.2, outcome, hollow=not completed)
            if not completed:
                canvas.text(right, line1 + 3, "abandoned", size=7.5,
                            fill=_ABANDONED, anchor="end")
            # a bar line under each step
            canvas.line(x_step - 30, y + h, right, y + h, _RULE, 0.4)
        else:
            # A run of steps that did the same thing. The band shows the first
            # and last value of anything that CHANGED across it, because a
            # timestep that grew by a factor of eight is a diagnostic, and a
            # symbol that only says "repeats" would throw it away.
            canvas.rect(_MARGIN, y, right - _MARGIN, h, fill=(0.965, 0.958, 0.948))
            first_step, last_step = steps[first], steps[last]
            canvas.text(x_step, y + 9,
                        f"{first_step.get('index', first)}", size=8, fill=_MUTED,
                        anchor="end")
            canvas.text(x_step, y + h - 3,
                        f"{last_step.get('index', last)}", size=8, fill=_MUTED,
                        anchor="end")
            canvas.text(x_step + 7, mid + 3, f"×{count}", size=7, fill=_MUTED)
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
    # the columns themselves, spanning the whole block
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
    canvas.note(_MARGIN + 4, y, 3.4, _ACCEPTED)
    canvas.text(_MARGIN + 16, y + 3,
                "ran; a step reads down, the path joining what ran in the "
                "order it ran", size=8, fill=_INK)
    y += 14
    canvas.mark(_MARGIN + 4, y, 4.2, "ok")
    canvas.text(_MARGIN + 16, y + 3, "solved, and converged", size=8, fill=_INK)
    y += 14
    canvas.mark(_MARGIN + 4, y, 4.2, "capped")
    canvas.text(_MARGIN + 16, y + 3,
                "converged, but a fieldsplit block hit its iteration cap — "
                "that block did not solve", size=8, fill=_INK)
    y += 14
    canvas.mark(_MARGIN + 4, y, 4.2, "diverged")
    canvas.text(_MARGIN + 16, y + 3,
                "did not converge; the reason is in the record", size=8,
                fill=_INK)
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

    if key_records:
        # The key: which model this is, by the quantities in its residuals.
        y += 6
        canvas.text(_MARGIN, y + 8, "Key — the parts, and what is in their residuals",
                    size=9.5, bold=True)
        y += 18
        for record in key_records:
            label = _short_operator(record.get("label", record.get("part", "?")))
            canvas.text(_MARGIN, y + 8, label, size=8.5, bold=True)
            step = record.get("at_step")
            if step is not None:
                canvas.text(right, y + 8, f"recorded at step {step}", size=7,
                            fill=_MUTED, anchor="end")
            y += 13
            statement = canvas.math(
                _MARGIN + 12, y + 9,
                r"\int F_0\,\phi + F_1 \cdot \nabla\phi = 0",
                size=7.5, fallback="residual: int F0 phi + F1 . grad(phi) = 0")
            canvas.text(_MARGIN + 12 + statement + 8, y + 9, "— with", size=7.5,
                        fill=_MUTED)
            y += 15
            seen = set()
            for form in record.get("forms", {}).values():
                if form.get("description"):
                    canvas.text(_MARGIN + 12, y + 8, form["description"][:110],
                                size=7.5, fill=_MUTED)
                    y += 11
                used = canvas.equation(_MARGIN + 12, y + 2, form.get("symbol") or "F",
                                       form.get("latex", ""), size=7.5,
                                       max_width=right - _MARGIN - 12)
                if used is None:
                    canvas.text(_MARGIN + 12, y + 8,
                                "(the form is in the record; uw.transcript_key "
                                "renders it)", size=7.5, fill=_MUTED)
                    used = 8
                y += used + 8
                for w in _flatten_where(form.get("where", [])):
                    if w["symbol"] in seen:
                        continue
                    seen.add(w["symbol"])
                    number, unit = _magnitude_and_unit(w.get("value"), w.get("units"))
                    latex, plain = str(w["symbol"]), _plain_symbol(w["symbol"])
                    if number not in (None, ""):
                        latex += " = " + _latex_value(w.get("latex"), w.get("value"),
                                                      w.get("units"))
                        plain += f" = {number}" + (f" {unit}" if unit else "")
                    drawn = canvas.math(_MARGIN + 12, y + 9, latex, size=7.5,
                                        fallback=plain[:60])
                    if w.get("description"):
                        canvas.text(_MARGIN + 12 + drawn + 8, y + 9,
                                    f"— {w['description']}"[:100], size=7.5,
                                    fill=_MUTED)
                    y += 12.5
            for bc in record.get("boundary_conditions") or []:
                line = f"{bc.get('type', bc.get('mechanism', '?'))} on {bc.get('boundary', '?')}"
                number, _u = _magnitude_and_unit(bc.get("text"), None)
                if number:
                    line += f": {number}"
                canvas.text(_MARGIN + 12, y + 8, line[:110], size=7.5, fill=_MUTED)
                y += 11
            y += 6

    return canvas, width, y + _MARGIN


def transcript_table(source, run=-1, width=11, collapse=True):
    """The transcript as a chart: parts across the page, steps down it.

    Post-hoc by design. The roster is not known until a run has happened — a
    part that first appears at step 300 must still have a column at step 1,
    empty — so this cannot be the thing streamed line-by-line as steps close.
    It is what you read afterwards, or partway through: rendering a transcript
    that is still being written gives what has happened so far, and says so.

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
    out.append(f"transcript · {_run_title(header, fallback='')}".rstrip(" ·"))
    if header.get("started"):
        out.append(f"started {header['started']}")
    if ended:
        out.append(f"complete — {ended.get('steps', len(steps))} step(s)")
    elif entry.get("live"):
        out.append(f"in progress — {len(steps)} step(s) so far")
    else:
        out.append("no terminator: this run is still going, or it was "
                   "interrupted. What follows is the transcript of a prefix.")
    out.append("")

    step_w, t_w = 5, 10
    width = max(width, max((len(label) for _, label in parts), default=width))
    head = (f"{'step':>{step_w}} {('t/' + short) if short else 't':>{t_w}} "
            f"{('dt/' + short) if short else 'dt':>{t_w}} │ "
            + " │ ".join(f"{label[:width]:^{width}}" for _, label in parts) + " │")
    out.append(head)
    out.append("─" * len(head))

    def cell_text(cell):
        """Positions and how they went. ASCII, because this view is columns in
        a terminal and an emoji is two cells wide in some of them and one in
        others — the alignment is the point here, and the figure is where the
        emoji belong."""
        if not cell:
            return "·"
        return ",".join(f"{order}{_TEXT_OUTCOME.get(outcome, '')}"
                        for order, outcome in cell)

    def row(step, cells):
        t1 = _converted(step.get("t1"), unit)
        dt = _converted(step.get("dt"), unit)
        body = " │ ".join(f"{cell_text(c):^{width}}" for c in cells)
        tail = "" if step.get("completed") else "   ABANDONED"
        return (f"{step.get('index', '?'):>{step_w}} {t1:>{t_w}.6g} "
                f"{dt:>{t_w}.6g} │ {body} │{tail}")

    i = 0
    while i < len(steps):
        step = steps[i]
        cells = _step_cells(step, parts)
        # How many steps that follow are IDENTICAL — same cells, same outcome,
        # and nothing happened between them.
        run_len = 0
        j = i + 1
        while (collapse
               and j < len(steps)
               and (j - 1) not in note_at
               and _step_cells(steps[j], parts) == cells
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
            out.append(f"{label:>{step_w}} {t_to:>{t_w}.6g} {dt_to:>{t_w}.6g} "
                       f"│ {carry} │   ×{run_len} unchanged"
                       + (f", dt {dt_from:.4g} → {dt_to:.4g}"
                          if abs(dt_to - dt_from) > 1e-12 * max(1.0, abs(dt_from))
                          else ""))
        for k in range(i, j):
            for note in note_at.get(k, []):
                out.append(f"{'':>{step_w}} {'':>{t_w}} {'':>{t_w}} │ "
                           f"{note.get('message', note.get('kind'))}")
        i = j

    out.append("")
    out.append(f"{len(steps)} step(s), {len(parts)} part(s): "
               + ", ".join(label for _, label in parts))
    out.append("↓  the steps between did exactly this, unchanged")
    out.append("·  this part did nothing in that step")
    out.append("digits are the order the parts ran within the step")
    outcomes = {o for s in steps for e in s.get("events", []) for o in [_outcome(e)] if o}
    if "capped" in outcomes:
        out.append("!  converged, but a fieldsplit block ended at its iteration "
                   "cap — that block did not solve")
    if "diverged" in outcomes:
        out.append("x  did not converge")
    return "\n".join(out)
