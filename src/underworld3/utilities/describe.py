r"""Structured descriptions of Underworld objects, rendered on demand.

An object says what it is once, as data, through ``describe()``. The
description is a plain tree: strings, numbers, lists and dicts, so it can
be written to a run transcript, returned by a query, or rendered for a
reader in whichever form the reader is using. ``render`` turns one tree
into Markdown with mathematics for a notebook, plain text for a terminal,
a LaTeX fragment for a note, or YAML and JSON for a record or a tool.
``view()`` on any object is ``render(describe())`` in the form the current
session wants.

The record
----------

Every description carries::

    kind      what sort of object: "solver", "mesh", "variable", ...
    name      the object's name
    summary   one line for a reader

and any of::

    facts        {label: value}       scalar facts, in a stable order
    terms        [{name, symbol, latex, text, units, description, where}]
                                      the named quantities the object holds
    forms        {name: {symbol, latex, text, description, where}}
                                      its equations
    conditions   [{type, boundary, latex, text, ...}]
                                      its boundary conditions
    children     [record, ...]        the objects it contains

``where`` is the list of named expressions inside a value, each with the
same keys as a term and its own ``where``, down to the depth the caller
asked for. A solver's description also keeps the keys the run transcript
records (``forms``, ``boundary_conditions``, ``terms``); ``conditions``
and ``boundary_conditions`` are the same list under either name.
"""

import json
import re

FORMATS = ("markdown", "text", "latex", "yaml", "json")


def record(kind, name, summary="", **fields):
    """A description with the shared keys first and the rest in the order given."""
    out = {"kind": str(kind), "name": str(name) if name is not None else None,
           "summary": str(summary or "")}
    for key, value in fields.items():
        if value is not None:
            out[key] = value
    return out


def term(name, value=None, description="", units=None, symbol=None, where=None):
    """One named quantity as a description entry: the value as LaTeX and as
    text, its units and description. ``value`` may be a SymPy expression, a
    named Underworld expression, a quantity or a number."""
    import sympy

    if hasattr(value, "sym") and hasattr(value, "symbol"):
        symbol = symbol or str(value.symbol)
        description = description or str(getattr(value, "description", "") or "")
        units = units or (str(value.units) if getattr(value, "units", None) else None)
        value = value.sym
    if description == "No description provided":
        description = ""
    if value is None:
        latex = text = None
    else:
        try:
            latex = sympy.latex(value)
        except Exception:
            latex = None
        text = str(value)
    return {"name": str(name), "symbol": symbol, "latex": latex, "text": text,
            "units": units, "description": description, "where": list(where or [])}


def plain(value):
    """``value`` with every leaf a string, number, boolean or None, so the
    tree can be written as YAML or JSON. Sequences and mappings are kept."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return str(value)


# --- rendering -----------------------------------------------------------

def render(description, format="markdown", depth=None):
    """One description as a string in ``format``.

    ``depth`` limits how many levels of children are rendered; ``None``
    renders them all. The ``where`` lists inside terms and forms are
    rendered as far as the description carries them.
    """
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, not {format!r}")
    if format == "json":
        return json.dumps(plain(_pruned(description, depth)), indent=2)
    if format == "yaml":
        import yaml
        return yaml.safe_dump(plain(_pruned(description, depth)), sort_keys=False,
                              allow_unicode=True, width=100)
    lines = _render_lines(description, format, level=0, depth=depth)
    return "\n".join(lines).rstrip() + "\n"


def _pruned(description, depth, level=0):
    if depth is None or not isinstance(description, dict):
        return description
    out = dict(description)
    if level >= depth:
        out.pop("children", None)
    else:
        out["children"] = [_pruned(c, depth, level + 1) for c in description.get("children", [])]
    return out


def _render_lines(d, mode, level, depth):
    out = []
    title = _title(d)
    if mode == "markdown":
        out.append(f"{'#' * min(level + 2, 6)} {title}")
    elif mode == "latex":
        macro = ["section*", "subsection*", "subsubsection*", "paragraph", "subparagraph"][min(level, 4)]
        out.append(f"\\{macro}{{{_tex_text(title)}}}")
    else:
        out.append(("  " * level) + title)
        out.append(("  " * level) + "-" * len(title))
    if d.get("summary"):
        out.append(_para(d["summary"], mode, level))
    facts = d.get("facts") or {}
    if facts:
        out.append("")
        for label, value in facts.items():
            out.append(_bullet(f"{label}: {_fact_text(value, mode)}", mode, level))
    forms = d.get("forms") or {}
    if forms:
        out.append("")
        if d.get("kind") == "solver":
            out.append(_para(_residual_statement(mode), mode, level))
        for name, form in forms.items():
            out.extend(_form_lines(name, form, mode, level))
    conditions = d.get("conditions")
    if conditions is None:
        conditions = d.get("boundary_conditions")
    if conditions:
        out.append("")
        out.append(_para(_heading_text("Boundary conditions", mode), mode, level))
        for bc in conditions:
            out.append(_bullet(_condition_text(bc, mode), mode, level))
    terms = d.get("terms")
    if terms:
        out.append("")
        out.append(_para(_heading_text("Given", mode), mode, level))
        for t in terms:
            out.append(_bullet(_term_text(t, mode), mode, level))
            out.extend(_where_lines(t.get("where", []), mode, level + 1))
    elif d.get("terms_declared") is False:
        out.append("")
        out.append(_para(_emph("this solver does not declare the terms it was given", mode), mode, level))
    children = d.get("children") or []
    if children and (depth is None or level < depth):
        for child in children:
            out.append("")
            out.extend(_render_lines(child, mode, level + 1, depth))
    return out


def _title(d):
    kind = str(d.get("kind") or "object").replace("_", " ")
    name = d.get("name")
    return f"{kind} {name}" if name else kind


def _residual_statement(mode):
    if mode == "text":
        return "residual: int F0 phi + F1 . grad phi = 0 with"
    return r"Residual $\int F_0\,\phi + F_1 \cdot \nabla\phi = 0$ with"


def _heading_text(text, mode):
    return {"markdown": f"**{text}**", "latex": f"\\textbf{{{text}}}", "text": text}[mode]


def _emph(text, mode):
    return {"markdown": f"*{text}*", "latex": f"\\emph{{{text}}}", "text": text}[mode]


def _para(text, mode, level):
    return ("  " * level + text) if mode == "text" else text


def _bullet(text, mode, level):
    if mode == "markdown":
        return f"- {text}"
    if mode == "latex":
        return f"\\item {text}"
    return "  " * level + "  " + text


def _fact_text(value, mode):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if mode == "latex":
        return _tex_text(str(value))
    return str(value)


def _math(latex, mode, display=False):
    if mode == "markdown":
        return f"$${latex}$$" if display else f"${latex}$"
    if mode == "latex":
        return f"\\[{latex}\\]" if display else f"${latex}$"
    return latex


def _value_text(entry, mode):
    """The value of a term or a where-entry in the mode's notation, a bare
    number shown compactly."""
    if mode == "text":
        text = entry.get("text")
        if text in (None, ""):
            return None
        try:
            return f"{float(text):.6g}"
        except (TypeError, ValueError):
            return _plain_math(text)
    latex = entry.get("latex")
    if latex in (None, ""):
        return None
    try:
        return f"{float(latex):.6g}"
    except (TypeError, ValueError):
        return latex


def _form_lines(name, form, mode, level):
    out = []
    symbol = form.get("symbol") or name
    what = form.get("description") or ""
    if mode == "text":
        text = form.get("text", "")
        out.append("  " * level + f"  {name}: {_plain_math(text)}")
        if what:
            out.append("  " * level + f"      {what}")
    else:
        out.append("")
        out.append(_math(f"{symbol} = {form.get('latex', '')}", mode, display=True))
        if what:
            out.append(_emph(what, mode))
    where = _where_lines(form.get("where", []), mode, level + 1)
    if where:
        if mode != "text":
            out.append("")
            out.append("where")
        out.extend(where)
    return out


def _where_lines(entries, mode, level):
    out = []
    for w in entries:
        symbol = w.get("symbol") or w.get("name") or "?"
        value = _value_text(w, mode)
        units = w.get("units")
        what = w.get("description") or ""
        if mode == "text":
            head = "  " * level + f"  {_plain_math(symbol)}"
            if value is not None:
                head += f" = {value}" + (f" {units}" if units else "")
        else:
            indent = "  " * (level - 1) if mode == "markdown" else ""
            span = symbol if value is None else f"{symbol} = {value}"
            head = f"{indent}- {_math(span, mode)}" if mode == "markdown" else f"\\item {_math(span, mode)}"
            if value is not None and units:
                head += f" {_units_text(units, mode)}"
        if what:
            head += f", {what}"
        out.append(head)
        out.extend(_where_lines(w.get("where", []), mode, level + 1))
    return out


def _term_text(t, mode):
    name = t.get("name") or t.get("symbol") or "?"
    value = _value_text(t, mode)
    units = t.get("units")
    what = t.get("description") or ""
    if mode == "text":
        head = str(name)
        if value is not None:
            head += f" = {value}" + (f" {units}" if units else "")
    else:
        head = f"`{name}`" if mode == "markdown" else f"\\texttt{{{_tex_text(name)}}}"
        span = t.get("symbol") or ""
        if value is not None:
            span = f"{span} = {value}"
        if span:
            head += f" {_math(span, mode)}"
            if value is not None and units:
                head += f" {_units_text(units, mode)}"
    if what:
        head += f", {what}"
    return head


def _condition_text(bc, mode):
    kind = bc.get("type") or bc.get("mechanism") or "?"
    where = bc.get("boundary", "?")
    value = _value_text(bc, mode)
    line = f"{kind} on {where}"
    if value:
        line += f": {value}" if mode == "text" else f": {_math(value, mode)}"
    if bc.get("normal") and bc["normal"] != "mesh":
        line += f" (normal {bc['normal']})"
    return line


def _units_text(units, mode):
    if mode == "latex":
        return f"\\,\\mathrm{{{_tex_text(str(units))}}}"
    return str(units)


def _tex_text(text):
    return re.sub(r"([#$%&_{}])", r"\\\1", str(text))


def _plain_math(text):
    """SymPy's text with the Greek and the sub/superscripts a terminal can
    show, through the transcript's own symbol printer."""
    try:
        from underworld3.utilities.transcript_report import _plain_symbol
        return _plain_symbol(text)
    except Exception:
        return str(text)


# --- viewing -------------------------------------------------------------

def view(target, format=None, depth=None, **describe_kwargs):
    """Show a description: ``target`` is an object with ``describe()`` or a
    description already made. With no ``format``, Markdown with mathematics
    in a notebook and plain text elsewhere; a named format prints it."""
    description = target.describe(**describe_kwargs) if hasattr(target, "describe") else target
    if format is None:
        from underworld3.utilities.docstring_utils import in_jupyter
        if in_jupyter():
            from IPython.display import Markdown, display
            display(Markdown(render(description, "markdown", depth)))
            return
        format = "text"
    print(render(description, format, depth), end="")
