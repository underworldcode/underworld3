r"""A local, read-only MCP server over run transcripts.

The server is a thin projection of :class:`underworld3.Transcript` and the
description layer: every tool reads a transcript file and returns the
query's own answer as YAML, so what a tool tells a model is what the
digest, a notebook and a test read. Nothing here runs a model or writes
to one.

Run it with ``python -m underworld3.mcp`` (stdio). The repository's
``.mcp.json`` registers it for Claude Code under the name ``underworld``.

Tools take a ``path`` that may be a transcript file, a run directory that
holds ``transcript.jsonl``, or a ``transcripts`` directory, in which case
the latest run is read. Paths are resolved against the working directory
the server was started in.
"""

import glob
import os

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from ..utilities.describe import plain, render
from ..utilities.transcript_query import Transcript

server = MCPServer(
    "underworld",
    instructions=(
        "Read-only questions about Underworld3 runs, answered from their transcripts. "
        "Start with uw_transcript_list to find runs and uw_transcript_summary for one run; "
        "use uw_transcript_problems for what went wrong, uw_transcript_part for the equation "
        "a solver was solving, with detail='exact' only when the full form is needed."
    ),
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                             idempotent_hint=True, open_world_hint=False)


def _yaml(value):
    import yaml
    return yaml.safe_dump(plain(value), sort_keys=False, allow_unicode=True, width=100)


def resolve(path):
    """The transcript file a path names: a file, a run directory, or a
    ``transcripts`` directory whose latest run is wanted."""
    p = os.path.expanduser(str(path or "."))
    if os.path.isdir(p):
        for candidate in (os.path.join(p, "transcript.jsonl"),
                          os.path.join(p, "latest", "transcript.jsonl"),
                          os.path.join(p, "transcripts", "latest", "transcript.jsonl")):
            if os.path.exists(candidate):
                return candidate
        runs = sorted(glob.glob(os.path.join(p, "*", "transcript.jsonl")))
        if runs:
            return runs[-1]
        raise FileNotFoundError(
            f"no transcript under {p!r}: pass a transcript.jsonl, a run directory, or a "
            f"'transcripts' directory (uw_transcript_list shows what is there)")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"{p!r} does not exist; uw_transcript_list finds transcripts under a directory")


def _transcript(path, run=-1):
    return Transcript(resolve(path), run=run)


def _magnitude(value):
    if isinstance(value, dict):
        return value.get("magnitude")
    return value


def _row(t, position, step):
    events = [e for e in step.get("events", []) if e.get("kind") != "warning"]
    return {
        "position": position,
        "index": step.get("index"),
        "label": step.get("label"),
        "t0": _magnitude(step.get("t0")),
        "dt": _magnitude(step.get("dt")),
        "wall_s": step.get("wall"),
        "completed": bool(step.get("completed")),
        "operators": t.sequence(step),
        "outcomes": [t.outcome(e) for e in events],
        **({"abandoned_by": step["abandoned_by"]} if step.get("abandoned_by") else {}),
    }


@server.tool(name="uw_transcript_list", annotations=_READ_ONLY)
def uw_transcript_list(directory: str = ".", limit: int = 20) -> str:
    """Find run transcripts under a directory (recursively, newest first):
    the file, when the run started, how many steps it holds and whether it
    ended. Use this first when the path of a run is not known."""
    root = os.path.expanduser(directory)
    files = sorted(glob.glob(os.path.join(root, "**", "transcript.jsonl"), recursive=True),
                   key=os.path.getmtime, reverse=True)
    out = []
    for f in files[:max(1, limit)]:
        try:
            t = Transcript(f)
            out.append({"path": f, "run": t.name, "started": t.header.get("started"),
                        "steps": len(t), "ended": bool(t.ended),
                        "abandoned": len(t.abandoned()), "backtracks": len(t.backtracks())})
        except Exception as exc:
            out.append({"path": f, "error": f"{type(exc).__name__}: {exc}"})
    if not out:
        return f"no transcript.jsonl under {root!r}"
    return _yaml(out)


@server.tool(name="uw_transcript_summary", annotations=_READ_ONLY)
def uw_transcript_summary(path: str = ".", run: int = -1) -> str:
    """One run in summary: when it ran, its scales as declared, how many
    steps, which were abandoned, how many backtracks, failed and capped
    solves, and the run's step patterns. The parts (solvers) are listed by
    name; uw_transcript_part gives what each solved."""
    t = _transcript(path, run)
    d = t.describe(depth=1)
    d["children"] = [{"part": c.get("part"), "label": c.get("label"), "solver": c.get("solver"),
                      "unknown": c.get("unknown"), "recorded_at_step": c.get("at_step")}
                     for c in d.get("children", [])]
    d["parts"] = d.pop("children")
    d["file"] = resolve(path)
    return _yaml(d)


@server.tool(name="uw_transcript_steps", annotations=_READ_ONLY)
def uw_transcript_steps(path: str = ".", start: int = 0, count: int = 40, run: int = -1) -> str:
    """Steps in sequence order from position `start`, `count` at a time:
    index, label, t0, dt, wall time, completion, the operators applied and
    the outcome of each. Indices repeat after a rewind, which is why rows
    carry a position as well."""
    t = _transcript(path, run)
    rows = [_row(t, i, s) for i, s in enumerate(t.steps)][start:start + max(1, count)]
    return _yaml({"total": len(t), "from": start, "rows": rows})


@server.tool(name="uw_transcript_patterns", annotations=_READ_ONLY)
def uw_transcript_patterns(path: str = ".", run: int = -1) -> str:
    """The run collapsed to its distinct step patterns: consecutive steps
    with the same operators, outcomes, label and completion, with nothing
    recorded between them, are one pattern with a count. A run that never
    changes has one; each extra pattern is something that happened."""
    return _yaml(_transcript(path, run).patterns())


@server.tool(name="uw_transcript_problems", annotations=_READ_ONLY)
def uw_transcript_problems(path: str = ".", run: int = -1) -> str:
    """Everything that went wrong or went back: abandoned steps with what
    stopped them, rewinds and restores with the reason and detail the
    caller gave, solves that diverged, solves that ran with an inner block
    at its iteration cap, and the count of warnings."""
    t = _transcript(path, run)
    return _yaml({
        "abandoned": [{"position": t.position(s), "index": s.get("index"), "label": s.get("label"),
                       "abandoned_by": s.get("abandoned_by")} for s in t.abandoned()],
        "backtracks": [{k: v for k, v in n.items() if k not in ("kind",)} for n in t.backtracks()],
        "failed": [{"step": i, "solver": e.get("name"), "reason": e.get("reason"),
                    "nonlinear_its": e.get("nl_its")} for i, e in t.failed()],
        "capped": [{"step": i, "solver": e.get("name"), "capped": e.get("capped"),
                    "deadline_expired": e.get("deadline_expired", False)} for i, e in t.capped()],
        "warnings": len(t.warnings()),
    })


@server.tool(name="uw_transcript_events", annotations=_READ_ONLY)
def uw_transcript_events(path: str = ".", kind: str = "", part: str = "", outcome: str = "",
                         step: int = -1, limit: int = 100, run: int = -1) -> str:
    """Events across the run, filtered: kind is solve, adjoint_solve,
    history_shift, swarm_advect or warning; outcome is ok, capped or
    diverged; part is a part id or a solver's label; step limits to one
    step index. Each event comes with its step index and the record's own
    fields (converged, reason, iterations, residual norms)."""
    t = _transcript(path, run)
    found = t.events(kind=kind or None, part=part or None, outcome=outcome or None,
                     step=None if step < 0 else step)
    rows = [{"step": i, **e} for i, e in found[:max(1, limit)]]
    return _yaml({"total": len(found), "events": rows})


@server.tool(name="uw_transcript_step", annotations=_READ_ONLY)
def uw_transcript_step(path: str, index: int, attempt: int = -1, run: int = -1) -> str:
    """One step as recorded, events and all. After a rewind an index can
    name several attempts: attempt=-1 is the one that stands, attempt=0 the
    first (rejected) one."""
    t = _transcript(path, run)
    try:
        s = t.step(index, attempt=attempt)
    except (KeyError, IndexError) as exc:
        return f"error: {exc}; the run has {len(t)} steps (uw_transcript_steps lists them)"
    return _yaml({"position": t.position(s), "row": _row(t, t.position(s), s), "record": s})


@server.tool(name="uw_transcript_compare", annotations=_READ_ONLY)
def uw_transcript_compare(path: str, a: int, b: int, attempt_a: int = -1, attempt_b: int = -1,
                          run: int = -1) -> str:
    """What differs between two steps: operators in one and not the other,
    outcomes that changed for the same operator, and the interval, wall
    time and completion of each."""
    t = _transcript(path, run)
    try:
        return _yaml(t.compare(t.step(a, attempt_a), t.step(b, attempt_b)))
    except (KeyError, IndexError) as exc:
        return f"error: {exc}"


@server.tool(name="uw_transcript_parts", annotations=_READ_ONLY)
def uw_transcript_parts(path: str = ".", run: int = -1) -> str:
    """The parts of the run, the solvers that acted, each with its label,
    the step it was first recorded at, its unknown and its boundary
    conditions in one line, and the steps at which its form or its
    parameter values changed."""
    t = _transcript(path, run)
    out = []
    for name in t.part_names():
        p = t.part(name)
        out.append({"part": name, "label": p.get("label"), "solver": p.get("solver"),
                    "unknown": p.get("unknown"), "dim": p.get("dim"),
                    "recorded_at_step": p.get("at_step"),
                    "boundary_conditions": [f"{bc.get('type')} on {bc.get('boundary')}"
                                            for bc in p.get("boundary_conditions", [])],
                    "changed_at_steps": [s for n, s in t.changes() if n == name]})
    return _yaml(out)


@server.tool(name="uw_transcript_part", annotations=_READ_ONLY)
def uw_transcript_part(path: str, part: str, at_step: int = -1, detail: str = "summary",
                       run: int = -1) -> str:
    """What a part was solving, as recorded: with at_step, the description
    in force at that step. detail='summary' gives the solver, unknown,
    boundary conditions, the named terms with values and units, and the
    run-time constants; detail='forms' adds the residual templates as
    LaTeX with the named expressions inside them; detail='exact' returns
    the whole record, which is large."""
    t = _transcript(path, run)
    try:
        p = t.part(part, at_step=None if at_step < 0 else at_step)
    except KeyError as exc:
        return f"error: {exc}; parts are {t.part_names()}"
    if detail == "exact":
        return _yaml(p)
    out = {"part": p.get("part"), "label": p.get("label"), "solver": p.get("solver"),
           "unknown": p.get("unknown"), "dim": p.get("dim"), "recorded_at_step": p.get("at_step"),
           "boundary_conditions": [{"type": bc.get("type"), "boundary": bc.get("boundary"),
                                    "value": bc.get("text")} for bc in p.get("boundary_conditions", [])],
           "terms": [{"name": x.get("name"), "value": x.get("text"), "units": x.get("units"),
                      "description": x.get("description")} for x in (p.get("terms") or [])],
           "constants": p.get("constants")}
    if detail == "forms":
        out["forms"] = {name: {"symbol": f.get("symbol"), "latex": f.get("latex"),
                               "description": f.get("description"),
                               "where": [f"{w.get('symbol')} = {w.get('value')}"
                                         + (f" {w.get('units')}" if w.get("units") else "")
                                         + (f", {w.get('description')}" if w.get("description") else "")
                                         for w in f.get("where", [])]}
                        for name, f in (p.get("forms") or {}).items()}
    elif detail != "summary":
        return "error: detail must be 'summary', 'forms' or 'exact'"
    return _yaml(out)


@server.tool(name="uw_transcript_key", annotations=_READ_ONLY)
def uw_transcript_key(path: str = ".", format: str = "text", run: int = -1) -> str:
    """The key to the run: every part's residual as implemented, the named
    expressions inside it with values and units, and its boundary
    conditions, as Markdown with LaTeX or as plain text. Long for a run
    with several solvers; uw_transcript_part reads one."""
    from ..utilities.transcript_report import transcript_key
    if format not in ("markdown", "text"):
        return "error: format must be 'markdown' or 'text'"
    return transcript_key(_transcript(path, run), format=format)


@server.tool(name="uw_transcript_adjoint_segments", annotations=_READ_ONLY)
def uw_transcript_adjoint_segments(path: str = ".", run: int = -1) -> str:
    """The run partitioned by adjoint support: the stretches over which
    every operator can be differentiated, and the refusals between them
    with their reasons."""
    return _yaml(_transcript(path, run).adjoint_segments())


@server.tool(name="uw_describe_render", annotations=_READ_ONLY)
def uw_describe_render(record_yaml: str, format: str = "markdown", depth: int = -1) -> str:
    """Render a description record (as returned by any uw_transcript tool
    or by an object's describe()) in another form: markdown, text, latex,
    yaml or json. depth limits levels of children; -1 renders all."""
    import yaml
    try:
        record = yaml.safe_load(record_yaml)
    except Exception as exc:
        return f"error: not YAML: {exc}"
    if not isinstance(record, dict):
        return "error: the record must be a mapping with kind, name and summary"
    try:
        return render(record, format, depth=None if depth < 0 else depth)
    except ValueError as exc:
        return f"error: {exc}"


def main():
    server.run(transport="stdio")
