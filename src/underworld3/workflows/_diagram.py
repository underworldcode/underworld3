"""Render a workflow module's produces/requires DAG as a flow chart.

Walks a workflow module's ``@workflow_step``-decorated functions and
emits a Graphviz DOT source string with one node per product and one
edge per ``requires`` link.  An optional *status_provider* callback
colours nodes by cache state ("cached", "loaded", "missing", ...).

Used both as a developer aid (``runner.diagram()`` for at-a-glance
DAG inspection) and as a documentation artefact (the DOT source can
be rendered to PNG/SVG via the ``dot`` command and embedded in
notebooks or papers).

The cetz/typst rendering path mentioned in the design memo would
emit a different source string from the same DAG walk; for now the
DOT path is the working primitive.

Example
-------

>>> from underworld3.workflows import diagram
>>> import convection_config as convection
>>> dot_src = diagram(convection)
>>> Path("convection_dag.dot").write_text(dot_src)
>>> # then in a shell: dot -Tpng convection_dag.dot -o convection_dag.png
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional


# Colours used when *status_provider* labels a node.
_STATUS_COLOURS = {
    "cached": "#bce6bc",      # green — sitting in memory
    "loaded": "#bcd7e6",      # blue — sitting on disk
    "on_disk": "#bcd7e6",     # alias
    "building": "#f7e389",    # yellow — currently running
    "missing": "#e6e6e6",     # grey — not yet built
    "stale": "#e6bcbc",       # red — input changed, needs rebuild
}


def _collect_dag(module) -> tuple[dict, list]:
    """Walk *module* for ``@workflow_step`` functions.

    Returns ``(producers, edges)`` where *producers* maps product
    name → producing step and *edges* is a list of ``(upstream,
    downstream)`` tuples.
    """
    producers: dict[str, Callable] = {}
    edges: list[tuple[str, str]] = []
    for _name, obj in inspect.getmembers(module, callable):
        if not getattr(obj, "_is_workflow_step", False):
            continue
        produces = getattr(obj, "workflow_produces", None) or []
        requires = getattr(obj, "workflow_requires", None) or []
        for prod in produces:
            producers[prod] = obj
            for req in requires:
                edges.append((req, prod))
    return producers, edges


def diagram(
    module,
    status_provider: Optional[Callable[[str], str]] = None,
    *,
    title: Optional[str] = None,
    rankdir: str = "LR",
) -> str:
    """Generate a Graphviz DOT source string for *module*'s DAG.

    Parameters
    ----------
    module : module
        A workflow module (e.g. ``import convection_config as cc``;
        pass ``cc``).
    status_provider : callable, optional
        ``status_provider(name)`` → status string for product *name*.
        Recognised statuses (and their fill colours) are listed in
        ``_STATUS_COLOURS`` — anything else falls through to the
        default "missing" grey.  Typically wired to
        :meth:`WorkflowRunner.status`.
    title : str, optional
        Graph title.  Defaults to ``module.__name__``.
    rankdir : str
        Graphviz ``rankdir`` — ``"LR"`` (left→right, default) or
        ``"TB"`` (top→bottom).

    Returns
    -------
    dot_source : str
        Compile with ``dot -Tpng input.dot -o output.png`` (or
        ``-Tsvg``) to render the diagram.
    """
    producers, edges = _collect_dag(module)
    if title is None:
        title = getattr(module, "__name__", "workflow")

    lines = [
        f'digraph "{title}" {{',
        f'  rankdir="{rankdir}";',
        '  node [shape=box, style="rounded,filled", fontname="monospace", fontsize=10];',
        '  edge [fontname="monospace", fontsize=8, color="#888888"];',
        f'  label="{title}"; labelloc="t"; fontname="sans-serif"; fontsize=12;',
    ]

    # Nodes.
    for name in sorted(producers):
        status = status_provider(name) if status_provider else None
        colour = _STATUS_COLOURS.get(status, _STATUS_COLOURS["missing"])
        step_name = producers[name].__name__
        # Tooltip carries the producing step's name so the SVG hover
        # text shows what built this product.
        lines.append(
            f'  "{name}" [fillcolor="{colour}", '
            f'tooltip="produced by {step_name}()"];'
        )

    # Add nodes for required-but-unproduced products (external inputs)
    # so dangling edges aren't silently dropped.
    referenced = {req for req, _ in edges}
    external = referenced - set(producers)
    for name in sorted(external):
        lines.append(
            f'  "{name}" [fillcolor="#fff8d0", '
            f'tooltip="external input (no producer in this module)", '
            f'style="rounded,filled,dashed"];'
        )

    # Edges.
    for upstream, downstream in edges:
        lines.append(f'  "{upstream}" -> "{downstream}";')

    lines.append("}")
    return "\n".join(lines)


def render(
    module,
    output_path,
    *,
    format: Optional[str] = None,
    status_provider: Optional[Callable[[str], str]] = None,
    rankdir: str = "LR",
) -> Path:
    """Render *module*'s DAG to an image file via the ``dot`` command.

    Convenience wrapper around :func:`diagram` plus ``subprocess``.
    Requires the ``dot`` binary on ``PATH`` (``apt install graphviz``,
    ``brew install graphviz``, etc.).

    Parameters
    ----------
    module : module
        A workflow module.
    output_path : str or Path
        Destination file.  Format inferred from the extension
        (``.png``, ``.svg``, ``.pdf``) unless overridden by *format*.
    format : str, optional
        Graphviz output format, e.g. ``"png"``, ``"svg"``, ``"pdf"``.
        Inferred from *output_path* extension if not given.
    status_provider, rankdir
        Forwarded to :func:`diagram`.

    Returns
    -------
    output_path : Path
        Resolved path of the written image.

    Raises
    ------
    FileNotFoundError
        If the ``dot`` command is not on PATH.
    """
    output_path = Path(output_path)
    if format is None:
        ext = output_path.suffix.lstrip(".").lower()
        format = ext or "png"

    dot_bin = shutil.which("dot")
    if dot_bin is None:
        raise FileNotFoundError(
            "Graphviz 'dot' binary not found on PATH.  "
            "Install with `brew install graphviz` (macOS), "
            "`apt install graphviz` (Debian/Ubuntu), or equivalent."
        )

    src = diagram(
        module, status_provider=status_provider, rankdir=rankdir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [dot_bin, f"-T{format}", "-o", str(output_path)],
        input=src.encode("utf-8"),
        check=True,
    )
    return output_path


__all__ = ["diagram", "render"]
