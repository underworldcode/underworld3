"""Workflow infrastructure for domain-specific simulation packages.

Provides a ``WorkflowConfig`` base class and small utilities so that
external packages (e.g. ``uw3-hydrogen``, ``uw3-groundwater``) can
define validated, serializable parameter sets on top of Underworld3.

See ``docs/developer/guides/workflow-packages.md`` for the full pattern.
"""

from ._base import WorkflowConfig
from ._utils import check_dependencies, parse_quantity, show_source, workflow_step


def view(module):
    """Display the workflow steps defined in *module*.

    Scans *module* for functions decorated with ``@workflow_step`` and
    lists them with their descriptions.  In Jupyter this renders as an
    HTML table; in a terminal as plain text.

    Parameters
    ----------
    module : module
        A workflow module (e.g. ``import convection_config as convection;
        uw.workflows.view(convection)``).
    """
    import inspect

    steps = []
    for name, obj in inspect.getmembers(module, callable):
        if getattr(obj, "_is_workflow_step", False):
            desc = obj.workflow_description or ""
            steps.append((name, desc))

    # Also find WorkflowConfig subclasses
    configs = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, WorkflowConfig) and obj is not WorkflowConfig:
            doc = (obj.__doc__ or "").strip().split("\n")[0]
            configs.append((name, doc))

    mod_name = getattr(module, "__name__", str(module))

    try:
        from IPython.display import HTML, display

        html = f"<h4>{mod_name}</h4>"

        if configs:
            html += "<p><strong>Config classes:</strong></p><ul>"
            for name, doc in configs:
                html += f'<li><code>{name}</code> — {doc}</li>'
            html += "</ul>"

        if steps:
            html += (
                '<table style="border-collapse:collapse;">'
                "<tr>"
                '<th style="text-align:left; padding:4px 12px 4px 0; '
                'border-bottom:2px solid #ccc;">Step</th>'
                '<th style="text-align:left; padding:4px 12px 4px 0; '
                'border-bottom:2px solid #ccc;">Description</th>'
                "</tr>"
            )
            for name, desc in steps:
                html += (
                    "<tr>"
                    f'<td style="padding:2px 12px 2px 0; font-family:monospace;">{name}</td>'
                    f'<td style="padding:2px 12px 2px 0;">{desc}</td>'
                    "</tr>"
                )
            html += "</table>"
            html += (
                '<p style="color:#888; font-size:0.9em;">'
                "Use <code>module.function.view()</code> to see the source of any step."
                "</p>"
            )
        elif not configs:
            html += "<p><em>No workflow steps found.</em></p>"

        display(HTML(html))

    except ImportError:
        print(mod_name)
        if configs:
            print("  Config classes:")
            for name, doc in configs:
                print(f"    {name} — {doc}")
        if steps:
            name_w = max(len(s[0]) for s in steps)
            print("  Steps:")
            for name, desc in steps:
                print(f"    {name:<{name_w}}  {desc}")
        if not configs and not steps:
            print("  No workflow steps found.")
        print()
        print("  Use module.function.view() to see the source of any step.")


__all__ = [
    "WorkflowConfig",
    "check_dependencies",
    "parse_quantity",
    "show_source",
    "workflow_step",
    "view",
]
