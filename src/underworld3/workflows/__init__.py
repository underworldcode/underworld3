"""Workflow infrastructure for domain-specific simulation packages.

Provides a ``WorkflowConfig`` base class and small utilities so that
external packages (e.g. ``uw3-hydrogen``, ``uw3-groundwater``) can
define validated, serializable parameter sets on top of Underworld3.

See ``docs/developer/guides/workflow-packages.md`` for the full pattern.
"""

from ._base import WorkflowConfig
from ._cache import config_cache_key, config_snapshot
from ._cli import cli_from_config, config_from_args
from ._diagram import diagram, render
from ._products import WorkflowProducts
from ._run import RUN_NAME, Manifest, Run
from ._runner import WorkflowRunner
from ._utils import check_dependencies, parse_quantity, show_source, workflow_step

# Public API version for the run-directory + product-graph primitives.
# Pre-1.0 — shape may shift as a second consumer exercises the design.
#
# 0.1 (step 5):    Run/Manifest lifted into the package; manifest
#                  carries workflow_api stamp.
# 0.2 (Phase A):   cache_key + inputs fields added to both Run.manifest
#                  and WorkflowProducts entries; config_cache_key /
#                  config_snapshot helpers exposed.
__api_version__ = "0.2"


def list_workflows(repo_root=None):
    """Discover all available workflows (builtin + external).

    See :mod:`underworld3.workflows.scaffold` for details.
    """
    from .scaffold import list_workflows as _list_workflows

    return _list_workflows(repo_root)


def init_workflow(name, target_dir=".", repo_root=None, force=False):
    """Scaffold a workflow into a target directory.

    See :mod:`underworld3.workflows.scaffold` for details.
    """
    from .scaffold import init_workflow as _init_workflow

    return _init_workflow(name, target_dir=target_dir, repo_root=repo_root, force=force)


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
            prod = getattr(obj, "workflow_produces", [])
            reqs = getattr(obj, "workflow_requires", [])
            steps.append((name, desc, prod, reqs))

    has_dag = any(s[2] or s[3] for s in steps)

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
            th = (
                '<th style="text-align:left; padding:4px 12px 4px 0; '
                'border-bottom:2px solid #ccc;">'
            )
            html += '<table style="border-collapse:collapse;"><tr>'
            html += f"{th}Step</th>{th}Description</th>"
            if has_dag:
                html += f"{th}Produces</th>{th}Requires</th>"
            html += "</tr>"

            for name, desc, prod, reqs in steps:
                td = '<td style="padding:2px 12px 2px 0;'
                html += (
                    f"<tr>"
                    f'{td} font-family:monospace;">{name}</td>'
                    f'{td}">{desc}</td>'
                )
                if has_dag:
                    prod_str = ", ".join(prod) if prod else "\u2014"
                    reqs_str = ", ".join(reqs) if reqs else "\u2014"
                    html += (
                        f'{td} font-family:monospace; color:#555;">{prod_str}</td>'
                        f'{td} font-family:monospace; color:#555;">{reqs_str}</td>'
                    )
                html += "</tr>"
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
            if has_dag:
                desc_w = max(len(s[1]) for s in steps)
                prod_w = max(len(", ".join(s[2]) or "\u2014") for s in steps)
                print("  Steps:")
                hdr = f"    {'Step':<{name_w}}  {'Description':<{desc_w}}  {'Produces':<{prod_w}}  Requires"
                print(hdr)
                print(f"    {'\u2500' * name_w}  {'\u2500' * desc_w}  {'\u2500' * prod_w}  {'\u2500' * 8}")
                for name, desc, prod, reqs in steps:
                    p = ", ".join(prod) or "\u2014"
                    r = ", ".join(reqs) or "\u2014"
                    print(f"    {name:<{name_w}}  {desc:<{desc_w}}  {p:<{prod_w}}  {r}")
            else:
                print("  Steps:")
                for name, desc, _, _ in steps:
                    print(f"    {name:<{name_w}}  {desc}")
        if not configs and not steps:
            print("  No workflow steps found.")
        print()
        print("  Use module.function.view() to see the source of any step.")


__all__ = [
    "Manifest",
    "RUN_NAME",
    "Run",
    "WorkflowConfig",
    "WorkflowProducts",
    "WorkflowRunner",
    "__api_version__",
    "check_dependencies",
    "cli_from_config",
    "config_cache_key",
    "config_from_args",
    "config_snapshot",
    "diagram",
    "init_workflow",
    "list_workflows",
    "parse_quantity",
    "render",
    "show_source",
    "workflow_step",
    "view",
]
