"""Common utilities for workflow packages."""

import functools
import inspect
from typing import Dict, Optional


def check_dependencies(packages: Dict[str, str]) -> None:
    """Check that optional packages are importable; raise with install hints.

    Parameters
    ----------
    packages : dict
        Mapping of import name to install instruction, e.g.
        ``{"geopandas": "pip install geopandas"}``.

    Raises
    ------
    ImportError
        If any package cannot be imported.
    """
    import importlib

    missing = []
    for module_name, install_hint in packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(f"  {module_name}  →  {install_hint}")

    if missing:
        msg = "Missing optional dependencies:\n" + "\n".join(missing)
        raise ImportError(msg)


def parse_quantity(s: str):
    """Parse a string like ``"1000 km"`` into a ``uw.quantity``.

    Parameters
    ----------
    s : str
        Value and unit separated by whitespace, e.g. ``"1e21 Pa*s"``.

    Returns
    -------
    UWQuantity
        The parsed quantity.
    """
    import underworld3 as uw

    parts = s.strip().split(None, 1)
    if len(parts) == 2:
        value, unit = parts
        return uw.quantity(float(value), unit)
    # Dimensionless number
    return uw.quantity(float(parts[0]), "")


# ---------------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------------


def show_source(fn) -> None:
    """Display the source code of *fn* with syntax highlighting.

    In Jupyter the source is rendered as a Python code block via
    ``IPython.display.Markdown``.  In a terminal it falls back to
    plain ``print``.

    Works on any function or method, not just workflow steps.

    Parameters
    ----------
    fn : callable
        The function whose source you want to inspect.
    """
    source = inspect.getsource(fn)

    try:
        from IPython.display import Markdown, display
        display(Markdown(f"```python\n{source}\n```"))
    except ImportError:
        print(source)


# ---------------------------------------------------------------------------
# Workflow step decorator
# ---------------------------------------------------------------------------


def workflow_step(fn=None, *, description=None, produces=None, requires=None):
    """Mark a function as a workflow helper step.

    Attaches a ``.view()`` method that displays the function source
    (syntax-highlighted in Jupyter).  The decorator is transparent —
    the wrapped function behaves identically to the original.

    Can be used with or without arguments::

        @workflow_step
        def create_mesh(config): ...

        @workflow_step(description="Build the simulation mesh")
        def create_mesh(config): ...

        @workflow_step(
            description="Adapt mesh near fault surfaces",
            produces=["adapted_mesh"],
            requires=["mesh", "fault_surfaces"],
        )
        def adapt_mesh(mesh, faults, config): ...

    The *description* is stored as ``fn.workflow_description`` and the
    *produces*/*requires* lists document the DAG of product
    dependencies (used by ``view()`` and ``WorkflowProducts``).

    Parameters
    ----------
    fn : callable, optional
        The function to decorate (when used without parentheses).
    description : str, optional
        Short human-readable description of this step.
    produces : list of str, optional
        Product names this step creates.
    requires : list of str, optional
        Product names this step depends on.
    """

    def _decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Attach metadata
        wrapper._is_workflow_step = True
        wrapper.workflow_description = description or func.__doc__ or ""
        wrapper.workflow_produces = list(produces) if produces else []
        wrapper.workflow_requires = list(requires) if requires else []

        # Attach view() — shows source in Jupyter or terminal
        def _view():
            show_source(func)

        wrapper.view = _view

        return wrapper

    # Support both @workflow_step and @workflow_step(description=...)
    if fn is not None:
        return _decorate(fn)
    return _decorate
