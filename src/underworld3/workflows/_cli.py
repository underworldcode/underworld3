"""Auto-derive an argparse CLI from a :class:`WorkflowConfig` subclass.

Walks the config's Pydantic ``model_fields`` and adds one argparse
argument per field, mapping types automatically:

- ``bool`` → ``--flag`` / ``--no-flag`` (BooleanOptionalAction)
- ``int`` / ``float`` / ``str`` → typed value
- ``Literal[...]`` → ``choices=`` constraint
- everything else → silently skipped (CLI may not be the right
  surface for that field; the user can still set it from Python)

Use case: a workflow's CLI driver is one ``cli_from_config(MyConfig)``
call away — no manual ``add_argument`` plumbing per field.

Usage
-----

::

    # in mymodule_cli.py
    from underworld3.workflows import cli_from_config
    import mymodule_config as mm

    def main():
        parser = cli_from_config(mm.MyConfig)
        parser.add_argument("--no-evolve", action="store_true")
        args = parser.parse_args()
        config = config_from_args(mm.MyConfig, args)
        ...
"""

from __future__ import annotations

import argparse
import typing as _typing


# Names that describe the workflow itself rather than user-tunable
# parameters; suppressed by default but callable code can override.
_DEFAULT_HIDDEN_FIELDS = frozenset({"workflow_name", "description"})


def _add_field_arg(
    parser: argparse.ArgumentParser,
    name: str,
    finfo,
) -> bool:
    """Add an argparse argument that mirrors a Pydantic field.

    Returns ``True`` if the field was mapped to a CLI flag, ``False``
    if its type wasn't recognised (silently skipped).
    """
    cli_name = "--" + name.replace("_", "-")
    annotation = finfo.annotation
    default = finfo.default
    help_text = finfo.description or ""
    help_text = (
        f"{help_text} (default: {default!r})" if help_text
        else f"default: {default!r}"
    )

    origin = _typing.get_origin(annotation)
    args = _typing.get_args(annotation)

    if annotation is bool:
        parser.add_argument(
            cli_name,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=help_text,
        )
        return True
    if annotation is int:
        parser.add_argument(cli_name, type=int, default=None, help=help_text)
        return True
    if annotation is float:
        parser.add_argument(cli_name, type=float, default=None, help=help_text)
        return True
    if annotation is str:
        parser.add_argument(cli_name, type=str, default=None, help=help_text)
        return True
    if origin is _typing.Literal:
        parser.add_argument(
            cli_name, choices=list(args), default=None, help=help_text,
        )
        return True
    return False


def cli_from_config(
    ConfigClass,
    *,
    description: _typing.Optional[str] = None,
    hidden_fields=_DEFAULT_HIDDEN_FIELDS,
) -> argparse.ArgumentParser:
    """Build an argparse parser auto-derived from *ConfigClass*' fields.

    Parameters
    ----------
    ConfigClass : type
        A :class:`WorkflowConfig` subclass (or any Pydantic ``BaseModel``
        with ``model_fields``).
    description : str, optional
        Parser description.  Defaults to the class' docstring's first
        line, if any.
    hidden_fields : iterable of str
        Field names that should *not* be exposed on the CLI.  Defaults
        to ``{"workflow_name", "description"}`` — the workflow-itself
        metadata.  Callers can pass their own list if they want to
        suppress (or expose) different fields.

    Returns
    -------
    parser : argparse.ArgumentParser
        Pre-populated with one argument per recognised Pydantic field.
        Callers can ``add_argument`` further (e.g. action flags like
        ``--movies`` or ``--no-evolve``) before parsing.
    """
    if description is None:
        doc = (ConfigClass.__doc__ or "").strip()
        description = doc.split("\n", 1)[0] if doc else None

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    hidden = set(hidden_fields)
    for name, finfo in ConfigClass.model_fields.items():
        if name in hidden:
            continue
        _add_field_arg(parser, name, finfo)
    return parser


def config_from_args(
    ConfigClass,
    args: argparse.Namespace,
    *,
    hidden_fields=_DEFAULT_HIDDEN_FIELDS,
):
    """Construct a *ConfigClass* instance from parsed CLI args.

    ``None`` values (i.e. fields the user didn't override on the
    command line) drop through to the config's default.  Fields not
    present in *args* are silently ignored.
    """
    hidden = set(hidden_fields)
    kwargs = {}
    for name in ConfigClass.model_fields:
        if name in hidden:
            continue
        val = getattr(args, name, None)
        if val is None:
            continue
        kwargs[name] = val
    return ConfigClass(**kwargs)


__all__ = ["cli_from_config", "config_from_args"]
