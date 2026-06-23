"""Scaffold workflow packages into user working directories.

Discovers available workflows (builtin examples and pip-installed packages)
and copies starter files into a user-specified directory.

CLI usage (called from ``./uw workflow``)::

    python -m underworld3.workflows.scaffold list --repo-root /path/to/uw3
    python -m underworld3.workflows.scaffold init convection --repo-root /path/to/uw3
    python -m underworld3.workflows.scaffold init h2ex ./my-project --force

Python API::

    from underworld3.workflows import list_workflows, init_workflow

    workflows = list_workflows()
    init_workflow("convection", target_dir="./my-study")
"""

import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class WorkflowInfo:
    """Metadata about a discovered workflow."""

    name: str
    description: str
    source: str  # "builtin" or "package:<dist-name>"
    scaffold_dir: Optional[Path] = None  # For external packages
    files: Dict[str, Path] = field(default_factory=dict)  # For builtins
    config_class: Optional[str] = None


# ---------------------------------------------------------------------------
# Builtin registry
# ---------------------------------------------------------------------------

_BUILTIN_WORKFLOWS = {
    "convection": {
        "description": "Rayleigh-Benard thermal convection (constant viscosity)",
        "files": {
            "config.py": "convection_config.py",
            "notebook.py": "convection_notebook.py",
        },
        "config_class": "ConvectionConfig",
    },
    "h2ex": {
        "description": "H2Ex fault-controlled flow with product persistence",
        "files": {
            "config.py": "h2ex_config.py",
            "notebook.py": "h2ex_notebook.py",
        },
        "config_class": "H2ExConfig",
    },
}

_EXAMPLES_SUBDIR = Path("docs") / "examples" / "workflows"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_builtins(repo_root: Optional[str] = None) -> Dict[str, WorkflowInfo]:
    """Find builtin workflows from the repo's docs/examples/workflows/.

    Only available when *repo_root* is provided (i.e. running from ``./uw``).
    """
    if repo_root is None:
        return {}

    examples_dir = Path(repo_root) / _EXAMPLES_SUBDIR
    if not examples_dir.is_dir():
        return {}

    workflows = {}
    for name, meta in _BUILTIN_WORKFLOWS.items():
        # Resolve file paths
        resolved_files = {}
        all_found = True
        for dest_name, src_name in meta["files"].items():
            src_path = examples_dir / src_name
            if src_path.exists():
                resolved_files[dest_name] = src_path
            else:
                all_found = False

        if not all_found:
            continue

        workflows[name] = WorkflowInfo(
            name=name,
            description=meta["description"],
            source="builtin",
            files=resolved_files,
            config_class=meta.get("config_class"),
        )

    return workflows


def _discover_external() -> Dict[str, WorkflowInfo]:
    """Find external workflows registered via entry_points.

    External packages register in their ``pyproject.toml``::

        [project.entry-points."underworld3.workflows"]
        hydrogen = "uw3_hydrogen:workflow_info"

    The callable must return a dict with ``"description"`` and
    ``"scaffold_dir"`` keys.
    """
    try:
        import importlib.metadata

        eps = importlib.metadata.entry_points(group="underworld3.workflows")
    except (ImportError, TypeError):
        return {}

    workflows = {}
    for ep in eps:
        try:
            info_fn = ep.load()
            info = info_fn()
            dist_name = ep.dist.name if ep.dist else "unknown"
            scaffold_dir = info.get("scaffold_dir")
            if scaffold_dir is not None:
                scaffold_dir = Path(scaffold_dir)

            workflows[ep.name] = WorkflowInfo(
                name=ep.name,
                description=info.get("description", ""),
                source=f"package:{dist_name}",
                scaffold_dir=scaffold_dir,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to load workflow entry point '{ep.name}': {exc}",
                stacklevel=2,
            )

    return workflows


def list_workflows(repo_root: Optional[str] = None) -> Dict[str, WorkflowInfo]:
    """Discover all available workflows (builtin + external).

    Parameters
    ----------
    repo_root : str or None
        Repository root for builtin discovery.  Passed automatically
        by ``./uw workflow list``.

    Returns
    -------
    dict
        Mapping of workflow name to :class:`WorkflowInfo`.
        External packages override builtins if names collide.
    """
    workflows = _discover_builtins(repo_root)
    workflows.update(_discover_external())  # externals win on collision
    return workflows


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


def init_workflow(
    name: str,
    target_dir: str = ".",
    repo_root: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Scaffold a workflow into a target directory.

    Creates ``<target_dir>/<name>/`` with editable copies of notebooks,
    configs, a default ``params.yaml``, and an ``output/`` directory.

    Parameters
    ----------
    name : str
        Workflow name (as shown by :func:`list_workflows`).
    target_dir : str
        Parent directory.  The workflow is placed in a subdirectory
        named *name*.  Defaults to current directory.
    repo_root : str or None
        Repository root for builtin workflows.
    force : bool
        If True, overwrite existing files.

    Returns
    -------
    Path
        The created workflow directory.

    Raises
    ------
    ValueError
        If the workflow name is not found.
    FileExistsError
        If the destination exists and *force* is False.
    """
    workflows = list_workflows(repo_root)

    if name not in workflows:
        available = ", ".join(sorted(workflows.keys())) or "(none)"
        raise ValueError(
            f"Unknown workflow: '{name}'\n"
            f"Available workflows: {available}"
        )

    info = workflows[name]
    dest = Path(target_dir) / name

    if info.source == "builtin":
        _scaffold_builtin(info, dest, force)
    else:
        _scaffold_external(info, dest, force)

    return dest


def _scaffold_builtin(info: WorkflowInfo, dest: Path, force: bool):
    """Copy builtin workflow files and generate supporting files."""
    dest.mkdir(parents=True, exist_ok=True)

    for dest_name, src_path in info.files.items():
        target = dest / dest_name
        if target.exists() and not force:
            raise FileExistsError(
                f"File already exists: {target}\n"
                f"Use --force to overwrite."
            )
        shutil.copy2(src_path, target)

    (dest / "output").mkdir(exist_ok=True)
    _generate_readme(info, dest)
    _generate_default_params(info, dest)


def _scaffold_external(info: WorkflowInfo, dest: Path, force: bool):
    """Copy external package's scaffold directory."""
    if info.scaffold_dir is None or not info.scaffold_dir.is_dir():
        raise ValueError(
            f"Workflow '{info.name}' has no scaffold directory.\n"
            f"The package must provide 'scaffold_dir' in its workflow_info()."
        )

    if dest.exists() and not force:
        raise FileExistsError(
            f"Directory already exists: {dest}\n"
            f"Use --force to overwrite."
        )

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(info.scaffold_dir, dest)

    (dest / "output").mkdir(exist_ok=True)


def _generate_readme(info: WorkflowInfo, dest: Path):
    """Write a starter README.md."""
    lines = [
        f"# {info.name.replace('_', ' ').title()} Workflow",
        "",
        info.description,
        "",
        "## Getting Started",
        "",
        "1. Review and edit `config.py` to set your parameters",
        "2. Run the notebook: `pixi run -e default python notebook.py`",
        "   or open in Jupyter",
        "3. Output will be saved to `output/`",
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "|------|---------|",
    ]

    for fname in sorted(info.files.keys()):
        purpose = "Configuration" if "config" in fname else "Notebook"
        lines.append(f"| `{fname}` | {purpose} |")

    lines.extend([
        "| `params.yaml` | Default parameters (editable) |",
        "| `output/` | Simulation output directory |",
        "",
        f"Scaffolded from: {info.source}",
        "",
    ])

    (dest / "README.md").write_text("\n".join(lines))


def _generate_default_params(info: WorkflowInfo, dest: Path):
    """Generate a default params.yaml by instantiating the config class."""
    params_path = dest / "params.yaml"
    if params_path.exists():
        return

    # Try to import and instantiate the config class from the copied file
    # This is best-effort — if it fails, skip silently
    if info.config_class is None:
        return

    try:
        import importlib.util
        import sys

        config_file = dest / "config.py"
        if not config_file.exists():
            return

        # Load the config module from the copied file
        spec = importlib.util.spec_from_file_location(
            f"_scaffold_{info.name}_config", config_file
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        config_cls = getattr(mod, info.config_class, None)
        if config_cls is not None and hasattr(config_cls, "save_yaml"):
            config_cls().save_yaml(str(params_path))
    except Exception:
        # Best-effort: don't fail the scaffold if params generation fails
        pass
    finally:
        # Clean up __pycache__ created by importing the config module
        pycache = dest / "__pycache__"
        if pycache.is_dir():
            shutil.rmtree(pycache)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for ``python -m underworld3.workflows.scaffold``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="uw workflow",
        description="Scaffold workflow packages into user directories",
    )
    sub = parser.add_subparsers(dest="command")

    list_p = sub.add_parser("list", help="Show available workflows")
    list_p.add_argument("--repo-root", default=None)

    init_p = sub.add_parser("init", help="Scaffold a workflow")
    init_p.add_argument("name", help="Workflow name")
    init_p.add_argument("target_dir", nargs="?", default=".")
    init_p.add_argument("--repo-root", default=None)
    init_p.add_argument(
        "--force", action="store_true", help="Overwrite existing files"
    )

    args = parser.parse_args()

    if args.command == "list":
        workflows = list_workflows(repo_root=args.repo_root)
        if not workflows:
            print("No workflows found.")
            sys.exit(0)

        name_w = max(len(w.name) for w in workflows.values())
        src_w = max(len(w.source) for w in workflows.values())
        print(f"\n{'Name':<{name_w}}  {'Source':<{src_w}}  Description")
        print(f"{'-' * name_w}  {'-' * src_w}  {'-' * 40}")
        for w in sorted(workflows.values(), key=lambda x: x.name):
            print(f"{w.name:<{name_w}}  {w.source:<{src_w}}  {w.description}")
        print()

    elif args.command == "init":
        try:
            dest = init_workflow(
                args.name,
                target_dir=args.target_dir,
                repo_root=args.repo_root,
                force=args.force,
            )
            print(f"\nWorkflow '{args.name}' scaffolded to: {dest}")
            print()
            print("Next steps:")
            print(f"  cd {dest}")
            print("  # Edit config.py and/or params.yaml")
            print("  # Run: pixi run -e default python notebook.py")
            print()
        except (ValueError, FileExistsError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
