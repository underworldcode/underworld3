"""CLI driver for a single Rayleigh-Bénard convection run.

Replaces the old ``run_ra_variant.py``-style monkey-patching with
straight config overrides — every field of :class:`ConvectionConfig`
becomes a CLI flag automatically (hyphens for underscores).

Usage
-----

::

    pixi run -e default python simulate.py [--option=VALUE ...]

Examples
--------

Run defaults (Ra=1e6, T_degree=3, output to ``output/convection/run``)::

    python simulate.py

Override Ra and refine T to degree 5::

    python simulate.py --rayleigh=1e5 --T-degree=5 \
        --output-dir=output/Ra1e5_T5

Resume / extend an existing run with a higher step cap::

    python simulate.py --output-dir=output/Ra1e5_T5 --max-steps=8000

Render frames + an mp4 after solving::

    python simulate.py --rayleigh=1e5 --movies

Use ``--help`` to see every available knob.
"""

from __future__ import annotations

import argparse
import sys
import typing as _typing
from pathlib import Path

import convection_config as cc


# Fields that are part of the config object but should not be exposed
# on the CLI — they describe the workflow itself rather than a knob.
_HIDDEN_FIELDS = {"workflow_name", "description"}


def _add_field_arg(parser: argparse.ArgumentParser, name: str, finfo) -> None:
    """Add an argparse argument that mirrors a ConvectionConfig field."""
    cli_name = "--" + name.replace("_", "-")
    annotation = finfo.annotation
    default = finfo.default
    help_text = finfo.description or ""
    help_text = f"{help_text} (default: {default!r})" if help_text else f"default: {default!r}"

    origin = _typing.get_origin(annotation)
    args = _typing.get_args(annotation)

    if annotation is bool:
        parser.add_argument(
            cli_name, action=argparse.BooleanOptionalAction,
            default=None, help=help_text,
        )
        return
    if annotation is int:
        parser.add_argument(cli_name, type=int, default=None, help=help_text)
        return
    if annotation is float:
        parser.add_argument(cli_name, type=float, default=None, help=help_text)
        return
    if annotation is str:
        parser.add_argument(cli_name, type=str, default=None, help=help_text)
        return
    if origin is _typing.Literal:
        parser.add_argument(
            cli_name, choices=list(args), default=None, help=help_text,
        )
        return
    # Unknown type — skip silently so the CLI never blocks on a new field
    # type (the caller can still set it via Python).


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser by introspecting ConvectionConfig fields."""
    parser = argparse.ArgumentParser(
        description="Run a single convection workflow to steady state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for name, finfo in cc.ConvectionConfig.model_fields.items():
        if name in _HIDDEN_FIELDS:
            continue
        _add_field_arg(parser, name, finfo)

    parser.add_argument(
        "--movies", action="store_true",
        help="Render temperature frames + mp4 after the run finishes",
    )
    parser.add_argument(
        "--no-evolve", action="store_true",
        help="Build the config and print it but skip the actual run",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> cc.ConvectionConfig:
    """Build a ConvectionConfig from parsed CLI args (Nones drop to defaults)."""
    config_kwargs = {}
    for name in cc.ConvectionConfig.model_fields:
        if name in _HIDDEN_FIELDS:
            continue
        val = getattr(args, name, None)
        if val is None:
            continue
        config_kwargs[name] = val
    return cc.ConvectionConfig(**config_kwargs)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    print(f"[simulate] ConvectionConfig:")
    for k in cc._IDENTITY_FIELDS:
        print(f"  {k:18s} = {getattr(config, k)!r}")
    print(f"  {'output_dir':18s} = {config.output_dir!r}")
    print(f"  {'max_steps':18s} = {config.max_steps}")
    print(f"  {'restart_policy':18s} = {config.restart_policy!r}")

    if args.no_evolve:
        print("[simulate] --no-evolve set; skipping run.")
        return 0

    # Defer heavy imports so --help is fast and --no-evolve can dry-run.
    from underworld3.workflows import WorkflowRunner

    runner = WorkflowRunner(cc, config)
    summary = runner.build("run_summary")

    print(f"\n[simulate] Final summary:")
    if summary is None:
        print("  (no summary — run hasn't reached steady state)")
    else:
        for k, v in summary.items():
            print(f"  {k:18s} = {v}")

    if args.movies:
        import convection_visualise as viz

        viz_cfg = viz.VisualiseConfig(run_dir=config.output_dir)
        frames_dir = viz.render_temperature_frames(viz_cfg)
        if frames_dir is not None:
            mp4 = viz.encode_movie(viz_cfg, kind="temperature")
            if mp4 is not None:
                print(f"[simulate] Movie: {mp4}")
            else:
                print(f"[simulate] Frames in {frames_dir} (movie encoding unavailable)")
        else:
            print("[simulate] Visualisation unavailable in this environment")

    return 0


if __name__ == "__main__":
    sys.exit(main())
