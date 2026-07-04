"""CLI driver for a single Rayleigh-Bénard convection run.

Replaces the old ``run_ra_variant.py``-style monkey-patching with
straight config overrides.  The argparse parser is auto-derived from
:class:`ConvectionConfig`'s Pydantic fields by
:func:`underworld3.workflows.cli_from_config` — every config knob
becomes a CLI flag automatically, and adding a new config field
extends the CLI for free.

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

import sys
from pathlib import Path

import convection_config as cc
from underworld3.workflows import cli_from_config, config_from_args


def build_parser():
    """Auto-derived parser plus this CLI's action flags.

    Suppresses fields that are pinned by the non-dimensional
    Boussinesq formulation (viscosity, diffusivity, T_top, T_bottom)
    and the workflow-itself metadata (workflow_name, description).
    Power users can still set those programmatically via Python.
    """
    hidden = {
        "workflow_name", "description",          # workflow self-metadata
        "viscosity", "diffusivity",              # absorbed into Rayleigh
        "T_top", "T_bottom",                     # set by the temperature scale
        "qdegree",                               # auto-derived from T_degree
    }
    parser = cli_from_config(
        cc.ConvectionConfig,
        description="Run a single convection workflow to steady state.",
        hidden_fields=hidden,
    )
    parser.add_argument(
        "--movies", action="store_true",
        help="Render temperature frames + mp4 after the run finishes",
    )
    parser.add_argument(
        "--no-evolve", action="store_true",
        help="Build the config and print it but skip the actual run",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(cc.ConvectionConfig, args)

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
