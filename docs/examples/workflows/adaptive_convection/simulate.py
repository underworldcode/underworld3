"""CLI driver for the adaptive-convection workflow.

Auto-derives its argument parser from ``AdaptiveConvectionConfig`` (one
flag per field) via ``underworld3.workflows.cli_from_config``, builds the
config, and runs the workflow DAG to the requested target.

Examples
--------
    # full run to steady state (or max_steps)
    python simulate.py --output-dir ~/+Simulations/AdaptiveConvection/wf_baseline \
        --rayleigh 1e6 --delta-eta 1e3 --cellsize 0.0417 \
        --resolution-ratio 5 --adapt-every 1 --dt-mult 4 \
        --max-steps 80 --max-t 0.06

    # just build/inspect the mesh (no time loop)
    python simulate.py --output-dir /tmp/x --target mesh
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as ac  # noqa: E402

from underworld3.workflows import (  # noqa: E402
    WorkflowRunner, cli_from_config, config_from_args,
)


def main(argv=None):
    parser = cli_from_config(ac.AdaptiveConvectionConfig)
    parser.add_argument("--target", default="run_summary",
                        help="Workflow product to build (default: run_summary). "
                             "Use 'mesh' or 'run_directory' for partial builds.")
    parser.add_argument("--dag", action="store_true",
                        help="Print the workflow DAG status and exit.")
    args = parser.parse_args(argv)

    config = config_from_args(ac.AdaptiveConvectionConfig, args)
    runner = WorkflowRunner(ac, config)

    if args.dag:
        runner.dag()
        return

    result = runner.build(args.target)
    if args.target == "run_summary":
        import underworld3 as uw
        uw.pprint(f"[simulate] summary: {result}")


if __name__ == "__main__":
    main()
