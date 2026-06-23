"""CLI driver for the fault-convection workflow (uniform-base, option 2).

Auto-derives its parser from ``FaultConvectionConfig`` and runs the DAG.

Example:
    python fault_simulate.py \
        --output-dir ~/+Simulations/FaultConvection/wf_fault_uniform \
        --rayleigh 1e6 --delta-eta 1e3 --cellsize 0.0417 \
        --resolution-ratio 5 --fault-anisotropy 8 --adapt-every 1 \
        --dt-mult 4 --max-steps 60 --max-t 0.06
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fault_config as fc  # noqa: E402

from underworld3.workflows import (  # noqa: E402
    WorkflowRunner, cli_from_config, config_from_args,
)


def main(argv=None):
    parser = cli_from_config(fc.FaultConvectionConfig)
    parser.add_argument("--target", default="run_summary",
                        help="Workflow product to build (default: run_summary).")
    parser.add_argument("--dag", action="store_true")
    args = parser.parse_args(argv)

    config = config_from_args(fc.FaultConvectionConfig, args)
    runner = WorkflowRunner(fc, config)
    if args.dag:
        runner.dag()
        return
    result = runner.build(args.target)
    if args.target == "run_summary":
        import underworld3 as uw
        uw.pprint(f"[fault_simulate] summary: {result}")


if __name__ == "__main__":
    main()
