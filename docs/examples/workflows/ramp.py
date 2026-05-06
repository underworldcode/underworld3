"""Recipe: sequential warm-started convection runs at increasing Ra.

Each leg starts from the previous leg's converged thermal field
(projected via :func:`warm_start.warm_start`) instead of a cold
boundary-layer initial condition.  This skips the early transient at
high Ra — where a cold start can take thousands of steps to spin up
the circulation — by handing each leg a thermally-organised seed
from a lower-Ra solve.

Like :func:`warm_start`, this is a *recipe* — an example script that
composes the public primitives (``Run`` / ``warm_start`` /
``WorkflowRunner``).  It is not part of the ``uw.workflow`` API
surface; if 3+ applications need a similar pattern, we promote the
shape at that point.

Example
-------

>>> from ramp import ramp_rayleigh
>>> summaries = ramp_rayleigh(
...     rayleigh_values=[1e4, 1e5, 1e6],
...     base_dir="output/ramp",
...     cellsize=1.0/24,
...     max_steps=4000,
... )
>>> for ra, summ in zip([1e4, 1e5, 1e6], summaries):
...     print(f"Ra={ra:.0e}  status={summ['status']}  Nu={summ['Nu_mean']:.2f}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import convection_config as cc
import underworld3 as uw
from underworld3.workflows import WorkflowRunner

from warm_start import warm_start


def ramp_rayleigh(
    rayleigh_values,
    *,
    base_dir,
    **shared: Any,
):
    """Run a sequence of convection cells at increasing Ra,
    each warm-starting from the previous.

    Parameters
    ----------
    rayleigh_values : iterable of float
        Ra values for each leg, in order.  Lowest first works best —
        the recipe assumes the next leg's circulation is reached more
        easily from the previous leg's converged state.
    base_dir : str | Path
        Parent directory.  Each leg lives at
        ``<base_dir>/Ra<value>/``.
    **shared
        Shared :class:`ConvectionConfig` fields applied to every leg
        (e.g. ``cellsize``, ``qdegree``, ``max_steps``,
        ``steady_tol_mean``).  Whatever Pydantic accepts.

    Returns
    -------
    summaries : list[dict | None]
        Per-leg run summary dict.  ``None`` for any leg that didn't
        reach steady state — invoke again with a higher ``max_steps``
        to extend.
    """
    base_dir = Path(base_dir)
    summaries: list = []
    prev_dir = None

    for ra in rayleigh_values:
        leg_dir = base_dir / f"Ra{ra:g}"
        leg_kwargs = {**shared, "rayleigh": ra, "output_dir": str(leg_dir)}

        if prev_dir is None or not Path(prev_dir).exists():
            uw.pprint(
                f"\n=== ramp leg Ra={ra:.0e} (cold start) -> {leg_dir} ===",
                flush=True,
            )
            cfg = cc.ConvectionConfig(**leg_kwargs)
        else:
            uw.pprint(
                f"\n=== ramp leg Ra={ra:.0e} (warm-start from {prev_dir}) "
                f"-> {leg_dir} ===",
                flush=True,
            )
            # warm_start projects T from prev_dir's last checkpoint,
            # builds the target ConvectionConfig from prev's identity
            # snapshot with our overrides, and persists step 0.
            cfg = warm_start(prev_dir, leg_dir, **leg_kwargs)

        runner = WorkflowRunner(cc, cfg)
        summary = runner.build("run_summary")
        summaries.append(summary)

        # Only chain forward from steady cells — a stalled leg can't
        # warm-start the next.  Caller can extend its max_steps and
        # re-run; the ramp picks back up.
        if summary is not None and summary.get("status") == "steady":
            prev_dir = leg_dir
        else:
            uw.pprint(
                f"[ramp] Ra={ra:.0e} did not reach steady state; "
                "subsequent legs will not warm-start from it.",
                flush=True,
            )
            prev_dir = None

    return summaries
