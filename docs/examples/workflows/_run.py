"""Thin wrappers for the on-disk run-directory format.

Future home: ``uw.workflow``.  Lives here while the convection example
still doubles as the workshop where the API is being shaped.

This module owns the bytes-on-disk for:

- ``manifest.yaml`` — run identity (workflow name, config hash, etc.).
- ``timeseries.csv`` — append-only per-step diagnostics.
- ``run_summary.yaml`` — steady-state "done" marker.
- discovery of the ``run.mesh.NNNNN.{h5,xdmf}`` checkpoint chain
  (the actual h5 read/write still goes through ``mesh.write_timestep``
  / ``var.read_timestep`` at the call sites — wrapping those is for
  ``Run.append_step`` / ``Run.load_field``, deferred to a later step).

The ``Run`` and ``Manifest`` classes here are pure adapters around what
``convection_config`` previously wrote with private helpers — same
``yaml.dump`` options, same CSV conventions ('---' for NaN), same
archive-timestamp format.  No new validation or schema enforcement;
that lands when this module is lifted into ``uw.workflow``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


# Filename stem for the h5 / xdmf checkpoint chain.  The convection
# workflow has always used "run"; defined here so visualisers and other
# consumers don't import it from ``convection_config``.
RUN_NAME = "run"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    """Wrapper around ``manifest.yaml`` contents.

    The wrapped ``data`` dict is whatever the workflow chose to write —
    no schema is enforced at this level.  Validation (``workflow_api``,
    ``config_hash`` cross-checking) lands when the type moves into
    ``uw.workflow``.  Convenience properties expose the keys the
    convection workflow uses today.
    """

    data: dict = field(default_factory=dict)

    @classmethod
    def read(cls, run_dir) -> Optional["Manifest"]:
        """Load ``manifest.yaml`` from *run_dir* or return ``None``."""
        p = Path(run_dir) / "manifest.yaml"
        if not p.exists():
            return None
        with open(p) as f:
            return cls(yaml.safe_load(f) or {})

    def write(self, run_dir) -> None:
        """Write ``manifest.yaml`` into *run_dir*."""
        p = Path(run_dir) / "manifest.yaml"
        with open(p, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

    # Dict-like access for code that just wants the raw fields.
    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    # Convenience properties for the keys the convection workflow uses.
    @property
    def workflow(self):
        return self.data.get("workflow")

    @property
    def config_hash(self):
        return self.data.get("config_hash")

    @property
    def config_snapshot(self) -> dict:
        return self.data.get("config_snapshot", {})

    @property
    def started_at(self):
        return self.data.get("started_at")


# ---------------------------------------------------------------------------
# CSV cell helpers (workflow-agnostic)
# ---------------------------------------------------------------------------


def _csv_to_float(s) -> float:
    """Parse a CSV cell as float; treat ``None`` / ``''`` / ``'---'`` as NaN."""
    if s is None:
        return float("nan")
    s = s.strip()
    if s in ("", "---"):
        return float("nan")
    return float(s)


def _format_for_csv(val):
    """Serialise a value for CSV: ``None`` and non-finite floats → ``'---'``."""
    if val is None:
        return "---"
    if isinstance(val, float) and not np.isfinite(val):
        return "---"
    return val


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class Run:
    """A run-output directory — thin wrapper, no IO on construction.

    Construction does not touch disk; ``Run.create`` is the helper that
    makes the directory.  Reads (``manifest``, ``steps``, ``timeseries``,
    ``summary``) all hit disk fresh on each access — no caching.
    """

    def __init__(self, path):
        self.path = Path(path)

    # --- construction -----------------------------------------------------

    @classmethod
    def open(cls, path) -> "Run":
        """Open an existing run directory.

        Currently a thin alias for ``Run(path)`` — no validation that
        the path exists or contains a manifest.  ``workflow_api`` /
        ``config_hash`` validation will be added when the type lands in
        ``uw.workflow``.
        """
        return cls(path)

    @classmethod
    def create(cls, path) -> "Run":
        """Create the directory (parents + exist_ok) and return a Run."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return cls(p)

    # --- manifest --------------------------------------------------------

    @property
    def manifest(self) -> Optional[Manifest]:
        """The manifest, or ``None`` if ``manifest.yaml`` doesn't exist."""
        return Manifest.read(self.path)

    def write_manifest(self, data: dict) -> None:
        """Write ``manifest.yaml`` with the given dict."""
        Manifest(dict(data)).write(self.path)

    # --- checkpoint discovery -------------------------------------------

    @property
    def steps(self) -> list[int]:
        """Indices of saved timesteps, derived from the xdmf files."""
        if not self.path.exists():
            return []
        steps = set()
        for p in self.path.glob(f"{RUN_NAME}.mesh.[0-9]*.xdmf"):
            try:
                steps.add(int(p.stem.split(".")[-1]))
            except ValueError:
                continue
        return sorted(steps)

    # --- timeseries ------------------------------------------------------

    @property
    def timeseries_path(self) -> Path:
        return self.path / "timeseries.csv"

    @property
    def timeseries(self) -> list[dict]:
        """Read ``timeseries.csv`` as a list of typed-row dicts.

        ``step`` is parsed as int; ``t`` and ``dt`` as float; every
        other column is parsed via :func:`_csv_to_float` (so '---' /
        empty cells become NaN).  Missing file → ``[]``.

        Columns absent from the file are absent from the row dicts —
        callers that need legacy compatibility should fill in defaults
        themselves.
        """
        path = self.timeseries_path
        if not path.exists():
            return []
        out = []
        with open(path) as f:
            for row in csv.DictReader(f):
                parsed = {
                    "step": int(row["step"]),
                    "t": float(row["t"]),
                    "dt": float(row["dt"]),
                }
                for k, v in row.items():
                    if k in parsed:
                        continue
                    parsed[k] = _csv_to_float(v)
                out.append(parsed)
        return out

    def append_timeseries_row(
        self, row: dict, fields: tuple[str, ...]
    ) -> None:
        """Append one row to ``timeseries.csv``, with schema migration.

        If the file exists with a header that doesn't match *fields*,
        the file is rewritten in place under the new schema (with
        missing columns serialising as ``'---'``).  Otherwise the row
        is appended, with the header auto-added on first write.

        ``None`` and non-finite floats in *row* are written as
        ``'---'`` (visually distinct from numeric zero).
        """
        path = self.timeseries_path

        if path.exists():
            with open(path) as f:
                existing_header = f.readline().rstrip("\n")
            existing_fields = tuple(existing_header.split(","))
            if existing_fields != tuple(fields):
                old_rows = self.timeseries
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=list(fields), extrasaction="ignore",
                    )
                    writer.writeheader()
                    for r in old_rows:
                        writer.writerow(
                            {k: _format_for_csv(r.get(k)) for k in fields}
                        )

        is_new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(fields), extrasaction="ignore",
            )
            if is_new:
                writer.writeheader()
            writer.writerow({k: _format_for_csv(row.get(k)) for k in fields})

    # --- summary --------------------------------------------------------

    @property
    def summary_path(self) -> Path:
        return self.path / "run_summary.yaml"

    @property
    def summary(self) -> Optional[dict]:
        """Contents of ``run_summary.yaml``, or ``None`` if absent."""
        p = self.summary_path
        if not p.exists():
            return None
        with open(p) as f:
            return yaml.safe_load(f)

    def write_summary(self, summary: dict) -> None:
        with open(self.summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    # --- lifecycle ------------------------------------------------------

    def archive(self) -> Optional[Path]:
        """Rename the directory to ``<name>.archive-<UTC stamp>/``.

        Never deletes.  Returns the new archive path, or ``None`` if
        the directory doesn't exist.
        """
        if not self.path.exists():
            return None
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = self.path.parent / f"{self.path.name}.archive-{ts}"
        self.path.rename(archive)
        return archive
