"""Unitary in-memory (and, later, on-disk) snapshot toolkit.

The first true unitary checkpoint in Underworld3 — captures enough state
that a Model can be put back exactly as it was, suitable for backtrack on
failure, multi-stage time integration, adaptive-Δt retry, and crash
recovery.

Distinct from the existing per-variable ``write_timestep`` /
``read_timestep`` path, which serves visualisation and partial restart.
That path stays in service of its existing role.

See ``docs/developer/design/in_memory_checkpoint_design.md`` for the
design rationale, scope, and roadmap. In v1 (this code), only an
in-memory backend is implemented and only mesh + mesh-variable state is
captured. Subsequent PRs add swarm coverage, solver-internal Python
state (DDt history, parameter mutation history), an on-disk full-state
backend, and schema versioning across UW3 releases.
"""

from .backend import CheckpointBackend, InMemoryBackend
from .snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    Snapshot,
    SnapshotInvalidatedError,
    snapshot,
    restore,
)
from .state import Snapshottable, SnapshottableState
from .tracker import ModelTracker, TrackerState

__all__ = [
    "CheckpointBackend",
    "InMemoryBackend",
    "SNAPSHOT_SCHEMA_VERSION",
    "Snapshot",
    "SnapshotInvalidatedError",
    "snapshot",
    "restore",
    "Snapshottable",
    "SnapshottableState",
    "ModelTracker",
    "TrackerState",
]
