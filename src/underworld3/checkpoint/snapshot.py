"""Unitary state capture and restore.

A :class:`Snapshot` is a plain-Python token holding numpy data and small
metadata. It contains no PETSc Vec or DM handles, so it survives object
lifecycle changes within a process. Within a process, a snapshot can be
restored back onto the same :class:`underworld3.Model` instance; across
processes (v1.1, on-disk backend) the model is initialised from the
snapshot rather than restored to a previous state.

Forward-compatibility for v1.2 (mesh-adapt rebuild on restore)
---------------------------------------------------------------
The snapshot module is structured so that v1.2 can replace the
``_mesh_version`` refusal with a true mesh-DM rebuild without touching
this module. Two principles support this:

- **Capture by stable name, not by Python id.** Meshes are keyed by
  ``mesh.name``, swarms by ``f"swarm_{instance_number}"``. Within a
  single process this is overkill (object id would work); but it
  trivialises v1.1 cross-process restore and v1.2 mesh-rebuild (where
  the wrapper object survives but its DM is destroyed and recreated).

- **Wrappers, not the snapshot module, decide how to apply a payload.**
  ``Mesh.apply_snapshot_payload()`` and ``Swarm.apply_snapshot_payload()``
  receive a self-contained dict and decide what to do with it. v1
  implementations are in-place writes; v1.2's mesh implementation can
  inspect the topology slot of the payload (left ``None`` by v1
  capture) and rebuild the DM if needed, without any change to the
  capture / orchestration here.

This module implements the v1 scope: mesh coordinates and mesh-variable
DOFs, plus swarm positions and user swarm-variable data with
rebuild-on-restore semantics. Solver-internal Python state, on-disk
backend, schema versioning, mesh-DM rebuild, and cross-process restore
are scheduled for follow-up PRs per the design note.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .backend import CheckpointBackend, InMemoryBackend
from .state import SnapshottableState


SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotInvalidatedError(RuntimeError):
    """Raised when a snapshot can no longer be restored faithfully.

    Triggers in v1:

    - A captured mesh / swarm / variable name is no longer present on
      the target :class:`underworld3.Model`.
    - Mesh ``_mesh_version`` differs from the snapshot's captured
      value. v1 treats this as fatal because the captured DOF arrays
      are sized for the pre-adapt section. **v1.2 will replace this
      refusal with a mesh-rebuild path** on the same principle as the
      swarm rebuild — see the design note's mesh-adapt scope section.

    Notably **not** a trigger: swarm population mutation
    (populate / migrate / add_particles / remesh) between capture and
    restore. The swarm restore path *rebuilds* the local particle
    population from the snapshot, so intervening mutations are exactly
    what restore is for. The ``_population_generation`` counter on the
    swarm is informational, not a restore gate.
    """


def _swarm_stable_name(swarm) -> str:
    """Per-process stable name for a swarm. Uses uw_object instance number."""
    return f"swarm_{swarm.instance_number}"


@dataclass
class Snapshot:
    """Unitary state token.

    Produced by :func:`snapshot`; consumed by :func:`restore`. Holds a
    backend (where the bulk arrays live) plus per-model bookkeeping —
    which meshes and swarms were captured, in what order, with what
    variable sets.

    Attributes
    ----------
    backend
        Where the captured arrays and small metadata live.
    schema_version
        Snapshot file-format version. Restore refuses on mismatch in
        v1; v1.1's migration registry will lift older versions to the
        current schema for on-disk restore only.
    mesh_names
        Capture order of mesh names. ``mesh.name`` is the stable key.
    mesh_versions
        Per-mesh ``_mesh_version`` at the moment of capture. v1
        compares strictly; v1.2 will rebuild on mismatch.
    meshvar_names
        Mapping ``mesh_name → [var clean_name, ...]``.
    swarm_names
        Capture order of swarm stable names
        (``f"swarm_{instance_number}"``).
    swarm_mesh_names
        Mapping ``swarm_name → mesh_name`` so restore can verify the
        swarm's parent mesh is still the captured one.
    swarm_generations
        Captured ``_population_generation`` per swarm — informational
        metadata; *not* a restore gate. Useful for logs and debugging
        ("this snapshot was taken at generation 7; the current swarm
        is at 12").
    swarmvar_names
        Mapping ``swarm_name → [user-var clean_name, ...]``. Internal
        DMSwarm-prefixed variables are filtered out.
    metadata
        Free-form user/system metadata (simulation time, step counter,
        ...). Not load-bearing for restore correctness.
    """

    backend: CheckpointBackend
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    mesh_names: list[str] = field(default_factory=list)
    mesh_versions: dict[str, int] = field(default_factory=dict)
    meshvar_names: dict[str, list[str]] = field(default_factory=dict)
    swarm_names: list[str] = field(default_factory=list)
    swarm_mesh_names: dict[str, str] = field(default_factory=dict)
    swarm_generations: dict[str, int] = field(default_factory=dict)
    swarmvar_names: dict[str, list[str]] = field(default_factory=dict)
    # State-bearer captures: list of (stable_key, state_dataclass).
    # stable_key is f"{type(obj).__name__}_{obj.instance_number}", matched
    # at restore against the same key derived from currently-registered
    # state-bearers. List preserves capture order — informational only,
    # since lookup is by key.
    state_bearers: list = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ----- Backend key conventions -----

def _mesh_coords_key(mesh_name: str) -> str:
    return f"mesh:{mesh_name}:coords"


def _meshvar_key(mesh_name: str, var_clean_name: str) -> str:
    return f"mesh:{mesh_name}:var:{var_clean_name}:gvec"


def _swarm_coords_key(swarm_name: str) -> str:
    return f"swarm:{swarm_name}:coords"


def _swarmvar_key(swarm_name: str, var_clean_name: str) -> str:
    return f"swarm:{swarm_name}:var:{var_clean_name}:data"


def _is_internal_swarmvar(var_name: str) -> bool:
    """Filter PETSc-managed internal swarm variables from user capture.

    ``DMSwarmPIC_coor`` is captured separately via the particle-coords
    path. ``DMSwarm_X0`` and ``DMSwarm_remeshed`` carry recycle-related
    bookkeeping that is regenerated on next solve and is out of scope
    for v1 capture.
    """
    return var_name.startswith("DMSwarm")


# ----- Capture (orchestration) -----

def snapshot(model, *, path: Optional[str] = None) -> Snapshot:
    """Capture a unitary snapshot of the model's current state.

    Captures, in v1: each registered mesh's deformed coordinates and
    every mesh-variable's global-vector DOFs; each registered swarm's
    per-rank particle coordinates and user swarm-variable arrays.

    Pass ``path=...`` once the v1.1 on-disk backend lands. v1 raises
    ``NotImplementedError``.

    See ``docs/developer/design/in_memory_checkpoint_design.md`` for
    the design rationale and scope boundaries.
    """
    if path is not None:
        raise NotImplementedError(
            "on-disk full-state snapshot is scheduled for v1.1; "
            "v1 supports the in-memory backend only"
        )

    snap = Snapshot(backend=InMemoryBackend())
    for mesh in list(model._meshes.values()):
        _capture_mesh(snap, mesh)
    for swarm in list(model._swarms.values()):
        _capture_swarm(snap, swarm)
    for obj in list(model._state_bearers):
        _capture_state_bearer(snap, obj)
    return snap


def _capture_mesh(snap: Snapshot, mesh) -> None:
    payload = mesh.snapshot_payload()
    name = payload["name"]
    if name in snap.mesh_names:
        raise RuntimeError(
            f"duplicate mesh name {name!r} in snapshot capture; mesh names "
            f"must be unique within a Model"
        )
    snap.mesh_names.append(name)
    snap.mesh_versions[name] = payload["mesh_version"]

    snap.backend.save_vector(_mesh_coords_key(name), payload["coords"])

    var_names: list[str] = []
    for var_clean_name, gvec_array in payload["vars"].items():
        snap.backend.save_vector(_meshvar_key(name, var_clean_name), gvec_array)
        var_names.append(var_clean_name)
    snap.meshvar_names[name] = var_names


def _state_bearer_key(obj) -> str:
    """Stable per-process key for a Snapshottable. ``instance_number``
    comes from ``uw_object`` and is unique across the run."""
    return f"{type(obj).__name__}_{obj.instance_number}"


def _capture_state_bearer(snap: Snapshot, obj) -> None:
    """Pull ``obj.state`` and store a deep copy.

    Deep copy ensures later mutations on the live state-bearer don't
    leak into the captured token. The dataclass itself is the storage
    here (no separate backend.save_vector call) because state
    dataclasses are small Python objects, not bulk numerical arrays.
    v1.1's on-disk backend will route the dataclass through the
    backend (HDF5 attrs/groups); v1 holds them in the in-memory Snapshot
    directly.
    """
    state = obj.state
    if not isinstance(state, SnapshottableState):
        raise TypeError(
            f"{type(obj).__name__}.state must be a SnapshottableState, "
            f"got {type(state).__name__}"
        )
    snap.state_bearers.append((_state_bearer_key(obj), copy.deepcopy(state)))


def _capture_swarm(snap: Snapshot, swarm) -> None:
    payload = swarm.snapshot_payload()
    name = payload["name"]
    if name in snap.swarm_names:
        raise RuntimeError(
            f"duplicate swarm name {name!r} in snapshot capture"
        )
    snap.swarm_names.append(name)
    snap.swarm_mesh_names[name] = payload["mesh_name"]
    snap.swarm_generations[name] = payload["population_generation"]

    snap.backend.save_vector(_swarm_coords_key(name), payload["coords"])

    var_names: list[str] = []
    for var_clean_name, data in payload["vars"].items():
        snap.backend.save_vector(_swarmvar_key(name, var_clean_name), data)
        var_names.append(var_clean_name)
    snap.swarmvar_names[name] = var_names


# ----- Restore (orchestration) -----

def restore(model, snap: Snapshot) -> None:
    """Restore the model from a snapshot.

    Mesh restore in v1 writes captured coords + DOFs back in place. If
    the mesh's ``_mesh_version`` has moved since capture, restore
    raises :class:`SnapshotInvalidatedError` — this becomes a rebuild
    path in v1.2.

    Swarm restore *rebuilds* the local particle population: clears
    current particles, re-adds at captured coords, writes captured
    per-variable data back in order. This is the rebuild-on-restore
    semantics described in the design note's "Restore semantics for
    swarms" section — restore is precisely *for* the case where
    particles have moved / been added / been removed since capture.

    Parameters
    ----------
    model
        The :class:`underworld3.Model` to restore. Must be the same
        instance the snapshot came from (within-process restore).
    snap
        Token returned by :func:`snapshot`.

    Raises
    ------
    SnapshotInvalidatedError
        Captured mesh / swarm / variable is no longer registered on
        the model, or mesh ``_mesh_version`` has moved since capture
        (mesh-adapt is v1.2 scope).
    TypeError
        ``snap`` is not a :class:`Snapshot`.
    """
    if not isinstance(snap, Snapshot):
        raise TypeError(
            f"expected underworld3.checkpoint.Snapshot, got {type(snap).__name__}"
        )
    if snap.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotInvalidatedError(
            f"snapshot schema version {snap.schema_version} does not match "
            f"current {SNAPSHOT_SCHEMA_VERSION}; on-disk migration is v1.1"
        )

    meshes_by_name = {m.name: m for m in model._meshes.values()}
    swarms_by_name = {_swarm_stable_name(s): s for s in model._swarms.values()}

    for mesh_name in snap.mesh_names:
        mesh = meshes_by_name.get(mesh_name)
        if mesh is None:
            raise SnapshotInvalidatedError(
                f"mesh {mesh_name!r} from snapshot is not registered on "
                f"this Model; within-process restore requires the originating "
                f"Model"
            )
        payload = _build_mesh_payload(snap, mesh_name)
        mesh.apply_snapshot_payload(payload)

    for swarm_name in snap.swarm_names:
        swarm = swarms_by_name.get(swarm_name)
        if swarm is None:
            raise SnapshotInvalidatedError(
                f"swarm {swarm_name!r} from snapshot is not registered on "
                f"this Model"
            )
        expected_mesh_name = snap.swarm_mesh_names[swarm_name]
        if swarm.mesh.name != expected_mesh_name:
            raise SnapshotInvalidatedError(
                f"swarm {swarm_name!r} parent mesh changed from "
                f"{expected_mesh_name!r} to {swarm.mesh.name!r} since "
                f"snapshot"
            )
        payload = _build_swarm_payload(snap, swarm_name)
        swarm.apply_snapshot_payload(payload)

    if snap.state_bearers:
        bearers_by_key = {
            _state_bearer_key(o): o for o in list(model._state_bearers)
        }
        for key, captured_state in snap.state_bearers:
            obj = bearers_by_key.get(key)
            if obj is None:
                raise SnapshotInvalidatedError(
                    f"state-bearer {key!r} from snapshot is not registered "
                    f"on this Model; restore requires the originating "
                    f"Model"
                )
            obj.state = copy.deepcopy(captured_state)


def _build_mesh_payload(snap: Snapshot, mesh_name: str) -> dict:
    return {
        "name": mesh_name,
        "captured_mesh_version": snap.mesh_versions[mesh_name],
        "coords": snap.backend.load_vector(_mesh_coords_key(mesh_name)),
        # Topology is None in v1; v1.2 mesh-rebuild path will populate
        # this slot (e.g., section view data) without bumping the
        # schema version, because v1 reads ignore the key.
        "topology": None,
        "vars": {
            var_clean_name: snap.backend.load_vector(
                _meshvar_key(mesh_name, var_clean_name)
            )
            for var_clean_name in snap.meshvar_names[mesh_name]
        },
    }


def _build_swarm_payload(snap: Snapshot, swarm_name: str) -> dict:
    return {
        "name": swarm_name,
        "mesh_name": snap.swarm_mesh_names[swarm_name],
        "captured_population_generation": snap.swarm_generations[swarm_name],
        "coords": snap.backend.load_vector(_swarm_coords_key(swarm_name)),
        "vars": {
            var_clean_name: snap.backend.load_vector(
                _swarmvar_key(swarm_name, var_clean_name)
            )
            for var_clean_name in snap.swarmvar_names[swarm_name]
        },
    }
