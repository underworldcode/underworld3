"""Unitary state capture and restore.

A :class:`Snapshot` is a plain-Python token holding numpy data and small
metadata. It contains no PETSc Vec or DM handles, so it survives object
lifecycle changes within a process. Within a process, a snapshot can be
restored back onto the same :class:`underworld3.Model` instance; across
processes (v1.1, on-disk backend) the model is initialised from the
snapshot rather than restored to a previous state.

This module implements the v1 scope: mesh coordinates and mesh-variable
DOFs. Swarm coverage, solver-internal Python state, on-disk backend,
schema versioning, and cross-process restore are scheduled for follow-up
PRs per the design note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .backend import CheckpointBackend, InMemoryBackend


SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotInvalidatedError(RuntimeError):
    """Raised when a snapshot can no longer be restored faithfully.

    Triggers in v1: mesh ``_mesh_version`` differs from the snapshot
    (mesh has been adapted; DM identity has changed), or a registered
    mesh / mesh-variable named in the snapshot is no longer present on
    the target :class:`underworld3.Model`.

    Future triggers (subsequent PRs): swarm population-generation
    counter mismatch, on-disk schema version that has no migration
    path.
    """


@dataclass
class Snapshot:
    """Unitary state token.

    Produced by :func:`snapshot`; consumed by :func:`restore`. Holds a
    backend (where the bulk arrays live) plus per-Model bookkeeping —
    which meshes were captured, which mesh variables were captured
    under each mesh, and the mesh-version counters that gate
    within-process restore.

    Attributes
    ----------
    backend
        Where the captured arrays live. v1 always uses
        :class:`InMemoryBackend`; v1.1 will add on-disk backends.
    schema_version
        Snapshot file-format version. Restore refuses on mismatch in
        v1; v1.1's migration registry will lift older versions to the
        current schema for on-disk restore only.
    mesh_keys
        Stable ordering of captured mesh identifiers (``id(mesh)``);
        determines restore order.
    mesh_versions
        Per-mesh ``_mesh_version`` at the moment of capture. Restore
        compares against the current value; mismatch ⇒
        :class:`SnapshotInvalidatedError`.
    meshvar_names
        Mapping ``mesh_id → [var.clean_name, ...]`` — the mesh
        variables captured for that mesh, in capture order.
    metadata
        User-visible bookkeeping (simulation time, step counter, free
        text). Not load-bearing for restore correctness.
    """

    backend: CheckpointBackend
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    mesh_keys: list[int] = field(default_factory=list)
    mesh_versions: dict[int, int] = field(default_factory=dict)
    meshvar_names: dict[int, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _mesh_coords_key(mesh_id: int) -> str:
    return f"mesh:{mesh_id}:coords"


def _meshvar_key(mesh_id: int, var_clean_name: str) -> str:
    return f"mesh:{mesh_id}:var:{var_clean_name}:gvec"


def snapshot(model, *, path: Optional[str] = None) -> Snapshot:
    """Capture a unitary snapshot of the model's current state.

    Parameters
    ----------
    model
        The :class:`underworld3.Model` whose registered meshes and
        mesh variables should be captured.
    path
        Reserved for the v1.1 on-disk backend. Passing a non-``None``
        value raises :class:`NotImplementedError` in v1.

    Returns
    -------
    Snapshot
        Token suitable for passing to :func:`restore` on the same
        ``model`` instance within the same process. v1 captures mesh
        coordinates and mesh-variable global-vector DOF values.
    """
    if path is not None:
        raise NotImplementedError(
            "on-disk full-state snapshot is scheduled for v1.1; "
            "v1 supports the in-memory backend only"
        )

    snap = Snapshot(backend=InMemoryBackend())
    for mesh_id, mesh in list(model._meshes.items()):
        _capture_mesh(snap, mesh_id, mesh)
    return snap


def _capture_mesh(snap: Snapshot, mesh_id: int, mesh) -> None:
    if mesh_id in snap.mesh_keys:
        return
    snap.mesh_keys.append(mesh_id)
    snap.mesh_versions[mesh_id] = int(getattr(mesh, "_mesh_version", 0))

    coords = np.asarray(mesh.X.coords)
    snap.backend.save_vector(_mesh_coords_key(mesh_id), coords)

    var_names: list[str] = []
    for var in mesh.vars.values():
        var._sync_lvec_to_gvec()
        gvec_array = np.asarray(var._gvec.array)
        snap.backend.save_vector(_meshvar_key(mesh_id, var.clean_name), gvec_array)
        var_names.append(var.clean_name)
    snap.meshvar_names[mesh_id] = var_names


def restore(model, snap: Snapshot) -> None:
    """Restore the model from a snapshot.

    Restore order (within-process; cross-process is v1.1):

    1. Mesh coordinates (via :meth:`Mesh._deform_mesh`, which rebuilds
       coordinate caches and notifies registered callbacks).
    2. Mesh-variable DOFs (global vector written, then synced to local
       vector via ``subdm.globalToLocal``).
    3. ``_mesh_version`` is verified equal to the capture value before
       any write; mismatch raises :class:`SnapshotInvalidatedError`.

    Future PRs extend the order to: swarm positions + migrate → swarm
    variable values → solver-internal Python state (DDt history,
    parameter mutation history) → generation-counter validation last.

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
        Mesh ``_mesh_version`` has changed since capture, or a
        captured mesh / variable is no longer registered on the model.
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

    for mesh_id in snap.mesh_keys:
        mesh = model._meshes.get(mesh_id)
        if mesh is None:
            raise SnapshotInvalidatedError(
                f"mesh id {mesh_id} from snapshot is not registered on this "
                f"Model; within-process restore requires the originating Model"
            )
        current_version = int(getattr(mesh, "_mesh_version", 0))
        captured_version = snap.mesh_versions[mesh_id]
        if current_version != captured_version:
            raise SnapshotInvalidatedError(
                f"mesh._mesh_version moved from {captured_version} to "
                f"{current_version} since snapshot — likely mesh.adapt() or "
                f"deform_mesh() invalidated the DM identity"
            )
        _restore_mesh(snap, mesh_id, mesh)


def _restore_mesh(snap: Snapshot, mesh_id: int, mesh) -> None:
    coords = snap.backend.load_vector(_mesh_coords_key(mesh_id))
    expected_shape = np.asarray(mesh.X.coords).shape
    if coords.shape != expected_shape:
        raise SnapshotInvalidatedError(
            f"mesh coordinate shape changed: snapshot {coords.shape} vs "
            f"current {expected_shape}"
        )
    mesh._deform_mesh(coords)

    current_vars = {var.clean_name: var for var in mesh.vars.values()}
    for var_clean_name in snap.meshvar_names[mesh_id]:
        var = current_vars.get(var_clean_name)
        if var is None:
            raise SnapshotInvalidatedError(
                f"mesh variable {var_clean_name!r} from snapshot is not "
                f"present on mesh; restore requires the same variable set"
            )
        var._sync_lvec_to_gvec()  # ensures _gvec exists with a current size
        saved = snap.backend.load_vector(_meshvar_key(mesh_id, var_clean_name))
        current_shape = np.asarray(var._gvec.array).shape
        if saved.shape != current_shape:
            raise SnapshotInvalidatedError(
                f"variable {var_clean_name!r} gvec shape changed: snapshot "
                f"{saved.shape} vs current {current_shape}"
            )
        var._gvec.array[...] = saved
        iset, subdm = mesh.dm.createSubDM(var.field_id)
        subdm.globalToLocal(var._gvec, var._lvec, addv=False)
        iset.destroy()
        subdm.destroy()
        mesh._stale_lvec = True
