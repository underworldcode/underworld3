"""On-disk snapshot format (v1.1) — metadata wrapper around PETSc bulk.

Design (see ``memory/project_snapshot_v1_1_disk_format.md`` and
``docs/developer/design/in_memory_checkpoint_design.md``):

  - Single self-contained HDF5 file (with the freedom to externalise
    bulky swarm data into companion files later — phase 3).
  - UW3-controlled rich metadata wrapper around PETSc-format bulk
    data, so ``h5ls`` / ``h5dump`` show useful information about a
    snapshot file without UW3 needing to be in the loop.
  - Bulk data layers (PETSc DMPlex topology, sections, vectors) are
    delegated to the primitives that landed in #146; this module owns
    the *layout* and the *metadata*, not the binary serialisation of
    fields.

File structure (target — phases 2+ fill in the bulk under these
groups; phase 1 writes the metadata and empty stub groups):

    my_run.snap.h5/
    ├── /metadata          (attrs: uw3_version, schema_version,
    │                       created_at, step, sim_time, dt, dim,
    │                       mesh_type, coordinate_system,
    │                       mpi_ranks_at_write, variables_summary, ...)
    ├── /mesh              (phase 2 — DMPlex topology + coords + labels)
    ├── /variables         (phase 2 — one subgroup per mesh-variable)
    ├── /swarms            (phase 3 — possibly @external_file refs)
    └── /python_state      (phase 3 — Snapshottable dataclasses as attrs)

Phase 1 (this commit): the metadata layer and the skeleton group
structure, with an inspectability acceptance test that asserts an
external reader (h5py here) sees meaningful information without any
UW3 imports.
"""

from __future__ import annotations

import datetime
import json
from typing import Any, Optional

import numpy as np

import underworld3 as uw


DISK_SNAPSHOT_SCHEMA_VERSION = 1

# Top-level group names — fixed; renaming would be a schema-version bump.
_GROUP_METADATA = "metadata"
_GROUP_MESH = "mesh"
_GROUP_VARIABLES = "variables"
_GROUP_SWARMS = "swarms"
_GROUP_PYTHON_STATE = "python_state"

_TOP_LEVEL_GROUPS = (
    _GROUP_METADATA,
    _GROUP_MESH,
    _GROUP_VARIABLES,
    _GROUP_SWARMS,
    _GROUP_PYTHON_STATE,
)


def _collect_metadata(model) -> dict:
    """Build the metadata dict that gets written into ``/metadata`` attrs.

    Stable, h5-friendly types only: strings, ints, floats, lists of
    strings (stored as JSON for compactness and to keep h5 attrs
    scalar-typed where possible). No pickling, no repr of UW3 objects.
    """
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )

    # Mesh-derived info (gracefully absent if no mesh registered).
    meshes = list(model._meshes.values())
    first_mesh = meshes[0] if meshes else None
    mesh_names = [m.name for m in meshes]
    if first_mesh is not None:
        dim = int(first_mesh.dim)
        mesh_type = type(first_mesh).__name__
        coord_system = (
            first_mesh.CoordinateSystem.type
            if hasattr(first_mesh, "CoordinateSystem")
            else "unknown"
        )
    else:
        dim = -1
        mesh_type = ""
        coord_system = ""

    # Swarm names (WeakValueDictionary, snapshot it).
    swarm_names = []
    for s in list(model._swarms.values()):
        name = getattr(s, "name", None) or f"swarm_{s.instance_number}"
        swarm_names.append(name)

    # State-bearer class summary (just the class names — useful in
    # h5ls without needing UW3 to interpret).
    state_bearer_classes = sorted(
        {type(o).__name__ for o in list(model._state_bearers)}
    )

    # Tracker conventions (the model-dwelling record).
    tracker = getattr(model, "tracker", None)
    if tracker is not None:
        sim_time = float(tracker.time) if tracker.time is not None else 0.0
        step = int(tracker.step) if tracker.step is not None else 0
        dt_val = tracker.dt
        dt = float(dt_val) if dt_val is not None else float("nan")
    else:
        sim_time, step, dt = 0.0, 0, float("nan")

    # Per-variable summary across all registered meshes.
    var_entries = []
    for m in meshes:
        for var in m.vars.values():
            kind = "vector" if var.num_components > 1 else "scalar"
            var_entries.append(
                f"{m.name}.{var.clean_name} ({kind}, "
                f"components={var.num_components}, degree={var.degree})"
            )
    variables_summary = "; ".join(var_entries) if var_entries else ""

    md = {
        # Versioning
        "uw3_version": str(getattr(uw, "__version__", "0.0.0")),
        "schema_version": int(DISK_SNAPSHOT_SCHEMA_VERSION),
        "created_at": now_iso,
        # Identity
        "run_name": str(getattr(model, "name", "default")),
        # Time / step (from the tracker — pre-seeded conventions)
        "step": step,
        "sim_time": sim_time,
        "dt": dt,
        # Geometry / topology
        "dim": dim,
        "mesh_type": mesh_type,
        "coordinate_system": str(coord_system),
        # MPI
        "mpi_ranks_at_write": int(uw.mpi.size),
        # Inventories — JSON for list-typed values so h5 attrs stay scalar.
        "mesh_names_json": json.dumps(mesh_names),
        "swarm_names_json": json.dumps(swarm_names),
        "state_bearer_classes_json": json.dumps(state_bearer_classes),
        "variables_summary": variables_summary,
    }
    return md


def _write_metadata_attrs(h5group, metadata: dict) -> None:
    """Write a metadata dict as HDF5 attrs on a group. Plain types only."""
    for k, v in metadata.items():
        h5group.attrs[k] = v


def write_snapshot_skeleton(model, path: str) -> str:
    """Phase 1: write the metadata + empty skeleton group structure.

    Returns the path written. Subsequent phases (2: mesh + meshvar
    bulk; 3: swarms + python_state) populate the empty top-level
    groups using PETSc primitives and dataclass serialisation
    respectively. Writing is rank-0-only at this phase since no
    collective PETSc operations are involved yet.
    """
    import h5py

    with uw.selective_ranks(0) as should_execute:
        if not should_execute:
            uw.mpi.barrier()
            return path

        metadata = _collect_metadata(model)

        with h5py.File(path, "w") as f:
            md_group = f.create_group(_GROUP_METADATA)
            _write_metadata_attrs(md_group, metadata)

            # Stub the other top-level groups so external readers can
            # see the file's intended shape from day one — phases 2/3
            # populate them.
            for name in (
                _GROUP_MESH,
                _GROUP_VARIABLES,
                _GROUP_SWARMS,
                _GROUP_PYTHON_STATE,
            ):
                grp = f.create_group(name)
                grp.attrs["filled_by"] = ""  # set to "phase2" / "phase3" later

    uw.mpi.barrier()
    return path


def read_snapshot_metadata(path: str) -> dict:
    """Read the ``/metadata`` group's attrs back as a plain dict.

    Validates the schema version. Lists stored as ``*_json`` are
    decoded back into Python lists for caller convenience but the
    on-disk form stays JSON for h5-tool friendliness.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if _GROUP_METADATA not in f:
            raise ValueError(
                f"{path}: not a UW3 snapshot file (no /{_GROUP_METADATA} group)"
            )
        md_group = f[_GROUP_METADATA]
        md = {}
        for k in md_group.attrs.keys():
            v = md_group.attrs[k]
            # h5py returns bytes for some string attrs; normalise to str.
            if isinstance(v, bytes):
                v = v.decode()
            elif isinstance(v, np.ndarray) and v.dtype.kind in ("S", "U"):
                v = [x.decode() if isinstance(x, bytes) else str(x) for x in v]
            md[k] = v

    schema = int(md.get("schema_version", -1))
    if schema != DISK_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: snapshot schema version {schema} does not match "
            f"current {DISK_SNAPSHOT_SCHEMA_VERSION}; on-disk schema "
            f"migration will land with phase 6 (not yet implemented)"
        )

    # Decode JSON-encoded list fields for caller convenience.
    for key in list(md.keys()):
        if key.endswith("_json"):
            try:
                decoded = json.loads(md[key])
                md[key[:-5]] = decoded   # e.g. "mesh_names" alongside "mesh_names_json"
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

    return md


def inspect_snapshot(path: str) -> str:
    """Human-readable one-shot summary of a snapshot file's metadata.

    Useful as a Python-side equivalent to running ``h5ls`` on the
    file; intended for ``print(uw.checkpoint.inspect_snapshot(path))``
    at a notebook prompt.
    """
    md = read_snapshot_metadata(path)
    lines = [
        f"UW3 snapshot: {path}",
        f"  run_name           : {md.get('run_name', '?')}",
        f"  created_at         : {md.get('created_at', '?')}",
        f"  uw3_version        : {md.get('uw3_version', '?')}",
        f"  schema_version     : {md.get('schema_version', '?')}",
        f"  step / sim_time / dt : {md.get('step', '?')}  /  "
        f"{md.get('sim_time', '?')}  /  {md.get('dt', '?')}",
        f"  dim / mesh_type    : {md.get('dim', '?')}  /  "
        f"{md.get('mesh_type', '?')}",
        f"  coordinate_system  : {md.get('coordinate_system', '?')}",
        f"  mpi_ranks_at_write : {md.get('mpi_ranks_at_write', '?')}",
        f"  meshes  : {md.get('mesh_names', [])}",
        f"  swarms  : {md.get('swarm_names', [])}",
        f"  state_bearer_classes : {md.get('state_bearer_classes', [])}",
        f"  variables_summary  : {md.get('variables_summary', '')}",
    ]
    return "\n".join(lines)
