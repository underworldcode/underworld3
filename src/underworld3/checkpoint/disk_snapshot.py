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
import os
from typing import Any, Optional

import numpy as np

import underworld3 as uw


DISK_SNAPSHOT_SCHEMA_VERSION = 1

# Top-level group names — fixed; renaming would be a schema-version bump.
# Variables are NOT a top-level group; they nest under each mesh:
# /meshes/{name}/variables/{var}.  Swarms similarly carry their own
# variables when phase 3 lands.
_GROUP_METADATA = "metadata"
_GROUP_MESHES = "meshes"
_GROUP_SWARMS = "swarms"
_GROUP_PYTHON_STATE = "python_state"

_TOP_LEVEL_GROUPS = (
    _GROUP_METADATA,
    _GROUP_MESHES,
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
                _GROUP_MESHES,
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


# ----- Phase 2: mesh + meshvar bulk via #146's PETSc primitives -----
#
# Layout convention:
#
#   /path/to/run.snap.h5              wrapper (metadata, h5py-readable)
#   /path/to/run.snap.bulk/           companion directory (one per snapshot)
#       {mesh_safe}.mesh.00000.h5     mesh DM dump (PETSc HDF5)
#       {mesh_safe}.{var_clean}.00000.h5  per-variable section + vec (PETSc HDF5)
#       ... one set per (mesh, var) ...
#
# The bulk-dir path is derived from the wrapper path by convention, so a
# user opening just the wrapper file can find the bulk. They are a unit
# for portability — move them together.


def _bulk_dir_for(wrapper_path: str) -> str:
    """Convention: wrapper at `run.snap.h5` ⇒ bulk at `run.snap.bulk/`."""
    base = wrapper_path[:-3] if wrapper_path.endswith(".h5") else wrapper_path
    return base + ".bulk"


def _sanitise(name: str) -> str:
    """Sanitise a mesh / variable name for use as a filename component.

    Replaces anything that isn't alphanumeric or in ``._-`` with ``_``.
    Falls back to ``unnamed`` if the result is empty. The original name
    is preserved in HDF5 group attrs as the ``@name`` field.
    """
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return safe or "unnamed"


def write_snapshot(model, path: str) -> str:
    """Write a complete on-disk snapshot of the model's mesh + mesh-variable
    state (phase 2 scope; swarms and python_state land in phase 3).

    Produces two artifacts:

    - ``path`` — the wrapper HDF5 file with rich metadata and the group
      structure inspectable via ``h5ls``.
    - ``_bulk_dir_for(path)`` — companion directory containing the
      PETSc HDF5 files (mesh DM + per-variable section/vec) produced
      by #146's :meth:`Mesh.write_checkpoint`.

    The two are a unit; move them together. Returns the wrapper path.
    """
    import h5py

    # Phase-1 layer: metadata + skeleton groups.
    write_snapshot_skeleton(model, path)
    bulk_dir = _bulk_dir_for(path)

    # rank-0 creates the bulk directory; collective ops below need it
    # to exist on the rank doing the PETSc-HDF5 write (which is rank 0
    # in this single-file write — actually PETSc's HDF5 viewer is
    # collective, so all ranks participate).
    with uw.selective_ranks(0) as rank0:
        if rank0:
            os.makedirs(bulk_dir, exist_ok=True)
    uw.mpi.barrier()

    # For each registered mesh, drive #146's write_checkpoint into the
    # bulk directory. write_checkpoint is collective (PETSc HDF5
    # viewer), so all ranks must participate.
    mesh_records: list[dict] = []
    for mesh in list(model._meshes.values()):
        mesh_safe = _sanitise(mesh.name)
        mesh_vars = list(mesh.vars.values())
        # Filter to allocated variables — same skip rule as the in-memory
        # path: lazy-allocated vars with _gvec == None have no data.
        mesh_vars = [v for v in mesh_vars if v._gvec is not None]

        mesh.write_checkpoint(
            mesh_safe,
            outputPath=bulk_dir,
            meshVars=mesh_vars,
            index=0,
        )

        mesh_records.append({
            "name": mesh.name,
            "safe_name": mesh_safe,
            "mesh_file": f"{mesh_safe}.mesh.00000.h5",
            "vars": [
                {
                    "name": v.clean_name,
                    "components": int(v.num_components),
                    "degree": int(v.degree),
                    "continuous": bool(v.continuous),
                    # Per-variable file produced by Mesh.write_checkpoint
                    # at outputPath: "{base}.{var.clean_name}.{index:05}.h5".
                    "external_file": (
                        f"{mesh_safe}.{v.clean_name}.00000.h5"
                    ),
                }
                for v in mesh_vars
            ],
        })

    # Reopen the wrapper to populate /meshes with the per-mesh records
    # and to mark the groups filled.
    with uw.selective_ranks(0) as rank0:
        if rank0:
            with h5py.File(path, "a") as f:
                meshes_group = f[_GROUP_MESHES]
                meshes_group.attrs["filled_by"] = "phase2"
                meshes_group.attrs["bulk_dir"] = os.path.basename(bulk_dir)

                for rec in mesh_records:
                    g = meshes_group.create_group(rec["safe_name"])
                    g.attrs["name"] = rec["name"]
                    g.attrs["mesh_file"] = rec["mesh_file"]

                    vars_g = g.create_group("variables")
                    for var_rec in rec["vars"]:
                        v = vars_g.create_group(_sanitise(var_rec["name"]))
                        v.attrs["name"] = var_rec["name"]
                        v.attrs["components"] = var_rec["components"]
                        v.attrs["degree"] = var_rec["degree"]
                        v.attrs["continuous"] = var_rec["continuous"]
                        v.attrs["external_file"] = var_rec["external_file"]

    uw.mpi.barrier()
    return path


def read_snapshot(model, path: str) -> None:
    """Load mesh-variable DOFs from an on-disk snapshot into the model.

    The model must already have the same meshes (by name) and the
    same variables (by ``clean_name``) registered — this is the
    same-rank-count restart path that mirrors :func:`restore` for the
    in-memory snapshot. Cross-run / rebuild-on-load is v1.2 scope.

    Bulk data is read via #146's :meth:`MeshVariable.read_checkpoint`;
    no KDTree remapping (that's phase 4's compatibility layer in
    ``read_timestep``).
    """
    import h5py

    md = read_snapshot_metadata(path)
    bulk_dir = _bulk_dir_for(path)
    if not os.path.isdir(bulk_dir):
        raise FileNotFoundError(
            f"snapshot bulk directory missing: {bulk_dir} (expected next "
            f"to wrapper {path})"
        )

    # Build {original_name -> registered Mesh} for lookup
    meshes_by_name = {m.name: m for m in model._meshes.values()}

    with h5py.File(path, "r") as f:
        meshes_group = f[_GROUP_MESHES]
        for mesh_safe in meshes_group.keys():
            g = meshes_group[mesh_safe]
            mesh_name = str(g.attrs.get("name", mesh_safe))
            mesh = meshes_by_name.get(mesh_name)
            if mesh is None:
                raise ValueError(
                    f"snapshot at {path} contains mesh {mesh_name!r} which "
                    f"is not registered on this model "
                    f"(registered: {sorted(meshes_by_name.keys())})"
                )

            current_vars = {v.clean_name: v for v in mesh.vars.values()}
            vars_g = g["variables"]
            for var_safe in vars_g.keys():
                v_attrs = vars_g[var_safe].attrs
                var_name = str(v_attrs["name"])
                external_file = str(v_attrs["external_file"])
                var = current_vars.get(var_name)
                if var is None:
                    raise ValueError(
                        f"snapshot variable {var_name!r} not registered on "
                        f"mesh {mesh_name!r}"
                    )
                var.read_checkpoint(
                    os.path.join(bulk_dir, external_file),
                    data_name=var_name,
                )


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
