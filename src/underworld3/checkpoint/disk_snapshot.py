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

File structure (the phased writers below fill the bulk under these
groups; ``write_snapshot_skeleton`` stubs each group with a
``filled_by`` attribute naming its writer):

    my_run.snap.h5/
    ├── /metadata          (attrs: uw3_version, schema_version,
    │                       created_at, step, sim_time, dt, dim,
    │                       mesh_type, coordinate_system,
    │                       mpi_ranks_at_write, variables_summary, ...)
    ├── /mesh              (phase 2 — DMPlex topology + coords + labels)
    ├── /variables         (phase 2 — one subgroup per mesh-variable)
    ├── /swarms            (phase 3 — possibly @external_file refs)
    └── /python_state      (phase 3 — Snapshottable dataclasses as attrs)

Phases 1 (metadata + skeleton), 2 (mesh + variables) and 3 (swarms +
python_state) are all implemented in this module; an inspectability
acceptance test asserts an external reader (h5py) sees meaningful
information without any UW3 imports.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import warnings
from contextlib import contextmanager
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
#       mesh_0000.mesh.00000.h5       mesh DM dump (PETSc HDF5)
#       mesh_0000.{var_clean}.00000.h5  per-variable section + vec (PETSc HDF5)
#       ... one set per (mesh, var) ...
#
# The bulk-dir path is derived from the wrapper path by convention, so a
# user opening just the wrapper file can find the bulk. They are a unit
# for portability — move them together.


def _bulk_dir_for(wrapper_path: str) -> str:
    """Convention: wrapper at `run.snap.h5` ⇒ bulk at `run.snap.bulk/`."""
    base = wrapper_path[:-3] if wrapper_path.endswith(".h5") else wrapper_path
    return base + ".bulk"


@contextmanager
def _short_io_path(path: str):
    """Expose ``path`` to native I/O as a basename from its parent directory.

    Some parallel PETSc/HDF5 stacks fail on valid absolute paths well below
    ``PATH_MAX``. Snapshot artifacts retain their normal locations, while the
    native reader or writer receives only the final path component.
    """
    absolute_path = os.path.abspath(path)
    previous_directory = os.getcwd()
    os.chdir(os.path.dirname(absolute_path))
    try:
        yield os.path.basename(absolute_path)
    finally:
        os.chdir(previous_directory)


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
    for mesh_index, mesh in enumerate(list(model._meshes.values())):
        # Loaded meshes commonly use their complete source path as ``name``.
        # Embedding that path in every PETSc-HDF5 bulk filename can exceed
        # practical MPI-I/O pathname limits even when the wrapper path itself
        # is valid. The wrapper preserves the original name for exact restore
        # matching, so bulk files only need a compact deterministic identifier.
        mesh_safe = f"mesh_{mesh_index:04d}"
        mesh_vars = list(mesh.vars.values())
        # Filter to allocated variables — same skip rule as the in-memory
        # path: lazy-allocated vars with _gvec == None have no data.
        mesh_vars = [v for v in mesh_vars if v._gvec is not None]

        # write_checkpoint is user-deprecated in favour of write_timestep, but
        # the snapshot backend is a legitimate internal user: it relies on the
        # `{base}.mesh.00000.h5` / `{base}.{var}.00000.h5` filename convention
        # (consumed by the reload path below). Suppress the FutureWarning for
        # this internal call rather than spam every snapshot. (Migrating the
        # snapshot to write_timestep is tracked in #252.)
        with _short_io_path(bulk_dir) as short_bulk_dir:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                mesh.write_checkpoint(
                    mesh_safe,
                    outputPath=short_bulk_dir,
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

                # Phase 3a: state-bearer dataclass serialisation.
                ps_group = f[_GROUP_PYTHON_STATE]
                ps_group.attrs["filled_by"] = "phase3a"
                for obj in list(model._state_bearers):
                    key = f"{type(obj).__name__}_{obj.instance_number}"
                    if key in ps_group:
                        continue   # idempotent if write_snapshot called twice
                    bg = ps_group.create_group(key)
                    _write_state_bearer_to_group(bg, obj)

    # Phase 3b + 6: swarms — per-rank sidecar files in the bulk dir,
    # referenced from /swarms/{swarm_safe}/ via a sidecar_pattern attr.
    # Every rank writes its own sidecar; all-ranks-participate so the
    # bulk dir is complete before the wrapper records the layout.
    rank = int(uw.mpi.rank)
    size = int(uw.mpi.size)
    swarm_records: list[dict] = []
    for swarm in list(model._swarms.values()):
        swarm_safe = _swarm_safe_name(swarm)
        sidecar_name = _swarm_sidecar_filename(swarm_safe, rank, size)
        sidecar_path = os.path.join(bulk_dir, sidecar_name)
        rec = _write_swarm_to_sidecar(swarm, sidecar_path)
        rec["safe_name"] = swarm_safe
        rec["sidecar_pattern"] = _swarm_sidecar_pattern(swarm_safe)
        swarm_records.append(rec)
    uw.mpi.barrier()

    # Wrapper update is rank-0-only; gather global counts across
    # ranks first so the wrapper carries a complete inventory.
    if swarm_records:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            global_counts = {
                rec["safe_name"]: int(
                    comm.allreduce(rec["num_particles_local"], op=MPI.SUM)
                )
                for rec in swarm_records
            }
        except ImportError:
            global_counts = {
                rec["safe_name"]: int(rec["num_particles_local"])
                for rec in swarm_records
            }

        with uw.selective_ranks(0) as rank0:
            if rank0:
                with h5py.File(path, "a") as f:
                    sw_group = f[_GROUP_SWARMS]
                    sw_group.attrs["filled_by"] = "phase3b+phase6"
                    sw_group.attrs["mpi_size_at_write"] = size
                    for rec in swarm_records:
                        g = sw_group.create_group(rec["safe_name"])
                        g.attrs["mesh_name"] = rec["mesh_name"]
                        g.attrs["num_particles_global"] = global_counts[
                            rec["safe_name"]
                        ]
                        g.attrs["population_generation"] = rec[
                            "population_generation"
                        ]
                        g.attrs["sidecar_pattern"] = rec["sidecar_pattern"]
                        g.attrs["mpi_size_at_write"] = size
                        vars_g = g.create_group("variables")
                        for var_rec in rec["vars"]:
                            v = vars_g.create_group(_sanitise(var_rec["name"]))
                            v.attrs["name"] = var_rec["name"]
                            v.attrs["num_components"] = var_rec[
                                "num_components"
                            ]

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
    write_size = int(md.get("mpi_ranks_at_write", 1))
    if write_size != int(uw.mpi.size):
        raise ValueError(
            f"snapshot at {path} was written on {write_size} MPI rank(s); "
            f"this run uses {uw.mpi.size}. Exact disk restart requires the "
            "same rank count; use mesh.write_timestep/read_timestep for "
            "coordinate-remapped field transfer."
        )

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
                with _short_io_path(
                    os.path.join(bulk_dir, external_file)
                ) as short_external_file:
                    var.read_checkpoint(
                        short_external_file,
                        data_name=var_name,
                    )

        # Phase 3a: restore state-bearer dataclasses.
        if _GROUP_PYTHON_STATE in f:
            ps_group = f[_GROUP_PYTHON_STATE]
            bearers_by_key = {
                f"{type(o).__name__}_{o.instance_number}": o
                for o in list(model._state_bearers)
            }
            for key in ps_group.keys():
                obj = bearers_by_key.get(key)
                if obj is None:
                    raise ValueError(
                        f"snapshot at {path} contains state-bearer {key!r} "
                        f"that is not registered on this model"
                    )
                _read_state_bearer_into(ps_group[key], obj)

        # Phase 3b + 6: restore swarms from per-rank sidecars.
        if _GROUP_SWARMS in f:
            sw_group = f[_GROUP_SWARMS]
            if "mpi_size_at_write" in sw_group.attrs:
                write_size = int(sw_group.attrs["mpi_size_at_write"])
                if write_size != int(uw.mpi.size):
                    raise ValueError(
                        f"snapshot at {path} was written on {write_size} "
                        f"MPI rank(s); this run is on {uw.mpi.size}. "
                        f"Cross-rank-count restore is not supported by the "
                        f"snapshot mechanism — use mesh.write_timestep for "
                        f"that case."
                    )

            swarms_by_safe = {
                _swarm_safe_name(s): s for s in list(model._swarms.values())
            }
            rank = int(uw.mpi.rank)
            size = int(uw.mpi.size)
            for swarm_safe in sw_group.keys():
                g = sw_group[swarm_safe]
                swarm = swarms_by_safe.get(swarm_safe)
                if swarm is None:
                    raise ValueError(
                        f"snapshot at {path} contains swarm {swarm_safe!r} "
                        f"that is not registered on this model"
                    )
                # Resolve the per-rank sidecar name from the pattern.
                pattern = str(g.attrs["sidecar_pattern"])
                sidecar_name = pattern.format(rank=rank, size=size)
                _read_swarm_from_sidecar(
                    swarm, os.path.join(bulk_dir, sidecar_name)
                )


# ----- Phase 3b: swarm sidecars --------------------------------------------
#
# Swarms always go to their own per-swarm sidecar file from day one,
# per Louis's "bulk is a problem with swarms, always" — no inline-vs-
# split toggle. The wrapper's /swarms/{swarm_safe}/ records metadata
# + an `@external_file` ref pointing at the sidecar in the bulk dir.
#
# The sidecar is h5py-direct (no PETSc). Swarms aren't DMPlex
# section/vec; they're per-particle numpy arrays.
#
# Single-rank now; MPI gets phase 6 treatment (per-rank sidecars or
# parallel-HDF5).


def _swarm_safe_name(swarm) -> str:
    """Stable name for a swarm in the snapshot layout. Mirrors the
    in-memory snapshot's `_snapshot_stable_name`."""
    raw = getattr(swarm, "name", None) or f"swarm_{swarm.instance_number}"
    return _sanitise(raw)


def _swarm_sidecar_filename(swarm_safe: str, rank: int, size: int) -> str:
    """Per-rank sidecar (phase 6 — parallel-safe).

    Each rank writes its own swarm sidecar; restoring requires the
    same rank count. The filename carries both the writer's rank and
    the total rank count so each file is self-describing.
    """
    return f"{swarm_safe}.swarm.rank{rank:04d}of{size:04d}.h5"


def _swarm_sidecar_pattern(swarm_safe: str) -> str:
    """Pattern stored in the wrapper for readers to fill in their own
    (rank, size) when locating their sidecar."""
    return f"{swarm_safe}.swarm.rank{{rank:04d}}of{{size:04d}}.h5"


def _write_swarm_to_sidecar(swarm, sidecar_path: str) -> dict:
    """Write a swarm's local-particle state to a sidecar h5 file.

    Returns a record dict the caller uses to populate the wrapper's
    /swarms/{name}/ group.
    """
    import h5py

    coord_field = swarm.dm.getField("DMSwarmPIC_coor").reshape(
        (-1, swarm.dim)
    )
    coords = np.asarray(coord_field).copy()
    swarm.dm.restoreField("DMSwarmPIC_coor")

    var_records: list[dict] = []
    with h5py.File(sidecar_path, "w") as f:
        # File-level metadata — h5ls -v on the sidecar tells you what
        # it is without needing UW3 (matches the wrapper's bar).
        f.attrs["num_particles_local"] = int(coords.shape[0])
        f.attrs["dim"] = int(swarm.dim)
        f.attrs["mesh_name"] = str(swarm.mesh.name)
        f.attrs["population_generation"] = int(swarm._population_generation)
        # Parallel-write provenance — each per-rank sidecar carries
        # its writer's identity so the reader can sanity-check it
        # opened the right file.
        f.attrs["mpi_rank"] = int(uw.mpi.rank)
        f.attrs["mpi_size_at_write"] = int(uw.mpi.size)

        f.create_dataset("coordinates", data=coords)

        vars_g = f.create_group("variables")
        for var in list(swarm._vars.values()):
            # Filter PETSc-internal variables — same rule as the
            # in-memory swarm capture.
            if var.name.startswith("DMSwarm"):
                continue
            data = np.asarray(var.data).copy()
            d = vars_g.create_dataset(var.clean_name, data=data)
            d.attrs["num_components"] = int(var.num_components)
            d.attrs["dtype"] = str(data.dtype)
            var_records.append({
                "name": var.clean_name,
                "num_components": int(var.num_components),
            })

    return {
        "num_particles_local": int(coords.shape[0]),
        "mesh_name": str(swarm.mesh.name),
        "population_generation": int(swarm._population_generation),
        "vars": var_records,
    }


def _read_swarm_from_sidecar(swarm, sidecar_path: str) -> None:
    """Restore swarm state from a sidecar file. Mirrors
    :meth:`Swarm.apply_snapshot_payload` exactly — clear local
    particles, re-add at saved coords, write var data back."""
    import h5py

    with h5py.File(sidecar_path, "r") as f:
        saved_coords = np.asarray(f["coordinates"][...])
        captured_mesh_name = str(f.attrs.get("mesh_name", ""))
        captured_rank = int(f.attrs.get("mpi_rank", 0))
        captured_size = int(f.attrs.get("mpi_size_at_write", 1))
        var_data: dict[str, np.ndarray] = {}
        if "variables" in f:
            for name in f["variables"].keys():
                var_data[name] = np.asarray(f["variables"][name][...])

    if captured_size != uw.mpi.size:
        raise ValueError(
            f"sidecar at {sidecar_path}: was written on "
            f"{captured_size} MPI rank(s); this run is on {uw.mpi.size}. "
            f"Cross-rank-count snapshot restore is out of scope — restart "
            f"on the same rank count or use mesh.write_timestep for the "
            f"flexible-restart path."
        )
    if captured_rank != uw.mpi.rank:
        raise ValueError(
            f"sidecar at {sidecar_path}: written by rank {captured_rank}; "
            f"this is rank {uw.mpi.rank}. Wrong per-rank sidecar opened."
        )
    if captured_mesh_name and captured_mesh_name != swarm.mesh.name:
        raise ValueError(
            f"sidecar at {sidecar_path}: parent mesh was "
            f"{captured_mesh_name!r}, target swarm is on {swarm.mesh.name!r}"
        )

    # Clear local population. removePoint is O(1) per call (last point),
    # so this is O(N) total — same approach as Swarm.apply_snapshot_payload.
    while swarm.dm.getLocalSize() > 0:
        swarm.dm.removePoint()

    n_saved = int(saved_coords.shape[0])
    if n_saved > 0:
        swarm.dm.finalizeFieldRegister()
        swarm.dm.addNPoints(npoints=n_saved)

        coord_field = swarm.dm.getField("DMSwarmPIC_coor").reshape(
            (-1, swarm.dim)
        )
        coord_field[...] = saved_coords
        swarm.dm.restoreField("DMSwarmPIC_coor")

        rank_field = swarm.dm.getField("DMSwarm_rank")
        rank_field[...] = uw.mpi.rank
        swarm.dm.restoreField("DMSwarm_rank")

    # Invalidate canonical-data caches so subsequent var.data reads
    # re-resolve to the rebuilt PETSc fields.
    swarm._invalidate_canonical_data()

    # Restore counted as a population change (matches in-memory path).
    swarm._population_generation += 1

    # Write per-variable captured data back into the freshly-rebuilt
    # swarm.
    current_vars = {v.clean_name: v for v in swarm._vars.values()}
    for var_name, saved in var_data.items():
        var = current_vars.get(var_name)
        if var is None:
            raise ValueError(
                f"sidecar variable {var_name!r} is not present on the "
                f"target swarm; restore requires the same variable set"
            )
        current = np.asarray(var.data)
        if current.shape != saved.shape:
            raise ValueError(
                f"swarm variable {var_name!r} shape mismatch on restore: "
                f"sidecar {saved.shape} vs current {current.shape}"
            )
        current[...] = saved


# ----- Phase 3a: state-bearer (Snapshottable) serialisation ----------------
#
# Each registered state-bearer (DDt instances, ModelTracker, future
# helpers) exposes a `.state` property returning a SnapshottableState
# dataclass. We serialise each dataclass's fields into a per-bearer
# HDF5 group under /python_state, keyed by the same stable name the
# in-memory snapshot uses: f"{type(obj).__name__}_{obj.instance_number}".
#
# Serialisation is *generic over dataclass fields* — no per-class
# special code. Handled value types: None, bool, int, float, str,
# numpy.ndarray, list (JSON-encoded), dict (recursive subgroup). Other
# types (notably sympy expressions in DDtSymbolicState.psi_star) are
# marked with `<field>_skipped` and not round-tripped — documented as
# a v1.x limitation; consumers either use a non-Symbolic DDt flavor
# or accept the psi_star reset.


_NULL_SENTINEL = "__none__"
_TYPE_ATTR = "__bearer_class__"
_DATACLASS_ATTR = "__state_class__"


def _is_h5_attr_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str)) and not isinstance(
        value, bool
    ) or isinstance(value, (bool, str))


def _serialise_field(h5group, name: str, value: Any) -> None:
    """Write a Python value into an HDF5 group as attr/dataset/subgroup.

    The shape of the storage records the type:
      - attr scalar (int/float/bool/str) for scalars
      - attr `<name>` = '__none__' for None
      - attr `<name>__json` for JSON-encodable lists / nested simple structures
      - dataset `<name>` for numpy arrays
      - subgroup `<name>` for dict values, recursing
      - attr `<name>__skipped` = '<type>' for anything else
    """
    if value is None:
        h5group.attrs[name] = _NULL_SENTINEL
        return
    if isinstance(value, (bool, int, float)):
        h5group.attrs[name] = value
        return
    if isinstance(value, str):
        h5group.attrs[name] = value
        return
    if isinstance(value, np.ndarray):
        if name in h5group:
            del h5group[name]
        h5group.create_dataset(name, data=value)
        return
    if isinstance(value, dict):
        if name in h5group:
            del h5group[name]
        sub = h5group.create_group(name)
        for k, v in value.items():
            _serialise_field(sub, str(k), v)
        return
    if isinstance(value, (list, tuple)):
        try:
            h5group.attrs[name + "__json"] = json.dumps(list(value))
            return
        except (TypeError, ValueError):
            h5group.attrs[name + "__skipped"] = (
                f"unserialisable list (len={len(value)}, "
                f"first-type={type(value[0]).__name__ if value else 'empty'})"
            )
            return
    h5group.attrs[name + "__skipped"] = (
        f"unserialisable type {type(value).__name__}"
    )


def _group_to_dict(h5group) -> dict:
    """Read a subgroup back as a plain dict — symmetric to the dict
    branch of :func:`_serialise_field`. Recurses for nested groups."""
    import h5py

    out: dict = {}
    for k in h5group.attrs.keys():
        if k.endswith("__skipped"):
            continue
        if k.endswith("__json"):
            out[k[: -len("__json")]] = json.loads(h5group.attrs[k])
            continue
        v = h5group.attrs[k]
        if isinstance(v, str) and v == _NULL_SENTINEL:
            out[k] = None
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    for k in h5group.keys():
        item = h5group[k]
        if isinstance(item, h5py.Group):
            out[k] = _group_to_dict(item)
        else:
            out[k] = np.asarray(item[...])
    return out


def _deserialise_field(h5group, name: str, fallback: Any) -> Any:
    """Inverse of :func:`_serialise_field`. Returns ``fallback`` if the
    field was skipped at write time, so we don't clobber a sensible
    default with a placeholder."""
    import h5py

    if name in h5group:
        item = h5group[name]
        if isinstance(item, h5py.Group):
            return _group_to_dict(item)
        return np.asarray(item[...])  # h5py.Dataset

    if name in h5group.attrs:
        v = h5group.attrs[name]
        if isinstance(v, str) and v == _NULL_SENTINEL:
            return None
        if isinstance(v, np.generic):
            return v.item()
        return v

    if (name + "__json") in h5group.attrs:
        return json.loads(h5group.attrs[name + "__json"])

    if (name + "__skipped") in h5group.attrs:
        # Skipped at write time — keep the current value rather than
        # clobber it with a placeholder.
        return fallback

    return fallback


def _write_state_bearer_to_group(group, obj) -> None:
    """Serialise a Snapshottable's .state into the given HDF5 group."""
    state = obj.state
    group.attrs[_TYPE_ATTR] = type(obj).__name__
    group.attrs[_DATACLASS_ATTR] = type(state).__name__
    group.attrs["instance_number"] = int(obj.instance_number)

    for f in dataclasses.fields(state):
        _serialise_field(group, f.name, getattr(state, f.name))


def _read_state_bearer_into(group, obj) -> None:
    """Restore a Snapshottable's .state from a group written by
    :func:`_write_state_bearer_to_group`. Uses the live ``obj.state``
    as a type template — fields that were skipped at write time keep
    their current value rather than being clobbered by a placeholder.
    """
    current_state = obj.state
    captured_class = str(group.attrs.get(_DATACLASS_ATTR, ""))
    if captured_class and captured_class != type(current_state).__name__:
        raise ValueError(
            f"state-bearer class mismatch: snapshot expects "
            f"{captured_class}, current is {type(current_state).__name__}"
        )

    overrides = {}
    for f in dataclasses.fields(current_state):
        new_val = _deserialise_field(group, f.name, getattr(current_state, f.name))
        overrides[f.name] = new_val
    obj.state = dataclasses.replace(current_state, **overrides)


def is_snapshot_wrapper(path: str) -> bool:
    """Quick check whether ``path`` is a v1.1 snapshot wrapper file.

    Used by :meth:`MeshVariable.read_timestep` to dispatch between
    the legacy per-variable layout and the v1.1 sidecar format —
    same user call, different storage, hidden behind the function.
    """
    import h5py

    try:
        with h5py.File(path, "r") as f:
            return _GROUP_METADATA in f and _GROUP_MESHES in f
    except (OSError, KeyError):
        return False


def extract_var_via_bridge(wrapper_path: str, var_name: str):
    """Bridge for selective per-variable reads of v1.1 snapshots.

    Given the wrapper path and a variable name, returns
    ``(coords, values)`` numpy arrays — exactly what
    :meth:`MeshVariable.read_timestep` produces on rank 0 from the
    legacy layout. The rest of read_timestep's swarm-routing +
    KDTree machinery is format-agnostic; this bridge is what makes
    ``read_timestep`` work transparently against new files.

    Mechanism: load the source mesh from the .mesh.h5 sidecar,
    rebuild the source variable with the correct shape, load DOFs
    via #146's ``read_checkpoint``, then read out ``var.coords`` +
    ``var.array``.
    """
    import h5py

    bulk_dir = _bulk_dir_for(wrapper_path)
    found = None
    with h5py.File(wrapper_path, "r") as f:
        for mesh_safe in f[_GROUP_MESHES].keys():
            mg = f[_GROUP_MESHES][mesh_safe]
            for var_safe in mg["variables"].keys():
                v_attrs = mg["variables"][var_safe].attrs
                if str(v_attrs["name"]) == var_name:
                    found = (
                        str(mg.attrs["mesh_file"]),
                        str(v_attrs["external_file"]),
                        int(v_attrs["degree"]),
                        int(v_attrs["components"]),
                        bool(v_attrs["continuous"]),
                    )
                    break
            if found:
                break
    if found is None:
        raise ValueError(
            f"variable {var_name!r} not found in v1.1 snapshot {wrapper_path}"
        )

    mesh_file_rel, var_file_rel, degree, components, continuous = found
    # Rebuild a transient source mesh + variable to read DOFs into.
    # We deliberately don't register them with the live model — these
    # are throwaway and exit scope on return.
    with _short_io_path(
        os.path.join(bulk_dir, mesh_file_rel)
    ) as short_mesh_file:
        src_mesh = uw.discretisation.Mesh(short_mesh_file)
    src_var = uw.discretisation.MeshVariable(
        var_name, src_mesh, components, degree=degree, continuous=continuous,
    )
    with _short_io_path(
        os.path.join(bulk_dir, var_file_rel)
    ) as short_var_file:
        src_var.read_checkpoint(short_var_file, data_name=var_name)

    coords = np.asarray(src_var.coords).copy()
    values = np.asarray(src_var.array[...]).reshape(
        coords.shape[0], components
    ).copy()
    return coords, values


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
