"""Phase 1 of the on-disk snapshot format (v1.1).

These tests assert the *inspectability bar* — an external h5 reader
(here, h5py — but the assertions translate directly to ``h5ls``
output) must see meaningful information about a UW3 snapshot file
without UW3 needing to be in the loop. They do not yet exercise any
PETSc bulk-data writes; that lands in phase 2.
"""

import json

import pytest
import numpy as np

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _fresh_model_with_state(tmp_path):
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    _ = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    _ = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=2)

    model.tracker.time = 3.14
    model.tracker.step = 42
    model.tracker.dt = 0.05
    return uw, model, mesh, swarm


def test_skeleton_writes_expected_group_structure(tmp_path):
    """The file an h5 tool would open has exactly the documented
    top-level group structure — no surprises for external readers."""
    import h5py

    uw, model, mesh, swarm = _fresh_model_with_state(tmp_path)
    path = str(tmp_path / "phase1.snap.h5")
    uw.checkpoint.write_snapshot_skeleton(model, path)

    with h5py.File(path, "r") as f:
        top = set(f.keys())
    assert top == {"metadata", "mesh", "variables", "swarms", "python_state"}


def test_metadata_is_inspectable_without_uw3(tmp_path):
    """The /metadata attrs h5py reads back are useful — i.e. an
    external reader sees the run identity, schema, step/time, geometry,
    MPI rank count, and the inventory of meshes/swarms/variables.

    Concretely: an h5py user (the proxy for h5ls/h5dump here) can
    answer 'what's in this file?' from /metadata alone."""
    import h5py

    uw, model, mesh, swarm = _fresh_model_with_state(tmp_path)
    path = str(tmp_path / "phase1.snap.h5")
    uw.checkpoint.write_snapshot_skeleton(model, path)

    with h5py.File(path, "r") as f:
        md = f["metadata"].attrs
        # Identity / versioning
        assert int(md["schema_version"]) == 1
        assert isinstance(str(md["created_at"]), str)
        assert str(md["run_name"]) != ""
        # Tracker conventions surfaced as scalars
        assert float(md["sim_time"]) == 3.14
        assert int(md["step"]) == 42
        assert float(md["dt"]) == 0.05
        # Geometry
        assert int(md["dim"]) == 2
        assert str(md["mesh_type"]) != ""
        # MPI
        assert int(md["mpi_ranks_at_write"]) >= 1
        # Inventories (JSON-encoded list-typed values)
        var_summary = str(md["variables_summary"])
        assert "T" in var_summary and "V" in var_summary
        swarm_names = json.loads(str(md["swarm_names_json"]))
        assert len(swarm_names) == 1
        state_classes = json.loads(str(md["state_bearer_classes_json"]))
        assert "ModelTracker" in state_classes


def test_read_snapshot_metadata_roundtrip(tmp_path):
    """write -> read returns the same content, with JSON-encoded list
    fields conveniently decoded for the caller."""
    import underworld3 as uw

    uw, model, mesh, swarm = _fresh_model_with_state(tmp_path)
    path = str(tmp_path / "phase1.snap.h5")
    uw.checkpoint.write_snapshot_skeleton(model, path)

    md = uw.checkpoint.read_snapshot_metadata(path)
    assert md["schema_version"] == 1
    assert md["sim_time"] == 3.14
    assert md["step"] == 42
    assert md["dim"] == 2
    # Convenience: JSON-encoded lists are also exposed as plain lists.
    assert isinstance(md["mesh_names"], list) and len(md["mesh_names"]) == 1
    assert isinstance(md["swarm_names"], list)
    assert "ModelTracker" in md["state_bearer_classes"]


def test_read_rejects_non_snapshot_file(tmp_path):
    """Pointing at an h5 file that isn't a UW3 snapshot raises
    cleanly, not with an obscure h5py error."""
    import h5py
    import underworld3 as uw

    path = str(tmp_path / "not-a-snapshot.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("random_dataset", data=np.zeros(3))

    with pytest.raises(ValueError, match="not a UW3 snapshot"):
        uw.checkpoint.read_snapshot_metadata(path)


def test_read_rejects_wrong_schema_version(tmp_path):
    """A future-version snapshot we cannot interpret raises, with a
    pointer to the (future) migration path."""
    import h5py
    import underworld3 as uw

    path = str(tmp_path / "future.snap.h5")
    with h5py.File(path, "w") as f:
        md = f.create_group("metadata")
        md.attrs["schema_version"] = 999

    with pytest.raises(ValueError, match="schema version 999"):
        uw.checkpoint.read_snapshot_metadata(path)


def test_inspect_snapshot_summary_includes_key_facts(tmp_path):
    """The human-readable summary surface (intended for notebook
    `print(...)` use) covers the same key facts external h5 inspection
    would surface."""
    import underworld3 as uw

    uw, model, mesh, swarm = _fresh_model_with_state(tmp_path)
    path = str(tmp_path / "phase1.snap.h5")
    uw.checkpoint.write_snapshot_skeleton(model, path)

    summary = uw.checkpoint.inspect_snapshot(path)
    assert "UW3 snapshot" in summary
    assert "schema_version     : 1" in summary
    assert "sim_time" in summary and "3.14" in summary
    assert "step" in summary and "42" in summary
    assert "ModelTracker" in summary


def test_skeleton_groups_have_filled_by_marker(tmp_path):
    """Empty top-level groups carry a `filled_by` attr so a phase-2/3
    reader knows whether their content is populated yet — and an
    external inspector sees 'this group is empty because phase 2
    hasn't run' rather than getting nothing."""
    import h5py
    import underworld3 as uw

    uw, model, mesh, swarm = _fresh_model_with_state(tmp_path)
    path = str(tmp_path / "phase1.snap.h5")
    uw.checkpoint.write_snapshot_skeleton(model, path)

    with h5py.File(path, "r") as f:
        for name in ("mesh", "variables", "swarms", "python_state"):
            assert "filled_by" in f[name].attrs
            assert str(f[name].attrs["filled_by"]) == ""
