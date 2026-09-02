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
    # Variables nest under each mesh (/meshes/{name}/variables/...) so
    # they are not a top-level group.
    assert top == {"metadata", "meshes", "swarms", "python_state"}


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
        for name in ("meshes", "swarms", "python_state"):
            assert "filled_by" in f[name].attrs
            assert str(f[name].attrs["filled_by"]) == ""


# ----- Phase 2: mesh + mesh-variable bulk via #146 -----


def _fresh_model_mesh_and_vars():
    import underworld3 as uw

    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    V = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    return uw, model, mesh, T, V


def test_write_snapshot_produces_wrapper_and_bulk_dir(tmp_path):
    """The two artifacts the convention promises: wrapper file +
    sibling .bulk/ directory containing PETSc HDF5 files."""
    import h5py
    import os
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    T.array[:, 0, 0] = 5.0
    V.array[:, 0, 0] = -3.0

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    bulk = str(tmp_path / "run.snap.bulk")
    assert os.path.exists(path)
    assert os.path.isdir(bulk)
    # #146-format files in the bulk dir.
    files = sorted(os.listdir(bulk))
    # At least: mesh file + one file per variable.
    assert any(f.endswith(".mesh.00000.h5") for f in files)
    assert any("T.00000.h5" in f for f in files)
    assert any("V.00000.h5" in f for f in files)

    for variable_name in ("T", "V"):
        variable_file = next(
            filename
            for filename in files
            if filename.endswith(f".{variable_name}.00000.h5")
        )
        with h5py.File(os.path.join(bulk, variable_file), "r") as h5:
            assert variable_name in h5["uw_checkpoint"]


def test_snapshot_bulk_filenames_do_not_expand_loaded_mesh_name(tmp_path):
    """A source pathname remains metadata, not a PETSc bulk filename."""
    import h5py
    import os

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    mesh.name = "/g/data/project/user/" + "nested_directory/" * 12 + "mesh.msh.h5"

    path = str(tmp_path / "compact.snap.h5")
    model.save_state(file=path)

    bulk = str(tmp_path / "compact.snap.bulk")
    files = sorted(os.listdir(bulk))
    assert files
    assert all(filename.startswith("mesh_0000.") for filename in files)
    assert max(map(len, files)) < 64

    with h5py.File(path, "r") as h5:
        assert list(h5["meshes"]) == ["mesh_0000"]
        assert h5["meshes"]["mesh_0000"].attrs["name"] == mesh.name


def test_write_snapshot_populates_wrapper_layout(tmp_path):
    """The wrapper carries the per-mesh + per-variable metadata that
    makes 'what's in this snapshot?' answerable from h5py alone."""
    import h5py
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    T.array[:, 0, 0] = 1.0
    V.array[:, 0, 0] = 2.0
    V.array[:, 0, 1] = 3.0

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    with h5py.File(path, "r") as f:
        assert f["meshes"].attrs["filled_by"] == "phase2"
        # One mesh subgroup
        mesh_names = list(f["meshes"].keys())
        assert len(mesh_names) == 1
        mg = f["meshes"][mesh_names[0]]
        # Per-mesh attrs
        assert mg.attrs["name"] == mesh.name
        assert mg.attrs["mesh_file"].endswith(".mesh.00000.h5")
        # Variables subgroup
        var_names = sorted(mg["variables"].keys())
        assert var_names == ["T", "V"]
        # Per-var attrs include shape info + external_file pointer.
        v_attrs = mg["variables"]["V"].attrs
        assert v_attrs["components"] == 2
        assert v_attrs["degree"] == 2
        assert v_attrs["external_file"].endswith("V.00000.h5")


def test_write_read_snapshot_bit_exact_roundtrip(tmp_path):
    """The core phase-2 guarantee: write a snapshot, scribble all
    variables, read snapshot back, all variables match write-time
    values bit-for-bit through the exact same-layout PETSc vector payload."""
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    T.array[:, 0, 0] = 5.0 * T.coords[:, 0] - 2.0
    V.array[:, 0, 0] = 3.0 * V.coords[:, 0]
    V.array[:, 0, 1] = 7.0 * V.coords[:, 1]
    T_pre = np.asarray(T.array[...]).copy()
    V_pre = np.asarray(V.array[...]).copy()

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    # Scribble.
    T.array[...] = -99.0
    V.array[...] = -99.0

    model.load_state(path)

    assert np.array_equal(np.asarray(T.array[...]), T_pre), (
        f"T not bit-exact after read_snapshot — max|d|="
        f"{float(np.max(np.abs(np.asarray(T.array[...]) - T_pre))):.3e}"
    )
    assert np.array_equal(np.asarray(V.array[...]), V_pre), (
        f"V not bit-exact after read_snapshot — max|d|="
        f"{float(np.max(np.abs(np.asarray(V.array[...]) - V_pre))):.3e}"
    )


def test_disk_restore_refreshes_packed_auxiliary_fields_before_solve(tmp_path):
    """A solve after disk restore must use restored coefficient fields."""
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5,
    )
    solution = uw.discretisation.MeshVariable("U", mesh, 1, degree=1)
    source = uw.discretisation.MeshVariable("source", mesh, 1, degree=1)

    poisson = uw.systems.Poisson(mesh, u_Field=solution)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = source.sym[0]
    poisson.add_dirichlet_bc(0.0, "All_Boundaries")

    source.array[...] = 1.0
    poisson.solve(zero_init_guess=True)
    reference = np.asarray(solution.array[...]).copy()

    path = str(tmp_path / "auxiliary.snap.h5")
    model.save_state(file=path)

    source.array[...] = 4.0
    poisson.solve(zero_init_guess=True)
    assert not np.allclose(np.asarray(solution.array[...]), reference)

    model.load_state(path)
    assert mesh._stale_lvec is True
    poisson.solve(zero_init_guess=True)

    assert np.allclose(
        np.asarray(solution.array[...]),
        reference,
        rtol=0.0,
        atol=1.0e-5,
    )


def test_read_snapshot_rejects_missing_bulk_dir(tmp_path):
    """If the user moves the wrapper without the bulk dir, read fails
    with a clear pointer rather than an obscure h5py error."""
    import os
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    # Delete the bulk dir to simulate the move-the-wrapper-only mistake.
    import shutil
    shutil.rmtree(str(tmp_path / "run.snap.bulk"))

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    with pytest.raises(FileNotFoundError, match="bulk directory missing"):
        model.load_state(path)


def test_read_snapshot_rejects_different_mpi_rank_count(tmp_path):
    """Exact disk restart must not silently remap a different MPI layout."""
    import h5py
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    path = str(tmp_path / "rank_count.snap.h5")
    model.save_state(file=path)

    with h5py.File(path, "r+") as h5:
        h5["metadata"].attrs["mpi_ranks_at_write"] = uw.mpi.size + 1

    with pytest.raises(ValueError, match="same rank count"):
        model.load_state(path)


def test_read_snapshot_rejects_mismatched_mesh(tmp_path):
    """If the target model's meshes don't match the snapshot's, raise
    clearly — mesh-rebuild on read is v1.2 scope."""
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    # Fresh model with a *different* mesh — write_snapshot's mesh.name
    # won't match.
    uw.reset_default_model()
    model2 = uw.get_default_model()
    other = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(2.0, 2.0), cellSize=1.0 / 3.0,
    )
    # Don't reuse the original mesh's name — make the lookup miss.
    other.name = "definitely_a_different_mesh"

    with pytest.raises(ValueError, match="not registered on this model"):
        model2.load_state(path)


# ----- Phase 3a: state-bearer (Snapshottable) serialisation -----


def test_tracker_round_trips_through_disk_snapshot(tmp_path):
    """ModelTracker is always auto-registered as a state-bearer, so
    every snapshot must round-trip its time/step/dt and any
    user-added managed quantities exactly."""
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    model.tracker.time = 3.14
    model.tracker.step = 42
    model.tracker.dt = 0.05
    model.tracker.my_diagnostic = 99.0
    model.tracker.history_arr = np.array([1.0, 2.0, 3.0])

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    # Scribble everything tracker-side.
    model.tracker.time = -1.0
    model.tracker.step = -1
    model.tracker.dt = -1.0
    model.tracker.my_diagnostic = -1.0
    model.tracker.history_arr = np.array([-1.0, -1.0, -1.0])

    model.load_state(path)

    assert model.tracker.time == 3.14
    assert model.tracker.step == 42
    assert model.tracker.dt == 0.05
    assert model.tracker.my_diagnostic == 99.0
    assert np.array_equal(
        np.asarray(model.tracker.history_arr), np.array([1.0, 2.0, 3.0])
    )


def test_python_state_group_is_inspectable(tmp_path):
    """An external h5py reader sees the per-bearer groups under
    /python_state with class info + a managed dict for the tracker."""
    import h5py
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    model.tracker.time = 1.0
    model.tracker.step = 2
    model.tracker.my_q = 7.0

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    with h5py.File(path, "r") as f:
        ps = f["python_state"]
        assert ps.attrs["filled_by"] == "phase3a"

        tracker_keys = [k for k in ps.keys() if k.startswith("ModelTracker_")]
        assert len(tracker_keys) == 1
        tg = ps[tracker_keys[0]]
        assert tg.attrs["__bearer_class__"] == "ModelTracker"
        assert tg.attrs["__state_class__"] == "TrackerState"
        # TrackerState.managed is a dict, stored as a sub-group.
        assert "managed" in tg and isinstance(tg["managed"], h5py.Group)
        managed = tg["managed"]
        # Pre-seeded conventions present.
        assert "time" in managed.attrs
        assert float(managed.attrs["time"]) == 1.0
        assert int(managed.attrs["step"]) == 2
        # User-added quantity present.
        assert "my_q" in managed.attrs
        assert float(managed.attrs["my_q"]) == 7.0


def test_ddt_symbolic_state_round_trips_primary_fields(tmp_path):
    """A Symbolic DDt has dt_history, history_initialised,
    n_solves_completed, dt round-tripped via the generic dataclass
    serialiser. psi_star (sympy) is documented as skipped — the
    primary BDF-control fields are what matter for re-continuing."""
    import underworld3 as uw
    import sympy

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    ddt = uw.systems.ddt.Symbolic(T.sym, order=2)
    ddt._dt_history = [0.05, 0.03]
    ddt._history_initialised = True
    ddt._n_solves_completed = 2
    ddt._dt = 0.05

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    # Scribble the primary fields.
    ddt._dt_history = [None, None]
    ddt._history_initialised = False
    ddt._n_solves_completed = 0
    ddt._dt = None

    model.load_state(path)

    assert ddt.state.dt_history == [0.05, 0.03]
    assert ddt.state.history_initialised is True
    assert ddt.state.n_solves_completed == 2
    assert ddt.state.dt == 0.05


def test_read_snapshot_rejects_missing_state_bearer(tmp_path):
    """If a state-bearer exists in the snapshot but not on the load-
    target model, raise — same-rank/same-model contract."""
    import underworld3 as uw

    # Source model has a Symbolic DDt; snapshot it.
    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    ddt = uw.systems.ddt.Symbolic(T.sym, order=2)
    ddt._dt_history = [0.05, 0.05]

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    # Target model has no DDt.
    uw, model2, mesh2, T2, V2 = _fresh_model_mesh_and_vars()
    # Force name match so the mesh part loads.
    mesh2.name = mesh.name

    with pytest.raises(ValueError, match="state-bearer .* not registered"):
        model2.load_state(path)


# ----- Phase 3b: swarms in sidecars -----


def _fresh_model_mesh_swarm():
    import underworld3 as uw

    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable("material", 1, dtype=float)
    swarm.populate(fill_param=2)
    material.data[:, 0] = swarm._particle_coordinates.data[:, 0]
    return uw, model, mesh, swarm, material


def test_swarm_sidecar_lands_in_bulk_dir(tmp_path):
    """A registered swarm produces its own h5 sidecar in the bulk dir
    with predictable name (per-rank); wrapper records the sidecar
    pattern + rank count."""
    import os
    import re
    import h5py
    import underworld3 as uw

    uw, model, mesh, swarm, material = _fresh_model_mesh_swarm()
    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    bulk = str(tmp_path / "run.snap.bulk")
    files = sorted(os.listdir(bulk))
    # Per-rank pattern: <safe>.swarm.rank{R:04d}of{S:04d}.h5
    swarm_sidecars = [
        f for f in files
        if re.match(r".*\.swarm\.rank\d{4}of\d{4}\.h5$", f)
    ]
    assert len(swarm_sidecars) >= 1  # at least rank-0 file on single-rank

    with h5py.File(path, "r") as f:
        sw = f["swarms"]
        assert sw.attrs["filled_by"] == "phase3b+phase6"
        assert "mpi_size_at_write" in sw.attrs
        assert len(sw.keys()) == 1
        swarm_safe = list(sw.keys())[0]
        sg = sw[swarm_safe]
        # Wrapper records pattern (not a specific file) — readers
        # fill in their own rank.
        assert "sidecar_pattern" in sg.attrs
        assert "{rank" in str(sg.attrs["sidecar_pattern"])
        assert sg.attrs["mesh_name"] == mesh.name
        assert int(sg.attrs["num_particles_global"]) > 0
        # User-added variables surface in /swarms/{name}/variables/
        assert "material" in sg["variables"]


def test_swarm_sidecar_is_inspectable(tmp_path):
    """The swarm sidecar itself has h5-inspectable structure: top-
    level attrs, /coordinates dataset, /variables/{name} datasets
    with attrs. h5ls -v on the sidecar tells you what it holds."""
    import os
    import h5py
    import underworld3 as uw

    uw, model, mesh, swarm, material = _fresh_model_mesh_swarm()
    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    bulk = str(tmp_path / "run.snap.bulk")
    swarm_sidecar = [
        f for f in os.listdir(bulk) if ".swarm.rank" in f
    ][0]

    with h5py.File(os.path.join(bulk, swarm_sidecar), "r") as f:
        # File-level attrs identify the file without UW3 in the loop.
        assert int(f.attrs["num_particles_local"]) > 0
        assert int(f.attrs["dim"]) == 2
        assert str(f.attrs["mesh_name"]) == mesh.name
        # Bulk in standard h5 places
        assert "coordinates" in f
        assert f["coordinates"].shape[1] == 2
        # User var present with metadata
        v = f["variables"]["material"]
        assert int(v.attrs["num_components"]) == 1


def test_swarm_round_trips_through_disk_snapshot(tmp_path):
    """The whole swarm — particle coords + svar data — round-trips
    exactly through write→scribble→read."""
    import os
    import underworld3 as uw

    uw, model, mesh, swarm, material = _fresh_model_mesh_swarm()
    coords_pre = swarm._particle_coordinates.data.copy()
    material_pre = np.asarray(material.data).copy()

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    # Scribble: corrupt coords directly via the PETSc field, scribble
    # material via the standard array view.
    coord_field = swarm.dm.getField("DMSwarmPIC_coor").reshape(
        (-1, swarm.dim)
    )
    coord_field[...] = -99.0
    swarm.dm.restoreField("DMSwarmPIC_coor")
    material.data[...] = -99.0

    model.load_state(path)

    assert np.array_equal(swarm._particle_coordinates.data, coords_pre), (
        "particle coords not bit-exact after disk-snapshot restore"
    )
    assert np.array_equal(np.asarray(material.data), material_pre), (
        "swarm material not bit-exact after disk-snapshot restore"
    )


def test_swarm_restore_recovers_after_particle_count_change(tmp_path):
    """Mirror of the in-memory rebuild-on-restore guarantee, but on
    disk: snapshot a swarm, mutate its population (add particles),
    restore — the original local population (count + coords + var
    data) is fully reconstructed."""
    import underworld3 as uw

    uw, model, mesh, swarm, material = _fresh_model_mesh_swarm()
    n_pre = swarm.dm.getLocalSize()
    coords_pre = swarm._particle_coordinates.data.copy()
    material_pre = np.asarray(material.data).copy()

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    # Mutate the population — add two extra particles. Local count
    # changes.
    swarm.add_particles_with_coordinates(
        np.array([[0.4, 0.4], [0.6, 0.6]])
    )
    assert swarm.dm.getLocalSize() != n_pre

    model.load_state(path)

    assert swarm.dm.getLocalSize() == n_pre, (
        "restore did not roll back to the captured particle count"
    )
    assert np.array_equal(swarm._particle_coordinates.data, coords_pre)
    assert np.array_equal(np.asarray(material.data), material_pre)


# ----- Phase 4: read_timestep is format-aware -----


def test_read_timestep_reads_v1_1_snapshot_via_dispatch(tmp_path):
    """The selective-read entry point users already know
    (``var.read_timestep(...)``) reads a v1.1 snapshot wrapper too —
    the format detection is hidden inside the call. Same-mesh read
    is bit-exact because the KDTree query lands exactly on the
    captured DOF coordinates."""
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    T.array[:, 0, 0] = 5.0 * T.coords[:, 0] + 3.0
    T_pre = np.asarray(T.array[...]).copy()

    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    # Scribble T, then re-load via the legacy read_timestep API —
    # but pointed at the v1.1 wrapper. Format dispatch handles it.
    T.array[...] = -99.0
    T.read_timestep(path, "T", 0)

    assert np.allclose(np.asarray(T.array[...]), T_pre, atol=1e-12), (
        f"read_timestep against v1.1 snapshot not bit-exact: "
        f"max|d| = {float(np.max(np.abs(np.asarray(T.array[...]) - T_pre))):.3e}"
    )


def test_read_timestep_legacy_path_unchanged(tmp_path):
    """Belt-and-braces: pointing read_timestep at a legacy
    write_timestep file still uses the legacy code path (not the v1.1
    bridge)."""
    import underworld3 as uw

    uw, model, mesh, T, V = _fresh_model_mesh_and_vars()
    T.array[:, 0, 0] = 7.0
    T_pre = np.asarray(T.array[...]).copy()

    # Write via legacy write_timestep — different format, same content.
    mesh.write_timestep(
        "legacy", index=0, outputPath=str(tmp_path), meshVars=[T]
    )

    T.array[...] = -99.0
    T.read_timestep("legacy", "T", 0, outputPath=str(tmp_path))
    assert np.allclose(np.asarray(T.array[...]), T_pre, atol=1e-12)


def test_swarm_internals_not_in_sidecar(tmp_path):
    """The PETSc-internal swarm variables (DMSwarmPIC_coor, DMSwarm_X0,
    DMSwarm_remeshed) are filtered at capture — they're not in the
    sidecar's /variables group. Same rule as in-memory capture."""
    import os
    import h5py
    import underworld3 as uw

    uw, model, mesh, swarm, material = _fresh_model_mesh_swarm()
    path = str(tmp_path / "run.snap.h5")
    model.save_state(file=path)

    bulk = str(tmp_path / "run.snap.bulk")
    sidecar = [f for f in os.listdir(bulk) if ".swarm.rank" in f][0]
    with h5py.File(os.path.join(bulk, sidecar), "r") as f:
        var_names = set(f["variables"].keys())
    assert "material" in var_names
    assert not any(n.startswith("DMSwarm") for n in var_names)
