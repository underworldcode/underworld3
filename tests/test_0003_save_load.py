import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
import h5py
import numpy as np
from pathlib import Path


def _shared_tmp_path(tmp_path, uw):
    """Use one rank-0 pytest tmp path for collective MPI I/O tests."""

    shared_path = str(tmp_path) if uw.mpi.rank == 0 else None
    shared_path = uw.mpi.comm.bcast(shared_path, root=0)
    shared_path = Path(shared_path)
    if uw.mpi.rank == 0:
        shared_path.mkdir(parents=True, exist_ok=True)
    uw.mpi.barrier()
    return shared_path


def test_mesh_save_and_load(tmp_path):
    import underworld3
    from underworld3.meshing import UnstructuredSimplexBox

    tmp_path = _shared_tmp_path(tmp_path, underworld3)

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0)

    mesh.write_timestep("test", meshUpdates=False, outputPath=tmp_path, index=0)

    mesh1 = underworld3.discretisation.Mesh(f"{tmp_path}/test.mesh.00000.h5")

    assert np.fabs(mesh1.get_min_radius() - mesh.get_min_radius()) < 1.0e-5


def test_meshvariable_save_and_read(tmp_path):
    import underworld3
    from underworld3.meshing import UnstructuredSimplexBox

    tmp_path = _shared_tmp_path(tmp_path, underworld3)

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0)

    X = underworld3.discretisation.MeshVariable("X", mesh, 1, degree=2)
    X2 = underworld3.discretisation.MeshVariable("X2", mesh, 1, degree=2)

    X.array[:, 0, 0] = X.coords[:, 0]

    mesh.write_timestep("test", meshUpdates=False, meshVars=[X], outputPath=tmp_path, index=0)

    X2.read_timestep("test", "X", 0, outputPath=tmp_path)

    assert np.allclose(X.array, X2.array)


def test_meshvariable_checkpoint_roundtrip(tmp_path):
    import underworld3 as uw
    from underworld3.meshing import UnstructuredSimplexBox

    tmp_path = _shared_tmp_path(tmp_path, uw)

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 8.0,
    )

    x = uw.discretisation.MeshVariable("x", mesh, 1, degree=1)
    u = uw.discretisation.MeshVariable("u", mesh, 2, degree=2)
    d = uw.discretisation.MeshVariable("d", mesh, 1, degree=1, continuous=False)

    x.data[:, 0] = x.coords[:, 0] + 2.0 * x.coords[:, 1]
    u.data[:, 0] = 3.0 * u.coords[:, 0] - u.coords[:, 1]
    u.data[:, 1] = u.coords[:, 0] + 4.0 * u.coords[:, 1]
    d.data[:, 0] = 5.0 * d.coords[:, 0] + 7.0 * d.coords[:, 1]

    def assert_reloaded_fields(x_reloaded, u_reloaded, d_reloaded):
        # Parallel DMPlex reloads may repartition the mesh, so compare against the
        # defining functions on the reloaded coordinates rather than local row order.
        np.testing.assert_allclose(
            x_reloaded.data[:, 0],
            x_reloaded.coords[:, 0] + 2.0 * x_reloaded.coords[:, 1],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            u_reloaded.data[:, 0],
            3.0 * u_reloaded.coords[:, 0] - u_reloaded.coords[:, 1],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            u_reloaded.data[:, 1],
            u_reloaded.coords[:, 0] + 4.0 * u_reloaded.coords[:, 1],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            d_reloaded.data[:, 0],
            5.0 * d_reloaded.coords[:, 0] + 7.0 * d_reloaded.coords[:, 1],
            atol=1.0e-12,
        )

    checkpoint_base = tmp_path / "restart"
    with pytest.warns(FutureWarning, match="write_checkpoint\\(\\) is deprecated"):
        mesh.write_checkpoint(
            "restart",
            outputPath=str(tmp_path),
            meshUpdates=False,
            meshVars=[x, u, d],
            index=0,
            separate_variable_files=False,
        )

    mesh_reloaded = uw.discretisation.Mesh(f"{checkpoint_base}.mesh.00000.h5")
    x_reloaded = uw.discretisation.MeshVariable("x", mesh_reloaded, 1, degree=1)
    u_reloaded = uw.discretisation.MeshVariable("u", mesh_reloaded, 2, degree=2)
    d_reloaded = uw.discretisation.MeshVariable("d", mesh_reloaded, 1, degree=1, continuous=False)

    x_reloaded.read_checkpoint(f"{checkpoint_base}.checkpoint.00000.h5", data_name="x")
    u_reloaded.read_checkpoint(f"{checkpoint_base}.checkpoint.00000.h5", data_name="u")
    d_reloaded.read_checkpoint(f"{checkpoint_base}.checkpoint.00000.h5", data_name="d")

    assert_reloaded_fields(x_reloaded, u_reloaded, d_reloaded)

    separate_base = tmp_path / "restart_separate"
    with pytest.warns(FutureWarning, match="write_checkpoint\\(\\) is deprecated"):
        mesh.write_checkpoint(
            "restart_separate",
            outputPath=str(tmp_path),
            meshUpdates=False,
            meshVars=[x, u, d],
            index=0,
        )

    mesh_reloaded = uw.discretisation.Mesh(f"{separate_base}.mesh.00000.h5")
    x_reloaded = uw.discretisation.MeshVariable("x", mesh_reloaded, 1, degree=1)
    u_reloaded = uw.discretisation.MeshVariable("u", mesh_reloaded, 2, degree=2)
    d_reloaded = uw.discretisation.MeshVariable("d", mesh_reloaded, 1, degree=1, continuous=False)

    x_reloaded.read_checkpoint(f"{separate_base}.x.00000.h5", data_name="x")
    u_reloaded.read_checkpoint(f"{separate_base}.u.00000.h5", data_name="u")
    d_reloaded.read_checkpoint(f"{separate_base}.d.00000.h5", data_name="d")

    assert_reloaded_fields(x_reloaded, u_reloaded, d_reloaded)


def test_timestep_with_petsc_reload_roundtrip(tmp_path):
    import underworld3 as uw
    from underworld3.meshing import UnstructuredSimplexBox

    tmp_path = _shared_tmp_path(tmp_path, uw)

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 8.0,
    )

    x = uw.discretisation.MeshVariable("x", mesh, 1, degree=1)
    x.data[:, 0] = x.coords[:, 0] + 2.0 * x.coords[:, 1]

    mesh.write_timestep(
        "unified",
        index=0,
        outputPath=str(tmp_path),
        meshVars=[x],
        create_xdmf=True,
        petsc_reload=True,
    )

    var_file = tmp_path / "unified.mesh.x.00000.h5"
    if uw.mpi.rank == 0:
        with h5py.File(var_file, "r") as h5f:
            assert "fields/x" in h5f
            assert "fields/coordinates" in h5f
            assert "vertex_fields/x_x" in h5f
            assert "topologies/uw_mesh/dms/x/section" in h5f
            assert "topologies/uw_mesh/dms/x/vecs/x" in h5f
    uw.mpi.barrier()

    mesh_reloaded = uw.discretisation.Mesh(f"{tmp_path}/unified.mesh.00000.h5")
    x_checkpoint = uw.discretisation.MeshVariable("x", mesh_reloaded, 1, degree=1)
    x_checkpoint.read_checkpoint(str(var_file), data_name="x")
    np.testing.assert_allclose(
        x_checkpoint.data[:, 0],
        x_checkpoint.coords[:, 0] + 2.0 * x_checkpoint.coords[:, 1],
        atol=1.0e-12,
    )

    x_timestep = uw.discretisation.MeshVariable("x_timestep", mesh, 1, degree=1)
    x_timestep.read_timestep("unified", "x", 0, outputPath=str(tmp_path))
    np.testing.assert_allclose(x_timestep.data[:, 0], x.data[:, 0], atol=1.0e-12)

    with pytest.warns(FutureWarning, match="write_checkpoint\\(\\) is deprecated"):
        mesh.write_checkpoint(
            "unified_checkpoint",
            index=0,
            outputPath=str(tmp_path),
            meshVars=[x],
            create_xdmf=True,
        )

    checkpoint_var_file = tmp_path / "unified_checkpoint.mesh.x.00000.h5"
    checkpoint_xdmf_file = tmp_path / "unified_checkpoint.mesh.00000.xdmf"
    if uw.mpi.rank == 0:
        assert checkpoint_xdmf_file.is_file()
        with h5py.File(checkpoint_var_file, "r") as h5f:
            assert "fields/x" in h5f
            assert "vertex_fields/x_x" in h5f
            assert "topologies/uw_mesh/dms/x/section" in h5f
    uw.mpi.barrier()

    mesh_reloaded = uw.discretisation.Mesh(
        f"{tmp_path}/unified_checkpoint.mesh.00000.h5"
    )
    x_checkpoint = uw.discretisation.MeshVariable("x", mesh_reloaded, 1, degree=1)
    x_checkpoint.read_checkpoint(str(checkpoint_var_file), data_name="x")
    np.testing.assert_allclose(
        x_checkpoint.data[:, 0],
        x_checkpoint.coords[:, 0] + 2.0 * x_checkpoint.coords[:, 1],
        atol=1.0e-12,
    )


def test_swarm_save_and_load(tmp_path):
    import underworld3 as uw
    from underworld3.meshing import UnstructuredSimplexBox

    tmp_path = _shared_tmp_path(tmp_path, uw)

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0)

    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=3)
    swarm.write_timestep("test", "swarm", swarmVars=[], outputPath=tmp_path, index=0)

    new_swarm = uw.swarm.Swarm(mesh)
    new_swarm.read_timestep("test", "swarm", 0, outputPath=tmp_path)

    # Restore must reproduce the checkpoint exactly once — the parallel
    # np-fold duplication regression (issue #324) is covered at np2/np4 by
    # tests/parallel/test_0757_swarm_read_timestep_mpi.py.
    saved_count = uw.mpi.comm.allreduce(max(swarm.dm.getLocalSize(), 0))
    restored_count = uw.mpi.comm.allreduce(max(new_swarm.dm.getLocalSize(), 0))
    assert restored_count == saved_count


def test_swarmvariable_save_and_load(tmp_path):
    import underworld3 as uw
    from underworld3 import swarm
    from underworld3.meshing import UnstructuredSimplexBox

    tmp_path = _shared_tmp_path(tmp_path, uw)

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0)
    swarm = swarm.Swarm(mesh)
    var = swarm.add_variable(name="X", size=1)
    var2 = swarm.add_variable(name="X2", size=1)

    swarm.populate(fill_param=2)

    var.array[:, 0, 0] = swarm._particle_coordinates.data[:, 0]

    swarm.write_timestep("test", "swarm", swarmVars=[var], outputPath=tmp_path, index=0)

    var2.read_timestep("test", "swarm", "X", 0, outputPath=tmp_path)

    assert np.allclose(var.array, var2.array)


@pytest.mark.tier_a
def test_write_timestep_missing_directory_raises(tmp_path):
    """write_timestep fails loudly when the output directory is absent.

    Regression for D-41/READ-83: the existence / write-access checks were
    written as inverted no-op guards (`if ok: pass else: raise`); the
    rewrite must still raise RuntimeError with the offending absolute path.
    """
    import underworld3 as uw

    # 4x4 keeps every rank populated up to np4 — a 2x2 mesh starves ranks
    # of cells and crashes in mesh construction (issue #399), which then
    # desynchronises every test after this one in an MPI run.
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(RuntimeError, match="does not exist"):
        mesh.write_timestep("chk", index=0, outputPath=str(missing_dir / "run"))


def test_dg0_read_timestep_across_units_sessions(tmp_path):
    """#269 regression: a DG0 field written under an active units model must
    read back correctly in a session WITHOUT that model.

    The remap contract stores model-frame coordinates (see
    ``MeshVariable.write``). The old writer stored ``.coords`` — dimensional
    under an active units model — so a units-free reading session queried in
    the model frame against a dimensionally-scaled file cloud and got a
    silently wrong, near-constant field (the Spiegelman ``edot`` symptom).
    """
    import underworld3 as uw

    tmp_path = _shared_tmp_path(tmp_path, uw)

    uw.reset_default_model()
    orchestration_model = uw.get_default_model()
    orchestration_model.set_reference_quantities(
        length=uw.quantity(30, "km"),
        velocity=uw.quantity(2.5, "cm/year"),
        viscosity=uw.quantity(1e21, "Pa*s"),
    )
    try:
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
            units="metre")
        v = uw.discretisation.MeshVariable("edot", mesh, 1, degree=0, continuous=False)
        c_nd = np.asarray(v.coords_nd)
        v.data[:, 0] = np.sin(3.0 * c_nd[:, 0]) + c_nd[:, 1]
        wrote = np.asarray(v.data)[:, 0].copy()

        mesh.write_timestep("chk269", index=0, outputPath=tmp_path, meshVars=[v])
    finally:
        # The reading "session" has no units model — the failing benchmark
        # context (post-processing without set_reference_quantities).
        uw.reset_default_model()

    mesh2 = uw.discretisation.Mesh(f"{tmp_path}/chk269.mesh.00000.h5")
    w = uw.discretisation.MeshVariable("edot", mesh2, 1, degree=0, continuous=False)
    w.read_timestep("chk269", "edot", 0, outputPath=tmp_path)
    got = np.asarray(w.data)[:, 0]

    # The two meshes partition differently, so compare GLOBAL statistics of
    # the per-cell value multiset (identical cells, exact nearest match).
    def _global_stats(a):
        from mpi4py import MPI

        n = uw.mpi.comm.allreduce(int(a.size), op=MPI.SUM)
        s = uw.mpi.comm.allreduce(float(a.sum()), op=MPI.SUM)
        ss = uw.mpi.comm.allreduce(float((a * a).sum()), op=MPI.SUM)
        mn = uw.mpi.comm.allreduce(float(a.min()) if a.size else np.inf, op=MPI.MIN)
        mx = uw.mpi.comm.allreduce(float(a.max()) if a.size else -np.inf, op=MPI.MAX)
        return n, s, ss, mn, mx

    n_w, s_w, ss_w, mn_w, mx_w = _global_stats(wrote)
    n_g, s_g, ss_g, mn_g, mx_g = _global_stats(got)

    assert n_g == n_w
    assert np.isclose(s_g, s_w, rtol=1e-10) and np.isclose(ss_g, ss_w, rtol=1e-10), (
        f"DG0 field corrupted on cross-session read: global (sum, sum-sq) "
        f"wrote ({s_w}, {ss_w}) vs read ({s_g}, {ss_g})"
    )
    assert np.isclose(mn_g, mn_w) and np.isclose(mx_g, mx_w)


def test_read_timestep_refuses_frame_mismatched_file(tmp_path):
    """#269 guard: a file whose saved coordinate cloud is on a wildly
    different scale from the live DOFs (a legacy dimensional-frame
    checkpoint) must raise, not silently return nearest-garbage."""
    import underworld3 as uw

    tmp_path = _shared_tmp_path(tmp_path, uw)

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    v = uw.discretisation.MeshVariable("phi269", mesh, 1, degree=0, continuous=False)
    v.data[:, 0] = np.asarray(v.coords_nd)[:, 0]
    mesh.write_timestep("chk269g", index=0, outputPath=tmp_path, meshVars=[v])

    # Emulate the legacy writer: scale the stored remap coordinates the way
    # an active units model did (metres for a km-scaled domain).
    if uw.mpi.rank == 0:
        with h5py.File(f"{tmp_path}/chk269g.mesh.phi269.00000.h5", "r+") as f:
            f["fields/coordinates"][...] = f["fields/coordinates"][...] * 30000.0
    uw.mpi.barrier()

    w = uw.discretisation.MeshVariable("phi269b", mesh, 1, degree=0, continuous=False)
    with pytest.raises(RuntimeError, match="coordinate frames do not match"):
        w.read_timestep("chk269g", "phi269", 0, outputPath=tmp_path)


def test_mesh_from_msh_file_writes(tmp_path):
    """#397: a mesh loaded directly from a .msh file must be writable.

    The coordinate system defaults to CARTESIAN (matching every uw.meshing
    constructor) instead of leaving ``CoordinateSystemType=None`` and
    crashing ``mesh.write()``'s metadata block.
    """
    import underworld3 as uw

    tmp_path = _shared_tmp_path(tmp_path, uw)

    msh = f"{tmp_path}/box397.msh"
    scratch = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, filename=msh)
    del scratch

    mesh = uw.discretisation.Mesh(msh)
    assert mesh.CoordinateSystemType is uw.coordinates.CoordinateSystemType.CARTESIAN
    mesh.write(f"{tmp_path}/box397.h5")
