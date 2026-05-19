import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
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
