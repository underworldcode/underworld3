import numpy as np


def test_mesh_save_and_load(tmp_path):
    import underworld3
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0
    )

    mesh.write_timestep("test", meshUpdates=False, outputPath=tmp_path, index=0)

    mesh1 = underworld3.discretisation.Mesh(f"{tmp_path}/test.mesh.00000.h5")

    assert np.fabs(mesh1.get_min_radius() - mesh.get_min_radius()) < 1.0e-5


def test_meshvariable_save_and_read(tmp_path):
    import underworld3
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0
    )

    X = underworld3.discretisation.MeshVariable("X", mesh, 1, degree=2)
    X2 = underworld3.discretisation.MeshVariable("X2", mesh, 1, degree=2)

    X.array[:, 0, 0] = X.coords[:, 0]

    mesh.write_timestep(
        "test", meshUpdates=False, meshVars=[X], outputPath=tmp_path, index=0
    )

    X2.read_timestep("test", "X", 0, outputPath=tmp_path)

    assert np.allclose(X.array, X2.array)


def test_swarm_save_and_load(tmp_path):
    import underworld3 as uw
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0
    )

    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=3)
    swarm.write_timestep("test", "swarm", swarmVars=[], outputPath=tmp_path, index=0)

    new_swarm = uw.swarm.Swarm(mesh)
    new_swarm.read_timestep("test", "swarm", 0, outputPath=tmp_path)


def test_swarmvariable_save_and_load(tmp_path):
    from underworld3 import swarm
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0
    )
    swarm = swarm.Swarm(mesh)
    var = swarm.add_variable(name="X", size=1)
    var2 = swarm.add_variable(name="X2", size=1)

    swarm.populate(fill_param=2)

    var.array[:, 0, 0] = swarm._particle_coordinates.data[:, 0]

    swarm.write_timestep("test", "swarm", swarmVars=[var], outputPath=tmp_path, index=0)

    var2.read_timestep("test", "swarm", "X", 0, outputPath=tmp_path)

    assert np.allclose(var.array, var2.array)
