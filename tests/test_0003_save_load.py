import numpy as np


def test_mesh_save_and_load():
    import underworld3
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0
    )

    mesh.write_timestep("test", meshUpdates=False, outputPath="", index=0)

    mesh1 = underworld3.discretisation.Mesh("test.mesh.00000.h5")

    assert np.fabs(mesh1.get_min_radius() - mesh.get_min_radius()) < 1.0e-5


def test_meshvariable_save_and_read():
    import underworld3
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0
    )

    X = underworld3.discretisation.MeshVariable("X", mesh, 1, degree=2)
    X2 = underworld3.discretisation.MeshVariable("X2", mesh, 1, degree=2)

    with mesh.access(X):
        X.data[:, 0] = X.coords[:, 0]

    mesh.write_timestep("test", meshUpdates=False, meshVars=[X], outputPath="", index=0)

    X2.read_timestep("test.mesh.X.00000.h5", "X")

    with mesh.access():
        assert np.allclose(X.data, X2.data)


def test_swarm_save_and_load():
    import underworld3 as uw
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0
    )

    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=3)
    swarm.write_timestep("test", "swarm", swarmVars=[], outputPath="", index=0)

    new_swarm = uw.swarm.Swarm(mesh)
    new_swarm.read_timestep(f"test.swarm.00000.h5")


def test_swarmvariable_save_and_load():
    from underworld3 import swarm
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0
    )
    swarm = swarm.Swarm(mesh)
    var = swarm.add_variable(name="X", size=1)
    var2 = swarm.add_variable(name="X2", size=1)

    swarm.populate(fill_param=2)

    with swarm.access(var):
        var.data[:, 0] = swarm.data[:, 0]

    swarm.write_timestep("test", "swarm", swarmVars=[var], outputPath="", index=0)

    with swarm.access(var2):
        var2.read_timestep("test.swarm.X.00000.h5", "test.swarm.00000.h5")

    with swarm.access():
        assert np.allclose(var.data, var2.data)
