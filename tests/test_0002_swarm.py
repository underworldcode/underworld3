import pytest
import os


@pytest.fixture
def setup_data():
    print("Build mesh and swarm")
    from underworld3 import swarm
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0
    )

    swarm = swarm.Swarm(mesh)
    yield swarm
    print("Destroy mesh and swarm")
    del swarm
    del mesh
    if os.path.exists("swarm.h5"):
        os.remove("swarm.h5")
    if os.path.exists("var.h5"):
        os.remove("var.h5")


def test_create_swarm(setup_data):
    swarm = setup_data
    swarm.populate(fill_param=3)
    swarm.save("swarm.h5")


def test_create_swarmvariable(setup_data):
    swarm = setup_data
    var = swarm.add_variable(name="test", size=2)

    # Fill param 2 -> 6 particles per triangle
    swarm.populate(fill_param=2)
    with swarm.access():
        shape = var.data.shape

    elements = swarm.mesh._centroids.shape[0]
    var.save("var.h5")
    assert shape == (elements * 6, 2)

def test_addNPoints(setup_data):
    
    from underworld3 import swarm

    swarm2 = setup_data
    var = swarm.SwarmVariable(name = "test", swarm = swarm2, size = 1)
    swarm2.dm.finalizeFieldRegister()

    swarm2.dm.addNPoints(10) # since swarm is initially empty, will add (10 - 1) points
    with swarm2.access():
        npts = swarm2.data.shape[0]
    assert npts == 9

    swarm2.dm.addNPoints(1) # already has particles, so will add 1 point
    with swarm2.access():
        npts = swarm2.data.shape[0]
    assert npts == 10

