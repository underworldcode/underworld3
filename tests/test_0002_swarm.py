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
    swarm.populate(fill_param=2)
    with swarm.access():
        shape = var.data.shape
    var.save("var.h5")
    assert shape == (7200, 2)
 
