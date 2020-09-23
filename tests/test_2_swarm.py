import pytest

def test_create_swarm():
    from underworld3 import Swarm, Mesh
    mesh = Mesh(elementRes=(10,10), minCoords=(0., 0.), maxCoords=(1.0, 1.0))
    swarm = Swarm(mesh)
    swarm.populate(ppcell=25)
    swarm.save("swarm.xmf")

def test_create_swarmvariable():
    from underworld3 import Swarm, Mesh
    mesh = Mesh(elementRes=(10,10), minCoords=(0., 0.), maxCoords=(1.0, 1.0))
    swarm = Swarm(mesh)
    var = swarm.add_variable(name='test', num_components=2)
    swarm.populate(ppcell=25)
    swarm.save("swarm.xmf")
