import pytest

def test_create_swarm():
    from underworld3 import Swarm, Mesh
    mesh = Mesh(elementRes=(10,10), minCoords=(0., 0.), maxCoords=(1.0, 1.0))
    swarm = Swarm(mesh)
    swarm.save("swarm.xmf")
