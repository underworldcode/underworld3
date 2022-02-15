
def test_create_swarm():
    from underworld3 import swarm, meshes
    mesh = meshes.Box(elementRes=(10,10), minCoords=(0., 0.), maxCoords=(1.0, 1.0))
    swarm = swarm.Swarm(mesh)
    swarm.populate(fill_param=5)
    #swarm.save("swarm.xmf")

def test_create_swarmvariable():
    from underworld3 import swarm, meshes
    mesh = meshes.Box(elementRes=(10,10), minCoords=(0., 0.), maxCoords=(1.0, 1.0))
    swarm = swarm.Swarm(mesh)
    var = swarm.add_variable(name='test', num_components=2)
    swarm.populate(fill_param=2)
    #swarm.save("swarm.xmf")
