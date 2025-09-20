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
    shape = var.data.shape

    elements = swarm.mesh._centroids.shape[0]
    var.save("var.h5")
    assert shape == (elements * 6, 2)


def test_addNPoints(setup_data):

    from underworld3 import swarm

    swarm2 = setup_data
    var = swarm.SwarmVariable(name="test", swarm=swarm2, size=1)
    swarm2.dm.finalizeFieldRegister()

    swarm2.dm.addNPoints(10)  # since swarm is initially empty, will add (10 - 1) points
    npts = swarm2.local_size
    assert npts == 9

    swarm2.dm.addNPoints(1)  # already has particles, so will add 1 point
    npts = swarm2.local_size
    assert npts == 10


def test_particle_position_setter(setup_data):
    import numpy as np

    swarm = setup_data
    swarm.populate(fill_param=2)
    swarm.clip_to_mesh = False

    # Get original positions
    original_positions = swarm.points[...].copy()
    npts = swarm.points[...].shape[0]

    # Create new positions (shift all particles by 0.1 in x and y)
    new_positions = original_positions + 10.0

    # Test the data setter (be careful that we don't delete the moved points)
    swarm.points = new_positions
    updated_positions = swarm.points[...]

    # Verify the positions were updated correctly
    np.testing.assert_allclose(updated_positions, new_positions, rtol=1e-15)
    assert updated_positions.shape == (npts, 2)

    # Verify positions actually changed
    assert not np.allclose(original_positions, updated_positions)


def test_particle_clip_context_manager(setup_data):
    import numpy as np

    swarm = setup_data
    swarm.populate(fill_param=2)

    # Get original positions
    original_positions = swarm.points.copy()
    npts0 = swarm.points.shape[0]

    # Create new positions (shift all particles by 0.1 in x and y)
    new_positions = original_positions + 10.0

    # Test the data setter (be careful that we don't delete the moved points)
    with swarm.dont_clip_to_mesh():
        # swarm.points[...] = new_positions[...]
        swarm.points = new_positions
        npts1 = swarm.points.shape[0]
        assert npts1 == npts0

    npts1 = swarm.points.shape[0]
    assert npts1 == 0
