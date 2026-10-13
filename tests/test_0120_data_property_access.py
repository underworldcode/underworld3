"""Writing a swarm variable through `.data`, twice, and reading back what was written.

This file used to be a converted debug script: the mesh, swarm and writes ran
at module level during pytest COLLECTION, and every statement was wrapped in a
`try/except` that printed the exception. It could not fail — a broken `.data`
property printed a cross and the run stayed green.
"""

import pytest
import numpy as np

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = pytest.mark.level_1


@pytest.fixture(scope="module")
def populated_swarm():
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 8.0,
    )
    swarm = uw.swarm.Swarm(mesh=mesh)
    values = uw.swarm.SwarmVariable("test", swarm, 1, proxy_degree=1)
    swarm.populate(fill_param=2)

    return swarm, values


def test_data_property_writes_and_reads_back(populated_swarm):
    """A write through `.data` is visible on the next read, and is the value written."""

    swarm, values = populated_swarm
    coords = swarm._particle_coordinates.data

    expected = np.cos(np.pi * coords[:, 0])
    values.data[:, 0] = expected

    assert np.allclose(values.data[:, 0], expected)


def test_second_write_replaces_the_first(populated_swarm):
    """The second write is not served a cached copy of the first.

    The cache is the point of the test: `.data` hands out a view whose validity
    is tracked, and a stale view would return the cosine below after the sine
    has been written.
    """

    swarm, values = populated_swarm
    coords = swarm._particle_coordinates.data

    values.data[:, 0] = np.cos(np.pi * coords[:, 0])
    replacement = np.sin(np.pi * coords[:, 1])
    values.data[:, 0] = replacement

    assert np.allclose(values.data[:, 0], replacement)
