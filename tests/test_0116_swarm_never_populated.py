"""A swarm that was never populated says so when advected (#702).

Before any particles are added on any rank the DMSwarm size is -1 and the
advection died inside numpy ("negative dimensions are not allowed"). An empty
rank of a populated swarm (size 0) is a different, valid case.

Run: pixi run python -m pytest tests/test_0116_swarm_never_populated.py -v
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_advecting_a_never_populated_swarm_is_a_clear_error():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    x, y = mesh.X
    swarm = uw.swarm.Swarm(mesh)
    assert swarm.local_size < 0
    with pytest.raises(RuntimeError, match="never been populated"):
        swarm.advection(sympy.Matrix([[-y, x]]), 0.01)
    # control: the same swarm, populated, advects
    swarm.populate(fill_param=1)
    before = np.array(swarm.data)
    swarm.advection(sympy.Matrix([[-y, x]]), 0.01)
    assert swarm.local_size > 0 and np.abs(np.asarray(swarm.data) - before).max() > 0
