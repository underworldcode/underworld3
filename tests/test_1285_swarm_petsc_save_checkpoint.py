"""swarm.petsc_save_checkpoint writes on PETSc 3.25, and the swarm reads back.

It called DMSwarm.viewXDMF, which PETSc 3.25 refuses for a swarm with no
active cell DM ("Swarm has no active cell DM"). Underworld's swarms are
DMSWARM_BASIC and have none, so every call failed
(docs/examples/solid_mechanics/advanced/Ex_VP-CM_slabSubduction.py, the
example sweep for #944, W1 2026-11-21). It now writes through write_timestep:
h5 data and an XDMF file at the same name, which read_timestep restores.
"""
import os

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _swarm():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    swarm = uw.swarm.Swarm(mesh)
    M = uw.swarm.SwarmVariable("M", swarm, 1, proxy_degree=0)
    return mesh, swarm, M


def test_the_checkpoint_writes_and_reads_back(tmp_path):
    mesh, swarm, M = _swarm()
    swarm.populate(2)
    M.array[:, 0, 0] = swarm.coords[:, 0] + 2.0 * swarm.coords[:, 1]
    count = swarm.dm.getLocalSize()
    coords = np.array(swarm.coords)
    values = np.array(M.array[:, 0, 0])

    out = str(tmp_path) + os.sep
    swarm.petsc_save_checkpoint(swarmName="swarm", index=3, outputPath=out)
    assert os.path.exists(f"{out}swarm_00003.xmf")

    _, restored, M2 = _swarm()
    restored.read_timestep("swarm", "swarm", 3, outputPath=out)
    M2.read_timestep("swarm", "swarm", "M", 3, outputPath=out)
    assert restored.dm.getLocalSize() == count
    order, back = np.lexsort(coords.T), np.lexsort(np.array(restored.coords).T)
    np.testing.assert_array_equal(np.array(restored.coords)[back], coords[order])
    np.testing.assert_array_equal(np.array(M2.array[:, 0, 0])[back], values[order])
