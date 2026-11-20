"""``read_timestep`` reads a field back onto a surface mesh (#870).

``MeshVariable.read_timestep`` sized every coordinate array by ``mesh.dim``. On
a surface mesh (``SphericalManifold``: dim 2, cdim 3) the saved coordinates
have three components, so the reshape scrambled them and adding the source
particles raised "must have shape (n, cdim)". The coordinates are now sized by
``mesh.cdim``, which equals ``dim`` on a volume mesh.

A linear field written and read back onto the same mesh must come back
exactly: every saved node coincides with a live one. The check that a saved
cloud has no coincident nodes read it the same way; with an odd node count
that reshape raised, and with an even one it scrambled the rows. Both counts
are run, on a cell size chosen for each.

Under ``mpirun`` every rank reads what rank 0's run wrote: the output directory
is rank 0's ``tmp_path``, and the parity is that of the global node count.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _global_node_count(mesh):
    """Nodes in the whole mesh, on every rank (a rank's own count includes its
    ghosts and differs between ranks)."""
    return mesh.dm.getCoordinateDM().createGlobalVector().getSize() // mesh.cdim


def _sphere_with_node_parity(parity):
    """A SphericalManifold whose node count has the given parity. gmsh builds
    different meshes on different hosts, so the cell size is chosen here. On a
    96-thread Linux workstation (Xeon 6240R) 0.4, 0.3, 0.35, 0.45 and 0.25 give
    even counts (162, 192, 272), and 0.28, 0.27, 0.31 and 0.32 odd ones (227,
    229, 197, 161)."""
    for cell_size in (0.4, 0.3, 0.28, 0.35, 0.27, 0.45, 0.31, 0.25, 0.32):
        mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=cell_size)
        if _global_node_count(mesh) % 2 == parity:
            return mesh
    pytest.skip(f"no candidate cell size gives a node count of parity {parity}")


@pytest.mark.parametrize("parity", [0, 1])
def test_a_field_on_a_spherical_manifold_reads_back_exactly(tmp_path, parity):
    mesh = _sphere_with_node_parity(parity)
    assert _global_node_count(mesh) % 2 == parity
    path = uw.mpi.comm.bcast(str(tmp_path), root=0)    # one directory for every rank
    T = uw.discretisation.MeshVariable("T870", mesh, 1, degree=1)
    X = np.asarray(T.coords)
    T.array[:, 0, 0] = X[:, 0] + 2.0 * X[:, 1] + 3.0 * X[:, 2]
    mesh.write_timestep("m870", index=0, outputPath=path, meshVars=[T])

    R = uw.discretisation.MeshVariable("R870", mesh, 1, degree=1)
    R.read_timestep("m870", "T870", 0, outputPath=path)
    np.testing.assert_allclose(np.asarray(R.array), np.asarray(T.array), rtol=0.0, atol=1.0e-12)
