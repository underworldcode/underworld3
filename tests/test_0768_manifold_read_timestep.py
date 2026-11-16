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
are run (164 and 195 nodes).
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.mark.parametrize("cell_size", [0.4, 0.3])
def test_a_field_on_a_spherical_manifold_reads_back_exactly(tmp_path, cell_size):
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=cell_size)
    T = uw.discretisation.MeshVariable("T870", mesh, 1, degree=1)
    X = np.asarray(T.coords)
    T.array[:, 0, 0] = X[:, 0] + 2.0 * X[:, 1] + 3.0 * X[:, 2]
    mesh.write_timestep("m870", index=0, outputPath=str(tmp_path), meshVars=[T])

    R = uw.discretisation.MeshVariable("R870", mesh, 1, degree=1)
    R.read_timestep("m870", "T870", 0, outputPath=str(tmp_path))
    np.testing.assert_allclose(np.asarray(R.array), np.asarray(T.array), rtol=0.0, atol=1.0e-12)
