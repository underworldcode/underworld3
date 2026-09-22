"""A mesh read from a file has no analytic closure for return_coords_to_bounds.

Every gmsh geometry (the benchmark meshes) is read from a file. Before the fix the
property returned None for them, so a trace-back foot leaving through an inlet was
never restored to the boundary and fell through to the evaluator's distance-weighted
fallback. The general facet restore already handles the case and leaves interior
points untouched, so it is the fallback.
"""
import os
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_a_file_mesh_restores_outside_points_to_its_boundary(tmp_path):
    path = os.path.join(str(tmp_path), "box.msh")
    uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                      cellSize=0.25, filename=path)
    mesh = uw.discretisation.Mesh(path)
    assert mesh._analytic_return_coords_to_bounds is None
    restore = mesh.return_coords_to_bounds
    assert restore is not None

    pts = np.array([[1.3, 0.5], [0.5, 0.5], [-0.2, 1.4]])
    out = np.asarray(restore(pts))
    # outside points land on the boundary (the facet restore steps 1e-4 inside)
    assert abs(out[0, 0] - 1.0) < 1.0e-3 and abs(out[0, 1] - 0.5) < 1.0e-9
    assert abs(out[2, 0] - 0.0) < 1.0e-3 and abs(out[2, 1] - 1.0) < 1.0e-3
    # an interior point is untouched
    assert np.array_equal(out[1], [0.5, 0.5])
