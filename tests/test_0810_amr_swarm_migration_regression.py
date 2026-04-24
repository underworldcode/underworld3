"""Regression test for issue #135.

After mesh.adapt(), swarm migration into the new mesh raised IndexError in
Mesh._test_if_points_in_cells_internal because the per-cell face control-point
caches (faces_outer_control_points / faces_inner_control_points) were not
invalidated by nuke_coords_and_rebuild. Cell IDs from the new mesh would index
into stale arrays sized for the old mesh's cell count.

Reproducer reduced from the user's report (bknight1, 2026-04-23). Skipped if
the running PETSc was not built with the pragmatic backend.
"""

import pytest
import numpy as np
import sympy

import underworld3 as uw
from test_0800_optional_modules import requires_amr


pytestmark = [pytest.mark.level_1, requires_amr]


@requires_amr
def test_swarm_migration_after_adapt_does_not_raise():
    """mesh.adapt() then swarm._force_migration_after_mesh_change() must not
    raise IndexError on the failing case from issue #135.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        regular=False,
        qdegree=3,
    )

    swarm = uw.swarm.Swarm(mesh=mesh)
    swarm.populate(fill_param=4)

    cells_before = mesh.dm.getHeightStratum(0)[1]

    x, h_min, h_max, sigma = sympy.symbols("x h_min h_max sigma", positive=True)
    h_expr = h_min + (h_max - h_min) * (1 - sympy.exp(-((x / sigma) ** 2)))
    h_subbed = h_expr.subs({h_min: 0.05, h_max: 0.5, sigma: 0.25})
    h_vals = uw.function.evaluate(h_subbed, mesh.X.coords)

    metric = uw.adaptivity.create_metric(mesh, h_vals)
    mesh.adapt(metric)

    cells_after = mesh.dm.getHeightStratum(0)[1]
    assert cells_after != cells_before, (
        "test premise: mesh.adapt should change cell count for this metric"
    )

    # Pre-fix: this line raises
    #   IndexError: index N is out of bounds for axis 1 with size M
    # because faces_*_control_points kept the old (cells_before-sized) arrays.
    swarm._force_migration_after_mesh_change()
