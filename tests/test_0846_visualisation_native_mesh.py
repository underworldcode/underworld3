"""A continuous P1 field is drawn on the mesh's OWN cells, not a Delaunay of them.

``meshVariable_to_pv_mesh_object`` triangulates a variable's nodal points so that
higher-order fields, whose DOFs the base mesh does not carry, can be plotted at
all. For a continuous P1 field that is the wrong thing to do: the DOFs *are* the
vertices, so the triangulation already exists, and re-deriving it is lossy.

Lossy specifically, not merely wasteful. ``delaunay_2d`` takes one ``alpha`` for
the whole domain and discards triangles whose circumradius exceeds it, so on a
graded mesh it deletes the coarse cells and they render as blank holes in the
middle of the field. That is why the fixture here is GRADED — on a uniform mesh
the two routes agree and the regression cannot be seen.

The point ORDER is asserted as well as the cell count, because the documented
usage attaches values by DOF index::

    pvm = vis.meshVariable_to_pv_mesh_object(T)
    pvm.point_data["T"] = np.asarray(T.data[:, 0])

Returning the right cells with the points in the DM's vertex order instead of the
variable's would draw a plausible-looking field with the values shuffled.
"""
import numpy as np
import pytest

import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.utilities import edge_split

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _graded_mesh():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.3,
        regular=False, qdegree=2)
    dm = base.dm
    for _ in range(20):
        cS, cE = dm.getHeightStratum(0)
        cen = np.array([dm.computeCellGeometryFVM(c)[1] for c in range(cS, cE)])
        d = np.linalg.norm(cen - np.array([0.35, 0.6]), axis=1)
        target = np.where(d < 0.2, 0.03, 0.4)
        sel = np.flatnonzero(edge_split.cell_diameters(dm) > target) + cS
        dm, n = edge_split.bisect_longest_edges(dm, sel)
        if n == 0:
            break
    return uw.discretisation.Mesh(dm, qdegree=2)


def _n_cells(mesh):
    cS, cE = mesh.dm.getHeightStratum(0)
    return cE - cS


def test_p1_uses_the_meshs_own_cells():
    mesh = _graded_mesh()
    p1 = uw.discretisation.MeshVariable("v1", mesh, 1, degree=1,
                                        continuous=True)
    pvm = vis.meshVariable_to_pv_mesh_object(p1)

    assert pvm.n_cells == _n_cells(mesh), (
        "the plotted mesh does not have the mesh's own cells")
    assert pvm.n_points == p1.coords.shape[0]


def test_delaunay_would_drop_cells_on_a_graded_mesh():
    """The control. Without it the test above could pass by coincidence.

    If this stops failing to reproduce the loss, the graded fixture has stopped
    being graded enough and the test above proves nothing.
    """
    mesh = _graded_mesh()
    p1 = uw.discretisation.MeshVariable("v2", mesh, 1, degree=1,
                                        continuous=True)
    cloud = vis.meshVariable_to_pv_cloud(p1)
    pts = np.asarray(cloud.points)
    alpha = (pts.max() - pts.min()) / max(10, len(pts) ** 0.5) * 2.0
    dropped = _n_cells(mesh) - cloud.delaunay_2d(alpha=alpha).n_cells
    assert dropped > 0, (
        "the Delaunay route loses no cells on this fixture, so it is not "
        "grading strongly enough to exercise the regression")


def test_values_line_up_with_the_points():
    """Attaching data by DOF index must land on the right vertices."""
    mesh = _graded_mesh()
    p1 = uw.discretisation.MeshVariable("v3", mesh, 1, degree=1,
                                        continuous=True)
    coords = np.asarray(p1.coords)
    p1.array[:, 0, 0] = coords[:, 0] + 2.0 * coords[:, 1]

    pvm = vis.meshVariable_to_pv_mesh_object(p1)
    pvm.point_data["f"] = np.asarray(p1.data[:, 0]).reshape(-1)
    exact = pvm.points[:, 0] + 2.0 * pvm.points[:, 1]

    assert np.abs(pvm.point_data["f"] - exact).max() == pytest.approx(0.0,
                                                                     abs=1e-12)


@pytest.mark.parametrize("degree,continuous", [(2, True), (0, False)])
def test_other_spaces_still_take_the_delaunay_route(degree, continuous):
    """Higher-order and discontinuous fields have no native triangulation.

    Their DOFs are not the vertices, so the mesh's cells cannot carry them and
    the helper must decline rather than return something the wrong shape.
    """
    mesh = _graded_mesh()
    var = uw.discretisation.MeshVariable(f"v4_{degree}", mesh, 1, degree=degree,
                                         continuous=continuous)
    assert vis.meshVariable_to_native_pv_mesh(var) is None
    assert vis.meshVariable_to_pv_mesh_object(var).n_points > 0
