"""``plot_scalar`` and ``plot_vector`` on a rank that holds no cells (#957, #1199).

``mesh_to_pv_mesh`` built its cell table with ``np.array`` of the cells' vertex
lists. On a rank with no cells the list is empty, the array has shape ``(0,)``,
and pyvista raised before the evaluation that the plot functions then make on
every rank, which is collective: the other ranks waited in it and the job hung
(measured at np 2 and np 3 on a 96-thread Linux workstation, MPICH).

The mesh is the right triangle of ``empty_rank_fixtures``, which leaves one rank
without cells (which one is PETSc's choice). Each rank checks that both plot functions return, and that
the rank with cells hands its cells to ``add_mesh``, through a recording
stand-in for ``pyvista.Plotter`` (nothing is rendered, as in ``test_0849``).
Serial and parallel alike: in serial no rank is empty.
"""

import numpy as np
import pytest

import underworld3 as uw
from empty_rank_fixtures import (cells_along_a_side_leaving_one_rank_empty, cells_per_rank,
                                 right_triangle_of_cells)

pv = pytest.importorskip("pyvista")
vis = pytest.importorskip("underworld3.visualisation")

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(120)]


class _RecordingPlotter:
    """Stands in for ``pyvista.Plotter``: keeps the meshes ``add_mesh`` is given."""

    meshes = []

    def __init__(self, *args, **kwargs):
        self.camera = self

    def add_mesh(self, grid, *args, **kwargs):
        _RecordingPlotter.meshes.append(grid)

    def _nothing(self, *args, **kwargs):
        return None

    show = add_text = add_arrows = screenshot = zoom = _nothing


@pytest.fixture
def triangle(monkeypatch):
    size = uw.mpi.size
    dm = right_triangle_of_cells(cells_along_a_side_leaving_one_rank_empty(size),
                                 max(size - 1, 1))
    mesh = uw.discretisation.Mesh(dm, degree=1, qdegree=3)
    _RecordingPlotter.meshes = []
    monkeypatch.setattr(pv, "Plotter", _RecordingPlotter)
    return mesh


def test_plot_scalar_and_plot_vector_return_on_a_rank_without_cells(triangle):
    mesh = triangle
    counts = cells_per_rank(mesh)
    if uw.mpi.size > 1:
        assert 0 in counts                           # the fixture leaves one rank empty
    T = uw.discretisation.MeshVariable("T_empty_rank", mesh, 1, degree=1)
    V = uw.discretisation.MeshVariable("V_empty_rank", mesh, 2, degree=1)
    X = np.asarray(T.coords)
    T.array[:, 0, 0] = 1.0 + X[:, 0]
    V.array[:, 0, 0] = 1.0

    vis.plot_scalar(mesh, T.sym[0], scalar_name="T")
    vis.plot_vector(mesh, V, vector_name="V")

    local = counts[uw.mpi.rank]
    drawn = [grid.n_cells for grid in _RecordingPlotter.meshes]
    assert drawn == ([local, local] if local else [])
