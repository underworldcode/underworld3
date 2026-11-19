"""The level set's interface thickness is the same at any rank count (review of #1234).

``interface_thickness`` carried each cell's thickness to the nodes from the
nearest cell centroid among a rank's own cells, so a node near a partition
boundary took a value that depended on the partition, and the initial level
set with it: its enclosed volume moved by up to 1.3e-3 relative at np 2
(measured in the review of #1234 on an Annulus). The value at a node is now
the mean over every cell holding it, summed over the ranks.

The fixture is the triangle of ``empty_rank_fixtures`` (built from a cell
list, not by gmsh) shared among every rank and graded by squaring one
coordinate, so the cell sizes vary and the mesh is the same on every machine.
Its integrals were measured serially, and the test asserts them at every rank
count.
"""

import numpy as np
import pytest
import underworld3 as uw
from empty_rank_fixtures import right_triangle_of_cells
from underworld3.systems import level_set

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]

# Measured serially on development plus this change (macOS, Apple M4).
EPSILON_INTEGRAL = 0.040477029043130185
VOLUME = 0.13180117601549332


def _graded_triangle():
    dm = right_triangle_of_cells(8, uw.mpi.size)
    coords = dm.getCoordinatesLocal()
    xy = coords.array.reshape(-1, 2)
    xy[:, 0] = xy[:, 0] ** 2                    # cells shrink towards x = 0
    dm.setCoordinatesLocal(coords)
    return uw.discretisation.Mesh(dm, qdegree=3)


def _epsilon_and_volume():
    mesh = _graded_triangle()
    psi = uw.discretisation.MeshVariable("psi_eps", mesh, 1, degree=2)
    eps = level_set.interface_thickness(mesh, psi, scale=2.0)
    X = np.asarray(psi.coords)
    level_set.initialise_psi(psi, eps,
                             signed_distance=0.2 - np.hypot(X[:, 0] - 0.25, X[:, 1] - 0.3))
    epsilon_integral = uw.maths.Integral(mesh, eps.sym[0]).evaluate()
    volume = uw.maths.Integral(mesh, psi.sym[0]).evaluate()
    return float(epsilon_integral), float(volume)


def test_the_interface_thickness_and_the_initial_volume_are_the_serial_ones():
    epsilon_integral, volume = _epsilon_and_volume()
    assert epsilon_integral == pytest.approx(EPSILON_INTEGRAL, rel=1e-12)
    assert volume == pytest.approx(VOLUME, rel=1e-12)


@pytest.mark.skipif(uw.mpi.size < 3, reason="the fixture empties a rank from np 3 only")
def test_a_rank_without_cells_does_not_change_the_thickness():
    # The same triangle with one rank empty and with every rank holding cells,
    # at the same rank count: the empty rank adds nothing to the node sums.
    from empty_rank_fixtures import cells_along_a_side_leaving_one_rank_empty, cells_per_rank

    def integrals(ranks_with_cells, tag):
        dm = right_triangle_of_cells(cells_along_a_side_leaving_one_rank_empty(uw.mpi.size),
                                     ranks_with_cells)
        mesh = uw.discretisation.Mesh(dm, qdegree=3)
        psi = uw.discretisation.MeshVariable(f"psi_eps_{tag}", mesh, 1, degree=2)
        eps = level_set.interface_thickness(mesh, psi, scale=2.0)
        return cells_per_rank(mesh), float(uw.maths.Integral(mesh, eps.sym[0]).evaluate())

    counts_empty, with_empty = integrals(uw.mpi.size - 1, "e")
    counts_full, without = integrals(uw.mpi.size, "f")
    assert 0 in counts_empty and 0 not in counts_full
    assert with_empty == pytest.approx(without, rel=1e-12)
