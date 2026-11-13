"""Field transfers between a moved refinement child and its parent are exact for
linear fields (#1102).

A child that moved parent nodes (``_refine_dofs_coincide = False``: a
conforming-surface cut, a fault, an add-on) is restricted to its parent by
interpolation over the child's DOF cloud. That builder's orphan repair, a
multigrid fix (#424), counted most child DOFs as orphans, because they
outnumber the parent's, and overwrote a parent row with a weight-one injection
for each: on this box, a linear field came back wrong by 0.27 on 16 rows (P1)
and by 0.195 on 39 rows (P2). Both directions now reproduce a linear field to
rounding, through the public ``copy_into`` and ``add_into``.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a, pytest.mark.timeout(300)]


def _linear(X):
    return 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.5


@pytest.fixture(scope="module")
def parent_and_child():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=2)
    surface = uw.meshing.Surface("f1102", mesh, np.array([[0.0, 0.27], [1.0, 0.71]]))
    child = mesh.add_conforming_surface(surface)
    return mesh, child


def test_the_child_moved_parent_nodes(parent_and_child):
    mesh, child = parent_and_child
    assert child.parent is mesh and child._relationship_kind == "refinement"
    assert child._refine_dofs_coincide is False


@pytest.mark.parametrize("degree", [1, 2])
def test_child_to_parent_reproduces_a_linear_field(parent_and_child, degree):
    if uw.mpi.size > 1:
        pytest.skip("the serial interpolation route")
    mesh, child = parent_and_child
    fp = uw.discretisation.MeshVariable(f"rp{degree}", mesh, 1, degree=degree)
    fc = uw.discretisation.MeshVariable(f"rc{degree}", child, 1, degree=degree)
    exact = _linear(np.asarray(fp.coords))
    fc.data[:, 0] = _linear(np.asarray(fc.coords))

    fp.data[:, 0] = 0.0
    fc.copy_into(fp)
    np.testing.assert_allclose(np.asarray(fp.data)[:, 0], exact, rtol=0, atol=1e-12)

    fp.data[:, 0] = 1.0
    fc.add_into(fp)
    np.testing.assert_allclose(np.asarray(fp.data)[:, 0], exact + 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("degree", [1, 2])
def test_parent_to_child_reproduces_a_linear_field(parent_and_child, degree):
    if uw.mpi.size > 1:
        pytest.skip("the serial interpolation route")
    mesh, child = parent_and_child
    fp = uw.discretisation.MeshVariable(f"pp{degree}", mesh, 1, degree=degree)
    fc = uw.discretisation.MeshVariable(f"pc{degree}", child, 1, degree=degree)
    fp.data[:, 0] = _linear(np.asarray(fp.coords))
    fc.data[:, 0] = 0.0
    fp.copy_into(fc)
    np.testing.assert_allclose(np.asarray(fc.data)[:, 0], _linear(np.asarray(fc.coords)),
                               rtol=0, atol=1e-12)
