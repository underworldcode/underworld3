"""A remapped variable is written in its own data layout.

global_evaluate returns the full array per node. A symmetric tensor stores only its
independent components, so the remap's reshape raised and left the move part done, and a
matrix stores by columns, so the reshape silently transposed it. See #977.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _linear_tensor(coords, shape):
    """A symmetric tensor linear in position, with distinct components."""
    x, y = coords[:, 0], coords[:, 1]
    values = np.zeros(shape)
    values[:, 0, 0] = 2.0 + x
    values[:, 1, 1] = 3.0 + 2.0 * y
    values[:, 0, 1] = values[:, 1, 0] = 5.0 + x - y
    return values


def test_a_symmetric_tensor_is_remapped_across_a_move():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    S = uw.discretisation.MeshVariable("S0769", mesh, vtype=uw.VarType.SYM_TENSOR, degree=1)
    S.array[...] = _linear_tensor(np.asarray(S.coords), np.asarray(S.array).shape)
    start = np.array(S.array)

    coords = np.array(mesh.X.coords)
    radius = np.linalg.norm(coords, axis=1)
    inside = (radius > 0.5 + 1.0e-9) & (radius < 1.0 - 1.0e-9)
    coords[inside] *= 1.02
    assert mesh.deform(coords) is True

    remapped = np.asarray(S.array)
    assert np.abs(remapped - start).max() > 1.0e-3      # the remap did something
    exact = _linear_tensor(np.asarray(S.coords), remapped.shape)
    interior = np.linalg.norm(np.asarray(S.coords), axis=1) < 0.99
    assert np.abs(remapped - exact)[interior].max() < 1.0e-9     # P1 is exact on a linear field
    assert np.abs(remapped[:, 0, 1] - remapped[:, 1, 0]).max() < 1.0e-12


@pytest.mark.parametrize("vtype", [uw.VarType.TENSOR, uw.VarType.MATRIX])
def test_a_full_array_keeps_its_components_across_a_move(vtype):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    kwargs = dict(varsymbol="M") if vtype == uw.VarType.MATRIX else {}
    size = 2
    M = uw.discretisation.MeshVariable(f"M0769{vtype.name}", mesh, (size, size), vtype=vtype,
                                       degree=1, **kwargs)

    def linear(coords, shape):
        x, y = coords[:, 0], coords[:, 1]
        values = np.zeros(shape)
        values[:, 0, 0] = 2.0 + x
        values[:, 0, 1] = 3.0 + 2.0 * y
        values[:, 1, 0] = 5.0 + x - y
        values[:, 1, 1] = 7.0 - y
        return values

    M.array[...] = linear(np.asarray(M.coords), np.asarray(M.array).shape)
    coords = np.array(mesh.X.coords)
    radius = np.linalg.norm(coords, axis=1)
    coords[(radius > 0.5 + 1.0e-9) & (radius < 1.0 - 1.0e-9)] *= 1.02
    assert mesh.deform(coords) is True

    remapped = np.asarray(M.array)
    exact = linear(np.asarray(M.coords), remapped.shape)
    interior = np.linalg.norm(np.asarray(M.coords), axis=1) < 0.99
    assert np.abs(remapped - exact)[interior].max() < 1.0e-9


def test_a_swarm_matrix_reads_back_what_was_written():
    """global_evaluate carries its answer on a MATRIX swarm variable, so a variable that
    did not round-trip transposed every non-symmetric tensor it returned."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 4, qdegree=3)
    swarm = uw.swarm.Swarm(mesh)
    D = uw.swarm.SwarmVariable("D0769", swarm, vtype=uw.VarType.MATRIX, size=(2, 2),
                               dtype=float, _proxy=False)
    swarm.populate(fill_param=1)
    values = np.zeros(np.asarray(D.array).shape)
    values[:, 0, 0], values[:, 0, 1], values[:, 1, 0], values[:, 1, 1] = 1.0, 2.0, 3.0, 4.0
    D.array[...] = values
    assert np.array_equal(np.asarray(D.array), values)


def test_global_evaluate_returns_a_tensor_in_the_order_evaluate_does():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 4, qdegree=3)
    M = uw.discretisation.MeshVariable("M0769g", mesh, (2, 2), vtype=uw.VarType.TENSOR, degree=1)
    values = np.zeros(np.asarray(M.array).shape)
    values[:, 0, 0], values[:, 0, 1], values[:, 1, 0], values[:, 1, 1] = 1.0, 2.0, 3.0, 4.0
    M.array[...] = values
    points = np.asarray(M.coords)[:3]
    assert np.array_equal(np.asarray(uw.function.global_evaluate(M.sym, points)),
                          np.asarray(uw.function.evaluate(M.sym, points)))
