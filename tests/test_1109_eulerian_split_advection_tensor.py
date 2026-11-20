"""The plain Eulerian history's split advection on a symmetric tensor, and the
commit's check that its frozen source is the history as it is now.

``Eulerian._apply_split_advection`` corrects ``psi_star[0] -= dt (V . grad) psi``
component by component. A 2x2 symmetric tensor stores three components; the
correction used to walk two data columns with the flat entries (xx, xy), so yy
was never corrected and took xy's correction, and xy took none. Here each
column is checked against a scalar history of that component alone, carried by
the same rigid rotation through the same evaluation.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

DT = 0.05


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


VELOCITY = lambda mesh: sympy.Matrix([[-mesh.X[1], mesh.X[0]]])     # a rigid rotation
COMPONENTS = {(0, 0): lambda x, y: x ** 2, (0, 1): lambda x, y: x * y, (1, 1): lambda x, y: y ** 2}


def _mesh():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 6, qdegree=3)


def _tensor_history(mesh):
    stress = uw.discretisation.MeshVariable("S", mesh, (2, 2), vtype=uw.VarType.SYM_TENSOR,
                                            degree=2)
    X = stress.coords
    for (i, j), f in COMPONENTS.items():
        stress.array[:, i, j] = f(X[:, 0], X[:, 1])
    history = uw.systems.ddt.Eulerian(mesh, stress, uw.VarType.SYM_TENSOR, degree=2,
                                      continuous=True, V_fn=VELOCITY(mesh))
    history.initialise_history()
    return history, stress


def _split_correction(history, field):
    """The correction alone, from the field's own nodal values (update_pre_solve
    would first re-record the field at the centroid-nudged nodes)."""
    level_0 = history.psi_star[0]
    level_0.data[...] = np.asarray(field.data)
    before = np.array(level_0.data)
    history._apply_split_advection(DT)
    return np.asarray(level_0.data) - before


def test_every_component_of_a_symmetric_tensor_takes_its_own_correction():
    """Each stored column of the tensor history takes the correction a scalar
    history of that component alone takes, through the same evaluation."""
    mesh = _mesh()
    history, stress = _tensor_history(mesh)
    tensor = _split_correction(history, stress)
    level_0 = history.psi_star[0]
    for (i, j), f in COMPONENTS.items():
        scalar = uw.discretisation.MeshVariable(f"s{i}{j}", mesh, 1, degree=2)
        scalar.array[:, 0, 0] = f(scalar.coords[:, 0], scalar.coords[:, 1])
        alone = uw.systems.ddt.Eulerian(mesh, scalar, uw.VarType.SCALAR, degree=2,
                                        continuous=True, V_fn=VELOCITY(mesh))
        alone.initialise_history()
        reference = _split_correction(alone, scalar)[:, 0]
        column = tensor[:, level_0._data_layout(i, j)]
        assert np.abs(reference).max() > 1.0e-3, (i, j)        # the flow moves this component
        # the scalar and tensor evaluations agree to 3e-8 (measured on the xx
        # column, which was right before); a misplaced column is off by 0.1
        assert np.abs(column - reference).max() < 1.0e-6, (i, j)


def test_the_commit_refuses_a_history_written_after_its_snapshot():
    history, _ = _tensor_history(_mesh())
    history.enable_source_snapshot()
    history.update_pre_solve(DT)
    history._check_snapshot_is_level_0()              # as refreshed: the same
    history.psi_star[0].data[:, 0] += 1.0
    with pytest.raises(RuntimeError, match="written after the source snapshot"):
        history._check_snapshot_is_level_0()
