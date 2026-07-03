"""
Regression test for the rbf (Clement) derivative-evaluation path selecting
the correct component of a multi-component variable.

Background
----------
``_clement_to_work_variable`` in ``_function.pyx`` is the quick derivative
path reached via ``uw.function.evaluate(expr, pts, rbf=True)``. It computes
per-component Clement gradients and stores them keyed by
``(source_var, component)``, and every derivative expression carries its own
data-column index (``diffcls.component``, set at ``UnderworldFunction``
registration).

Until this fix the retrieval loop discarded that index through a dead
ternary (``comp = 0 if ... else 0``), so ``v[1].diff(x)`` with ``rbf=True``
silently returned the component-0 gradient ``d(v[0])/dx``. The accurate L2
path (``rbf=False``) was unaffected.

The same fix also repairs the adjacent nodal-sampling step for higher-degree
variables: ``var.sym[comp, 0]`` mis-indexed the (1, cdim) row-matrix ``sym``
(IndexError for any component > 0), so the rbf derivative path crashed
outright whenever a multi-component P2+ variable existed on the mesh. The
P2 fixture below exercises that path; the P1 test isolates the ternary bug.

The test field v = (x, 2y) has distinguishable constant gradients per
component, so any cross-component mix-up produces an O(1) error:

    dv0/dx = 1   dv0/dy = 0
    dv1/dx = 0   dv1/dy = 2

See: docs/reviews/2026-07/LOOSE-ENDS-AUDIT.md (LE-01),
docs/reviews/2026-07/REMEDIATION-WORKLIST.md (BF-01).
"""
import numpy as np
import pytest
import underworld3 as uw


@pytest.fixture(scope="module")
def mesh_and_vector_var():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1
    )
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
    v.data[:, 0] = v.coords[:, 0]  # v_0 = x
    v.data[:, 1] = 2.0 * v.coords[:, 1]  # v_1 = 2y
    return mesh, v


# Interior points, away from boundaries where Clement recovery degrades.
PTS = np.array([[0.31, 0.37], [0.55, 0.62], [0.73, 0.29]])


@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize(
    "component,direction,expected",
    [
        (0, 0, 1.0),  # dv0/dx = 1  (control: component 0 was always correct)
        (0, 1, 0.0),  # dv0/dy = 0
        (1, 0, 0.0),  # dv1/dx = 0  (bug returned dv0/dx = 1)
        (1, 1, 2.0),  # dv1/dy = 2  (bug returned dv0/dy = 0)
    ],
)
def test_rbf_derivative_uses_requested_component(
    mesh_and_vector_var, component, direction, expected
):
    """Each rbf-path derivative must come from its own component's gradient."""
    mesh, v = mesh_and_vector_var
    expr = v.sym[component].diff(mesh.X[direction])

    result = np.asarray(uw.function.evaluate(expr, PTS, rbf=True)).flatten()

    # Gradients are constant, so Clement recovery + rbf interpolation are
    # essentially exact in the interior.
    assert np.allclose(result, expected, atol=1e-6), (
        f"d(v[{component}])/d(x[{direction}]) with rbf=True: "
        f"expected {expected}, got {result}"
    )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_rbf_derivative_scalar_variable(mesh_and_vector_var):
    """Single-component variables still work through the rbf path."""
    mesh, _ = mesh_and_vector_var
    x, y = mesh.X

    T = uw.discretisation.MeshVariable("T_rbfderiv", mesh, 1, degree=1)
    T.data[:, 0] = 3.0 * T.coords[:, 0] + 1.0  # T = 3x + 1

    result = np.asarray(uw.function.evaluate(T.sym[0].diff(x), PTS, rbf=True)).flatten()
    assert np.allclose(result, 3.0, atol=1e-6)


@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize(
    "component,direction,expected",
    [
        (0, 0, 1.0),
        (0, 1, 0.0),
        (1, 0, 0.0),
        (1, 1, 2.0),
    ],
)
def test_rbf_derivative_p1_vector_component(component, direction, expected):
    """P1 vector on its own mesh: isolates the dead-ternary retrieval bug
    from the higher-degree nodal-sampling bug (both fixed together)."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1
    )
    v = uw.discretisation.MeshVariable("v_p1", mesh, mesh.dim, degree=1)
    v.data[:, 0] = v.coords[:, 0]
    v.data[:, 1] = 2.0 * v.coords[:, 1]

    expr = v.sym[component].diff(mesh.X[direction])
    result = np.asarray(uw.function.evaluate(expr, PTS, rbf=True)).flatten()

    assert np.allclose(result, expected, atol=1e-6), (
        f"d(v[{component}])/d(x[{direction}]) with rbf=True: "
        f"expected {expected}, got {result}"
    )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_rbf_derivative_mixed_component_expression(mesh_and_vector_var):
    """An expression mixing derivatives of both components (e.g. divergence)."""
    mesh, v = mesh_and_vector_var
    x, y = mesh.X

    div_v = v.sym[0].diff(x) + v.sym[1].diff(y)  # = 1 + 2 = 3
    result = np.asarray(uw.function.evaluate(div_v, PTS, rbf=True)).flatten()

    # Bug returned dv0/dx + dv0/dy = 1 + 0 = 1.
    assert np.allclose(result, 3.0, atol=1e-6)
