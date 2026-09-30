"""IntegrationPointVariable: a field stored at the mesh integration points.

What is checked, and why each check is the one that matters:

- layout: one value per rule point per cell, coordinates from the assembler's
  own cell geometry (the rule-weighted mean of a cell's points is its centroid);
- the assembler reads the stored values exactly: the integral of random
  point data equals the quadrature sum done by hand, and a P2 projection of
  P2 point data is exact to solver tolerance;
- ``evaluate`` is the nearest integration point of the owning cell, exact at
  the variable's own points;
- the two guards: a derivative of the symbol is refused by the JIT, and a
  solver on a different rule is refused by the mesh.
"""

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _mesh(kind):
    if kind == "triangle":
        return uw.meshing.UnstructuredSimplexBox(cellSize=0.25, qdegree=2)
    if kind == "tetrahedron":
        return uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.5, qdegree=2
        )
    if kind == "quadrilateral":
        return uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=2)
    raise ValueError(kind)


def _ncells(mesh):
    c0, c1 = mesh.dm.getHeightStratum(0)
    return c1 - c0


@pytest.mark.parametrize("kind", ["triangle", "tetrahedron", "quadrilateral"])
def test_layout_and_geometry(kind):
    mesh = _mesh(kind)
    h = uw.discretisation.IntegrationPointVariable("h", mesh)
    Nq = len(np.asarray(mesh.integration_rule.getData()[1]))
    n = _ncells(mesh)

    assert h.is_integration_point
    assert h.num_points_per_cell == Nq
    assert h.data.shape == (n * Nq, 1)
    assert h.cell_data.shape == (n, Nq, 1)
    assert np.asarray(h.coords).shape == (n * Nq, mesh.dim)
    assert np.array_equal(np.asarray(h.coords_nd), h.integration_points.reshape(-1, mesh.dim))

    # Affine cells: the rule-weighted mean of the points is the centroid.
    w = np.asarray(mesh.integration_rule.getData()[1]).reshape(-1)
    cent = (h.integration_points * w[None, :, None]).sum(1) / w.sum()
    assert np.allclose(cent, np.asarray(mesh._centroids)[:n], atol=1e-12)


@pytest.mark.skipif(uw.mpi.size > 1, reason="the hand quadrature sum is over the rank's local cells; serial only")
def test_assembler_reads_the_stored_values():
    """Integral of random point data == the quadrature sum done by hand."""
    mesh = _mesh("triangle")
    h = uw.discretisation.IntegrationPointVariable("h", mesh)
    rng = np.random.default_rng(1)
    h.data[:, 0] = rng.uniform(-1.0, 2.0, size=h.data.shape[0])

    # Cell areas from the vertices, rule weights scaled by area / reference area.
    verts = np.asarray(mesh._get_coords_for_basis(1, True))
    rows = np.asarray(mesh._cell_node_indices(1, True))
    p = verts[rows]  # (ncells, 3, 2)
    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    w = np.asarray(mesh.integration_rule.getData()[1]).reshape(-1)
    by_hand = ((h.cell_data[:, :, 0] * w[None, :]).sum(1) * area / w.sum()).sum()

    assembled = uw.maths.Integral(mesh, h.sym[0]).evaluate()
    assert abs(assembled - by_hand) < 1e-12 * max(1.0, abs(by_hand))
    # Negative control: perturb one point and the integral must move by
    # exactly that point's weight.
    c, q = 3, 2
    h.cell_data[c, q, 0] += 1.0
    moved = uw.maths.Integral(mesh, h.sym[0]).evaluate()
    assert abs((moved - assembled) - w[q] * area[c] / w.sum()) < 1e-12


def test_projection_of_p2_point_data_is_exact():
    mesh = _mesh("triangle")
    x, y = mesh.X
    h = uw.discretisation.IntegrationPointVariable("h", mesh)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    f = lambda X: 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 0] ** 2 + X[:, 0] * X[:, 1]
    h.data[:, 0] = f(np.asarray(h.coords))

    proj = uw.systems.solvers.SNES_Projection(mesh, T)
    proj.uw_function = h.sym[0]
    proj.smoothing = 0.0
    proj.petsc_options["ksp_rtol"] = 1e-13
    proj.petsc_options["snes_rtol"] = 1e-13
    proj.solve()
    assert np.abs(T.data[:, 0] - f(np.asarray(T.coords))).max() < 1e-9


def test_evaluate_is_nearest_point_of_owning_cell():
    mesh = _mesh("triangle")
    h = uw.discretisation.IntegrationPointVariable("h", mesh)
    rng = np.random.default_rng(2)
    h.data[:, 0] = rng.uniform(size=h.data.shape[0])

    # Exact at its own points (the index selection is exact; the evaluator
    # pipeline can add an ulp of round-off on the way out).
    own = uw.function.evaluate(h.sym[0], np.asarray(h.coords)).reshape(-1)
    assert np.allclose(own, h.data[:, 0], rtol=0, atol=1e-14)

    # Nearest point of the owning cell elsewhere. In parallel keep only the
    # points this rank owns (the locator returns -1 for the others).
    pts = rng.uniform(0.05, 0.95, size=(300, 2))
    cells = np.asarray(mesh._robust_owning_cells(pts)).reshape(-1)
    pts, cells = pts[cells >= 0], cells[cells >= 0]
    assert len(pts) > 20
    ipc = h.integration_points
    j = ((ipc[cells] - pts[:, None, :]) ** 2).sum(-1).argmin(1)
    expected = h.cell_data[cells, j, 0]
    got = uw.function.evaluate(h.sym[0], pts).reshape(-1)
    assert np.allclose(got, expected, rtol=0, atol=1e-14)

    # Negative control: a nodal P1 interpolant of the same data does not
    # match this definition (it smooths), so the test discriminates.
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    T.data[:, 0] = uw.function.evaluate(h.sym[0], np.asarray(T.coords)).reshape(-1)
    smooth = uw.function.evaluate(T.sym[0], pts).reshape(-1)
    assert not np.allclose(smooth, expected)


def test_derivative_is_refused_by_the_jit():
    mesh = _mesh("triangle")
    x, y = mesh.X
    h = uw.discretisation.IntegrationPointVariable("h", mesh)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    proj = uw.systems.solvers.SNES_Projection(mesh, T)
    proj.uw_function = h.sym[0].diff(x)
    with pytest.raises(RuntimeError, match="integration-point"):
        proj.solve()


def test_other_rule_is_refused():
    mesh = _mesh("triangle")
    uw.discretisation.IntegrationPointVariable("h", mesh)
    same = PETSc.FE().createDefault(2, 1, True, mesh.qdegree, "same_", PETSc.COMM_SELF)
    other = PETSc.FE().createDefault(2, 1, True, mesh.qdegree + 1, "other_", PETSc.COMM_SELF)
    mesh._verify_integration_rule(same)
    with pytest.raises(RuntimeError, match="integration rule"):
        mesh._verify_integration_rule(other)


def test_vector_variable_layout_projection_and_evaluate():
    """A two-component integration-point variable: (ncells*Nq, 2) layout,
    each component reproduced exactly by a P2 projection of P2 point data,
    and evaluate exact at its own points."""
    mesh = _mesh("triangle")
    x, y = mesh.X
    v = uw.discretisation.IntegrationPointVariable("v", mesh, num_components=2)
    Nq = len(np.asarray(mesh.integration_rule.getData()[1]))
    n = _ncells(mesh)
    assert v.data.shape == (n * Nq, 2)
    assert v.cell_data.shape == (n, Nq, 2)
    X = np.asarray(v.coords)
    f0 = lambda X: 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 0] ** 2
    f1 = lambda X: -2.0 + X[:, 0] * X[:, 1] + X[:, 1] ** 2
    v.data[:, 0] = f0(X)
    v.data[:, 1] = f1(X)

    own = np.asarray(uw.function.evaluate(v.sym, X)).reshape(-1, 2)
    assert np.allclose(own, np.asarray(v.data), rtol=0, atol=1e-14)

    T = uw.discretisation.MeshVariable("Tv", mesh, 2, degree=2)
    proj = uw.systems.solvers.SNES_Vector_Projection(mesh, T)
    proj.uw_function = v.sym
    proj.smoothing = 0.0
    proj.petsc_options["ksp_rtol"] = 1e-13
    proj.petsc_options["snes_rtol"] = 1e-13
    proj.solve()
    Xt = np.asarray(T.coords)
    assert np.abs(T.data[:, 0] - f0(Xt)).max() < 1e-9
    assert np.abs(T.data[:, 1] - f1(Xt)).max() < 1e-9
