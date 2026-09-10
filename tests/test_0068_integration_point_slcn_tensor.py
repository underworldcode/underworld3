"""Vector and tensor histories at the integration points.

``IntegrationPointSemiLagrangian`` used to refuse anything but a scalar. It
now carries a vector or a tensor, which is what a Navier-Stokes momentum
history or a viscoelastic stress history needs.

The properties checked here are the same ones the scalar case rests on — each
slot holds the snapshot evaluated exactly at the traced departure point — plus
the one the shape introduces: a field with N independent components is stored
in N columns, not in one per matrix entry, and the packing has to survive the
round trip. A symmetric tensor in 2-D is 2x2 symbolically and 3 columns in
storage; getting that wrong transposes a stress silently.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems.ddt import _storage_components

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

V_UNIFORM = np.array([1.0, 0.5])
DT = 0.1


def _velocity():
    return sympy.Matrix([[V_UNIFORM[0], V_UNIFORM[1]]])


# Fields chosen to lie in the P2 space, so the sample at the departure point is
# exact to round-off, and with every component DISTINCT so that a packing or
# transposition error cannot hide.
def _scalar_field(X):
    return 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 0] ** 2


def _vector_field(X):
    return np.column_stack([
        1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 0] ** 2,
        -2.0 + X[:, 0] * X[:, 1] + X[:, 1] ** 2,
    ])


def _tensor_entries(X):
    """The 2x2 symbolic entries, keyed by (i, j)."""
    xx = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1]
    yy = -2.0 + X[:, 1] + 0.5 * X[:, 1] ** 2
    xy = 0.25 + X[:, 0] * X[:, 1] - 0.5 * X[:, 0] ** 2
    return {(0, 0): xx, (1, 1): yy, (0, 1): xy, (1, 0): xy}


def _pack(entries, columns):
    return np.column_stack([entries[ij] for ij in columns])


# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vtype,dim,expected",
    [
        (uw.VarType.SCALAR, 2, [(0, 0)]),
        (uw.VarType.VECTOR, 2, [(0, 0), (0, 1)]),
        (uw.VarType.SYM_TENSOR, 2, [(0, 0), (1, 1), (0, 1)]),
        (uw.VarType.SYM_TENSOR, 3, [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]),
    ],
)
def test_the_storage_order_is_what_the_symbol_reconstructs(vtype, dim, expected):
    """Pin the column -> (i, j) convention against the variable itself.

    If the storage order ever changes, this fails here rather than quietly
    transposing a stress history somewhere downstream.
    """
    if dim == 2:
        mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.4, qdegree=3)
    else:
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
            cellSize=0.5, qdegree=3,
        )
    var = uw.discretisation.MeshVariable(f"sv{vtype.value}{dim}", mesh,
                                         vtype=vtype, degree=1)
    columns = _storage_components(vtype, tuple(var.sym.shape))
    assert columns == expected
    assert len(columns) == var.num_components

    # a distinct marker per column, read back through the symbol
    with uw.synchronised_array_update():
        for c in range(var.num_components):
            var.data[:, c] = 10.0 * (c + 1)
    point = np.full((1, mesh.dim), 0.5)
    got = np.asarray(uw.function.evaluate(var.sym, point)).reshape(var.sym.shape)
    for c, (i, j) in enumerate(columns):
        assert got[i, j] == pytest.approx(10.0 * (c + 1)), (c, i, j, got)


def test_a_vector_history_holds_the_departure_point_values():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=3)
    U = uw.discretisation.MeshVariable("Uv", mesh, vtype=uw.VarType.VECTOR, degree=2)
    with uw.synchronised_array_update():
        U.data[...] = _vector_field(np.asarray(U.coords))

    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(
        mesh, U, _velocity(), vtype=uw.VarType.VECTOR, degree=2, order=2)
    assert ddt.num_components == 2
    assert all(ps.is_integration_point for ps in ddt.psi_star)
    assert ddt.bdf().shape == (1, 2)

    ddt.update_pre_solve(DT)
    X = np.asarray(ddt.psi_star[0].coords)
    foot = X - V_UNIFORM * DT
    inside = (foot > 0.0).all(1) & (foot < 1.0).all(1)
    assert inside.sum() > 100
    got = np.asarray(ddt.psi_star[0].data)[inside]
    assert np.abs(got - _vector_field(foot[inside])).max() < 1e-12

    # two segments back, from the older snapshot
    ddt.update_post_solve(DT)
    ddt.update_pre_solve(DT)
    foot2 = X - V_UNIFORM * 2 * DT
    inside2 = (foot2 > 0.0).all(1) & (foot2 < 1.0).all(1)
    got2 = np.asarray(ddt.psi_star[1].data)[inside2]
    assert np.abs(got2 - _vector_field(foot2[inside2])).max() < 1e-12


def test_a_symmetric_tensor_history_transports_every_component():
    """Three independent components, all different, so a packing error or a
    transposed off-diagonal cannot pass."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=3)
    S = uw.discretisation.MeshVariable("St", mesh, vtype=uw.VarType.SYM_TENSOR,
                                       degree=2)
    columns = _storage_components(uw.VarType.SYM_TENSOR, (2, 2))
    with uw.synchronised_array_update():
        S.data[...] = _pack(_tensor_entries(np.asarray(S.coords)), columns)

    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(
        mesh, S, _velocity(), vtype=uw.VarType.SYM_TENSOR, degree=2, order=1)
    assert ddt.num_components == 3
    assert ddt._components == columns
    assert ddt.bdf().shape == (2, 2)

    ddt.update_pre_solve(DT)
    X = np.asarray(ddt.psi_star[0].coords)
    foot = X - V_UNIFORM * DT
    inside = (foot > 0.0).all(1) & (foot < 1.0).all(1)
    assert inside.sum() > 100
    expected = _pack(_tensor_entries(foot[inside]), columns)
    got = np.asarray(ddt.psi_star[0].data)[inside]
    assert np.abs(got - expected).max() < 1e-12

    # and the symbol reads back as the right 2x2 matrix, off-diagonal included
    point = foot[inside][0].reshape(1, -1)
    entries = _tensor_entries(point)
    sym = np.asarray(
        uw.function.evaluate(ddt.psi_star[0].sym, X[inside][0].reshape(1, -1))
    ).reshape(2, 2)
    assert sym[0, 1] == pytest.approx(entries[(0, 1)][0], abs=1e-10)
    assert sym[1, 0] == pytest.approx(sym[0, 1])
    assert sym[0, 0] == pytest.approx(entries[(0, 0)][0], abs=1e-10)
    assert sym[1, 1] == pytest.approx(entries[(1, 1)][0], abs=1e-10)
    assert abs(sym[0, 0] - sym[1, 1]) > 0.1        # the components are distinct


def test_a_scalar_history_is_unchanged():
    """The generalisation must not move the scalar answer."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=3)
    T = uw.discretisation.MeshVariable("Ts", mesh, 1, degree=2)
    with uw.synchronised_array_update():
        T.data[:, 0] = _scalar_field(np.asarray(T.coords))

    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(
        mesh, T, _velocity(), degree=2, order=1)
    assert ddt.num_components == 1
    assert ddt.bdf().shape == (1, 1)

    ddt.update_pre_solve(DT)
    X = np.asarray(ddt.psi_star[0].coords)
    foot = X - V_UNIFORM * DT
    inside = (foot > 0.0).all(1) & (foot < 1.0).all(1)
    assert np.abs(
        np.asarray(ddt.psi_star[0].data)[inside, 0] - _scalar_field(foot[inside])
    ).max() < 1e-12


@pytest.mark.parametrize("vtype", [uw.VarType.VECTOR, uw.VarType.SYM_TENSOR])
def test_the_history_symbol_participates_in_expressions(vtype):
    """A shaped history has to be usable, not merely storable: its symbol goes
    where a mesh variable's symbol goes."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2, qdegree=3)
    var = uw.discretisation.MeshVariable(f"Pe{vtype.value}", mesh, vtype=vtype,
                                         degree=2)
    columns = _storage_components(vtype, tuple(var.sym.shape))
    with uw.synchronised_array_update():
        var.data[...] = 2.0
    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(
        mesh, var, _velocity(), vtype=vtype, degree=2, order=1)
    ddt.update_pre_solve(DT)

    star = ddt.psi_star[0].sym
    trace = sum(star[i, i] for i in range(star.shape[0]))
    assert float(uw.maths.Integral(mesh, trace).evaluate()) == pytest.approx(
        2.0 * star.shape[0], rel=1e-8)

    # the second invariant of the difference is a legitimate weak-form term
    expr = (star - ddt.bdf()).T * (star - ddt.bdf())
    assert expr.shape[0] == star.shape[1]
    assert len(columns) == ddt.num_components


def test_the_refusal_is_gone_but_the_rule_check_is_not():
    """An undersampled quadrature rule is still refused, whatever the shape:
    a delta field cannot carry more values than the rule has points."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2, qdegree=2)
    U = uw.discretisation.MeshVariable("Ur", mesh, vtype=uw.VarType.VECTOR, degree=2)
    with pytest.raises(RuntimeError, match="qdegree|rule|oversample"):
        uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, U, _velocity(), vtype=uw.VarType.VECTOR, degree=2, order=1)


def test_the_storage_map_follows_the_shape_not_the_mesh_dimension():
    """On a manifold the topological and embedding dimensions differ — a
    spherical surface is dim 2, cdim 3 — and the variable sizes its storage by
    the embedding one, which is what ``.sym`` is shaped by. The map therefore
    reads the tensor dimension off the shape and never touches the mesh.
    """
    assert len(_storage_components(uw.VarType.SYM_TENSOR, (2, 2))) == 3
    assert len(_storage_components(uw.VarType.SYM_TENSOR, (3, 3))) == 6
    assert _storage_components(uw.VarType.VECTOR, (1, 3)) == [(0, 0), (0, 1), (0, 2)]


def test_a_vtype_that_does_not_match_psi_fn_is_refused():
    """And refused BEFORE any variable is allocated: a mesh variable created
    and then abandoned leaves its field on the DM (#1058)."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, qdegree=3)
    before = len(mesh.vars)

    with pytest.raises(ValueError, match="psi_fn has shape"):
        uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, sympy.Matrix([[1.0]]), _velocity(),
            vtype=uw.VarType.VECTOR, degree=2, order=1)

    with pytest.raises(ValueError, match="psi_fn has shape"):
        uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, sympy.Matrix([[1.0, 2.0]]), _velocity(),
            vtype=uw.VarType.SYM_TENSOR, degree=2, order=1)

    assert len(mesh.vars) == before, "a refused history left variables behind"
