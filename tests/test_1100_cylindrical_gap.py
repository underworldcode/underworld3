"""Analytical serial/MPI gates for physical-circle mixed Stokes integration.

The independent pressure-gradient problem has u=0, p=x and f=(1,0).
No Kramer reference or spectral coefficient is used by the solver.
"""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw

pytestmark = pytest.mark.level_2


class _GradientObserver(uw.diagnostics.StokesObserver):
    def solve_started(self, ksp, state):
        self.started = True
    def solve_finished(self, ksp, vector, state):
        physical = state.gauged_copy(vector)
        try:
            assert state.residual(physical)['original_relative_residual'] < 1e-10
        finally:
            physical.destroy()
    def finalized(self, result):
        self.result = result


@pytest.mark.tier_b
@pytest.mark.parametrize("force_degree", [None, 1, 2])
def test_cylindrical_gap_preserves_pressure_gradient_and_null_modes(force_degree):
    mesh = uw.meshing.Annulus(
        radiusInner=1.22, radiusOuter=2.22, cellSize=0.125, qdegree=3
    )
    velocity = uw.discretisation.MeshVariable("GapU", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("GapP", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.bodyforce = sp.Matrix([1, 0])
    if force_degree is not None:
        force = uw.discretisation.MeshVariable(
            "GapForce", mesh, 2, degree=force_degree, continuous=force_degree == 1
        )
        values = np.zeros_like(np.asarray(force.array))
        values[:, 0, 0] = 1.0
        force.array[:, :, :] = values
        stokes.bodyforce = force.sym
    observer = _GradientObserver()
    with stokes.observe_stokes(observer):
        result = stokes.solve_cylindrical_gap(
            inner_boundary="Lower",
            outer_boundary="Upper",
            radius_inner=1.22,
            radius_outer=2.22,
            viscosity=1.0,
            pressure_jump=sp.S.Zero,
            pressure_jump_integral=0.0,
            cap_bodyforce=sp.Matrix([1, 0]),
            quadrature=8,
            linear_solver_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
    assert observer.started and observer.result is result
    assert not hasattr(stokes, '_stokes_observer')
    area = uw.maths.Integral(mesh, sp.S.One).evaluate()
    velocity_rms = np.sqrt(
        uw.maths.Integral(mesh, velocity.sym.dot(velocity.sym)).evaluate() / area
    )
    pressure_rms = np.sqrt(
        uw.maths.Integral(mesh, (pressure.sym[0] - mesh.N.x) ** 2).evaluate() / area
    )
    assert (
        velocity_rms < 2e-8
    ), "A pure pressure gradient must not drive appreciable flow"
    assert (
        pressure_rms < 3e-8
    ), "Physical pressure must reproduce p=x with zero circular-domain mean"
    assert result.checks["relative_residual"] < 1e-10
    assert result.checks["pressure_null_relative_residual"] < 1e-12
    assert result.checks["rotation_null_relative_residual"] < 1e-12
    assert abs(result.checks["physical_area"] - np.pi * (2.22**2 - 1.22**2)) < 1e-11


@pytest.mark.tier_b
def test_circular_cap_area_and_empty_partition():
    from underworld3.utilities._cylindrical_gap import _cap_quadrature

    radius, half_angle = 2.22, 0.06
    edges = radius * np.array(
        [
            [
                [np.cos(half_angle), -np.sin(half_angle)],
                [np.cos(half_angle), np.sin(half_angle)],
            ]
        ]
    )
    rule = _cap_quadrature(edges, radius, 12, 4)
    exact = radius**2 * (2 * half_angle - np.sin(2 * half_angle)) / 2
    np.testing.assert_allclose(rule["cap_weight"].sum(), exact, atol=1e-14, rtol=0)
    empty = _cap_quadrature(np.empty((0, 2, 2)), radius, 8, 4)
    assert empty["cap_points"].shape == (0, 32, 2)


@pytest.mark.tier_b
def test_mixed_element_transpose_is_inserted_in_row_order():
    """PETSc Mat.setValues must receive C-contiguous rectangular block data."""
    from petsc4py import PETSc
    from underworld3.utilities._cylindrical_gap import _insert_blocks

    matrix = PETSc.Mat().createAIJ(size=(15, 15), nnz=15, comm=PETSc.COMM_SELF)
    b = np.arange(36, dtype=float).reshape(1, 3, 12)
    _insert_blocks(
        matrix,
        np.arange(12, dtype=PETSc.IntType)[None],
        np.arange(12, 15, dtype=PETSc.IntType)[None],
        {"A": np.zeros((1, 12, 12)), "B": b},
        1.0,
        1,
    )
    matrix.assemble()
    actual = matrix.getValues(
        np.arange(12, dtype=PETSc.IntType), np.arange(12, 15, dtype=PETSc.IntType)
    )
    np.testing.assert_array_equal(actual, b[0].T)
    assert matrix.isSymmetric(tol=1e-14)
    matrix.destroy()


@pytest.mark.tier_b
def test_dg_source_reproduces_a_quadratic_at_nonstandard_nodes():
    from underworld3.utilities._cylindrical_gap import _field_values, _triangle_rule

    mesh = uw.meshing.Annulus(
        radiusInner=1.22, radiusOuter=2.22, cellSize=0.25, qdegree=3
    )
    force = uw.discretisation.MeshVariable(
        "QuadraticGapForce", mesh, 2, degree=2, continuous=False
    )
    x, y = np.asarray(force.coords).T
    values = np.column_stack((x * x + x * y, y * y - x * y))
    force.array[:, 0, :] = values
    rows = mesh._cell_node_indices(1, True)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[rows]
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    points, _ = _triangle_rule(vertices[owned], 4)
    x, y = points[..., 0], points[..., 1]
    exact = np.stack((x * x + x * y, y * y - x * y), axis=-1)
    actual = _field_values(force, vertices, points, owned)
    np.testing.assert_allclose(actual, exact, atol=2e-13, rtol=0)


@pytest.mark.tier_b
def test_constant_pressure_lift_keeps_physical_pressure_and_reaction_zero():
    mesh = uw.meshing.Annulus(
        radiusInner=1.22, radiusOuter=2.22, cellSize=0.125, qdegree=3
    )
    velocity = uw.discretisation.MeshVariable("LiftGapU", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("LiftGapP", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.bodyforce = sp.zeros(2, 1)
    area = uw.maths.Integral(mesh, sp.S.One).evaluate()
    result = stokes.solve_cylindrical_gap(
        inner_boundary="Lower",
        outer_boundary="Upper",
        radius_inner=1.22,
        radius_outer=2.22,
        viscosity=1.0,
        pressure_jump=sp.S.One,
        pressure_jump_integral=area,
        cap_bodyforce=sp.zeros(2, 1),
        quadrature=8,
        linear_solver_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    np.testing.assert_allclose(np.asarray(velocity.array), 0, atol=1e-11, rtol=0)
    np.testing.assert_allclose(np.asarray(pressure.array) + 1, 0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        np.asarray(result.normal_reaction.array), 0, atol=1e-10, rtol=0
    )


@pytest.mark.tier_b
def test_zero_slip_gradient_load_has_no_rotation_gauge():
    mesh = uw.meshing.Annulus(
        radiusInner=1.22, radiusOuter=2.22, cellSize=0.125, qdegree=4
    )
    u = uw.discretisation.MeshVariable("ZeroGapU", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("ZeroGapP", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.bodyforce = sp.Matrix([1, 0])
    observer = _GradientObserver()
    with stokes.observe_stokes(observer):
        result = stokes.solve_cylindrical_gap_zero_slip(
            inner_boundary="Lower",
            outer_boundary="Upper",
            radius_inner=1.22,
            radius_outer=2.22,
            viscosity=1.0,
            pressure_jump=sp.S.Zero,
            pressure_jump_integral=0.0,
            cap_bodyforce=sp.Matrix([1, 0]),
            quadrature=8,
            linear_solver_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
    assert observer.started and observer.result is result
    assert not hasattr(stokes, '_stokes_observer')
    np.testing.assert_allclose(np.asarray(u.array), 0, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        np.asarray(p.array)[:, 0, 0], p.coords[:, 0], atol=1e-7, rtol=0
    )
    assert result.checks["rotation_is_null"] is False
    assert result.checks["relative_residual"] < 1e-10


@pytest.mark.tier_b
def test_zero_slip_balances_torque_and_converges_to_azimuthal_solution():
    errors = []
    for h in (0.25, 0.125):
        mesh = uw.meshing.Annulus(
            radiusInner=1.22, radiusOuter=2.22, cellSize=h, qdegree=4
        )
        u = uw.discretisation.MeshVariable("TorqueGapU", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable("TorqueGapP", mesh, 1, degree=1)
        stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        x, y = mesh.N.x, mesh.N.y
        force = sp.Matrix([-y, x])
        stokes.bodyforce = force
        result = stokes.solve_cylindrical_gap_zero_slip(
            inner_boundary="Lower",
            outer_boundary="Upper",
            radius_inner=1.22,
            radius_outer=2.22,
            viscosity=1.0,
            pressure_jump=sp.S.Zero,
            pressure_jump_integral=0.0,
            cap_bodyforce=force,
            quadrature=8,
            linear_solver_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        rr = x * x + y * y
        ri, ro = sp.Rational(122, 100), sp.Rational(222, 100)
        profile = (ri**2 + ro**2 - ri**2 * ro**2 / rr - rr) / 8
        exact = sp.Matrix([[-y * profile, x * profile]])
        error = u.sym - exact
        errors.append(
            np.sqrt(
                uw.maths.Integral(mesh, error.dot(error)).evaluate()
                / uw.maths.Integral(mesh, exact.dot(exact)).evaluate()
            )
        )
        # Match the public solver's unpinned-residual contract. Distributed
        # MUMPS on Gadi gives ~1.003e-10 here; accuracy and rate are checked below.
        assert result.checks["relative_residual"] < 1e-9
    assert errors[1] < 1e-3, "Manufactured azimuthal no-slip velocity must be accurate"
    assert (
        np.log2(errors[0] / errors[1]) > 2.5
    ), "P2 velocity must approach third-order convergence"


@pytest.mark.tier_b
@pytest.mark.parametrize("empty_partition", [False, True])
def test_gap_matrix_preallocates_remote_mixed_connectivity(empty_partition):
    """The known mixed graph must require no numerical insertion reallocations."""
    from petsc4py import PETSc
    from underworld3.utilities._cylindrical_gap import (
        _create_gap_matrix,
        _insert_blocks,
    )

    ranks = uw.mpi.size
    owner = (uw.mpi.rank + 1) % ranks
    uids = (12 * owner + np.arange(12, dtype=PETSc.IntType))[None]
    pids = (12 * ranks + 3 * owner + np.arange(3, dtype=PETSc.IntType))[None]
    lids = (15 * ranks + 3 * owner + np.arange(3, dtype=PETSc.IntType))[None]
    count = int(not empty_partition or uw.mpi.rank == 0)
    uids, pids, lids = uids[:count], pids[:count], lids[:count]
    matrix = _create_gap_matrix(
        18 * ranks, uids, pids, [(uids, lids)], PETSc.COMM_WORLD
    )
    a = np.diag(np.arange(1, 13, dtype=float))[None]
    b = np.arange(1, 37, dtype=float).reshape(1, 3, 12)
    c = b[0] / 2
    _insert_blocks(matrix, uids, pids, {"A": a[:count], "B": b[:count]}, 1.0, 1)
    for u, l in zip(uids, lids, strict=True):
        matrix.setValues(l, u, c, addv=PETSc.InsertMode.ADD_VALUES)
        matrix.setValues(
            u, l, np.ascontiguousarray(c.T), addv=PETSc.InsertMode.ADD_VALUES
        )
    matrix.assemble()
    lo, hi = matrix.getOwnershipRange()
    exact = np.zeros((hi - lo, 18 * ranks))
    for cell in ((1 % ranks),) if empty_partition else range(ranks):
        u = 12 * cell + np.arange(12)
        p = 12 * ranks + 3 * cell + np.arange(3)
        l = 15 * ranks + 3 * cell + np.arange(3)
        for rows, cols, values in (
            (u, u, a[0]),
            (p, u, b[0]),
            (u, p, b[0].T),
            (l, u, c),
            (u, l, c.T),
        ):
            selected = (rows >= lo) & (rows < hi)
            exact[np.ix_(rows[selected] - lo, cols)] += values[selected]
    actual = matrix.getValues(
        np.arange(lo, hi, dtype=PETSc.IntType),
        np.arange(18 * ranks, dtype=PETSc.IntType),
    )
    np.testing.assert_array_equal(actual, exact)
    assert matrix.getInfo(PETSc.Mat.InfoType.LOCAL)["mallocs"] == 0
    pinned = matrix.copy()
    pinned.zeroRowsColumns(np.asarray([12 * ranks], dtype=PETSc.IntType), diag=1.0)
    assert pinned.getInfo(PETSc.Mat.InfoType.LOCAL)["mallocs"] == 0
    pinned.destroy()
    matrix.destroy()
