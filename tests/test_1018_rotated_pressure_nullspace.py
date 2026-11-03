"""Pressure gauges require right and left nullity of the constrained operator."""

import numpy as np
import pytest
import sympy as sp
from petsc4py import PETSc
import underworld3 as uw
from underworld3.utilities import rotated_bc as rbc

from test_1018_rotated_solver_options import _hydrostatic_stokes, _BLOCK_OPTIONS

pytestmark = pytest.mark.level_2


@pytest.mark.tier_b
def test_empty_pressure_space_is_not_a_null_basis():
    # The identity has no kernel. An empty pressure IS must not turn its zero
    # candidate into an accepted basis vector, including on empty local batches.
    size = 2*uw.mpi.size
    matrix = PETSc.Mat().createAIJ([size, size], nnz=1, comm=PETSc.COMM_WORLD)
    lo, hi = matrix.getOwnershipRange()
    for row in range(lo, hi):
        matrix.setValue(row, row, 1)
    matrix.assemble()
    pressure_is = PETSc.IS().createGeneral([], comm=matrix.comm)
    try:
        assert pressure_is.getSize() == 0
        checks = rbc._pressure_nullspace_checks(matrix, pressure_is, requested=True)
        assert checks["requested"] and not checks["verified"]
        assert checks["reason"] == "no free pressure degrees of freedom"
        assert checks["right_action_norm"] is checks["left_action_norm"] is None
    finally:
        pressure_is.destroy()
        matrix.destroy()


@pytest.mark.tier_b
@pytest.mark.parametrize("entries, accepted", [
    ([[1, 0], [0, 0]], True),
    ([[1, 0], [1, 0]], False),  # Right null, but not left null.
    ([[1, 1], [0, 0]], False),  # Left null, but not right null.
])
def test_pressure_mode_requires_both_operator_actions(entries, accepted):
    size = 2*uw.mpi.size
    matrix = PETSc.Mat().createAIJ([size, size], nnz=2, comm=PETSc.COMM_WORLD)
    lo, hi = matrix.getOwnershipRange()
    for row in range(lo, hi):
        if row < 2:
            matrix.setValues([row], [0, 1], [entries[row]])
        else:
            matrix.setValue(row, row, 1)
    matrix.assemble()
    pressure_is = PETSc.IS().createGeneral([1] if lo <= 1 < hi else [], comm=matrix.comm)
    try:
        checks = rbc._pressure_nullspace_checks(matrix, pressure_is, requested=True)
        assert checks["verified"] is accepted
        assert checks["right_action_norm"] == abs(entries[0][1])
        assert checks["left_action_norm"] == abs(entries[1][0])
    finally:
        pressure_is.destroy()
        matrix.destroy()


def _assert_report_fields(stokes):
    info = stokes._rotated_freeslip_info
    report = stokes.solve_report
    gauge = report.config["gauge"]
    for key in ("pressure_nullspace_requested", "pressure_nullspace_verified",
                "pressure_nullspace_attached", "pressure_nullspace_projection_active",
                "pressure_nullspace_checks"):
        assert gauge[key] == info[key]
    assert gauge["operator_scope"] == info["pressure_nullspace_checks"]["operator_scope"]
    preconditioner = report.config["rotated_preconditioner"]
    for key in ("pressure_coarse_correction_active", "pressure_coarse_correction_metrics",
                "outer_pc_type", "retained_base_pc_type"):
        assert preconditioner[key] == info[key]
    residual = report.config["rotated_residual"]
    assert residual["fresh_unprojected_norm"] == info["fresh_unprojected_rnorm"]
    assert residual["fresh_projected_norm"] == info["fresh_projected_rnorm"]
    assert residual["full_reference_norm"] == info["fresh_unprojected_reference_rnorm"]
    assert report.fnorm == info["fresh_projected_rnorm"]
    assert report.converged == info["converged"]


def _assert_hydrostatic_nodes(stokes, amplitude):
    values = np.asarray(stokes.p.array)[:, 0, 0]
    coordinates = np.asarray(stokes.p.coords)
    offset = values-amplitude*coordinates[:, 0]
    extrema = uw.mpi.comm.allgather((float(offset.min(initial=np.inf)),
                                    float(offset.max(initial=-np.inf))))
    assert max(pair[1] for pair in extrema)-min(pair[0] for pair in extrema) < 1e-7*amplitude
    velocity = float(np.max(np.abs(np.asarray(stokes.u.array)), initial=0))
    assert max(uw.mpi.comm.allgather(velocity)) < 1e-8*amplitude


def _assert_flat_solution(stokes, amplitude):
    _assert_report_fields(stokes)
    _assert_hydrostatic_nodes(stokes, amplitude)
    info = stokes._rotated_freeslip_info
    assert info["pressure_nullspace_requested"] and info["pressure_nullspace_verified"]
    assert info["pressure_nullspace_attached"]
    assert not info["pressure_coarse_correction_active"]
    assert info["pressure_coarse_correction_metrics"] is None
    space = stokes._rotated_linear_cache["Ahat"].getNullSpace()
    assert space is not None
    pressure_content = 0.0
    pressure_is = stokes._subdict["pressure"][0]
    for vector in space.getVecs():
        part = vector.getSubVector(pressure_is)
        pressure_content += part.norm()**2
        vector.restoreSubVector(pressure_is, part)
    assert pressure_content == pytest.approx(1, rel=0, abs=256*np.finfo(float).eps)
    assert info["fresh_unprojected_rnorm"] <= stokes.tolerance*info["fresh_reference_rnorm"]


@pytest.mark.tier_b
@pytest.mark.skipif(uw.mpi.size > 1, reason="Existing rotated direct-LU datum is serial-only")
@pytest.mark.parametrize("requested", [False, True])
def test_lu_datum_follows_verification_and_preserves_hydrostatics(monkeypatch, requested):
    stokes, _load = _hydrostatic_stokes()
    stokes.petsc_use_pressure_nullspace = requested
    stokes._rotated_use_lu = True
    checked, selected = [], []
    original_checks = rbc._pressure_nullspace_checks
    original_pin = rbc._naive_pressure_pin

    def check(operator, pressure_is, requested):
        result = original_checks(operator, pressure_is, requested)
        checked.append(result["verified"])
        return result

    def select(dm):
        assert checked and checked[-1], "A datum must follow actual operator verification"
        pin = original_pin(dm)
        selected.append(pin)
        return pin

    monkeypatch.setattr(rbc, "_pressure_nullspace_checks", check)
    monkeypatch.setattr(rbc, "_naive_pressure_pin", select)
    stokes.solve()
    _assert_report_fields(stokes)
    _assert_hydrostatic_nodes(stokes, 1)
    info = stokes._rotated_freeslip_info
    assert info["converged"] and len(selected) == 1 and selected[0] is not None
    assert info["pressure_nullspace_requested"] is requested
    assert info["pressure_nullspace_verified"]
    assert not info["pressure_nullspace_attached"]
    assert not info["pressure_nullspace_projection_active"]
    assert info["pressure_nullspace_checks"]["operator_scope"] == "constrained operator before direct-LU pressure datum"
    assert info["pressure_nullspace_checks"]["pressure_datum_global_dof"] == selected[0]
    assert stokes._rotated_pressure_gauge_residual == 0


@pytest.mark.tier_b
def test_true_pressure_gauge_survives_warm_cache_and_collective_refresh():
    stokes, load = _hydrostatic_stokes()
    for key, value in _BLOCK_OPTIONS.items():
        stokes.petsc_options[key] = value
    stokes.solve()
    _assert_flat_solution(stokes, 1)
    context = stokes._rotated_linear_cache["ctx"]
    stokes.solve(zero_init_guess=False)
    _assert_flat_solution(stokes, 1)
    load.array[:, 0, 0] = 2
    stokes.solve(zero_init_guess=False)
    _assert_flat_solution(stokes, 2)
    assert stokes._rotated_linear_cache["ctx"] is context
    assert stokes._rotated_freeslip_info["workspace_reused"]
    # A local cache-state mismatch must make EVERY rank recheck the operator.
    # The analytic hydrostatic solution remains unchanged by this invalidation.
    if uw.mpi.rank == 0:
        stokes._rotated_linear_cache["pressure_operator_state"] -= 1
    stokes.solve(zero_init_guess=False)
    _assert_flat_solution(stokes, 2)
    assert not stokes._rotated_freeslip_info["workspace_reused"]
    assert stokes._rotated_linear_cache["ctx"] is context


@pytest.mark.tier_b
def test_finalized_fields_must_pass_a_fresh_original_residual(monkeypatch):
    stokes, _load = _hydrostatic_stokes()
    for key, value in _BLOCK_OPTIONS.items():
        stokes.petsc_options[key] = value
    original = rbc._finalize_rotated_solution

    def incorrect_finalizer(solver, solution, rotation, rows, remove_rotation):
        # Negative control: doubling p doubles its gradient, so it cannot
        # balance the unchanged unit body force even after a converged solve.
        pressure_is = solver._subdict["pressure"][0]
        part = solution.getSubVector(pressure_is)
        part.scale(2)
        solution.restoreSubVector(pressure_is, part)
        return original(solver, solution, rotation, rows, remove_rotation)

    monkeypatch.setattr(rbc, "_finalize_rotated_solution", incorrect_finalizer)
    stokes.solve()
    info = stokes._rotated_freeslip_info
    _assert_report_fields(stokes)
    assert not info["converged"] and not stokes.solve_report.converged
    assert not info["fresh_projected_residual_passed"]
    assert info["fresh_unprojected_rnorm"] > stokes.tolerance*info["fresh_unprojected_reference_rnorm"]
    assert info["rnorm"] == info["fresh_projected_rnorm"]
    assert int(stokes.snes.getConvergedReason()) <= 0
    assert not stokes._rotated_linear_cache["linear_hint"]


@pytest.mark.tier_b
@pytest.mark.parametrize("use_lu", [False, True])
def test_radial_sphere_rejects_pressure_before_zero_residual_exit(monkeypatch, use_lu):
    mesh = uw.meshing.SphericalShell(radiusInner=.55, radiusOuter=1,
                                    cellSize=1/8, degree=1, qdegree=4)
    velocity = uw.discretisation.MeshVariable("PressureProbeU", mesh, 3, degree=2)
    pressure = uw.discretisation.MeshVariable("PressureProbeP", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = sp.Integer(1)
    stokes.bodyforce = sp.zeros(1, 3)
    stokes.petsc_use_pressure_nullspace = True
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall, normal=mesh.CoordinateSystem.unit_e_0)
    stokes._rotated_pressure_gauge_residual = 123.0
    stokes._rotated_use_lu = use_lu
    observed = []
    original = rbc._pressure_nullspace_checks

    def probe(operator, pressure_is, requested):
        # Independent construction of p=1 on the actual eliminated operator.
        mode = operator.createVecRight()
        mode.set(0)
        part = mode.getSubVector(pressure_is)
        part.set(1)
        mode.restoreSubVector(pressure_is, part)
        right, left = mode.duplicate(), mode.duplicate()
        try:
            operator.mult(mode, right)
            operator.multTranspose(mode, left)
            observed.append((right.norm(PETSc.NormType.INFINITY),
                             left.norm(PETSc.NormType.INFINITY)))
        finally:
            right.destroy()
            left.destroy()
            mode.destroy()
        return original(operator, pressure_is, requested)

    monkeypatch.setattr(rbc, "_pressure_nullspace_checks", probe)
    if use_lu:
        def forbidden_pin(dm):
            pytest.fail("An invalid pressure mode must be rejected before selecting a datum")
        monkeypatch.setattr(rbc, "_naive_pressure_pin", forbidden_pin)
        with pytest.raises(NotImplementedError, match="constant pressure.*not a right and left null mode"):
            stokes.solve()
        assert observed
        assert stokes._rotated_pressure_gauge_residual == 0
        return
    stokes.solve()
    info = stokes._rotated_freeslip_info
    _assert_report_fields(stokes)
    assert observed, "Even an initially zero residual must verify the assembled operator"
    checks = info["pressure_nullspace_checks"]
    assert info["pressure_nullspace_requested"] and not info["pressure_nullspace_verified"]
    assert not info["pressure_nullspace_attached"]
    assert max(checks["right_relative"], checks["left_relative"]) > checks["tolerance"]
    assert stokes._rotated_pressure_gauge_residual == 0
    assert not info["pressure_coarse_correction_active"]
    assert info["pressure_coarse_correction_metrics"] is None
    assert info["fresh_unprojected_rnorm"] == info["fresh_projected_rnorm"] == 0
    assert info["rotation_gauge"]["orthonormal"] == 3
    assert all(mode["accepted"] and mode["op_viol"] is not None
               for mode in info["rotation_gauge"]["modes"])
    nullspace = rbc._rotated_nullspace(stokes, info["Q"], info["normal_rows"],
                                     pressure_null_verified=False)
    try:
        assert len(nullspace.getVecs()) == 3
    finally:
        nullspace.destroy()
