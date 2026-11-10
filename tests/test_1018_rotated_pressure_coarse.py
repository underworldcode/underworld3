"""Real PETSc component gates; no UW model or production-equation changes.

R7 requires operationwise/Cassembly/coarse backward errors at declared gamma bounds.
Original full equation bound1e-10 and field bound1e-7 are unchanged; exact/roundedC
mapforward differences remain conditioning diagnostics, not10-digit PCmap claims.
accounts for the prescribed1e-4 weak coupling (system condition~1e8).
The positive diagonal is M0's test norm only; the original singular A is intact.
"""
import json
from _rotated_pressure_coarse_reference import audit_reference
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
import underworld3 as uw
from underworld3.utilities._rotated_pressure_coarse import _BalancedPressureCoarse as BalancedPressureCoarse
from underworld3.utilities.rotated_bc import _pressure_coarse_correction_needed
pytestmark = pytest.mark.level_2


@pytest.mark.tier_b
def test_correction_requires_requested_measured_nonnullity_on_both_sides():
    for requested, verified, right, left, expected in (
            (True, False, 1.0, 1.0, True),
            (False, False, 1.0, 1.0, False),
            (True, True, 0.0, 0.0, False),
            (True, False, 0.0, 1.0, False),
            (True, False, 1.0, 0.0, False),
            (True, False, None, None, False)):
        checks = dict(requested=requested, verified=verified, right_relative=right,
                      left_relative=left, tolerance=256*np.finfo(float).eps)
        assert _pressure_coarse_correction_needed(checks) is expected

def matrix_from_dense(dense):
    n = len(dense)
    layout = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)
    lo, hi = layout.getOwnershipRange()
    layout.destroy()
    matrix = PETSc.Mat().createAIJ(size=((hi - lo, n), (hi - lo, n)), nnz=(hi - lo, n - (hi - lo)), comm=PETSc.COMM_WORLD)
    lo, hi = matrix.getOwnershipRange()
    for row in range(lo, hi):
        matrix.setValues([row], np.arange(n, dtype=PETSc.IntType), dense[row:row + 1])
    matrix.assemble()
    return matrix

def vector_from_values(matrix, values):
    vector = matrix.createVecRight()
    lo, hi = vector.getOwnershipRange()
    vector.array[:] = values[lo:hi]
    return vector

class PseudoDiagonal:

    def __init__(self, diagonal):
        self.diagonal = diagonal

    def apply(self, pc, rhs, result):
        lo, hi = rhs.getOwnershipRange()
        diagonal = self.diagonal[lo:hi]
        result.array[:] = np.divide(rhs.getArray(readonly=True), diagonal, out=np.zeros_like(rhs.getArray(readonly=True)), where=diagonal != 0)

def toy_values(vector):
    """Eleven-entry manufactured test only; production helper never gathers."""
    assert vector.getSize() == 11
    values = np.zeros(11)
    lo, hi = vector.getOwnershipRange()
    values[lo:hi] = vector.getArray(readonly=True)
    vector.comm.tompi4py().Allreduce(MPI.IN_PLACE, values, op=MPI.SUM)
    return values

def independent_audit(A, dense, helper, rhs, result, values, norm):
    V = np.column_stack([toy_values(vector) for vector in helper.v])
    actual = toy_values(result)
    image = result.duplicate()
    A.mult(result, image)
    image.axpy(-1, rhs)
    raw = np.array([vector.dot(image) for vector in helper.v])
    image.destroy()
    scales = np.abs(V).T @ (np.abs(dense) @ np.abs(actual) + np.abs(values))
    eta = np.array([abs(value) / scale if scale else 0.0 if value == 0 else np.inf for value, scale in zip(raw, scales)])
    assert np.max(eta) <= helper.roundoff['gamma_k']
    np.testing.assert_allclose(scales, helper.coarse_histories[-1]['history'][-1]['componentwise_scales'], rtol=1e-12, atol=0)
    reference, exact_reference, oracle = audit_reference(dense, V, values, 1 / np.diag(norm), actual, helper.c, np.column_stack([toy_values(vector) for vector in helper.av]), np.column_stack([toy_values(vector) for vector in helper.atv]), toy_values(helper.r_s), toy_values(helper.s), helper.last_operation)
    reference_vector = vector_from_values(A, reference)
    difference = result.copy()
    difference.axpy(-1, reference_vector)
    hp_map_error = difference.norm() / max(reference_vector.norm(), 1.0)
    exact_vector = vector_from_values(A, exact_reference)
    exact_difference = result.copy()
    exact_difference.axpy(-1, exact_vector)
    exact_map_error = exact_difference.norm() / max(exact_vector.norm(), 1.0)
    exact_vector.destroy()
    exact_difference.destroy()
    kC = helper.roundoff['mmax'] + helper.roundoff['N'] + helper.roundoff['P'] + 2
    u = helper.roundoff['u']
    gammaC = kC * u / (1 - kC * u)
    oracle.update(actual_rounded_C_map_relative=hp_map_error, exact_C_map_relative_diagnostic=exact_map_error, old_r5_exact_C_map_gate_would_pass=exact_map_error < 1e-10, old_r6_rounded_C_map_gate_would_pass=hp_map_error < 1e-10, Cassembly_k=kC, Cassembly_gamma=gammaC)
    counts = dict(cached_AV_backward=helper.roundoff['mmax'], cached_ATV_backward=helper.roundoff['cmax'], q_dot_backward=helper.roundoff['N'] + helper.roundoff['P'] + 1, alpha_solve_backward=6, rs_axpy_backward=4, M0_apply_backward=1, beta_rhs_backward=helper.roundoff['N'] + helper.roundoff['P'] + 1, beta_solve_backward=6, grouping_backward=4)
    bounds = {name: count * u / (1 - count * u) for name, count in counts.items()}
    oracle.update(operation_counts=counts, operation_gamma=bounds, acceptance_scope='r7 componentwise backward stability plus original-equation/direct-solution gates; mapforward differences are diagnostics')
    if A.comm.rank == 0:
        print('RANK2_ORACLE_STAGE ' + json.dumps(oracle), flush=True)
    assert max((max(row) for row in oracle['decimal80_C_assembly_backward'])) <= gammaC
    assert max(oracle['decimal80_actual_result_backward']) <= helper.roundoff['gamma_k']
    assert all((error <= bounds[name] for name, error in oracle['operation_component_backward'].items()))
    reference_vector.destroy()
    difference.destroy()
    return dict(actual_raw_defects=raw.tolist(), old_rhs_forward=(np.abs(raw) / (np.linalg.norm(V, axis=0) * rhs.norm())).tolist(), componentwise_scales=scales.tolist(), dense_backward=eta.tolist(), decimal80_actual_rounded_C_map_relative=hp_map_error, decimal80_oracle=oracle)

def problem(sign, zero_coupling, n_rotations):
    d = np.array([0.0] * n_rotations + [1.0] * (3 - n_rotations) + [1.0, 2.0, 3.0, 4.0, 5.0])
    z = np.ones(3) / np.sqrt(3)
    q1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    q2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
    g0 = np.array([1.0, 0.2, -0.1, 0.3, 0.25])
    g1 = np.array([0.2, 1.0, 0.1, 0.2, -0.3])
    g2 = np.array([0.1, -0.1, 1.0, 0.3, 0.4])
    coupling = np.zeros((8, 3))
    coupling[3:] = 0.0001 * np.outer(g0, z) + np.outer(g1, q1) + np.outer(g2, q2)
    if zero_coupling:
        coupling[:] = 0
    dense = np.block([[np.diag(d), coupling], [sign * coupling.T, np.zeros((3, 3))]])
    A = matrix_from_dense(dense)
    lo, hi = A.getOwnershipRange()
    uis = PETSc.IS().createGeneral(np.array([row for row in range(lo, hi) if row < 8], dtype=PETSc.IntType), comm=A.comm)
    pis = PETSc.IS().createGeneral(np.array([row for row in range(lo, hi) if row >= 8], dtype=PETSc.IntType), comm=A.comm)
    avv = A.createSubMatrix(uis, uis)
    velocity = PETSc.KSP().create(comm=A.comm)
    velocity.setType('preonly')
    velocity.setOperators(avv)
    velocity.getPC().setType('python')
    velocity.getPC().setPythonContext(PseudoDiagonal(d))
    velocity.setUp()
    norm = np.diag(np.r_[np.where(d == 0, 1, d), np.ones(3)])
    P = matrix_from_dense(norm)
    base = PETSc.PC().create(comm=A.comm)
    base.setType('jacobi')
    base.setOperators(P)
    base.setUp()
    rotations = [vector_from_values(A, np.eye(11)[index]) for index in range(n_rotations)]
    ns = PETSc.NullSpace().create(vectors=rotations, constant=False, comm=A.comm)
    A.setNullSpace(ns)
    A.setTransposeNullSpace(ns)
    return (dense, A, avv, velocity, P, base, uis, pis, rotations, ns, norm)


@pytest.mark.tier_b
def test_velocity_pseudoinverse_projects_only_a_copy_of_its_source():
    """Roundoff along a true rotation must not enter the singular inner solve."""
    dense, A, avv, velocity, P, base, uis, pis, rotations, ns, norm = problem(1.0, False, 1)
    perturbation = np.finfo(float).eps**2
    lo, hi = A.getOwnershipRange()
    if lo <= 0 < hi:
        A.setValue(0, 8, perturbation)
    if lo <= 8 < hi:
        A.setValue(8, 0, perturbation)
    A.assemble()

    class RecordedVelocitySolve:
        def solve(self, rhs, target):
            self.source = rhs.copy()
            velocity.solve(rhs, target)

        def getConvergedReason(self):
            return velocity.getConvergedReason()

        def getOperators(self):
            return velocity.getOperators()

    recorded = RecordedVelocitySolve()
    helper = BalancedPressureCoarse(A, base, recorded, uis, pis, [], rotations, 1e-10)
    mode = rotations[0].getSubVector(uis)
    assert recorded.source.dot(mode) == 0.0
    rotations[0].restoreSubVector(uis, mode)
    # The original A*z remains observable; projecting it in place would change
    # the coarse operator rather than constructing its velocity pseudoinverse.
    assert helper.g.dot(rotations[0]) == pytest.approx(perturbation/np.sqrt(3), rel=1e-14, abs=0)
    expected = np.divide(toy_values(helper.g)[:8], np.diag(dense)[:8],
                         out=np.zeros(8), where=np.diag(dense)[:8] != 0)
    np.testing.assert_allclose(toy_values(helper.w)[:8], expected, rtol=1e-14, atol=0)
    recorded.source.destroy()
    helper.destroy(None)
    for item in [ns, *rotations, velocity, base, avv, P, A, uis, pis]:
        item.destroy()

@pytest.mark.tier_b
@pytest.mark.parametrize('n_rotations', [0, 1, 3])
@pytest.mark.parametrize('sign', [1.0, -1.0], ids=['symmetric', 'nonsymmetric'])
def test_balanced_coarse_equations_and_original_direct_solution(sign, n_rotations):
    dense, A, avv, velocity, P, base, uis, pis, rotations, ns, norm = problem(sign, False, n_rotations)
    base_id = base.getId()
    a_state = A.stateGet()
    helper = BalancedPressureCoarse(A, base, velocity, uis, pis, [], rotations, 1e-10)
    pc = PETSc.PC().create(comm=A.comm)
    pc.setType('python')
    pc.setOperators(A)
    pc.setPythonContext(helper)
    pc.setUp()
    z = np.r_[np.zeros(8), np.ones(3) / np.sqrt(3)]
    g = dense @ z
    w = np.r_[np.divide(g[:8], np.diag(dense)[:8], out=np.zeros(8), where=np.diag(dense)[:8] != 0), np.zeros(3)]
    V = np.column_stack((z, w))
    C = V.T @ dense @ V
    np.testing.assert_allclose(toy_values(helper.w), w, rtol=1e-10, atol=1e-14)
    assert helper.velocity_response_error < 1e-12 and helper.g_dot_w > 0
    independent_coarse_errors = []
    cases = [np.r_[np.zeros(3), np.array([0.4, -0.2, 0.7, 0.1, 0.3]), np.array([0.2, -0.4, 0.6])], z, dense @ np.r_[np.zeros(3), np.ones(8)]]
    for values in cases:
        rhs = vector_from_values(A, values)
        result = rhs.duplicate()
        pc.apply(rhs, result)
        independent_coarse_errors.append(independent_audit(A, dense, helper, rhs, result, values, norm))
        for vector in (rhs, result):
            vector.destroy()
    ksp = PETSc.KSP().create(comm=A.comm)
    ksp.setOperators(A)
    ksp.setType('fgmres')
    ksp.setTolerances(rtol=1e-12, atol=1e-14, max_it=100)
    ksp.setGMRESRestart(30)
    ksp.setPCSide(PETSc.PC.Side.RIGHT)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setPC(base)
    ksp.setPC(pc)
    truth = np.r_[np.zeros(3), np.array([0.6, -0.4, 0.3, 1.1, -0.2]), np.array([0.9, -0.2, 0.4]) + 0.7 / np.sqrt(3)]
    values = dense @ truth
    rhs = vector_from_values(A, values)
    answer = rhs.duplicate()
    answer.set(0)
    ksp.solve(rhs, answer)
    assert ksp.getConvergedReason() > 0
    residual = rhs.duplicate()
    A.mult(answer, residual)
    residual.axpy(-1, rhs)
    relative_residual = residual.norm() / rhs.norm()
    assert relative_residual < 1e-10
    direct = np.zeros(11)
    direct[3:] = np.linalg.solve(dense[3:, 3:], values[3:])
    direct_vector = vector_from_values(A, direct)
    difference = answer.copy()
    difference.axpy(-1, direct_vector)
    field_error = difference.norm() / direct_vector.norm()
    assert field_error < 1e-07
    assert base.getId() == base_id and base.getType() == 'jacobi' and (A.stateGet() == a_state)
    for changed in (A, base):
        if A.comm.rank == 0:
            changed.stateIncrease()
        with pytest.raises(ArithmeticError, match='Stale'):
            helper.apply(pc, rhs, answer)
        helper.refresh()
    lo, hi = A.getOwnershipRange()
    updated = dense.copy()
    updated[:8, :8] *= 1.3
    for row in range(lo, hi):
        A.setValues([row], np.arange(11, dtype=PETSc.IntType), updated[row:row + 1])
    A.assemble()
    with pytest.raises(ArithmeticError, match='Stale'):
        helper.apply(pc, rhs, answer)
    A.createSubMatrix(uis, uis, submat=avv)
    velocity.getPC().getPythonContext().diagonal *= 1.3
    velocity.setOperators(avv)
    base.setUp()
    helper.refresh()
    helper.apply(pc, rhs, answer)
    refresh_audit = independent_audit(A, updated, helper, rhs, answer, values, norm)
    refreshed = helper.metrics()
    assert refreshed['refreshes'] >= 3 and refreshed['coarse_condition'] < 10
    if A.comm.rank == 0:
        print('RANK2_COMPONENT_GATE ' + json.dumps(dict(sign=sign, criterion_version='r7_operationwise_backward_plus_original_solution_oracles', full_equation_relative=relative_residual, direct_field_relative=field_error, independent_coarse_audits=independent_coarse_errors, refresh_audit=refresh_audit, ksp_iterations=ksp.getIterationNumber(), base_pc_retained=True, metrics=refreshed)), flush=True)
    for vector in (rhs, answer, residual, direct_vector, difference):
        vector.destroy()
    ksp.destroy()
    pc.destroy()
    base.destroy()
    velocity.destroy()
    avv.destroy()
    P.destroy()
    A.destroy()
    uis.destroy()
    pis.destroy()
    ns.destroy()
    for vector in rotations:
        vector.destroy()

@pytest.mark.tier_b
def test_degenerate_coarse_response_hard_fails_without_regularization():
    dense, A, avv, velocity, P, base, uis, pis, rotations, ns, norm = problem(1.0, True, 3)
    with pytest.raises(ArithmeticError, match='Degenerate'):
        BalancedPressureCoarse(A, base, velocity, uis, pis, [], rotations, 1e-10)
    base.destroy()
    velocity.destroy()
    avv.destroy()
    P.destroy()
    A.destroy()
    uis.destroy()
    pis.destroy()
    ns.destroy()
    for vector in rotations:
        vector.destroy()
