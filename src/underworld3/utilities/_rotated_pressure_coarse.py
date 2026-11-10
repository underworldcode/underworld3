"""Balanced pressure correction around an independent borrowed preconditioner."""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

class _BalancedPressureCoarse:
    """Borrow A, M0, velocity KSP and field IS; own only vectors and tiny coarse data.

    V=[z,w], w approximates Auu^dagger(Az)_u with its natural response scale.
    Apply E+(I-EA)M0^-1(I-AE), E=V(V^TAV)^-1V^T. The only gauge
    projection removes only the supplied verified rigid rotations, which may be empty.
    """

    def __init__(self, operator, base_pc, velocity_ksp, velocity_is, pressure_is, normal_rows, rotations, coarse_solve_backward_bound):
        if coarse_solve_backward_bound != 1e-10:
            raise ValueError('Require the validated 1e-10 coarse-solve backward bound')
        self.a, self.base, self.velocity_ksp = (operator, base_pc, velocity_ksp)
        self.comm = operator.comm.tompi4py()
        self.base_id = base_pc.getId()
        self.velocity_is, self.pressure_is = (velocity_is, pressure_is)
        self.normal_rows = list(normal_rows)
        self.bound = coarse_solve_backward_bound
        self.rotations = []
        for mode in rotations:
            vector = mode.copy()
            for previous in self.rotations:
                vector.axpy(-previous.dot(vector), previous)
            norm = vector.norm()
            if norm == 0:
                raise ValueError('Dependent true rotation modes')
            vector.scale(1 / norm)
            self.rotations.append(vector)
            pressure = vector.getSubVector(pressure_is)
            pressure_norm = pressure.norm()
            vector.restoreSubVector(pressure_is, pressure)
            if pressure_norm != 0:
                raise ValueError('True rotations must have exactly zero pressure components')
        self.true_space = PETSc.NullSpace().create(vectors=self.rotations, constant=False, comm=operator.comm) if self.rotations else None
        if self.true_space is not None and (not self.true_space.test(operator)):
            raise ValueError('Rotation modes are not true null vectors')
        self.z = operator.createVecRight()
        self.z.set(0)
        part = self.z.getSubVector(pressure_is)
        part.set(1)
        self.z.restoreSubVector(pressure_is, part)
        self.z.normalize()
        self.w = self.z.duplicate()
        self.g = self.z.duplicate()
        self.v = (self.z, self.w)
        self.av = [self.z.duplicate(), self.z.duplicate()]
        self.atv = [self.z.duplicate(), self.z.duplicate()]
        self.abs_atv = [self.z.duplicate(), self.z.duplicate()]
        self.r_s = self.z.duplicate()
        self.s = self.z.duplicate()
        self.actual_image = self.z.duplicate()
        self.refreshes = 0
        self.applications = 0
        self.closed = False
        self.maximum_coarse_relative = 0.0
        self.maximum_backward_error = 0.0
        self.maximum_initial_coarse_relative = 0.0
        self.coarse_refinement_steps = 0
        self.coarse_histories = []
        self.maximum_componentwise_backward = 0.0
        try:
            self.refresh()
        except Exception:
            self.destroy(None)
            raise

    def _zero_normal(self, vector):
        lo, hi = vector.getOwnershipRange()
        indices = [row - lo for row in self.normal_rows if lo <= row < hi]
        vector.getArray()[indices] = 0

    def refresh(self):
        """Caller must refresh after ANY A/PC update; native Python-PC setUp does so."""
        if self.true_space is not None and not self.true_space.test(self.a):
            raise ArithmeticError('Supplied rotations are not null modes of the refreshed operator')
        self.a.mult(self.z, self.g)
        self.w.set(0)
        # H^dagger acts on the range of H. A*z can carry assembly roundoff
        # along an exact rotation; asking singular H to reduce that component
        # stalls even when its compatible residual is already tiny. Keep A*z
        # intact for the coarse equations and project only this solve's copy.
        response_source = self.g.copy()
        if self.true_space is not None:
            self.true_space.remove(response_source)
        source = response_source.getSubVector(self.velocity_is)
        target = self.w.getSubVector(self.velocity_is)
        try:
            self.velocity_ksp.solve(source, target)
            if self.velocity_ksp.getConvergedReason() <= 0:
                raise ArithmeticError('Velocity response PC failed')
        finally:
            self.w.restoreSubVector(self.velocity_is, target)
            response_source.restoreSubVector(self.velocity_is, source)
            response_source.destroy()
        self._zero_normal(self.w)
        if self.true_space is not None:
            self.true_space.remove(self.w)
        if not np.isfinite(self.w.norm()) or self.w.norm() == 0:
            raise ArithmeticError('Degenerate coarse velocity response; no regularization or fallback')
        source = self.g.getSubVector(self.velocity_is)
        target = self.w.getSubVector(self.velocity_is)
        actual = target.duplicate()
        self.velocity_ksp.getOperators()[0].mult(target, actual)
        actual.axpy(-1, source)
        response_error = float(actual.norm() / max(source.norm(), np.finfo(float).tiny))
        actual.destroy()
        self.w.restoreSubVector(self.velocity_is, target)
        self.g.restoreSubVector(self.velocity_is, source)
        for index, vector in enumerate(self.v):
            self.a.mult(vector, self.av[index])
            self.a.multTranspose(vector, self.atv[index])
        self.c = np.array([[left.dot(image) for image in self.av] for left in self.v], dtype=float)
        singular = np.linalg.svd(self.c, compute_uv=False)
        if not np.isfinite(singular).all() or singular[0] == 0 or singular[-1] <= 64 * np.finfo(float).eps * singular[0]:
            raise ArithmeticError('Degenerate scaled2x2 coarse block; no regularization or fallback')
        self.condition = float(singular[0] / singular[-1])
        self.vnorm = np.array([vector.norm() for vector in self.v])
        self.g_dot_w = float(self.g.dot(self.w))
        self.velocity_response_error = response_error
        self._build_componentwise_scales()
        self.a_handle = self.a.handle
        self.a_state = self.a.stateGet()
        self.pc_state = self.base.stateGet()
        self.refreshes += 1
        self.w_norm = float(self.w.norm())

    def _build_componentwise_scales(self):
        """Cache |A|^T|vi| using owned rows and bounded256-row ADD batches."""
        lo, hi = self.a.getOwnershipRange()
        if self.z.getOwnershipRange() != (lo, hi):
            raise ValueError('Balanced correction requires matching row and column ownership layouts')
        for target in self.abs_atv:
            target.set(0)
        self.actual_image.set(0)
        local_mmax = 0
        owned = [vector.getArray(readonly=True) for vector in self.v]
        for start in range(lo, hi, 256):
            columns = []
            weights = [[], []]
            for row in range(start, min(start + 256, hi)):
                cols, values = self.a.getRow(row)
                cols = cols.copy()
                absolute = np.abs(values)
                local_mmax = max(local_mmax, len(cols))
                columns.append(cols)
                for index in range(2):
                    weights[index].append(absolute * abs(owned[index][row - lo]))
            indices = np.concatenate(columns)
            for index, target in enumerate(self.abs_atv):
                target.setValues(indices, np.concatenate(weights[index]), addv=PETSc.InsertMode.ADD_VALUES)
            self.actual_image.setValues(indices, np.ones(len(indices)), addv=PETSc.InsertMode.ADD_VALUES)
        for target in [*self.abs_atv, self.actual_image]:
            target.assemblyBegin()
            target.assemblyEnd()
        comm = self.a.comm.tompi4py()
        mmax = int(comm.allreduce(local_mmax, op=MPI.MAX))
        cmax = int(round(self.actual_image.norm(PETSc.NormType.INFINITY)))
        N = self.a.getSize()[0]
        P = self.a.comm.size
        u = float(np.finfo(PETSc.RealType).eps / 2)
        k = mmax + cmax + 2 * N + 2 * P + 8
        if k * u >= 1:
            raise ArithmeticError('Roundoff envelope is undefined')
        self.roundoff = dict(u=u, k=k, mmax=mmax, cmax=cmax, N=N, P=P, gamma_k=k * u / (1 - k * u), expression='k=mmax+cmax+2N+2P+8; gamma_k=ku/(1-ku)')

    def _audit(self, rhs, result, q, rhs_norm):
        self.a.mult(result, self.actual_image)
        self.actual_image.axpy(-1, rhs)
        defect = np.array([vector.dot(self.actual_image) for vector in self.v])
        absolute_y = np.abs(result.getArray(readonly=True))
        absolute_r = np.abs(rhs.getArray(readonly=True))
        local = np.array([np.dot(target.getArray(readonly=True), absolute_y) + np.dot(np.abs(vector.getArray(readonly=True)), absolute_r) for target, vector in zip(self.abs_atv, self.v)])
        scales = self.a.comm.tompi4py().allreduce(local, op=MPI.SUM)
        if not np.isfinite(scales).all() or np.any(scales < 0):
            raise ArithmeticError('Invalid componentwise backward scales')
        eta = np.array([abs(value) / scale if scale > 0 else 0.0 if value == 0 else np.inf for value, scale in zip(defect, scales)])
        return dict(backward_error=float(np.max(eta)), componentwise_eta=eta.tolist(), componentwise_scales=scales.tolist(), raw_defect=defect.tolist(), old_rhs_forward=(np.abs(defect) / (self.vnorm * rhs_norm)).tolist(), cached_defect=(np.array([image.dot(result) for image in self.atv]) - q).tolist())

    def setUp(self, pc):
        self.base.setUp()
        changed = self.a.handle != self.a_handle or self.a.stateGet() != self.a_state or self.base.stateGet() != self.pc_state
        if self.comm.allreduce(changed, op=MPI.LOR):
            self.refresh()

    def _solve_coarse(self, rhs):
        coefficients = np.linalg.solve(self.c, rhs)
        scale = np.linalg.norm(self.c, np.inf) * np.linalg.norm(coefficients, np.inf) + np.linalg.norm(rhs, np.inf)
        error = float(np.linalg.norm(self.c @ coefficients - rhs, np.inf) / max(scale, np.finfo(float).tiny))
        if not np.isfinite(error) or error > self.bound:
            raise ArithmeticError('Coarse linear solve failed backward-error bound')
        self.maximum_backward_error = max(self.maximum_backward_error, error)
        return coefficients

    def apply(self, pc, rhs, result):
        stale = not self.a.handle or self.a.handle != self.a_handle or self.a.stateGet() != self.a_state or self.base.stateGet() != self.pc_state
        if self.comm.allreduce(stale, op=MPI.LOR):
            raise ArithmeticError('Stale coarse cache detected collectively; explicit refresh required')
        rhs_norm = rhs.norm()
        if rhs_norm == 0:
            result.set(0)
            return
        q = np.array([vector.dot(rhs) for vector in self.v])
        c_r = self._solve_coarse(q)
        rhs.copy(self.r_s)
        for coefficient, image in zip(c_r, self.av):
            self.r_s.axpy(-coefficient, image)
        self.base.apply(self.r_s, self.s)
        s_rhs = np.array([image.dot(self.s) for image in self.atv])
        c_s = self._solve_coarse(s_rhs)
        self.last_operation = dict(q=q.tolist(), alpha=c_r.tolist(), beta_rhs=s_rhs.tolist(), beta=c_s.tolist())
        self.s.copy(result)
        for coefficient, vector in zip(c_r - c_s, self.v):
            result.axpy(coefficient, vector)
        if self.true_space is not None:
            self.true_space.remove(result)
        audit = self._audit(rhs, result, q, rhs_norm)
        history = [audit]
        relative = audit['backward_error']
        bound = self.roundoff['gamma_k']
        for step in range(3):
            if np.isfinite(relative) and relative <= bound:
                break
            if not np.isfinite(relative):
                raise ArithmeticError('Nonfinite balanced coarse residual')
            coefficients = self._solve_coarse(-np.asarray(audit['raw_defect']))
            for coefficient, vector in zip(coefficients, self.v):
                result.axpy(coefficient, vector)
            audit = self._audit(rhs, result, q, rhs_norm)
            new_relative = audit['backward_error']
            history.append(audit)
            if not np.isfinite(new_relative) or (new_relative >= relative and new_relative > bound):
                raise ArithmeticError(f'Coarse-only refinement nonimprovement: {history}; Ccondition={self.condition}')
            relative = new_relative
        if relative > bound:
            raise ArithmeticError(f'Coarse-only refinement exhausted3steps: {history}; Ccondition={self.condition}')
        steps = len(history) - 1
        self.coarse_refinement_steps += steps
        self.coarse_histories.append(dict(steps=steps, history=history, Ccondition=self.condition))
        self.coarse_histories = self.coarse_histories[-32:]
        self.maximum_initial_coarse_relative = max(self.maximum_initial_coarse_relative, max(history[0]['old_rhs_forward']))
        self.maximum_coarse_relative = max(self.maximum_coarse_relative, max(history[-1]['old_rhs_forward']))
        self.maximum_componentwise_backward = max(self.maximum_componentwise_backward, relative)
        self.applications += 1

    def metrics(self):
        return dict(coarse_matrix=self.c.tolist(), coarse_condition=self.condition, g_dot_w=self.g_dot_w, velocity_response_relative_error=self.velocity_response_error, criterion_version='componentwise_backward_roundoff', roundoff=self.roundoff, maximum_componentwise_backward=self.maximum_componentwise_backward, backward_scale='|vi|^T(|A||y|+|r|); zero scale passes only exact zero defect', maximum_old_rhs_forward_relative=self.maximum_coarse_relative, old_forward_scale='max_i |v_i^T(Ay-r)|/(||v_i||2 ||r||2)', coarse_solve_backward_bound=self.bound, maximum_coarse_solve_backward_error=self.maximum_backward_error, maximum_initial_coarse_relative=self.maximum_initial_coarse_relative, coarse_refinement_steps=self.coarse_refinement_steps, last_coarse_histories=self.coarse_histories, refreshes=self.refreshes, applications=self.applications, natural_w_norm=self.w_norm, true_null_modes=len(self.rotations), base_pc_id=self.base_id)

    def destroy(self, pc):
        if not self.closed:
            for vector in [*self.rotations, self.z, self.w, self.g, *self.av, *self.atv, *self.abs_atv, self.r_s, self.s, self.actual_image]:
                vector.destroy()
            if self.true_space is not None:
                self.true_space.destroy()
            self.closed = True

class _RotatedBalancedPressureCoarse(_BalancedPressureCoarse):
    """Refresh the retained native fieldsplit before the balanced correction."""

    def setUp(self, pc):
        operator = pc.getOperators()[0]
        if pc.handle == self.base.handle or self.base.getType() != 'fieldsplit':
            raise ArithmeticError('Balanced correction must retain a distinct fieldsplit PC')
        changed = operator.handle != self.a_handle or operator.stateGet() != self.a_state or self.base.stateGet() != self.pc_state
        if self.comm.allreduce(changed, op=MPI.LOR):
            self.a = operator
            self.base.setOperators(operator)
            self.base.setUp()
            self.velocity_ksp = self.base.getFieldSplitSubKSP()[0]
            schur = self.base.getFieldSplitSubKSP()[1].getOperators()[0]
            schur.setNullSpace(PETSc.NullSpace())
            self.refresh()
