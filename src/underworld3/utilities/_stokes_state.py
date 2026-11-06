"""Borrowed mixed Stokes state and original-equation observations."""
from __future__ import annotations
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, ExitStack
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import underworld3 as uw

class StokesSolveState:
    """Solver-owned borrowed state; subclasses own physical gauge restoration."""
    def restore_gauge(self, vector):
        raise NotImplementedError

    def __init__(self, environment: Mapping[str, object]) -> None:
        self._environment = environment
        self._lift_integral = None


    def __getitem__(self, key: str) -> object:
        """Read-only compatibility for historical experimental recorder adapters."""
        return self._environment[key]


    @property
    def rhs(self) -> PETSc.Vec:
        return self._environment['rhs']


    @property
    def lift_integral(self) -> float:
        e = self._environment
        if self._lift_integral is None:
            self._lift_integral = float(e['pressure_jump_integral'])+e['mpi'].allreduce(float(e['lift_gap']), op=MPI.SUM)
        return self._lift_integral


    def gauged_copy(self, vector: PETSc.Vec) -> PETSc.Vec:
        result = vector.copy()
        self.restore_gauge(result)
        return result


    def build_iterate(self, ksp: PETSc.KSP) -> PETSc.Vec:
        result = self.rhs.duplicate()
        ksp.buildSolution(result)
        return result


    @contextmanager
    def _residual_vector(self, vector: PETSc.Vec) -> Iterator[PETSc.Vec]:
        work = self.rhs.duplicate()
        try:
            self._environment['matrix'].mult(vector, work)
            work.axpy(-1, self.rhs)
            yield work
        finally:
            work.destroy()


    def residual_norms(self, vector: PETSc.Vec) -> dict[str, float]:
        with self._residual_vector(vector) as work:
            scale, actual = self.rhs.norm(), work.norm()
            relative = actual/max(scale, np.finfo(float).tiny)
            if not np.isfinite(relative):
                raise ArithmeticError('Nonfinite original spherical Stokes residual')
            return dict(original_rhs_norm=scale, original_residual_norm=actual,
                        original_relative_residual=relative)


    def residual(self, vector: PETSc.Vec) -> dict[str, object]:
        e = self._environment
        with self._residual_vector(vector) as work:
            scale, actual = self.rhs.norm(), work.norm()
            denominator = max(scale, np.finfo(float).tiny)
            relative = actual/denominator
            if not np.isfinite(relative):
                raise ArithmeticError('Nonfinite original spherical Stokes residual')
            blocks = {}
            lo, hi = work.getOwnershipRange()
            nu, npres = e['nu'], e['npres']
            for name, first, last in (('velocity',0,nu),('pressure',nu,nu+npres),('multiplier',nu+npres,work.getSize())):
                indices = PETSc.IS().createGeneral(np.arange(max(lo,first), min(hi,last), dtype=PETSc.IntType), comm=work.comm)
                block = work.getSubVector(indices)
                try:
                    norm = block.norm()
                    blocks[name] = dict(residual_norm=norm, residual_over_full_rhs=norm/denominator)
                finally:
                    work.restoreSubVector(indices, block)
                    indices.destroy()
            return dict(original_rhs_norm=scale, original_residual_norm=actual,
                        original_relative_residual=relative, blocks=blocks,
                        physical_pressure_integral=e['pmean'].dot(vector)+self.lift_integral,
                        rotation_integrals=[weight.dot(vector) for weight in e['rotation_weights']])

