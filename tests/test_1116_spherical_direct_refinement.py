"""Actual approximate LU inverses refine fixed equations with a known null gauge."""

from contextlib import contextmanager, ExitStack

import numpy as np
import pytest
from petsc4py import PETSc
import underworld3 as uw

from underworld3.utilities._spherical_stokes_gap import _refine_direct_solution

pytestmark = pytest.mark.level_1


class _CountSolves:
    """Count applications while delegating every solve to an actual PETSc LU."""

    def __init__(self, ksp):
        self.ksp, self.calls = ksp, 0

    def solve(self, rhs, solution):
        self.calls += 1
        self.ksp.solve(rhs, solution)

    def getConvergedReason(self):
        return self.ksp.getConvergedReason()


@contextmanager
def _problem(factor_scale):
    # One original null row represents the paired pressure/reaction gauge.
    # Its artificial pin is zero, while the physical gauge prescribes x[0]=7.
    # All other equations have the exact solution x[i]=1+i/N.
    with ExitStack() as cleanup:
        size = 4*uw.mpi.size
        pinned = PETSc.Mat().createAIJ((size, size), nnz=1, comm=PETSc.COMM_WORLD)
        cleanup.callback(pinned.destroy)
        lo, hi = pinned.getOwnershipRange()
        for row in range(lo, hi):
            pinned.setValue(row, row, 1 if row == 0 else 2+row % 3)
        pinned.assemble()
        matrix = pinned.copy(); cleanup.callback(matrix.destroy)
        if lo == 0:
            matrix.setValue(0, 0, 0)
        matrix.assemble()
        approximate = pinned.copy(); cleanup.callback(approximate.destroy)
        approximate.scale(factor_scale)
        exact = pinned.createVecRight(); cleanup.callback(exact.destroy)
        exact.array[:] = 1+np.arange(lo, hi)/size
        if lo == 0:
            exact.setValue(0, 7)
        exact.assemble()
        rhs = exact.duplicate(); cleanup.callback(rhs.destroy)
        matrix.mult(exact, rhs)
        pinned_rhs = rhs.copy(); cleanup.callback(pinned_rhs.destroy)
        ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD); cleanup.callback(ksp.destroy)
        ksp.setOperators(approximate); ksp.setType("preonly")
        ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps")
        ksp.setUp()
        ungauged = rhs.duplicate(); cleanup.callback(ungauged.destroy)
        ksp.solve(pinned_rhs, ungauged)
        assert ksp.getConvergedReason() > 0
        physical = ungauged.copy(); cleanup.callback(physical.destroy)

        def restore_gauge(vector):
            if lo == 0:
                vector.setValue(0, 7)
            vector.assemble()

        restore_gauge(physical)
        counted = _CountSolves(ksp)
        yield counted, pinned, pinned_rhs, matrix, rhs, ungauged, physical, restore_gauge, exact


@pytest.mark.tier_b
@pytest.mark.parametrize("factor_scale,steps", ((1., 0), (1.001, 2)))
def test_same_equations_recover_known_solution_and_preserve_accepted_fields(factor_scale, steps):
    with _problem(factor_scale) as args:
        counted, pinned, pinned_rhs, matrix, rhs, ungauged, physical, gauge, exact = args
        original_fields, original_rhs = physical.array.copy(), rhs.array.copy()
        factor_handle = counted.ksp.getPC().getFactorMatrix().handle
        history = _refine_direct_solution(*args[:-1])
        assert len(history)-1 == counted.calls == steps
        assert history[-1]["full_relative"] <= 1e-8
        assert all(np.isfinite(value) for record in history for value in record.values())
        assert counted.ksp.getPC().getFactorMatrix().handle == factor_handle
        np.testing.assert_array_equal(rhs.array, original_rhs)
        if steps == 0:
            np.testing.assert_array_equal(physical.array, original_fields)
        else:
            assert history[0]["full_relative"] > 1e-8
            assert history[-1]["pinned_relative"] < history[0]["pinned_relative"]
        physical.axpy(-1, exact)
        assert physical.norm(PETSc.NormType.INFINITY) <= 1e-8*exact.norm(PETSc.NormType.INFINITY)


@pytest.mark.tier_b
@pytest.mark.parametrize("factor_scale,message,calls", ((-1., "nonimproving", 1), (2., "five corrections", 5)))
def test_fixed_residual_guard_rejects_growth_and_bounded_exhaustion(factor_scale, message, calls):
    with _problem(factor_scale) as args:
        with pytest.raises(ArithmeticError, match=message):
            _refine_direct_solution(*args[:-1])
        assert args[0].calls == calls


@pytest.mark.tier_b
def test_nonfinite_original_equation_residual_is_rejected_before_correction():
    with _problem(1.) as args:
        args[6].set(float("inf"))
        with pytest.raises(ArithmeticError, match="Nonfinite"):
            _refine_direct_solution(*args[:-1])
        assert args[0].calls == 0
