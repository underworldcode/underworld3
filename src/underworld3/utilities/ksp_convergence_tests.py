"""One Python convergence test per KSP, shared by the library's KSP tests.

petsc4py allows exactly one ``KSP.addConvergenceTest`` per KSP and offers no way to take
it off again. The wall-clock guard (``systems/solver_health.py``) and the convergence
policy's linear floor (``SolverBaseClass._install_linear_floor_test``) each need one, so
both register here. The first registration on a KSP prepends a single dispatcher, and
every test registered on that KSP runs in it, in the order registered, in front of
PETSc's own test. A test returns ``ITERATING`` to hand the decision on; the first other
reason ends the iteration, and when every test returns ``ITERATING`` the KSP's native
test decides exactly as it would have (the guard's design note verified this for one
prepended test).
"""

from petsc4py import PETSc

_ITERATING = PETSc.KSP.ConvergedReason.ITERATING


def _chain(ksp):
    """The dispatcher's ordered tests on ``ksp``, None when it has none, or False when
    a convergence test the library did not set owns the KSP."""
    current = ksp.getConvergenceTest()
    if current is None:
        return None
    return getattr(current[0], "_uw_ksp_tests", False)


def register(ksp, name, test):
    """Run ``test(ksp, its, rnorm)`` in ``ksp``'s convergence test under ``name``,
    replacing a test registered under that name before. Returns False, and registers
    nothing, when a convergence test the library did not set owns the KSP."""
    tests = _chain(ksp)
    if tests is False:
        return False
    if tests is None:
        tests = {}

        def _dispatch(ksp, its, rnorm):
            for fn in tuple(tests.values()):
                reason = fn(ksp, its, rnorm)
                if reason != _ITERATING:
                    return reason
            return _ITERATING

        _dispatch._uw_ksp_tests = tests
        ksp.addConvergenceTest(_dispatch, prepend=True)
    tests[name] = test
    return True


def registered(ksp, name):
    """Whether a test is registered on ``ksp`` under ``name``."""
    tests = _chain(ksp)
    return bool(tests) and name in tests
