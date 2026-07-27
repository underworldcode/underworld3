"""Solver difficulty / convergence reporting.

Lives *with the solvers* (not ``utilities``): a :class:`SolveReport` is the record every
SNES-based solver leaves after a solve. It is populated by default on every solve (read via
``solver.solve_report``) and returned by ``solver.estimate_difficulty()``.

Pure-Python (imports only the standard library) so it can be used without rebuilding the
Cython solver extension, and consumed by continuation / homotopy drivers.

See ``underworld3.cython.petsc_generic_snes_solvers.SolverBaseClass``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

# PETSc SNESConvergedReason codes -> short names. Mirrors the compact map in
# SolverBaseClass._convergence_reasons; duplicated here so this module imports without the
# Cython solver extension present.
REASON_STRINGS = {
    1: "CONVERGED_FNORM_ABS",
    2: "CONVERGED_FNORM_RELATIVE",
    3: "CONVERGED_SNORM_RELATIVE",
    4: "CONVERGED_ITS",
    0: "ITERATING",
    -1: "DIVERGED_FUNCTION_DOMAIN",
    -2: "DIVERGED_FUNCTION_COUNT",
    -3: "DIVERGED_LINEAR_SOLVE",
    -4: "DIVERGED_FNORM_NAN",
    -5: "DIVERGED_MAX_IT",
    -6: "DIVERGED_LINE_SEARCH",
    -7: "DIVERGED_INNER",
    -8: "DIVERGED_LOCAL_MIN",
    -9: "DIVERGED_DTOL",
    -10: "DIVERGED_JACOBIAN_DOMAIN",
    -11: "DIVERGED_TR_DELTA",
}


def reason_string(reason: int) -> str:
    """Human-readable name for a PETSc SNES converged-reason code."""
    return REASON_STRINGS.get(int(reason), f"UNKNOWN_{reason}")


# PETSc KSPConvergedReason codes -> short names. This is a DIFFERENT namespace from the
# SNES table above: the integers overlap but name different outcomes (e.g. -3 is
# SNES DIVERGED_LINEAR_SOLVE but KSP DIVERGED_MAX_IT). Rotated free-slip solves report
# KSP codes and must be rendered with this table. Names carry the KSP_ prefix so a
# report string is unambiguous about which namespace it came from. Verified against
# petsc4py's PETSc.KSP.ConvergedReason (see test_1055).
KSP_REASON_STRINGS = {
    1: "KSP_CONVERGED_RTOL_NORMAL_EQUATIONS",
    2: "KSP_CONVERGED_RTOL",
    3: "KSP_CONVERGED_ATOL",
    4: "KSP_CONVERGED_ITS",
    5: "KSP_CONVERGED_NEG_CURVE",
    6: "KSP_CONVERGED_STEP_LENGTH",
    7: "KSP_CONVERGED_HAPPY_BREAKDOWN",
    9: "KSP_CONVERGED_ATOL_NORMAL_EQUATIONS",
    0: "KSP_ITERATING",
    -2: "KSP_DIVERGED_NULL",
    -3: "KSP_DIVERGED_MAX_IT",
    -4: "KSP_DIVERGED_DTOL",
    -5: "KSP_DIVERGED_BREAKDOWN",
    -6: "KSP_DIVERGED_BREAKDOWN_BICG",
    -7: "KSP_DIVERGED_NONSYMMETRIC",
    -8: "KSP_DIVERGED_INDEFINITE_PC",
    -9: "KSP_DIVERGED_NANORINF",
    -10: "KSP_DIVERGED_INDEFINITE_MAT",
    -11: "KSP_DIVERGED_PCSETUP_FAILED",
}


def ksp_reason_string(reason: int) -> str:
    """Human-readable name for a PETSc KSP converged-reason code."""
    return KSP_REASON_STRINGS.get(int(reason), f"KSP_UNKNOWN_{reason}")


def contraction(history) -> Optional[float]:
    """Geometric-mean per-iteration contraction factor ρ from a residual ladder.

    ρ = (‖F_last‖ / ‖F_first‖) ** (1 / (n-1)). Returns ``None`` when there are fewer than
    two finite, positive residuals to compare (ρ is undefined for a single point).
    """
    r = [h for h in history if h == h and h > 0.0]  # finite & positive
    if len(r) < 2:
        return None
    return (r[-1] / r[0]) ** (1.0 / (len(r) - 1))


@dataclass(frozen=True)
class SolveReport:
    """Difficulty / convergence record from a single solve.

    Populated by ``SolverBaseClass`` after every solve (default-on) and returned by
    ``estimate_difficulty()``. Read-only; access the most recent via ``solver.solve_report``
    and the recent history via ``solver.solve_history``.

    Attributes
    ----------
    reason, reason_str, converged
        PETSc SNES converged-reason code, its name, and whether it indicates convergence.
    nl_its, ksp_its
        Nonlinear (SNES) iterations and cumulative linear (KSP) iterations of this solve.
    fnorm, fnorm0, reduction
        Final ‖F‖, initial ‖F‖ (first entry of the residual ladder, if captured), and their
        ratio ``fnorm / fnorm0``.
    rho
        Geometric-mean per-iteration contraction of the residual ladder (``None`` if < 2
        points). A cheap difficulty indicator: ρ ≪ 1 is easy, ρ → 1 is a struggling solve.
    fev
        Function evaluations (includes line-search backtracks), if available.
    history
        The residual ladder ``getConvergenceHistory()`` recorded for this solve.
    bounded
        ``True`` when this report came from an iteration-capped (``estimate_difficulty``)
        solve — i.e. work was truncated, not a genuine failure.
    sub
        Work done by each fieldsplit sub-solve, keyed by block name (``"velocity"``,
        ``"pressure"`` for Stokes). ``ksp_its`` above counts only the OUTER Krylov
        iterations, which Eisenstat–Walker collapses to about one per Newton step — the
        real cost of a Stokes solve is ``sub["velocity"].its`` multigrid cycles. Empty
        for solvers with no fieldsplit preconditioner. See
        ``underworld3.systems.solver_health``.
    deadline_expired
        ``True`` when a wall-clock guard armed with ``solver.guard(...)`` cut this solve
        short. Distinguishes "the budget ran out" from a genuine divergence: both report
        ``DIVERGED_LINEAR_SOLVE``.
    """

    reason: int
    reason_str: str
    converged: bool
    nl_its: int
    ksp_its: int
    fnorm: float
    fnorm0: Optional[float] = None
    reduction: Optional[float] = None
    rho: Optional[float] = None
    fev: Optional[int] = None
    history: Tuple[float, ...] = ()
    bounded: bool = False
    sub: Mapping[str, "SubSolveReport"] = field(default_factory=dict)
    deadline_expired: bool = False

    def __str__(self) -> str:
        rho = f"{self.rho:.3f}" if self.rho is not None else "n/a"
        sub = "".join(f", {r.name}={r.its}" for r in self.sub.values())
        return (
            f"SolveReport({self.reason_str}, nl={self.nl_its}, ksp={self.ksp_its}, "
            f"|F|={self.fnorm:.3e}, rho={rho}"
            + sub
            + (", bounded" if self.bounded else "")
            + (", deadline" if self.deadline_expired else "")
            + ")"
        )
