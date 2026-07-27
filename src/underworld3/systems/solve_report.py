"""Solver difficulty / convergence reporting.

Lives *with the solvers* (not ``utilities``): a :class:`SolveReport` is the record every
SNES-based solver leaves after a solve. It is populated by default on every solve (read via
``solver.solve_report``) and returned by ``solver.estimate_difficulty()``.

Pure-Python (imports only the standard library) so it can be used without rebuilding the
Cython solver extension, and consumed by continuation / homotopy drivers.

See ``underworld3.cython.petsc_generic_snes_solvers.SolverBaseClass``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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

    def __str__(self) -> str:
        rho = f"{self.rho:.3f}" if self.rho is not None else "n/a"
        return (
            f"SolveReport({self.reason_str}, nl={self.nl_its}, ksp={self.ksp_its}, "
            f"|F|={self.fnorm:.3e}, rho={rho}"
            + (", bounded" if self.bounded else "")
            + ")"
        )
