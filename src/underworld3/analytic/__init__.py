r"""Exact solutions for validating Underworld3 solves.

Every solution here is a closed form we can compare a numerical answer against —
the code's source of truth for benchmarking and convergence testing. They share
one contract (:class:`AnalyticSolution`), so a validation run reads the same way
whatever the solution::

    sol = uw.analytic.SolCx(mesh, eta_A=1.0, eta_B=1.0e6)

    stokes.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    stokes.bodyforce = sol.fn_bodyforce
    sol.apply_boundary_conditions(stokes)

    stokes.solve()
    rel = sol.error("velocity", stokes.u)

:func:`available` lists what is here; :func:`describe` summarises one solution
without constructing it.

These are worth reaching for whenever a solver change needs evidence. The SolCx
port alone caught a direct ``ksponly + lu`` solve silently mangling the singular
saddle on a free-slip problem — a wrong answer that no residual norm reported.

See Also
--------
underworld3.analytic._base : the contract each solution satisfies.
"""

from ._base import AnalyticSolution, FreeSlipWalls, FixedWalls

from .inclusion import EllipticalInclusion
from .velic import SolA, SolB, SolC, SolCx, SolDA, SolDB2d, SolDB3d, SolKx, SolKz, SolM, SolNL

__all__ = [
    "AnalyticSolution",
    "FreeSlipWalls",
    "FixedWalls",
    "EllipticalInclusion",
    "SolA",
    "SolB",
    "SolC",
    "SolCx",
    "SolDA",
    "SolDB2d",
    "SolDB3d",
    "SolKx",
    "SolKz",
    "SolM",
    "SolNL",
    "available",
    "describe",
]

# The solutions this namespace offers. Explicit rather than introspected, so the
# listing stays truthful while solutions are still being migrated onto the
# contract, and so a solution needing an optional dependency can be listed
# without being importable.
_SOLUTIONS = {
    "EllipticalInclusion": EllipticalInclusion,
    "SolA": SolA,
    "SolB": SolB,
    "SolC": SolC,
    "SolCx": SolCx,
    "SolDA": SolDA,
    "SolDB2d": SolDB2d,
    "SolDB3d": SolDB3d,
    "SolKx": SolKx,
    "SolKz": SolKz,
    "SolM": SolM,
    "SolNL": SolNL,
}


def available():
    """Names of the solutions in this namespace, in alphabetical order.

    Returns
    -------
    list of str
        Every name that can be constructed as ``uw.analytic.<name>(mesh, ...)``.
    """

    return sorted(_SOLUTIONS)


def describe(name):
    """One-line summary of a solution, without constructing it.

    Parameters
    ----------
    name : str
        A name from :func:`available`.

    Returns
    -------
    str
        The first line of the solution's docstring.
    """

    try:
        solution = _SOLUTIONS[name]
    except KeyError:
        raise ValueError(
            f"no analytic solution named {name!r}; available: "
            f"{', '.join(available())}"
        ) from None

    docstring = solution.__doc__ or ""
    return docstring.strip().split("\n")[0]
