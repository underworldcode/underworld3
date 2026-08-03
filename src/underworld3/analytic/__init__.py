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
from .kramer import CylindricalStokes
from .richards import GardnerSteady, GardnerTransient
from .transport import AdvectedFront, ErfcDiffusion, Poisson1D, TwoLayerDarcy
from .velic import SolA, SolB, SolC, SolCx, SolDA, SolDB2d, SolDB3d, SolH, SolKx, SolKz, SolM, SolNL

__all__ = [
    "AnalyticSolution",
    "FreeSlipWalls",
    "FixedWalls",
    "AdvectedFront",
    "EllipticalInclusion",
    "ErfcDiffusion",
    "CylindricalStokes",
    "GardnerSteady",
    "GardnerTransient",
    "Poisson1D",
    "SolA",
    "SolB",
    "SolC",
    "SolCx",
    "SolDA",
    "SolDB2d",
    "SolDB3d",
    "SolH",
    "SolKx",
    "SolKz",
    "SolM",
    "SolNL",
    "TwoLayerDarcy",
    "available",
    "describe",
    "is_available",
]

# The solutions this namespace offers. Explicit rather than introspected, so the
# listing stays truthful while solutions are still being migrated onto the
# contract, and so a solution needing an optional dependency can be listed
# without being importable.
_SOLUTIONS = {
    "AdvectedFront": AdvectedFront,
    "CylindricalStokes": CylindricalStokes,
    "EllipticalInclusion": EllipticalInclusion,
    "ErfcDiffusion": ErfcDiffusion,
    "GardnerSteady": GardnerSteady,
    "GardnerTransient": GardnerTransient,
    "Poisson1D": Poisson1D,
    "SolA": SolA,
    "SolB": SolB,
    "SolC": SolC,
    "SolCx": SolCx,
    "SolDA": SolDA,
    "SolDB2d": SolDB2d,
    "SolDB3d": SolDB3d,
    "SolH": SolH,
    "SolKx": SolKx,
    "SolKz": SolKz,
    "SolM": SolM,
    "SolNL": SolNL,
    "TwoLayerDarcy": TwoLayerDarcy,
}


def is_available(name):
    """Whether *name* can actually be constructed here and now.

    False only when a solution needs an optional package that is not installed.
    Kept separate from :func:`available` because a missing optional dependency
    should not make a solution disappear — silently shortening the listing hides
    the very thing the user needs to be told.

    Parameters
    ----------
    name : str
        A name from :func:`available`.

    Returns
    -------
    bool
    """

    solution = _SOLUTIONS[name]
    requires = getattr(solution, "requires", None)

    if requires is None:
        return True

    import importlib.util

    return importlib.util.find_spec(requires) is not None


def available(installed_only=False):
    """Names of the solutions in this namespace, in alphabetical order.

    Parameters
    ----------
    installed_only : bool
        Omit solutions whose optional dependency is missing. The default lists
        everything; use :func:`is_available` to tell them apart, or
        :func:`describe`, which says so in words.

    Returns
    -------
    list of str
    """

    names = sorted(_SOLUTIONS)

    if installed_only:
        return [name for name in names if is_available(name)]

    return names


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
    summary = docstring.strip().split("\n")[0]

    if not is_available(name):
        return f"{summary}  [unavailable: needs the '{solution.requires}' package]"

    return summary
