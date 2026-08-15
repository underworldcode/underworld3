r"""Deprecated location for the analytic solutions — use :mod:`underworld3.analytic`.

The suite moved out of ``underworld3.function`` because it outgrew it: it now
carries boundary conditions, error norms and a registry, none of which belong
under "symbolic function evaluation". Everything here resolves to the *same
objects* as the new location, so ``isinstance`` and pickling work across both
paths.

.. deprecated::
   Import from :mod:`underworld3.analytic` instead::

       sol = uw.analytic.SolCx(mesh, eta_A=1.0, eta_B=1.0e6)

Notes
-----
This is a package directory rather than a module file on purpose. The old
``underworld3.function.analytic`` was a compiled extension, and an orphaned
``.so`` left behind by an earlier install would otherwise be imported in place
of this shim — Python's path finder checks for a package directory *before* it
checks for an extension of the same name, so the directory always wins.
"""

import warnings

__all__ = [
    "AnalyticSolCx_base",
    "AnalyticSolCx_pressure",
    "AnalyticSolCx_stress_xx",
    "AnalyticSolCx_stress_xy",
    "AnalyticSolCx_stress_yy",
    "AnalyticSolCx_velocity",
    "AnalyticSolCx_velocity_x",
    "AnalyticSolCx_velocity_y",
    "AnalyticSolCx_viscosity",
    "AnalyticSolNL_base",
    "AnalyticSolNL_bodyforce",
    "AnalyticSolNL_bodyforce_x",
    "AnalyticSolNL_bodyforce_y",
    "AnalyticSolNL_velocity",
    "AnalyticSolNL_velocity_x",
    "AnalyticSolNL_velocity_y",
    "AnalyticSolNL_viscosity",
    "SolCx",
    "sympy_function_printable",
]

# Names are resolved through __getattr__ rather than imported here, so that the
# deprecation warning fires when one is *used*. Ten test files import this
# module at collection time; a warning there is noise the reader cannot act on.
_warned = False


def __getattr__(name):
    global _warned

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if not _warned:
        warnings.warn(
            "underworld3.function.analytic has moved to underworld3.analytic; "
            "import from there instead (uw.analytic.SolCx, ...).",
            DeprecationWarning,
            stacklevel=2,
        )
        _warned = True

    if name == "SolCx":
        # The one class, not a second copy: resolving it anywhere else would
        # break isinstance for objects built through the other path.
        from underworld3.analytic import SolCx

        return SolCx

    from underworld3.analytic._reference import _velic

    return getattr(_velic, name)


def __dir__():
    return sorted(__all__)
