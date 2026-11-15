"""Helpers for tests that set dimensional parameters in a units model.

Since the units ruling of 2026-11-13, a bare number assigned to a parameter that
declares a dimension is refused in a model with reference quantities. A test
that meant "1 in model units" says so explicitly with ``model_units``.
"""
import underworld3 as uw


def model_units(value, units):
    """``value`` model units of the dimension of ``units``.

    In a model with reference quantities, the dimensional quantity whose
    non-dimensional value is ``value``, so the problem's numbers are unchanged.
    Without reference quantities, ``value`` itself.
    """
    if not uw.get_default_model().has_units():
        return value
    dimensionality = dict((1.0 * uw.units(units)).dimensionality)
    return uw.dimensionalise(value, target_dimensionality=dimensionality)
