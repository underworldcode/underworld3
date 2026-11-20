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


def model_length():
    """One model length, to divide a coordinate by inside a function argument.

    In a model with reference quantities a coordinate carries length, so
    ``sin(pi*x)`` is refused and is written ``sin(pi*x/L)`` (ruling, 2026-11-19);
    with ``L = model_length()`` the argument has the value it had before. Without
    reference quantities, 1.
    """
    if not uw.get_default_model().has_units():
        return 1
    return uw.expression(r"L_{\mathrm{model}}", model_units(1.0, "m"), "one model length")
