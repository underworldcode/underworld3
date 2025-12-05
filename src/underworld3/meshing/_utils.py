"""
Utility functions for mesh generation that handle UWQuantity objects.

Note on Unit Conversion for Mesh Creation (2025-12)
---------------------------------------------------
For mesh creation with non-dimensionalization, use `model.to_model_magnitude()`:

    model = uw.get_default_model()
    radiusOuter = model.to_model_magnitude(radiusOuter)
    cellSize = model.to_model_magnitude(cellSize)

This approach:
1. Handles UWQuantity, UWexpression, Pint Quantity, and plain numbers
2. Uses the reference quantities from model.set_reference_quantities()
3. Is consistent with the pattern in cartesian.py

The previous `_to_model_float()` function in this module was redundant.
"""

import underworld3 as uw
from underworld3.function.unit_conversion import _extract_value


def _to_base_units(value):
    """
    Convert Pint quantity to base SI units, or pass through plain numbers.

    Note: For mesh creation, prefer model.to_model_magnitude() instead,
    which converts to model units for correct non-dimensionalization.

    Parameters
    ----------
    value : float, int, Quantity, or tuple/list
        Value to convert to base units

    Returns
    -------
    float, int, or tuple/list
        Value in base SI units (or unchanged if plain number)
    """
    # Handle tuples/lists recursively
    if isinstance(value, (list, tuple)):
        return type(value)(_to_base_units(v) for v in value)

    # Check if it's a Pint Quantity
    if hasattr(value, "to_base_units") and hasattr(value, "magnitude"):
        try:
            return value.to_base_units().magnitude
        except:
            return value.magnitude

    # Check if it's a UWQuantity
    if hasattr(value, "_pint_qty"):
        try:
            return value._pint_qty.to_base_units().magnitude
        except:
            return value.value

    # Plain number - pass through
    return value


def extract_coordinates(coords, target_units=None):
    """
    Extract numerical values from coordinate tuples, converting to base SI units.

    Note: For mesh creation, prefer model.to_model_magnitude() instead,
    which converts to model units for correct non-dimensionalization.

    Parameters
    ----------
    coords : tuple
        Tuple of coordinates that may be floats, ints, or UWQuantity objects
    target_units : str, optional
        If provided, converts quantities to these units instead of base SI units.
        Use 'km', 'm', 'cm', etc.

    Returns
    -------
    tuple
        Tuple of numerical values (floats) in base SI units or target units
    """
    if target_units is not None:
        # User specified target units - use _extract_value with conversion
        result = _extract_value(coords, target_units)
    else:
        # Default: convert to base SI units for consistency
        result = _to_base_units(coords)

    # Convert to floats for mesh API consistency
    return tuple(float(x) for x in result)


def extract_scalar(value, target_units=None):
    """
    Extract numerical value from a scalar, converting to base SI units.

    Note: For mesh creation, prefer model.to_model_magnitude() instead,
    which converts to model units for correct non-dimensionalization.

    Parameters
    ----------
    value : float, int, or UWQuantity-like
        Scalar value that may be a UWQuantity object
    target_units : str, optional
        If provided, converts quantity to these units instead of base SI units

    Returns
    -------
    float
        Numerical value in base SI units or target units
    """
    if target_units is not None:
        # User specified target units
        result = _extract_value(value, target_units)
    else:
        # Default: convert to base SI units for consistency
        result = _to_base_units(value)

    return float(result)
