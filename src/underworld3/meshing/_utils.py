"""
Utility functions for mesh generation that handle UWQuantity objects.

All mesh creation functions convert quantities to BASE SI UNITS (meters, seconds, etc.)
to ensure consistency. This means 1000*km becomes 1000000 (meters).
"""

from underworld3.function.unit_conversion import _extract_value


def _to_base_units(value):
    """
    Convert Pint quantity to base SI units, or pass through plain numbers.

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

    For mesh creation, all coordinates must be in a consistent unit system. This function
    converts Pint Quantities to base SI units (meters) by default, or to the specified
    target units if provided.

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

    Examples
    --------
    >>> extract_coordinates((0.0, 0.0))
    (0.0, 0.0)
    >>> extract_coordinates((1000 * uw.units.km, 500 * uw.units.km))
    (1000000.0, 500000.0)  # Converted to meters (base SI)
    >>> extract_coordinates((1000 * uw.units.km, 500 * uw.units.km), 'km')
    (1000.0, 500.0)  # Kept in kilometers as requested
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

    For mesh creation, all scalars (like cellSize) must be in a consistent unit system.
    This function converts Pint Quantities to base SI units by default, or to the
    specified target units if provided.

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

    Examples
    --------
    >>> extract_scalar(5.0)
    5.0
    >>> extract_scalar(100 * uw.units.km)
    100000.0  # Converted to meters (base SI)
    >>> extract_scalar(100 * uw.units.km, 'km')
    100.0  # Kept in kilometers as requested
    """
    if target_units is not None:
        # User specified target units
        result = _extract_value(value, target_units)
    else:
        # Default: convert to base SI units for consistency
        result = _to_base_units(value)

    return float(result)
