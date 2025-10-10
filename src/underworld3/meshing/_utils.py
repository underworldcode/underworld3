"""
Utility functions for mesh generation that handle UWQuantity objects.
"""

def extract_coordinates(coords):
    """
    Extract numerical values from coordinate tuples that may contain UWQuantity objects.

    Parameters
    ----------
    coords : tuple
        Tuple of coordinates that may be floats, ints, or UWQuantity objects

    Returns
    -------
    tuple
        Tuple of numerical values (floats)
    """
    def extract_value(coord):
        # Check if it's a UWQuantity-like object (has .value attribute)
        if hasattr(coord, 'value'):
            return float(coord.value)
        # Check if it's a Pint quantity (has .magnitude attribute)
        elif hasattr(coord, 'magnitude'):
            return float(coord.magnitude)
        # Otherwise assume it's already a number
        else:
            return float(coord)

    return tuple(extract_value(coord) for coord in coords)


def extract_scalar(value):
    """
    Extract numerical value from a scalar that may be a UWQuantity object.

    Parameters
    ----------
    value : float, int, or UWQuantity-like
        Scalar value that may be a UWQuantity object

    Returns
    -------
    float
        Numerical value
    """
    # Check if it's a UWQuantity-like object (has .value attribute)
    if hasattr(value, 'value'):
        return float(value.value)
    # Check if it's a Pint quantity (has .magnitude attribute)
    elif hasattr(value, 'magnitude'):
        return float(value.magnitude)
    # Otherwise assume it's already a number
    else:
        return float(value)