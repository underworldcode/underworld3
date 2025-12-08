"""
Unit conversion utilities for automatic coordinate and quantity conversion.

This module provides the core functionality for universal unit tracking and
conversion throughout the UW3 system.
"""

import numpy as np
import underworld3 as uw

# NOTE: UnitAwareArray is imported where needed to avoid circular imports
# (utilities.unit_aware_array imports from this module)


def _extract_value(value, target_units=None):
    """
    Universal helper to extract numeric value from quantities or pass through numbers.

    This is a lightweight, zero-side-effect function that makes APIs accept both
    plain numbers and unit-aware quantities transparently.

    Parameters
    ----------
    value : float, int, Quantity, UWQuantity, tuple, list, or None
        Value to extract. Can be:
        - Plain number → returned as-is
        - Pint Quantity → magnitude extracted
        - UWQuantity → value extracted
        - tuple/list → recursively processed
        - None → returned as-is
    target_units : str, optional
        If provided, converts quantity to these units before extracting value.
        Ignored if value has no units.

    Returns
    -------
    float, int, tuple, list, or None
        Plain numeric value(s), ready to use in numeric APIs

    Examples
    --------
    >>> # Plain numbers pass through unchanged
    >>> _extract_value(5.0)
    5.0

    >>> # Quantities have magnitude extracted
    >>> _extract_value(5.0 * uw.units.km)
    5000.0  # in meters

    >>> # With target units
    >>> _extract_value(5.0 * uw.units.km, 'cm')
    500000.0

    >>> # Tuples/lists processed recursively
    >>> _extract_value((0.0, 5.0 * uw.units.km))
    (0.0, 5000.0)

    >>> # UWQuantity support
    >>> _extract_value(uw.quantity(5.0, "km"))
    5000.0
    """
    # Handle None
    if value is None:
        return None

    # Handle tuples/lists recursively
    if isinstance(value, (list, tuple)):
        extracted = [_extract_value(v, target_units) for v in value]
        return type(value)(extracted)

    # Check if it's a Pint Quantity (has both units and magnitude)
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        if target_units is not None:
            try:
                return value.to(target_units).magnitude
            except:
                # Conversion failed, just return magnitude
                return value.magnitude
        return value.magnitude

    # Check if it's a UWQuantity
    if hasattr(value, "value") and hasattr(value, "_has_pint_qty"):
        if target_units is not None and hasattr(value, "_pint_qty"):
            try:
                return value._pint_qty.to(target_units).magnitude
            except:
                # Conversion failed, return value as-is
                return value.value
        return value.value

    # Already numeric (int, float, numpy scalar, etc.)
    return value


def has_units(obj):
    """
    Check if an object has unit information.

    Parameters
    ----------
    obj : any
        Object to check for unit information

    Returns
    -------
    bool
        True if object has detectable units
    """
    # Check for UWQuantity
    if hasattr(obj, "_has_pint_qty") and obj._has_pint_qty:
        return True

    # Check for Pint quantity
    if hasattr(obj, "units") and hasattr(obj, "magnitude"):
        return True

    # Check for array with unit metadata
    if hasattr(obj, "_units") or hasattr(obj, "units"):
        return True

    # Check for NDArray with unit information
    if hasattr(obj, "__array__") and hasattr(obj, "_unit_metadata"):
        return True

    return False
# DEPRECATED: Old get_units() from function.unit_conversion has been removed.
# Use uw.get_units() from the units module instead, which provides the unified,
# high-level API for extracting units from any object type. The units module
# version delegates to _extract_units_info() which provides the same smart
# extraction strategy (prioritizing variables over atoms, avoiding blind tree-walking).


def compute_expression_units(expr):
    """
    Compute units for compound SymPy expressions using dimensional analysis.

    This function traverses the expression tree and uses Pint to perform
    dimensional arithmetic on the units of sub-expressions.

    Parameters
    ----------
    expr : sympy expression
        Expression to analyze (e.g., temperature / y)

    Returns
    -------
    pint.Unit or None
        Computed unit object, or None if cannot determine

    Examples
    --------
    >>> # T.sym has units 'kelvin', y has units 'kilometer'
    >>> compute_expression_units(T.sym / y)
    <Unit('kelvin / kilometer')>

    >>> # velocity has units 'm/s', time has units 's'
    >>> compute_expression_units(velocity * time)
    <Unit('meter')>

    >>> # Derivative: dT/dx
    >>> compute_expression_units(T.sym.diff(mesh.N.x))
    <Unit('kelvin / kilometer')>

    Changed in 2025-10-16: Now returns pint.Unit objects instead of strings.
    """
    try:
        import sympy
        import underworld3 as uw

        # Helper to check if a pint.Unit is dimensionless
        def is_dimensionless_unit(unit):
            if unit is None:
                return True
            if hasattr(unit, "dimensionality"):
                return len(unit.dimensionality) == 0
            return str(unit) == "dimensionless"

        # Priority -1: Check for DERIVATIVES first (before general UnderworldFunction)
        # Derivatives are UnderworldFunctions with a diffindex attribute
        # Units of derivative = units(variable) / units(coordinate)
        if hasattr(expr, "diffindex"):
            try:
                # Get variable units from meshvar
                variable = expr.meshvar()
                if variable and hasattr(variable, "units"):
                    var_units = variable.units
                    if var_units is not None:
                        # Get the coordinate it's derived with respect to
                        deriv_index = expr.diffindex
                        if deriv_index < len(expr.args):
                            coord = expr.args[deriv_index]
                            coord_units = get_units(coord)
                            if coord_units is not None:
                                # Convert to pint.Unit if needed
                                if not hasattr(var_units, "dimensionality"):
                                    var_units = uw.units(str(var_units))
                                if not hasattr(coord_units, "dimensionality"):
                                    coord_units = uw.units(str(coord_units))
                                # Compute derivative units: var_units / coord_units
                                result_qty = (1 * var_units) / (1 * coord_units)
                                return result_qty.units
            except Exception:
                pass

        # Priority 0: Check for UnderworldFunction (has meshvar) FIRST
        # This must come before is_Atom check because UnderworldFunctions
        # have args (coordinates) so is_Atom returns False
        if hasattr(expr, "meshvar"):
            try:
                variable = expr.meshvar()
                if variable and hasattr(variable, "units"):
                    var_units = variable.units
                    if var_units is not None:
                        # Convert to pint.Unit if it's a string
                        if not hasattr(var_units, "dimensionality"):
                            return uw.units(str(var_units))
                        return var_units
            except (ReferenceError, AttributeError):
                pass

        # Handle Matrix expressions - check first element
        if isinstance(expr, sympy.MatrixBase):
            if expr.shape[0] > 0 and expr.shape[1] > 0:
                return compute_expression_units(expr[0, 0])
            return None

        # Base case: atomic expression (coordinate or constant)
        if expr.is_Atom or (not expr.args):
            # CRITICAL: Do NOT blindly extract units from BaseScalar coordinates
            # Coordinates should only contribute their units in specific compound
            # expressions (like T.sym / y for derivatives), not as atomic atoms
            #
            # Instead, prioritize UWexpressions (registered variables) and only
            # extract coordinate units if explicitly requested for a pure coordinate

            # FIRST: Check if this symbol IS a UWexpression (direct isinstance check)
            # This is the DRY approach - UWexpression inherits from Symbol and has
            # has_units/units properties, so we just ask it directly for units.
            # No need for dictionary lookups by name - the object knows its own units.
            from underworld3.function.expressions import UWexpression

            if isinstance(expr, UWexpression):
                # Case 1: UWexpression with explicit units (e.g., uw.expression("r_i", uw.quantity(3000, "km")))
                if expr.has_units and expr.units is not None:
                    units_from_uw = expr.units
                    # Ensure we return a Pint Unit object
                    if not hasattr(units_from_uw, "dimensionality"):
                        return uw.units(str(units_from_uw))
                    return units_from_uw

                # Case 2: UWexpression wrapping a SymPy expression that has units
                # (e.g., r - inner_radius where r = sqrt(x**2 + y**2) with coordinate units)
                # Check if ._sym is a SymPy expression (not a number) and recurse
                if hasattr(expr, '_sym') and expr._sym is not None:
                    inner_sym = expr._sym
                    # Don't recurse on numbers or other simple types
                    if isinstance(inner_sym, sympy.Basic) and not isinstance(inner_sym, sympy.Number):
                        inner_units = compute_expression_units(inner_sym)
                        if inner_units is not None:
                            return inner_units
                # UWexpression without discoverable units - fall through to other checks

            # Numbers are dimensionless
            if expr.is_Number:
                return uw.units.dimensionless

            # Check for UWCoordinate (has .units property that delegates to BaseScalar._units)
            from underworld3.coordinates import UWCoordinate
            if isinstance(expr, UWCoordinate):
                coord_units = expr.units
                if coord_units is not None:
                    if not hasattr(coord_units, 'dimensionality'):
                        return uw.units(str(coord_units))
                    return coord_units

            # Check for coordinate symbols (BaseScalar with _units attribute)
            # These carry length units from the mesh coordinate system
            if hasattr(expr, '_units') and expr._units is not None:
                coord_units = expr._units
                # Ensure we return a Pint Unit object
                if not hasattr(coord_units, 'dimensionality'):
                    return uw.units(str(coord_units))
                return coord_units

            # No units detected for this atom
            return None

        # Recursive case: compound expression
        # Handle different operation types

        if isinstance(expr, sympy.Mul):
            # Multiplication: multiply units
            result_unit = None
            found_any_units = False  # Track if we found any unit-aware terms

            for arg in expr.args:
                arg_units = compute_expression_units(arg)
                if arg_units is None:
                    continue

                found_any_units = True

                if is_dimensionless_unit(arg_units):
                    continue

                # Multiply units using pint arithmetic
                try:
                    if result_unit is None:
                        result_unit = arg_units
                    else:
                        # Multiply by creating quantities
                        result_qty = (1 * result_unit) * (1 * arg_units)
                        result_unit = result_qty.units
                except Exception:
                    continue

            # IMPORTANT: Check if result is dimensionless
            # This handles both: (1) all units canceled → result_unit is None
            #                   (2) mixed units that are dimensionless (kg*km³/m⁴/Pa/s²)
            if found_any_units:
                if result_unit is None:
                    # All units canceled out - return dimensionless Unit (not Quantity)
                    return uw.units.dimensionless
                elif hasattr(result_unit, 'dimensionality'):
                    # Check if Pint says it's dimensionless
                    dim = result_unit.dimensionality
                    # UnitsContainer is empty (len == 0) for dimensionless units
                    if len(dim) == 0:
                        return uw.units.dimensionless

            return result_unit

        elif isinstance(expr, sympy.Pow):
            # Power: raise units to power
            base, exponent = expr.args
            base_units = compute_expression_units(base)

            if base_units is None or is_dimensionless_unit(base_units):
                return None

            # Exponent must be a number
            if not exponent.is_Number:
                return None

            try:
                # Raise units to power
                result_qty = (1 * base_units) ** float(exponent)
                return result_qty.units
            except Exception:
                return None

        elif isinstance(expr, sympy.Add):
            # Addition: all terms must have same units
            # Return units of first non-dimensionless term
            for arg in expr.args:
                arg_units = compute_expression_units(arg)
                if arg_units and not is_dimensionless_unit(arg_units):
                    return arg_units
            return None

        elif hasattr(expr, "func"):
            # Function application (sin, cos, etc.) - usually dimensionless result
            # But argument should be dimensionless too
            return uw.units.dimensionless

        # For other expression types, try to get units from args recursively
        for arg in expr.args:
            arg_units = compute_expression_units(arg)
            if arg_units:
                return arg_units

        return None

    except Exception:
        # If anything fails, fall back to None
        return None


def get_mesh_coordinate_units(mesh_or_expr):
    """
    Get the coordinate units expected by a mesh or expression.

    Parameters
    ----------
    mesh_or_expr : Mesh or sympy expression
        Mesh object or expression containing mesh variables

    Returns
    -------
    dict or None
        Dictionary with coordinate unit information, or None if not available
    """
    # Try to extract mesh from expression
    if not hasattr(mesh_or_expr, "CoordinateSystem"):
        try:
            mesh, _ = uw.function.expressions.mesh_vars_in_expression(mesh_or_expr)
            if mesh is None:
                return None
            mesh_or_expr = mesh
        except:
            return None

    # Get coordinate system information
    coord_sys = mesh_or_expr.CoordinateSystem

    # Check if mesh has coordinate scaling (units applied)
    if hasattr(coord_sys, "_scaled") and coord_sys._scaled:
        if hasattr(coord_sys, "_length_scale"):
            scale_factor = coord_sys._length_scale
            # Return unit information - internal units are what the mesh expects
            return {
                "length_scale": scale_factor,
                "scaled": True,
                "units": "model_units",  # Mesh expects internal model units
            }

    # No scaling - mesh uses whatever units were used to create it
    return {"scaled": False, "units": "native"}  # No specific unit system


def convert_coordinates_to_mesh_units(coords, mesh_info, coord_units=None):
    """
    Convert coordinate array to mesh unit system with explicit unit specification.

    Following UW3 policy: no implicit unit conversions. Coordinates must have
    explicit units or are assumed to be in model units.

    Parameters
    ----------
    coords : array-like
        Coordinate array
    mesh_info : dict
        Mesh coordinate unit information from get_mesh_coordinate_units()
    coord_units : str, optional
        Explicit coordinate units. If None, assumes model coordinates.

    Returns
    -------
    numpy.ndarray
        Coordinates converted to mesh unit system

    Raises
    ------
    ValueError
        If coordinate units are specified but mesh has no scaling context
    """
    # Extract coordinate values and ensure float64
    coord_values = np.asarray(coords, dtype=np.float64)

    # If no mesh info or mesh is not scaled, coordinates must be model units
    if mesh_info is None or not mesh_info.get("scaled", False):
        if coord_units is not None:
            raise ValueError(
                f"Cannot convert coordinates with units '{coord_units}' - "
                "mesh has no scaling context (no model.set_reference_quantities() called)"
            )
        return coord_values

    # For scaled meshes with explicit coordinate units
    if coord_units is not None:
        # Convert physical coordinates to model coordinates
        scale_factor = mesh_info["length_scale"]

        # Create a temporary quantity for proper unit conversion
        import underworld3 as uw

        coord_qty = uw.function.quantity(coord_values, coord_units)

        # Convert coordinates to meters (scale factor is always in meters)
        # The scale factor represents the characteristic length in meters
        coord_in_meters = convert_quantity_units(coord_qty, "m")

        # Get the magnitude for division
        if hasattr(coord_in_meters, "_pint_qty"):
            coord_magnitude = coord_in_meters._pint_qty.magnitude
        else:
            coord_magnitude = coord_in_meters

        # Handle scale factor magnitude (should already be in meters)
        if hasattr(scale_factor, "_pint_qty"):
            scale_magnitude = scale_factor._pint_qty.magnitude
        else:
            scale_magnitude = scale_factor  # Already in meters

        # physical_coords_in_meters / scale_factor_in_meters = model_coords
        model_coords = coord_magnitude / scale_magnitude

        return np.asarray(model_coords, dtype=np.float64)
    else:
        # No explicit units - assume coordinates are already in model units
        return coord_values


def detect_coordinate_units(coords):
    """
    Detect what unit system coordinates are in.

    Parameters
    ----------
    coords : array-like
        Coordinate array

    Returns
    -------
    dict
        Information about coordinate units
    """
    if has_units(coords):
        units_str = get_units(coords)
        return {"has_units": True, "units": units_str, "is_physical": True}
    else:
        return {"has_units": False, "units": None, "is_physical": False}


def add_expression_units_to_result(result, expression, mesh_info):
    """
    Add appropriate units to evaluation result based on expression analysis.

    Analyzes the expression to determine its physical units and converts
    the model-unit result back to appropriate physical units.

    Parameters
    ----------
    result : numpy.ndarray
        Raw evaluation result in model units from PETSc
    expression : sympy expression
        Expression that was evaluated
    mesh_info : dict
        Mesh coordinate unit information

    Returns
    -------
    array or UWQuantity
        Result with appropriate units if detectable, otherwise plain array
    """
    try:
        # Try to determine the units of the expression
        expr_units = determine_expression_units(expression, mesh_info)

        if expr_units is not None:
            # Convert result from model units to physical units
            import underworld3 as uw

            return uw.function.quantity(result, expr_units)
        else:
            # No units detectable - return plain array (likely dimensionless)
            return result

    except Exception:
        # If unit analysis fails, return plain result
        return result


def determine_expression_units(expression, mesh_info):
    """
    Determine the physical units of a SymPy expression.

    Analyzes the expression to infer what units the result should have
    based on the constituent variables and operations. This now uses
    dimensional arithmetic for compound expressions.

    Parameters
    ----------
    expression : sympy expression
        Expression to analyze
    mesh_info : dict
        Mesh coordinate unit information (optional, can be None)

    Returns
    -------
    str or None
        Unit string if determinable, None if dimensionless or unknown

    Notes
    -----
    This function now delegates to compute_expression_units() which performs
    dimensional arithmetic using Pint. This ensures consistent behavior between
    get_units() and determine_expression_units().
    """
    try:
        # Use the unified dimensional analysis function
        return compute_expression_units(expression)
    except Exception:
        # If analysis fails, assume dimensionless
        return None


def add_units_to_result(result, expression):
    """
    Add appropriate units to evaluation result based on expression.

    Parameters
    ----------
    result : numpy.ndarray
        Raw evaluation result
    expression : sympy expression
        Expression that was evaluated

    Returns
    -------
    array or UWQuantity
        Result with appropriate units if detectable
    """
    # This is the old function - kept for backward compatibility
    # New function is add_expression_units_to_result
    return result


def convert_quantity_units(quantity, target_units):
    """
    Convert UWQuantity or Pint quantity to target units.

    Parameters
    ----------
    quantity : UWQuantity, Pint quantity, or array-like
        The quantity to convert
    target_units : str or Pint unit
        Target units to convert to

    Returns
    -------
    converted quantity
        Quantity converted to target units
    """
    import underworld3 as uw

    # Handle UWQuantity
    if hasattr(quantity, "_has_pint_qty") and quantity._has_pint_qty:
        if hasattr(quantity, "_pint_qty"):
            # Convert using Pint
            converted_pint = quantity._pint_qty.to(target_units)
            # Return new UWQuantity
            return uw.function.quantity(converted_pint.magnitude, str(converted_pint.units))

    # Handle direct Pint quantity
    if hasattr(quantity, "to") and hasattr(quantity, "units"):
        return quantity.to(target_units)

    # Handle plain arrays - assume they're already in target units
    return quantity


def detect_quantity_units(obj):
    """
    Detect units of any object (UWQuantity, Pint, array with metadata).

    Parameters
    ----------
    obj : any
        Object to detect units from

    Returns
    -------
    dict
        Dictionary with unit information:
        - 'has_units': bool
        - 'units': str or None
        - 'is_dimensionless': bool
        - 'unit_type': str ('UWQuantity', 'Pint', 'metadata', 'none')
    """
    # Check for UWQuantity
    if hasattr(obj, "_has_pint_qty") and obj._has_pint_qty:
        if hasattr(obj, "_pint_qty"):
            units_str = str(obj._pint_qty.units)
            return {
                "has_units": True,
                "units": units_str,
                "is_dimensionless": units_str == "dimensionless",
                "unit_type": "UWQuantity",
            }

    # Check for Pint quantity
    if hasattr(obj, "units") and hasattr(obj, "magnitude"):
        units_str = str(obj.units)
        return {
            "has_units": True,
            "units": units_str,
            "is_dimensionless": units_str == "dimensionless",
            "unit_type": "Pint",
        }

    # Check for array with unit metadata
    if hasattr(obj, "_units"):
        units_str = str(obj._units) if obj._units is not None else None
        return {
            "has_units": units_str is not None,
            "units": units_str,
            "is_dimensionless": units_str == "dimensionless" if units_str else False,
            "unit_type": "metadata",
        }

    # Check for NDArray with unit information
    if hasattr(obj, "__array__") and hasattr(obj, "_unit_metadata"):
        units_str = obj._unit_metadata.get("units")
        return {
            "has_units": units_str is not None,
            "units": units_str,
            "is_dimensionless": units_str == "dimensionless" if units_str else False,
            "unit_type": "metadata",
        }

    # No units detected
    return {"has_units": False, "units": None, "is_dimensionless": False, "unit_type": "none"}


def make_dimensionless(quantity, reference_scales):
    """
    Convert physical quantity to dimensionless using reference scales.

    Parameters
    ----------
    quantity : UWQuantity or Pint quantity
        Physical quantity to make dimensionless
    reference_scales : dict or Model
        Dictionary of reference scales or Model with reference quantities

    Returns
    -------
    UWQuantity
        Dimensionless quantity
    """
    import underworld3 as uw

    # Handle Model object
    if hasattr(reference_scales, "get_fundamental_scales"):
        scales = reference_scales.get_fundamental_scales()
    else:
        scales = reference_scales

    # Get quantity info
    quantity_info = detect_quantity_units(quantity)
    if not quantity_info["has_units"]:
        # Already dimensionless
        return uw.function.quantity(quantity, "dimensionless")

    # Extract Pint quantity
    if quantity_info["unit_type"] == "UWQuantity":
        pint_qty = quantity._pint_qty
    elif quantity_info["unit_type"] == "Pint":
        pint_qty = quantity
    else:
        raise ValueError(
            f"Cannot make dimensionless: unsupported quantity type {quantity_info['unit_type']}"
        )

    # Determine appropriate scale based on quantity dimensions
    dimensionality = pint_qty.dimensionality

    # Map dimensions to scale factors
    scale_factor = None

    if "[length]" in str(dimensionality):
        if "length" in scales:
            scale_factor = scales["length"]
    elif "[time]" in str(dimensionality):
        if "time" in scales:
            scale_factor = scales["time"]
    elif "[temperature]" in str(dimensionality):
        if "temperature" in scales:
            scale_factor = scales["temperature"]
    elif "[mass]" in str(dimensionality):
        if "mass" in scales:
            scale_factor = scales["mass"]

    if scale_factor is None:
        raise ValueError(
            f"No appropriate reference scale found for dimensionality: {dimensionality}"
        )

    # Convert to dimensionless
    if hasattr(scale_factor, "_pint_qty"):
        scale_pint = scale_factor._pint_qty
    else:
        scale_pint = scale_factor

    # Ensure both quantities use the same registry
    try:
        dimensionless_value = (pint_qty / scale_pint).to("dimensionless")
    except Exception:
        # Handle registry mismatch by converting through magnitude and units
        import underworld3 as uw

        scale_magnitude = scale_pint.magnitude if hasattr(scale_pint, "magnitude") else scale_pint
        scale_units = str(scale_pint.units) if hasattr(scale_pint, "units") else str(scale_pint)

        # Create new scale quantity in same registry as input
        registry = pint_qty._REGISTRY
        scale_in_same_registry = registry.Quantity(scale_magnitude, scale_units)

        dimensionless_value = (pint_qty / scale_in_same_registry).to("dimensionless")

    return uw.function.quantity(dimensionless_value.magnitude, "dimensionless")


def convert_array_units(array, from_units, to_units):
    """
    Convert array from one unit system to another.

    Parameters
    ----------
    array : array-like
        Array values to convert
    from_units : str or Pint unit
        Source units
    to_units : str or Pint unit
        Target units

    Returns
    -------
    numpy.ndarray
        Converted array values
    """
    import underworld3 as uw

    # Create a temporary quantity for conversion
    temp_qty = uw.function.quantity(array, from_units)
    converted_qty = convert_quantity_units(temp_qty, to_units)

    # Extract the magnitude
    if hasattr(converted_qty, "_pint_qty"):
        return converted_qty._pint_qty.magnitude
    elif hasattr(converted_qty, "magnitude"):
        return converted_qty.magnitude
    else:
        return converted_qty


def auto_convert_to_mesh_units(array, mesh):
    """
    Convert array coordinates to mesh unit system.

    Parameters
    ----------
    array : array-like
        Coordinate array that may have units
    mesh : Mesh
        Mesh to get unit system from

    Returns
    -------
    numpy.ndarray
        Coordinates converted to mesh unit system
    """
    mesh_info = get_mesh_coordinate_units(mesh)
    return convert_coordinates_to_mesh_units(array, mesh_info)


def convert_evaluation_result(result, target_units):
    """
    Convert evaluation results to target unit system.

    Parameters
    ----------
    result : array-like or UWQuantity
        Evaluation result to convert
    target_units : str or Pint unit
        Target units to convert to

    Returns
    -------
    converted result
        Result converted to target units
    """
    return convert_quantity_units(result, target_units)


def add_units(array, units_str):
    """
    Add unit metadata to plain array.

    Parameters
    ----------
    array : array-like
        Plain array to add units to
    units_str : str
        Unit string to associate with array

    Returns
    -------
    UWQuantity
        Array wrapped with unit information
    """
    import underworld3 as uw

    return uw.function.quantity(array, units_str)


def make_evaluate_unit_aware(original_evaluate_func):
    """
    Decorator to make evaluate functions unit-aware with explicit unit specification.

    This wraps the original evaluate function to:
    1. Accept explicit coordinate units parameter
    2. Convert coordinates to mesh units if units are specified
    3. Assume model coordinates if no units specified
    4. Convert results back to appropriate physical units

    Following UW3 policy: no implicit unit detection or conversion.

    Parameters
    ----------
    original_evaluate_func : callable
        Original evaluate function to wrap

    Returns
    -------
    callable
        Unit-aware version of the function
    """

    def unit_aware_evaluate(expr, coords=None, coord_sys=None, coord_units=None, **kwargs):
        """
        Unit-aware wrapper for evaluate function.

        Parameters
        ----------
        expr : sympy expression or MeshVariable
            Expression to evaluate. If MeshVariable, will use .sym automatically.
        coords : array-like, optional
            Coordinate array for evaluation
        coord_sys : CoordinateSystem, optional
            Coordinate system context
        coord_units : str, optional
            Explicit coordinate units. If None, assumes model coordinates.
        **kwargs
            Additional arguments passed to original evaluate function

        Returns
        -------
        numpy.ndarray
            Evaluation result as plain numpy array (unit metadata NOT attached as wrapper)

        Raises
        ------
        ValueError
            If coordinate units are specified but mesh has no scaling context
        """
        # Auto-extract .sym from MeshVariable for user convenience
        import underworld3 as uw

        if hasattr(expr, "sym") and hasattr(expr, "mesh"):
            # This is likely a MeshVariable - extract the symbolic representation
            expr = expr.sym

        # Handle the case where no coordinates are provided
        if coords is None:
            return original_evaluate_func(expr, coords, coord_sys, **kwargs)

        # Get mesh coordinate unit information
        mesh_info = None

        # Approach 1: Use coord_sys if provided
        if coord_sys is not None:
            mesh_info = get_mesh_coordinate_units(coord_sys)

        # Approach 2: Try to extract from expression
        if mesh_info is None:
            mesh_info = get_mesh_coordinate_units(expr)

        # Approach 3: Use a default model's current mesh if available
        if mesh_info is None:
            try:
                import underworld3 as uw

                model = uw.get_default_model()
                # Look for any mesh in the model
                if hasattr(model, "_meshes") and model._meshes:
                    # Get the first mesh (assuming single mesh context for now)
                    first_mesh = next(iter(model._meshes.values()))
                    mesh_info = get_mesh_coordinate_units(first_mesh)
            except:
                pass

        # Convert coordinates to mesh units with explicit unit specification
        internal_coords = convert_coordinates_to_mesh_units(coords, mesh_info, coord_units)

        # Call original evaluate function with converted coordinates
        # Filter out coord_units from kwargs since Cython function doesn't accept it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "coord_units"}
        result = original_evaluate_func(expr, internal_coords, coord_sys, **filtered_kwargs)

        # Determine what units the result should have based on expression analysis
        try:
            expr_units = determine_expression_units(expr, mesh_info)
            if expr_units is not None:
                # Return UnitAwareArray with unit metadata attached
                # This allows uw.get_units(result) to work while maintaining
                # full compatibility with numpy operations
                from underworld3.utilities.unit_aware_array import UnitAwareArray
                return UnitAwareArray(result, units=expr_units)
        except Exception:
            # If unit detection fails, return plain array
            pass

        # Return plain numpy array if no units detected
        return result

    return unit_aware_evaluate


def _convert_coords_to_si(coords):
    """
    Convert coordinate input to numpy array in model coordinates.
    
    This function handles unit-aware coordinates by converting them to model units,
    not SI units. This ensures coordinates work correctly with meshes that use
    reference quantities for scaling.
    
    Accepts:
    - numpy arrays (assumed to be in model coordinates if no units)
    - lists/tuples of coordinates (each coordinate can be UWQuantity, Pint Quantity, or float/int)
    - lists/tuples of tuples (for multiple points)
    
    Returns numpy array of shape (n_points, n_dims) with dtype=np.double in model coordinates.
    """
    import pint
    
    # Get the model for unit conversion
    model = uw.get_default_model()
    
    # Helper function to convert a single coordinate value
    def convert_single_coord(coord):
        if isinstance(coord, uw.function.quantities.UWQuantity) or isinstance(coord, pint.Quantity):
            # Unit-aware coordinate - convert to model units
            model_qty = model.to_model_units(coord)
            # Extract the magnitude
            if hasattr(model_qty, '_pint_qty'):
                return model_qty._pint_qty.magnitude
            elif hasattr(model_qty, 'value'):
                return float(model_qty.value)
            else:
                # Conversion returned plain number - use it
                return float(model_qty)
        elif isinstance(coord, (float, int, np.number)):
            # Plain number - assume it's already in model coordinates
            return float(coord)
        else:
            raise TypeError(f"Unsupported coordinate type: {type(coord)}. "
                          f"Expected UWQuantity, pint.Quantity, or numeric value.")
    
    # If already numpy array with correct dtype, assume it's in model coordinates
    if isinstance(coords, np.ndarray):
        if coords.dtype == np.double:
            return coords
        else:
            return np.array(coords, dtype=np.double)
    
    # Convert list/tuple input
    if isinstance(coords, (list, tuple)):
        # Check if it's a list of coordinates or a single point
        if len(coords) > 0:
            first_elem = coords[0]
            
            # Multiple points: [(x1, y1), (x2, y2), ...]
            if isinstance(first_elem, (list, tuple)):
                model_coords = []
                for point in coords:
                    model_point = []
                    for coord in point:
                        model_value = convert_single_coord(coord)
                        model_point.append(model_value)
                    model_coords.append(model_point)
                return np.array(model_coords, dtype=np.double)
            else:
                # Flat list of coordinates - could be single point [x, y] or [x, y, z]
                # Check if all elements are coordinate values (not lists/tuples)
                all_coords = all(
                    isinstance(elem, (uw.function.quantities.UWQuantity, pint.Quantity, float, int, np.number))
                    for elem in coords
                )
                
                if all_coords and len(coords) in [2, 3]:
                    # This is a single point like [x, y] or [x, y, z]
                    model_point = []
                    for coord in coords:
                        model_value = convert_single_coord(coord)
                        model_point.append(model_value)
                    # Return as 2D array with single point
                    return np.array([model_point], dtype=np.double)
                else:
                    raise TypeError(f"Unable to parse coordinate format. Expected list of tuples like "
                                  f"[(x1,y1), (x2,y2)] or single point like [x, y].")
    
    raise TypeError(f"coords must be numpy array, list, or tuple. Got {type(coords)}")
