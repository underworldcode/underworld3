"""
Unit conversion utilities for automatic coordinate and quantity conversion.

This module provides the core functionality for universal unit tracking and
conversion throughout the UW3 system.
"""

import numpy as np
import underworld3 as uw


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
    if hasattr(obj, '_has_pint_qty') and obj._has_pint_qty:
        return True

    # Check for Pint quantity
    if hasattr(obj, 'units') and hasattr(obj, 'magnitude'):
        return True

    # Check for array with unit metadata
    if hasattr(obj, '_units') or hasattr(obj, 'units'):
        return True

    # Check for NDArray with unit information
    if hasattr(obj, '__array__') and hasattr(obj, '_unit_metadata'):
        return True

    return False


def get_units(obj):
    """
    Extract unit information from any object.

    Parameters
    ----------
    obj : any
        Object to extract units from

    Returns
    -------
    str or None
        Unit string if detectable, None otherwise
    """
    if hasattr(obj, '_has_pint_qty') and obj._has_pint_qty:
        if hasattr(obj, '_pint_qty'):
            return str(obj._pint_qty.units)

    if hasattr(obj, 'units'):
        units = obj.units
        return str(units) if units is not None else None

    if hasattr(obj, '_units'):
        return str(obj._units)

    if hasattr(obj, '_unit_metadata'):
        return obj._unit_metadata.get('units')

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
    if not hasattr(mesh_or_expr, 'CoordinateSystem'):
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
    if hasattr(coord_sys, '_scaled') and coord_sys._scaled:
        if hasattr(coord_sys, '_length_scale'):
            scale_factor = coord_sys._length_scale
            # Return unit information - internal units are what the mesh expects
            return {
                'length_scale': scale_factor,
                'scaled': True,
                'units': 'model_units'  # Mesh expects internal model units
            }

    # No scaling - mesh uses whatever units were used to create it
    return {
        'scaled': False,
        'units': 'native'  # No specific unit system
    }


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
    if mesh_info is None or not mesh_info.get('scaled', False):
        if coord_units is not None:
            raise ValueError(
                f"Cannot convert coordinates with units '{coord_units}' - "
                "mesh has no scaling context (no model.set_reference_quantities() called)"
            )
        return coord_values

    # For scaled meshes with explicit coordinate units
    if coord_units is not None:
        # Convert physical coordinates to model coordinates
        scale_factor = mesh_info['length_scale']

        # Create a temporary quantity for proper unit conversion
        import underworld3 as uw
        coord_qty = uw.function.quantity(coord_values, coord_units)

        # Convert coordinates to meters (scale factor is always in meters)
        # The scale factor represents the characteristic length in meters
        coord_in_meters = convert_quantity_units(coord_qty, 'm')

        # Get the magnitude for division
        if hasattr(coord_in_meters, '_pint_qty'):
            coord_magnitude = coord_in_meters._pint_qty.magnitude
        else:
            coord_magnitude = coord_in_meters

        # Handle scale factor magnitude (should already be in meters)
        if hasattr(scale_factor, '_pint_qty'):
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
        return {
            'has_units': True,
            'units': units_str,
            'is_physical': True
        }
    else:
        return {
            'has_units': False,
            'units': None,
            'is_physical': False
        }


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
    based on the constituent variables and operations.

    Parameters
    ----------
    expression : sympy expression
        Expression to analyze
    mesh_info : dict
        Mesh coordinate unit information

    Returns
    -------
    str or None
        Unit string if determinable, None if dimensionless or unknown
    """
    try:
        import sympy
        import underworld3 as uw

        # Check if expression contains UW variables with known units

        # First check if the expression itself is a matrix containing UnderworldFunctions
        if hasattr(expression, 'shape') and len(expression.shape) == 2:
            for i in range(expression.shape[0]):
                for j in range(expression.shape[1]):
                    element = expression[i, j]
                    # Check if this element is an UnderworldFunction with meshvar reference
                    if hasattr(element, 'meshvar'):
                        try:
                            # meshvar is a weak reference, get the actual variable
                            variable = element.meshvar()
                            if variable and hasattr(variable, 'units'):
                                # Found a variable with units attribute
                                # Return its units, even if None (explicitly unitless)
                                # This prevents falling through to coordinate detection
                                if variable.units:
                                    return str(variable.units)
                                else:
                                    # Variable explicitly has no units - return None
                                    return None
                        except ReferenceError:
                            # Weak reference is dead
                            continue

        # Check if the expression itself is a coordinate (e.g., N.x, N.y)
        if hasattr(expression, 'name'):
            expr_name = str(expression.name)
            is_coordinate = (
                expr_name in ['x', 'y', 'z'] or  # Simple coordinate names
                expr_name.endswith('.x') or expr_name.endswith('.y') or expr_name.endswith('.z')  # BaseScalar coordinates
            )

            if is_coordinate and mesh_info and mesh_info.get('scaled'):
                # Coordinate variable - has length units
                try:
                    model = uw.get_default_model()
                    # Get the length scale from the model's fundamental scales
                    fundamental_scales = model.get_fundamental_scales()
                    if 'length' in fundamental_scales:
                        length_scale = fundamental_scales['length']
                        if hasattr(length_scale, 'units'):
                            return str(length_scale.units)
                except:
                    pass

        # Also check atoms for other function types
        atoms = expression.atoms()

        for atom in atoms:
            # Check if this is an UnderworldFunction with meshvar reference
            if hasattr(atom, 'meshvar'):
                try:
                    # meshvar is a weak reference, get the actual variable
                    variable = atom.meshvar()
                    if variable and hasattr(variable, 'units'):
                        # Found a variable with units attribute
                        # Return its units, even if None (explicitly unitless)
                        if variable.units:
                            return str(variable.units)
                        else:
                            # Variable explicitly has no units - return None
                            return None
                except ReferenceError:
                    # Weak reference is dead
                    continue

            # Check if this is an UnderworldFunction (mesh/swarm variable)
            if hasattr(atom, '_parent') and hasattr(atom._parent, 'units'):
                # Found a variable with units metadata
                var_units = atom._parent.units
                if var_units:
                    return str(var_units)

            # Legacy check for _units attribute (backward compatibility)
            if hasattr(atom, '_parent') and hasattr(atom._parent, '_units'):
                var_units = atom._parent._units
                if var_units:
                    return str(var_units)

            # Check for UW mesh coordinates (which have length units)
            if hasattr(atom, 'name'):
                # Check for coordinate symbols like 'x', 'y', 'z' or 'N.x', 'N.y', 'N.z'
                atom_name = str(atom.name)
                is_coordinate = (
                    atom_name in ['x', 'y', 'z'] or  # Simple coordinate names
                    atom_name.endswith('.x') or atom_name.endswith('.y') or atom_name.endswith('.z')  # BaseScalar coordinates
                )
                if is_coordinate and mesh_info and mesh_info.get('scaled'):
                    # Coordinate variable - has length units
                    try:
                        model = uw.get_default_model()
                        # Get the length scale from the model's fundamental scales
                        fundamental_scales = model.get_fundamental_scales()
                        if 'length' in fundamental_scales:
                            length_scale = fundamental_scales['length']
                            if hasattr(length_scale, 'units'):
                                return str(length_scale.units)
                    except:
                        pass

        # Advanced: Check if expression is a simple operation on known quantities
        # This would require more sophisticated analysis for compound expressions
        # For now, if no direct variable units found, assume dimensionless
        return None

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
    if hasattr(quantity, '_has_pint_qty') and quantity._has_pint_qty:
        if hasattr(quantity, '_pint_qty'):
            # Convert using Pint
            converted_pint = quantity._pint_qty.to(target_units)
            # Return new UWQuantity
            return uw.function.quantity(converted_pint.magnitude, str(converted_pint.units))

    # Handle direct Pint quantity
    if hasattr(quantity, 'to') and hasattr(quantity, 'units'):
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
    if hasattr(obj, '_has_pint_qty') and obj._has_pint_qty:
        if hasattr(obj, '_pint_qty'):
            units_str = str(obj._pint_qty.units)
            return {
                'has_units': True,
                'units': units_str,
                'is_dimensionless': units_str == 'dimensionless',
                'unit_type': 'UWQuantity'
            }

    # Check for Pint quantity
    if hasattr(obj, 'units') and hasattr(obj, 'magnitude'):
        units_str = str(obj.units)
        return {
            'has_units': True,
            'units': units_str,
            'is_dimensionless': units_str == 'dimensionless',
            'unit_type': 'Pint'
        }

    # Check for array with unit metadata
    if hasattr(obj, '_units'):
        units_str = str(obj._units) if obj._units is not None else None
        return {
            'has_units': units_str is not None,
            'units': units_str,
            'is_dimensionless': units_str == 'dimensionless' if units_str else False,
            'unit_type': 'metadata'
        }

    # Check for NDArray with unit information
    if hasattr(obj, '__array__') and hasattr(obj, '_unit_metadata'):
        units_str = obj._unit_metadata.get('units')
        return {
            'has_units': units_str is not None,
            'units': units_str,
            'is_dimensionless': units_str == 'dimensionless' if units_str else False,
            'unit_type': 'metadata'
        }

    # No units detected
    return {
        'has_units': False,
        'units': None,
        'is_dimensionless': False,
        'unit_type': 'none'
    }


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
    if hasattr(reference_scales, 'get_fundamental_scales'):
        scales = reference_scales.get_fundamental_scales()
    else:
        scales = reference_scales

    # Get quantity info
    quantity_info = detect_quantity_units(quantity)
    if not quantity_info['has_units']:
        # Already dimensionless
        return uw.function.quantity(quantity, 'dimensionless')

    # Extract Pint quantity
    if quantity_info['unit_type'] == 'UWQuantity':
        pint_qty = quantity._pint_qty
    elif quantity_info['unit_type'] == 'Pint':
        pint_qty = quantity
    else:
        raise ValueError(f"Cannot make dimensionless: unsupported quantity type {quantity_info['unit_type']}")

    # Determine appropriate scale based on quantity dimensions
    dimensionality = pint_qty.dimensionality

    # Map dimensions to scale factors
    scale_factor = None

    if '[length]' in str(dimensionality):
        if 'length' in scales:
            scale_factor = scales['length']
    elif '[time]' in str(dimensionality):
        if 'time' in scales:
            scale_factor = scales['time']
    elif '[temperature]' in str(dimensionality):
        if 'temperature' in scales:
            scale_factor = scales['temperature']
    elif '[mass]' in str(dimensionality):
        if 'mass' in scales:
            scale_factor = scales['mass']

    if scale_factor is None:
        raise ValueError(f"No appropriate reference scale found for dimensionality: {dimensionality}")

    # Convert to dimensionless
    if hasattr(scale_factor, '_pint_qty'):
        scale_pint = scale_factor._pint_qty
    else:
        scale_pint = scale_factor

    # Ensure both quantities use the same registry
    try:
        dimensionless_value = (pint_qty / scale_pint).to('dimensionless')
    except Exception:
        # Handle registry mismatch by converting through magnitude and units
        import underworld3 as uw
        scale_magnitude = scale_pint.magnitude if hasattr(scale_pint, 'magnitude') else scale_pint
        scale_units = str(scale_pint.units) if hasattr(scale_pint, 'units') else str(scale_pint)

        # Create new scale quantity in same registry as input
        registry = pint_qty._REGISTRY
        scale_in_same_registry = registry.Quantity(scale_magnitude, scale_units)

        dimensionless_value = (pint_qty / scale_in_same_registry).to('dimensionless')

    return uw.function.quantity(dimensionless_value.magnitude, 'dimensionless')


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
    if hasattr(converted_qty, '_pint_qty'):
        return converted_qty._pint_qty.magnitude
    elif hasattr(converted_qty, 'magnitude'):
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
        expr : sympy expression
            Expression to evaluate
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
        array or UWQuantity
            Evaluation result, potentially with units if expression has physical dimensions

        Raises
        ------
        ValueError
            If coordinate units are specified but mesh has no scaling context
        """
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
                if hasattr(model, '_meshes') and model._meshes:
                    # Get the first mesh (assuming single mesh context for now)
                    first_mesh = next(iter(model._meshes.values()))
                    mesh_info = get_mesh_coordinate_units(first_mesh)
            except:
                pass

        # Convert coordinates to mesh units with explicit unit specification
        internal_coords = convert_coordinates_to_mesh_units(coords, mesh_info, coord_units)

        # Call original evaluate function with converted coordinates
        # Filter out coord_units from kwargs since Cython function doesn't accept it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'coord_units'}
        result = original_evaluate_func(expr, internal_coords, coord_sys, **filtered_kwargs)

        # Convert result back to appropriate physical units
        result_with_units = add_expression_units_to_result(result, expr, mesh_info)

        return result_with_units

    return unit_aware_evaluate