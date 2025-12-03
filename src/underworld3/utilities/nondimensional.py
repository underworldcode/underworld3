"""
Helper functions for non-dimensionalization and scaling coefficient derivation.

This module provides functions to automatically derive scaling coefficients
from reference quantities using dimensional analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Any


def derive_scaling_coefficients(model):
    """
    Automatically derive scaling coefficients for all variables and parameters
    based on the model's reference quantities.

    Args:
        model: The Model object with reference quantities set

    This function performs dimensional analysis to find the appropriate
    combination of reference quantities that produces the correct
    dimensionality for each variable/parameter.
    """
    if not hasattr(model, "_reference_quantities") or not model._reference_quantities:
        return  # No reference quantities to work with

    # Get fundamental scales from reference quantities
    if not hasattr(model, "_fundamental_scales"):
        return  # No fundamental scales derived yet

    fundamental_scales = model._fundamental_scales

    # Process all registered variables
    for name, var in model._variables.items():
        if hasattr(var, "dimensionality") and var.dimensionality:
            # Try to derive scaling coefficient from dimensionality
            scale = _derive_scale_for_variable(var, fundamental_scales, model._reference_quantities)
            if scale is not None and hasattr(var, "set_reference_scale"):
                var.set_reference_scale(scale)


def _derive_scale_for_variable(var, fundamental_scales, reference_quantities):
    """
    Derive the scaling coefficient for a variable based on its dimensionality.

    Args:
        var: Variable with dimensionality property
        fundamental_scales: Dictionary of fundamental scales (length, time, mass, etc.)
        reference_quantities: Dictionary of reference quantities

    Returns:
        Scaling coefficient or None if unable to derive
    """
    # First check if any reference quantity matches the variable's dimensionality exactly
    var_dim = var.dimensionality
    for name, qty in reference_quantities.items():
        if hasattr(qty, "dimensionality"):
            if str(qty.dimensionality) == str(var_dim):
                # Direct match found - convert to base units first
                try:
                    base_qty = qty.to_base_units()
                    if hasattr(base_qty, "magnitude"):
                        return float(base_qty.magnitude)
                    elif hasattr(base_qty, "value"):
                        return float(base_qty.value)
                except:
                    # Fallback if conversion fails
                    if hasattr(qty, "magnitude"):
                        return float(qty.magnitude)
                    elif hasattr(qty, "value"):
                        return float(qty.value)

    # Try to construct from fundamental scales using Pint dimensionality objects
    scale = _construct_from_fundamental_scales(var_dim, fundamental_scales)
    if scale is not None:
        return scale

    # For common variable types, use heuristics
    var_name = var.name.lower() if hasattr(var, "name") else ""
    scale = _heuristic_scaling(var_name, var_dim, fundamental_scales, reference_quantities)

    return scale


def _construct_from_fundamental_scales(var_dimensionality, fundamental_scales):
    """
    Construct a scaling coefficient from fundamental scales using dimensional powers.

    Uses LINEAR ALGEBRA: scale = L^a * T^b * M^c * θ^d where [a,b,c,d] are the
    dimensional exponents from the Pint dimensionality object.

    Args:
        var_dimensionality: Pint dimensionality object, dict, or string representation
        fundamental_scales: Dict with 'length', 'time', 'mass', 'temperature' Pint Quantity scales

    Returns:
        Scaling coefficient (float) or None if scales are incomplete
    """
    if not fundamental_scales:
        return None

    # Map Pint dimension names to our fundamental_scales keys
    dim_to_key = {
        "[length]": "length",
        "[time]": "time",
        "[mass]": "mass",
        "[temperature]": "temperature",
    }

    # Convert dimensionality to a dict of {dimension: power}
    if isinstance(var_dimensionality, dict):
        dim_dict = var_dimensionality
    elif isinstance(var_dimensionality, str):
        # Parse string like "[length] / [time]" or "[length] ** 2 / [time]"
        dim_dict = _parse_dimensionality_string(var_dimensionality)
        if dim_dict is None:
            return None
    elif hasattr(var_dimensionality, "items"):
        # Pint UnitsContainer or similar dict-like object
        dim_dict = dict(var_dimensionality.items())
    else:
        # Try direct dict conversion as fallback
        try:
            dim_dict = dict(var_dimensionality)
        except (TypeError, ValueError):
            return None

    # Compute scale = product of (fundamental_scale ^ power) for each dimension
    result = 1.0
    for pint_dim, power in dim_dict.items():
        if power == 0:
            continue

        key = dim_to_key.get(pint_dim)
        if key is None or key not in fundamental_scales:
            # Missing a required fundamental scale
            return None

        scale = fundamental_scales[key]
        # Extract magnitude from Pint Quantity
        if hasattr(scale, "magnitude"):
            scale_value = float(scale.magnitude)
        else:
            scale_value = float(scale)

        result *= scale_value ** power

    return result


def _parse_dimensionality_string(dim_str):
    """
    Parse a dimensionality string like "[length] / [time]" into a dict.

    Examples:
        "[length]" -> {"[length]": 1}
        "[length] / [time]" -> {"[length]": 1, "[time]": -1}
        "[length] ** 2 / [time]" -> {"[length]": 2, "[time]": -1}
        "[mass] / [length] ** 3" -> {"[mass]": 1, "[length]": -3}

    Returns:
        dict mapping dimension names to powers, or None if parsing fails
    """
    import re

    if not dim_str or not isinstance(dim_str, str):
        return None

    result = {}

    # Handle dimensionless
    if dim_str.strip() == "dimensionless" or dim_str.strip() == "":
        return {}

    # Split by "/" to get numerator and denominator
    parts = dim_str.split("/")

    # Process numerator (positive powers)
    if len(parts) >= 1:
        numerator = parts[0].strip()
        _parse_dim_part(numerator, result, positive=True)

    # Process denominator (negative powers)
    if len(parts) >= 2:
        denominator = "/".join(parts[1:]).strip()
        _parse_dim_part(denominator, result, positive=False)

    return result if result else None


def _parse_dim_part(part, result_dict, positive=True):
    """Parse a part of dimensionality string (numerator or denominator)."""
    import re

    # Find all dimensions with optional exponents
    # Matches: [length], [length] ** 2, [time] ** 3, etc.
    pattern = r"\[(\w+)\](?:\s*\*\*\s*(\d+))?"

    for match in re.finditer(pattern, part):
        dim_name = f"[{match.group(1)}]"
        power = int(match.group(2)) if match.group(2) else 1

        if not positive:
            power = -power

        # Accumulate powers for same dimension
        result_dict[dim_name] = result_dict.get(dim_name, 0) + power


def get_required_reference_quantities(units_str, reference_quantities=None):
    """
    Determine which reference quantities are required for a given unit string.

    Uses pure dimensional analysis - NOT hardcoded parameter names.
    Checks if the given unit's dimensionality can be derived from available
    reference quantities through dimensional analysis alone.

    Args:
        units_str: Unit string (e.g., 'Pa', 'm/s', 'K')
        reference_quantities: Optional dict of reference quantities to check against

    Returns:
        tuple: (can_be_derived, descriptive message)
               can_be_derived=True if dimensionality can be derived from available quantities
               message contains explanation of what is available or missing
    """
    try:
        from ..scaling import units as ureg
    except ImportError:
        import pint

        ureg = pint.UnitRegistry()

    # Parse the target units
    try:
        qty = ureg(units_str)
        target_dim = qty.dimensionality
    except:
        return (True, "")  # If we can't parse, don't validate

    # If no reference quantities provided, we can't determine if derivable
    if reference_quantities is None:
        return (False, f"No reference quantities provided to derive units '{units_str}'")

    if not reference_quantities:
        return (False, f"Reference quantities not set for units '{units_str}'")

    # Extract dimensionalities from all reference quantities
    ref_dims = {}
    for name, ref_qty in reference_quantities.items():
        try:
            if hasattr(ref_qty, "dimensionality"):
                ref_dims[name] = ref_qty.dimensionality
        except:
            pass

    # Check if target dimensionality exactly matches any reference quantity
    for name, ref_dim in ref_dims.items():
        if target_dim == ref_dim:
            return (True, f"Dimensionality '{units_str}' matches reference quantity '{name}'")

    # Check if target dimensionality can be derived from reference quantities
    # by analyzing the fundamental scales that can be extracted
    can_derive, explanation = _check_derivability_from_dimensions(target_dim, ref_dims, ureg)

    return (can_derive, explanation)


def _check_derivability_from_dimensions(target_dim, ref_dimensions, ureg):
    """
    Check if a target dimensionality can be derived from reference quantity dimensions.

    Uses LINEAR ALGEBRA on the dimensional matrix - the same approach as
    _comprehensive_dimensional_analysis(). If the reference quantities span
    the dimensional space (matrix rank = 4), any dimensionality can be derived.

    Args:
        target_dim: Target Pint dimensionality object
        ref_dimensions: Dict mapping reference quantity names to their dimensionalities
        ureg: Pint UnitRegistry

    Returns:
        tuple: (can_derive, explanation_message)
    """
    import numpy as np

    # The four fundamental dimensions
    fundamental_dims = ["[length]", "[time]", "[mass]", "[temperature]"]

    # Build dimensional matrix from reference quantities (same as _comprehensive_dimensional_analysis)
    matrix = []
    for name, dim in ref_dimensions.items():
        dim_dict = dict(dim)
        row = [dim_dict.get(fund_dim, 0) for fund_dim in fundamental_dims]
        matrix.append(row)

    if not matrix:
        return (False, "No reference quantities with valid dimensionality")

    matrix = np.array(matrix)

    # Check matrix rank - if rank == 4, all fundamental dimensions are covered
    rank = np.linalg.matrix_rank(matrix)

    if rank >= 4:
        # Full coverage - any dimensionality can be derived
        return (True, f"Reference quantities span all fundamental dimensions (rank {rank}/4)")

    # Partial coverage - identify which dimensions are missing
    covered_dims = []
    for i, dim_name in enumerate(fundamental_dims):
        if np.any(matrix[:, i] != 0):
            covered_dims.append(dim_name.strip("[]"))

    missing_dims = [d.strip("[]") for d in fundamental_dims if d.strip("[]") not in covered_dims]

    # Check if the target dimensionality only uses covered dimensions
    target_dict = dict(target_dim)
    target_needs = []
    for fund_dim in fundamental_dims:
        if target_dict.get(fund_dim, 0) != 0:
            target_needs.append(fund_dim.strip("[]"))

    # If target only needs dimensions we have, it can be derived
    missing_for_target = [d for d in target_needs if d not in covered_dims]
    if not missing_for_target:
        return (True, f"Target dimensionality uses only covered dimensions: {covered_dims}")

    # Cannot derive - report what's missing
    return (
        False,
        f"Incomplete dimensional coverage (rank {rank}/4). "
        f"Missing: {missing_dims}. Target needs: {missing_for_target}",
    )


def validate_variable_reference_quantities(var_name, units_str, model):
    """
    Validate that a variable's units can be properly scaled by reference quantities.

    Uses pure dimensional analysis (NOT hardcoded parameter names) to determine
    if the variable's dimensionality can be derived from available reference quantities.

    Args:
        var_name: Name of the variable
        units_str: Unit string for the variable
        model: Model object to check for reference quantities

    Returns:
        tuple: (is_valid, warning_message)
               is_valid=True if units can be scaled properly
               warning_message contains details if validation fails
    """
    if units_str is None:
        return (True, "")

    # Check if model has any reference quantities set
    has_ref_quantities = (
        hasattr(model, "_reference_quantities")
        and model._reference_quantities
        and len(model._reference_quantities) > 0
    )

    if not has_ref_quantities:
        return (
            False,
            f"Variable '{var_name}' has units '{units_str}' but no reference quantities are set.\n"
            f"Call model.set_reference_quantities() before creating variables with units.",
        )

    # Use dimensional analysis to check if units can be derived
    can_derive, message = get_required_reference_quantities(units_str, model._reference_quantities)

    if can_derive:
        # All good - dimensionality can be derived
        return (True, "")
    else:
        # Cannot derive the needed dimensionality
        return (
            False,
            f"Variable '{var_name}' with units '{units_str}' cannot be properly scaled.\n"
            f"  Issue: {message}\n"
            f"  Available reference quantities: {list(model._reference_quantities.keys())}\n"
            f"  Ensure you provide enough reference quantities to define all fundamental scales [L], [M], [T], [θ].",
        )

    return (True, "")


def _heuristic_scaling(var_name, dimensionality, fundamental_scales, reference_quantities):
    """
    Use heuristics based on variable name and type to determine scaling.

    Args:
        var_name: Name of the variable (lowercase)
        dimensionality: Dimensionality string
        fundamental_scales: Dictionary of fundamental scales
        reference_quantities: Dictionary of reference quantities

    Returns:
        Scaling coefficient or None
    """
    # Velocity-like variables
    if any(v in var_name for v in ["velocity", "vel", "flow"]):
        # Look for velocity reference
        for name, qty in reference_quantities.items():
            if "velocity" in name.lower() or "vel" in name.lower():
                if hasattr(qty, "magnitude"):
                    return float(qty.magnitude)
        # Fallback to L/T
        if "length" in fundamental_scales and "time" in fundamental_scales:
            return fundamental_scales["length"] / fundamental_scales["time"]

    # Pressure-like variables
    if any(p in var_name for p in ["pressure", "stress", "sigma", "tau"]):
        # Look for pressure/stress reference
        for name, qty in reference_quantities.items():
            if any(p in name.lower() for p in ["pressure", "stress", "viscosity"]):
                if "viscosity" in name.lower():
                    # Pressure scale from viscosity: eta * v / L
                    if all(k in fundamental_scales for k in ["length", "time", "mass"]):
                        L = fundamental_scales["length"]
                        T = fundamental_scales["time"]
                        M = fundamental_scales["mass"]
                        # Stress ~ viscosity * strain_rate ~ (M/(L*T)) * (1/T)
                        return M / (L * T**2)
        # Fallback
        if all(k in fundamental_scales for k in ["mass", "length", "time"]):
            return fundamental_scales["mass"] / (
                fundamental_scales["length"] * fundamental_scales["time"] ** 2
            )

    # Temperature-like variables
    if any(t in var_name for t in ["temperature", "temp", "thermal"]):
        if "temperature" in fundamental_scales:
            return fundamental_scales["temperature"]
        # Look for temperature reference
        for name, qty in reference_quantities.items():
            if "temperature" in name.lower() or "temp" in name.lower():
                if hasattr(qty, "magnitude"):
                    return float(qty.magnitude)

    return None


def apply_nondimensional_scaling(expr, model):
    """
    Apply non-dimensional scaling to a SymPy expression.

    This replaces all variables and parameters with their non-dimensional
    equivalents by dividing by their scaling coefficients.

    Args:
        expr: SymPy expression
        model: Model object with scaling coefficients set

    Returns:
        Scaled SymPy expression
    """
    import sympy

    substitutions = {}

    # Find all function symbols (variables)
    for func_symbol in expr.atoms(sympy.Function):
        # Try to find corresponding variable in model
        for var_name, var in model._variables.items():
            if hasattr(var, "sym"):
                # Check if this function matches the variable's symbolic form
                if str(func_symbol.func) == str(var.sym.func):
                    if hasattr(var, "scaling_coefficient") and var.scaling_coefficient != 1.0:
                        # Create non-dimensional symbol
                        nd_func = sympy.Function(f"{func_symbol.func}_star")
                        # Replace with scaled version
                        substitutions[func_symbol] = func_symbol / var.scaling_coefficient
                    break

    # Find all regular symbols (parameters)
    for symbol in expr.atoms(sympy.Symbol):
        # Check if this is a tracked parameter
        # This would need model to track parameters separately
        # For now, we'll look for common parameter patterns
        if hasattr(model, "_constitutive_models"):
            for const_model in model._constitutive_models:
                if hasattr(const_model, "parameters"):
                    for param_name, param in const_model.parameters.items():
                        if str(symbol) == str(param.sym) if hasattr(param, "sym") else str(param):
                            if (
                                hasattr(param, "scaling_coefficient")
                                and param.scaling_coefficient != 1.0
                            ):
                                substitutions[symbol] = symbol / param.scaling_coefficient

    return expr.subs(substitutions) if substitutions else expr
