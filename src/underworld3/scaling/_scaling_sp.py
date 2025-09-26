"""
Utilities to convert between dimensional and non-dimensional values.
"""
from __future__ import print_function, absolute_import
import underworld3 as uw
from ._utils_sp import TransformedDict, expr_dimension, unit, ensure_to_base_units
from sympy.physics.units.systems.si import dimsys_SI, SI
from sympy.physics.units import convert_to
import sympy.physics.units as u_sp

__all__ = [
    "get_coefficients_sp",
    "non_dimensionalise_sympy",
    "dimensionalise_sympy",
    "ndargs"
]

_COEFFICIENTS = None

def get_coefficients_sp():
    """
    Returns the global scaling dictionary.
    """
    global _COEFFICIENTS
    if _COEFFICIENTS is None:
        _COEFFICIENTS = TransformedDict()
        _COEFFICIENTS["[length]"] = 1.0 * u_sp.meter
        _COEFFICIENTS["[mass]"] = 1.0 * u_sp.kilogram
        _COEFFICIENTS["[time]"] = 1.0 * u_sp.year
        _COEFFICIENTS["[temperature]"] = 1.0 * u_sp.kelvin
        _COEFFICIENTS["[substance]"] = 1.0 * u_sp.mole
    return _COEFFICIENTS

def non_dimensionalise_sympy(value):
    """
    Non-dimensionalise a SymPy value using scaling coefficients.
    Args:
        value: SymPy quantity (e.g., 9.81*u.meter/u.second**2)
    Returns:
        float: dimensionless magnitude
    """
    scaling_coefficients = get_coefficients_sp()
    value_SI = convert_to(value, SI._base_units)
    mag_value, _ = value_SI.as_coeff_Mul()
    dim = expr_dimension(unit(value_SI))
    deps = dimsys_SI.get_dimensional_dependencies(dim)

    # Build a mapping from dimension to scaling coefficient
    dim_map = {expr_dimension(val): val for val in scaling_coefficients.values()}

    scale = 1.0
    for dep, exp in deps.items():
        if dep not in dim_map:
            raise ValueError(f"No scaling coefficient provided for dimension {dep}")
        factor = dim_map[dep] ** exp
        factor_SI = convert_to(factor, SI._base_units)
        mag_factor, _ = factor_SI.as_coeff_Mul()
        scale *= mag_factor

    return float(mag_value) / scale

def dimensionalise_sympy(nd_value, target_units):
    """
    Dimensionalise a dimensionless value using target_units and scaling_coefficients keyed by Dimension objects.
    Args:
        nd_value: float or symbolic, the non-dimensional value.
        target_units: SymPy units expression (e.g., u.meter/u.second)
        scaling_coefficients: dict mapping Dimension objects to SymPy quantities
    Returns:
        SymPy quantity: dimensionalised value with physical units.
    """
    scaling_coefficients = get_coefficients_sp()
    dim = expr_dimension(target_units)
    deps = dimsys_SI.get_dimensional_dependencies(dim)
    dim_map = {expr_dimension(val): val for val in scaling_coefficients.values()}

    scale = 1
    for dep, exp in deps.items():
        if dep not in dim_map:
            raise ValueError(f"No scaling coefficient provided for dimension {dep}")
        scale *= dim_map[dep] ** exp

    scale_target = convert_to(scale, target_units)
    return nd_value * scale_target



def ndargs(f):
    """ Decorator used to non-dimensionalise the arguments of a function"""

    def convert(obj):
        if isinstance(obj, (list, tuple)):
            return type(obj)([convert(val) for val in obj])
        else:
            return non_dimensionalise(obj)

    def new_f(*args, **kwargs):
        nd_args = [convert(arg) for arg in args]
        nd_kwargs = {name:convert(val) for name, val in kwargs.items()}
        return f(*nd_args, **nd_kwargs)
    new_f.__name__ = f.__name__
    return new_f
