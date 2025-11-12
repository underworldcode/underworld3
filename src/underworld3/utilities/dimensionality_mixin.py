"""
Dimensionality tracking mixin for non-dimensionalization support.
This shadows the units system and adds scaling coefficient tracking.

This is designed to have ZERO side effects on existing functionality.
"""

import sympy
import numpy as np
from typing import Optional, Union


class DimensionalityMixin:
    """
    Mixin to add dimensionality tracking and non-dimensionalization capability
    to any class that has units.

    Attributes:
        _scaling_coefficient: Reference scale for non-dimensionalization
        _is_nondimensional: Whether currently in non-dimensional state
        _original_units: Stores units before non-dimensionalization
    """

    def __init__(self, *args, **kwargs):
        """Initialize dimensionality tracking attributes"""
        super().__init__(*args, **kwargs)
        self._scaling_coefficient = 1.0  # Default: no scaling
        self._is_nondimensional = False
        self._original_units = None
        self._original_dimensionality = None

    @property
    def dimensionality(self):
        """
        Get dimensionality from units if available.
        Returns None for dimensionless quantities.
        """
        if self._is_nondimensional:
            return None  # Non-dimensional has no dimensionality

        if hasattr(self, "units") and self.units:
            # Try to get from Pint if using UWQuantity
            if hasattr(self, "_pint_qty"):
                return str(self._pint_qty.dimensionality)
            # Try from units backend
            elif hasattr(self, "_units_backend") and self._units_backend:
                try:
                    from underworld3.scaling import units

                    qty = units.Quantity(1.0, self.units)
                    return str(qty.dimensionality)
                except:
                    pass
            # For simple string units
            return f"[{self.units}]"
        return None  # Dimensionless

    @property
    def scaling_coefficient(self):
        """Get the reference scale for non-dimensionalization"""
        return self._scaling_coefficient

    @scaling_coefficient.setter
    def scaling_coefficient(self, value):
        """Set the reference scale for non-dimensionalization"""
        if value is None or value == 0:
            raise ValueError("Scaling coefficient must be non-zero")

        # Handle UWQuantity or Pint quantities
        if hasattr(value, "magnitude"):
            # Convert to same units as self if possible
            if hasattr(self, "units") and self.units and hasattr(value, "to"):
                try:
                    value_in_my_units = value.to(self.units)
                    self._scaling_coefficient = float(value_in_my_units.magnitude)
                except:
                    self._scaling_coefficient = float(value.magnitude)
            else:
                self._scaling_coefficient = float(value.magnitude)
        else:
            self._scaling_coefficient = float(value)

    @property
    def is_nondimensional(self):
        """Check if currently in non-dimensional state"""
        return self._is_nondimensional

    @property
    def nd_array(self):
        """
        Get non-dimensional array values.
        Convenience property for accessing array data in non-dimensional form.
        """
        if not hasattr(self, "array"):
            raise AttributeError(f"{type(self).__name__} does not have array property")

        import numpy as np

        return np.array(self.array) / self._scaling_coefficient

    def from_nd(self, nd_value):
        """
        Convert a non-dimensional value back to dimensional form.

        Args:
            nd_value: Non-dimensional value to convert

        Returns:
            Dimensional value
        """
        return nd_value * self._scaling_coefficient

    def set_reference_scale(self, scale):
        """
        Set the reference scale for non-dimensionalization.

        Args:
            scale: Reference scale (can be UWQuantity, Pint quantity, or number)
        """
        self.scaling_coefficient = scale

    def _create_nondimensional_expression(self):
        """Create non-dimensional symbolic expression"""
        if not hasattr(self, "sym"):
            return self

        # Create starred symbol for non-dimensional version
        if isinstance(self.sym, sympy.Function):
            # For functions like T(x,y,z)
            base_name = str(self.sym.func)
            nd_func = sympy.Function(f"{base_name}_star")
            # Use same arguments
            nd_sym = nd_func(*self.sym.args)
        elif isinstance(self.sym, sympy.Symbol):
            # For symbols like eta
            nd_sym = sympy.Symbol(f"{self.sym.name}_star")
        else:
            # For expressions, divide by scale
            nd_sym = self.sym / self._scaling_coefficient

        # Create wrapper that knows it's non-dimensional
        from underworld3.function import expression

        nd_expr = expression(
            nd_sym, None, f"Non-dimensional {getattr(self, 'name', 'expression')}"  # No units
        )
        nd_expr._scaling_coefficient = self._scaling_coefficient
        nd_expr._is_nondimensional = True
        nd_expr._original_expression = self

        return nd_expr


class NonDimensionalView:
    """
    A view of a variable in non-dimensional form.
    This allows accessing arrays in non-dimensional form without modifying the original.
    """

    def __init__(self, original):
        self._original = original
        self._scaling_coefficient = original.scaling_coefficient
        self._is_nondimensional = True

    @property
    def array(self):
        """Return non-dimensionalized array values"""
        import numpy as np

        # Get the underlying numpy array and scale it
        original_array = np.array(self._original.array)
        return original_array / self._scaling_coefficient

    @array.setter
    def array(self, values):
        """Set values in non-dimensional form (converts back to dimensional)"""
        self._original.array[...] = values * self._scaling_coefficient

    @property
    def data(self):
        """Return non-dimensionalized data values"""
        import numpy as np

        # Get the underlying numpy array and scale it
        original_data = np.array(self._original.data)
        return original_data / self._scaling_coefficient

    @data.setter
    def data(self, values):
        """Set data in non-dimensional form"""
        self._original.data[...] = values * self._scaling_coefficient

    @property
    def sym(self):
        """Return non-dimensional symbolic representation"""
        if hasattr(self._original, "sym"):
            base_sym = self._original.sym
            if isinstance(base_sym, sympy.Function):
                # Create starred version
                nd_func = sympy.Function(f"{base_sym.func}_star")
                return nd_func(*base_sym.args)
            else:
                return base_sym / self._scaling_coefficient
        return None

    @property
    def dimensionality(self):
        """Non-dimensional quantities have no dimensionality"""
        return None

    @property
    def units(self):
        """Non-dimensional quantities have no units"""
        return None

    @property
    def is_nondimensional(self):
        """Always True for non-dimensional view"""
        return True

    def to_dimensional(self):
        """Convert back to dimensional form"""
        return self._original

    def __repr__(self):
        return f"NonDimensionalView({self._original.name if hasattr(self._original, 'name') else 'variable'})"
