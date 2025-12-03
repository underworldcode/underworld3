"""
UWQuantity - Simplified unit-aware quantities for Underworld3

This is a simplified implementation following the MeshVariable pattern:
- Stores dimensional values (what user sees via .value)
- Provides non-dimensional values (what solver sees via .data)
- All arithmetic delegated to Pint
- No UnitAwareExpression complexity

Design Principles:
1. UWQuantity is just a number with units - nothing more
2. Arithmetic uses Pint directly, returns UWQuantity
3. .value = dimensional (user view), .data = non-dimensional (solver view)
4. No symbolic complexity - that belongs in UWexpression
"""

import sympy
import numpy as np
from typing import Union, Optional, Any


class UWQuantity:
    """
    A number with units.

    Simple, clean, Pint-backed. Follows the MeshVariable pattern:
    - .value → dimensional (what user sees)
    - .data → non-dimensional (what solver sees)
    - .units → Pint Unit object

    All arithmetic is delegated to Pint. No symbolic complexity.

    Parameters
    ----------
    value : float, int, array-like
        The numerical value (dimensional)
    units : str or Pint Unit, optional
        Units specification (e.g., "Pa*s", "cm/year", "K")

    Examples
    --------
    >>> viscosity = uw.quantity(1e21, "Pa*s")
    >>> viscosity.value  # 1e21 (dimensional)
    >>> viscosity.data   # 1.0 (non-dimensional, if model is set up)
    >>> viscosity.units  # <Unit('pascal * second')>

    >>> # Arithmetic via Pint
    >>> T1 = uw.quantity(1000, "kelvin")
    >>> T2 = uw.quantity(273, "kelvin")
    >>> dT = T1 - T2  # UWQuantity(727, "kelvin")
    """

    def __init__(
        self,
        value: Union[float, int, np.ndarray],
        units: Optional[str] = None
    ):
        """
        Initialize a UWQuantity.

        Parameters
        ----------
        value : float, int, or array-like
            The dimensional value
        units : str or Pint Unit, optional
            Units specification
        """
        from ..scaling import units as ureg

        # Store value as numpy array or scalar
        if isinstance(value, (list, tuple)):
            self._value = np.asarray(value)
        elif isinstance(value, np.ndarray):
            self._value = value
        else:
            # Scalar - store directly
            self._value = value

        # Handle units
        if units is not None:
            # Accept both strings and Pint Unit objects
            if hasattr(units, 'dimensionality'):
                # Already a Pint Unit
                self._pint_unit = units
            else:
                # String - parse it
                self._pint_unit = ureg.parse_expression(units).units

            # Create Pint Quantity for arithmetic
            self._pint_qty = self._value * self._pint_unit
        else:
            self._pint_unit = None
            self._pint_qty = None

        # Cache for non-dimensional value (computed lazily)
        self._nd_value_cache = None
        self._nd_value_valid = False

        # Units backend for protocol compatibility (units.py uses this)
        self._units_backend = "pint"

    @classmethod
    def _from_pint(cls, pint_qty, model_registry=None):
        """
        Create UWQuantity from a Pint Quantity object.

        This is used by Model.to_model_units() and other internal methods
        that work with Pint quantities directly.

        Parameters
        ----------
        pint_qty : pint.Quantity
            A Pint Quantity object
        model_registry : pint.UnitRegistry, optional
            Model-specific registry (for model units)

        Returns
        -------
        UWQuantity
            New quantity with the Pint quantity's value and units
        """
        value = pint_qty.magnitude
        units = pint_qty.units
        return cls(value, units)

    # =========================================================================
    # Core Properties - The MeshVariable Pattern
    # =========================================================================

    @property
    def value(self) -> Union[float, np.ndarray]:
        """
        Dimensional value (what the user sees).

        Returns
        -------
        float or np.ndarray
            The value in the quantity's units
        """
        return self._value

    @property
    def data(self) -> Union[float, np.ndarray]:
        """
        Non-dimensional value (what the solver sees).

        Returns the value scaled by the model's reference quantities.
        If no model is registered or no scaling is active, returns the
        dimensional value.

        Returns
        -------
        float or np.ndarray
            Non-dimensional value for solver use
        """
        if self._nd_value_valid:
            return self._nd_value_cache

        # Compute non-dimensional value
        self._nd_value_cache = self._compute_nd_value()
        self._nd_value_valid = True
        return self._nd_value_cache

    def _compute_nd_value(self) -> Union[float, np.ndarray]:
        """Compute the non-dimensional value using model scaling."""
        import underworld3 as uw

        # If no units, value is already "non-dimensional"
        if self._pint_unit is None:
            return self._value

        # Try to get scaling from model
        try:
            model = uw.get_default_model()
            if model is not None and model.has_units():
                # Get the scale factor for our dimensionality
                scale = model.get_scale_for_dimensionality(self.dimensionality)
                if scale is not None:
                    # Extract scalar from scale (may be Pint Quantity)
                    if hasattr(scale, 'magnitude'):
                        scale_value = scale.magnitude
                    else:
                        scale_value = float(scale)

                    if scale_value != 0:
                        # Convert to base units first, then scale
                        base_value = self._pint_qty.to_base_units().magnitude
                        return base_value / scale_value
        except Exception:
            pass

        # Fallback: return dimensional value
        return self._value

    @property
    def magnitude(self) -> Union[float, np.ndarray]:
        """Alias for .value (Pint compatibility)."""
        return self.value

    @property
    def units(self):
        """
        Get the Pint Unit object.

        Returns
        -------
        pint.Unit or None
            The unit, or None if dimensionless
        """
        return self._pint_unit

    @property
    def has_units(self) -> bool:
        """Check if this quantity has units."""
        return self._pint_unit is not None

    @property
    def dimensionality(self) -> dict:
        """
        Get the Pint dimensionality dictionary.

        Returns
        -------
        dict
            e.g., {'[length]': 1, '[time]': -1} for velocity
        """
        if self._pint_qty is not None:
            return dict(self._pint_qty.dimensionality)
        return {}

    # =========================================================================
    # Unit Conversion Methods
    # =========================================================================

    def to(self, target_units: str) -> 'UWQuantity':
        """
        Convert to different units.

        Parameters
        ----------
        target_units : str
            Target units (e.g., "m/s", "km", "degC")

        Returns
        -------
        UWQuantity
            New quantity with converted value and units
        """
        if self._pint_qty is None:
            raise ValueError("Cannot convert dimensionless quantity")

        converted = self._pint_qty.to(target_units)
        return UWQuantity(converted.magnitude, converted.units)

    def to_base_units(self) -> 'UWQuantity':
        """Convert to SI base units."""
        if self._pint_qty is None:
            return self

        base = self._pint_qty.to_base_units()
        return UWQuantity(base.magnitude, base.units)

    def to_reduced_units(self) -> 'UWQuantity':
        """Simplify units by canceling common factors."""
        if self._pint_qty is None:
            return self

        reduced = self._pint_qty.to_reduced_units()
        return UWQuantity(reduced.magnitude, reduced.units)

    def to_compact(self) -> 'UWQuantity':
        """Convert to most readable unit representation."""
        if self._pint_qty is None:
            return self

        compact = self._pint_qty.to_compact()
        return UWQuantity(compact.magnitude, compact.units)

    # =========================================================================
    # Arithmetic - Pure Pint Delegation
    # =========================================================================

    def __add__(self, other: Union['UWQuantity', float, int]) -> 'UWQuantity':
        """Addition via Pint."""
        from .expressions import UWexpression

        if isinstance(other, UWQuantity):
            if self._pint_qty is not None and other._pint_qty is not None:
                result = self._pint_qty + other._pint_qty
                return UWQuantity(result.magnitude, result.units)
            else:
                # One or both dimensionless
                return UWQuantity(self._value + other._value, self._pint_unit or other._pint_unit)

        # Handle UWexpression
        if isinstance(other, UWexpression):
            # Delegate to UWexpression's __radd__
            return NotImplemented

        # Handle SymPy expressions - LAZY EVALUATION approach
        if isinstance(other, sympy.Basic):
            if self._pint_qty is not None:
                # Wrap self in UWexpression first, then add symbolically
                # This preserves unit information AND the symbolic structure for diff/subs
                # Use Pint's LaTeX format for readable names (e.g., "300\ \mathrm{K}")
                # Uniqueness handled by _unique_name_generation via \hspace trick
                latex_name = f"{self._pint_qty:~L}"
                wrapped_self = UWexpression(
                    latex_name,
                    self,  # Store the full UWQuantity - Transparent Container
                    _unique_name_generation=True
                )
                # Symbolic addition: UWexpression (symbol) + sympy expr
                # Result is a sympy Add - preserves structure for differentiation
                return wrapped_self + other
            else:
                # No units - return plain SymPy result
                return self._value + other

        # Scalar addition
        if self._pint_qty is not None:
            result = self._pint_qty + other
            return UWQuantity(result.magnitude, result.units)
        else:
            return UWQuantity(self._value + other)

    def __radd__(self, other):
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Union['UWQuantity', float, int]) -> 'UWQuantity':
        """Subtraction via Pint."""
        from .expressions import UWexpression

        if isinstance(other, UWQuantity):
            if self._pint_qty is not None and other._pint_qty is not None:
                result = self._pint_qty - other._pint_qty
                return UWQuantity(result.magnitude, result.units)
            else:
                return UWQuantity(self._value - other._value, self._pint_unit or other._pint_unit)

        # Handle UWexpression: UWQuantity - UWexpression → UWexpression
        if isinstance(other, UWexpression):
            from ..units import get_units
            other_units = other.units

            # For subtraction, units must be compatible - convert other to self's units
            if self._pint_unit is not None and other_units is not None:
                try:
                    from ..scaling import units as ureg
                    # Convert other's value to self's units
                    other_converted = (other.value * other_units).to(self._pint_unit).magnitude
                except Exception:
                    # Units incompatible - use raw value (will be wrong but won't crash)
                    other_converted = other.value
            else:
                other_converted = other.value

            result_sym = self._value - other_converted
            combined_units = self._pint_unit  # Result in self's units
            return UWexpression(
                f"(qty-{other.name})",
                result_sym,
                _unique_name_generation=True,
                units=combined_units
            )

        # Handle SymPy expressions
        if isinstance(other, sympy.Basic):
            if self._pint_qty is not None:
                from ..units import get_units
                other_units = get_units(other)
                combined_units = self._pint_unit  # Subtraction preserves units
                result_sym = self._value - other
                return UWexpression(
                    f"(qty-sympy)",
                    result_sym,
                    _unique_name_generation=True,
                    units=combined_units
                )
            else:
                return self._value - other

        if self._pint_qty is not None:
            result = self._pint_qty - other
            return UWQuantity(result.magnitude, result.units)
        else:
            return UWQuantity(self._value - other)

    def __rsub__(self, other):
        """Right subtraction: other - self."""
        from .expressions import UWexpression

        # Handle UWexpression: UWexpression - UWQuantity is handled by UWexpression.__sub__
        # This handles: sympy.Basic - UWQuantity
        if isinstance(other, sympy.Basic):
            if self._pint_qty is not None:
                from ..units import get_units
                other_units = get_units(other)
                # For subtraction, if other has units, use those; otherwise use inverted self units
                if other_units is not None:
                    combined_units = other_units  # other - self has other's units
                else:
                    combined_units = self._pint_unit  # Fallback to self's units
                result_sym = other - self._value
                return UWexpression(
                    f"(sympy-qty)",
                    result_sym,
                    _unique_name_generation=True,
                    units=combined_units
                )
            else:
                return other - self._value

        if self._pint_qty is not None:
            result = other - self._pint_qty
            return UWQuantity(result.magnitude, result.units)
        else:
            return UWQuantity(other - self._value)

    def __mul__(self, other: Union['UWQuantity', float, int]) -> 'UWQuantity':
        """Multiplication via Pint."""
        if isinstance(other, UWQuantity):
            if self._pint_qty is not None and other._pint_qty is not None:
                result = self._pint_qty * other._pint_qty
                return UWQuantity(result.magnitude, result.units)
            elif self._pint_qty is not None:
                result = self._pint_qty * other._value
                return UWQuantity(result.magnitude, result.units)
            elif other._pint_qty is not None:
                result = self._value * other._pint_qty
                return UWQuantity(result.magnitude, result.units)
            else:
                return UWQuantity(self._value * other._value)
        else:
            # Handle UnitAwareArray - Pint doesn't properly combine units with UnitAwareArray
            # We need to manually combine units and multiply values
            from ..utilities.unit_aware_array import UnitAwareArray
            if isinstance(other, UnitAwareArray):
                other_units = other.units
                if self._pint_qty is not None and other_units is not None:
                    # Both have units - combine them via Pint
                    from ..scaling import units as ureg
                    combined_units = (1 * self._pint_unit * other_units).units
                    # Multiply numeric values (extract numpy from UnitAwareArray)
                    result_values = self._value * np.array(other)
                    # Return UnitAwareArray with combined units
                    return UnitAwareArray(result_values, units=combined_units)
                elif self._pint_qty is not None:
                    # Only self has units
                    result_values = self._value * np.array(other)
                    return UnitAwareArray(result_values, units=self._pint_unit)
                elif other_units is not None:
                    # Only other has units
                    result_values = self._value * np.array(other)
                    return UnitAwareArray(result_values, units=other_units)
                else:
                    # Neither has units - return plain numpy
                    return self._value * np.array(other)

            # Check if other is a UWexpression - handle specially to preserve units
            # Import here to avoid circular import
            from .expressions import UWexpression
            if isinstance(other, UWexpression):
                # Both may have units - combine them via Pint
                other_units = other.units
                if self._pint_qty is not None and other_units is not None:
                    # Both have units - compute combined units
                    from ..scaling import units as ureg
                    combined_units = (1 * self._pint_unit * other_units).units
                    # Create SymPy product using values
                    sympy_product = self._value * other.value
                    # Return UWexpression with combined units
                    return UWexpression(
                        f"({self}*{other.name})",
                        UWQuantity(sympy_product, combined_units),
                        _unique_name_generation=True
                    )
                elif self._pint_qty is not None:
                    # Only self has units
                    sympy_product = self._value * other.value
                    return UWexpression(
                        f"({self}*{other.name})",
                        UWQuantity(sympy_product, self._pint_unit),
                        _unique_name_generation=True
                    )
                elif other_units is not None:
                    # Only other has units
                    sympy_product = self._value * other.value
                    return UWexpression(
                        f"({self}*{other.name})",
                        UWQuantity(sympy_product, other_units),
                        _unique_name_generation=True
                    )
                else:
                    # Neither has units - just delegate to SymPy
                    return NotImplemented

            # Handle SymPy expressions - LAZY EVALUATION approach
            # Keep UWQuantity in the expression tree so get_units() can find it later
            if isinstance(other, sympy.Basic):
                if self._pint_qty is not None:
                    # Check if the result will be numeric (concrete) or symbolic
                    test_product = self._value * other

                    if test_product.is_number:
                        # Concrete numeric result - compute combined units directly
                        from ..units import get_units
                        other_units = get_units(other)
                        if other_units is not None:
                            combined_units = self._pint_unit * other_units
                        else:
                            combined_units = self._pint_unit
                        return UWexpression(
                            f"({self}*sympy)",
                            UWQuantity(float(test_product), combined_units),
                            _unique_name_generation=True
                        )
                    else:
                        # LAZY EVALUATION: Keep UWQuantity in expression tree
                        # Wrap self in UWexpression first, then do symbolic multiplication
                        # This preserves unit information for later get_units() traversal
                        # Use Pint's LaTeX format for readable symbol names (e.g., "300\ \mathrm{K}")
                        latex_name = f"{self._pint_qty:~L}"
                        wrapped_self = UWexpression(
                            latex_name,
                            self,  # Store the full UWQuantity - Transparent Container
                            _unique_name_generation=True
                        )
                        # Symbolic multiplication: UWexpression (symbol) * sympy expr
                        # Result is sympy expression containing wrapped_self
                        return wrapped_self * other
                else:
                    # No units - return plain SymPy result
                    return self._value * other

            # Scalar multiplication
            if self._pint_qty is not None:
                result = self._pint_qty * other
                return UWQuantity(result.magnitude, result.units)
            else:
                return UWQuantity(self._value * other)

    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Union['UWQuantity', float, int]) -> 'UWQuantity':
        """Division via Pint."""
        if isinstance(other, UWQuantity):
            if self._pint_qty is not None and other._pint_qty is not None:
                result = self._pint_qty / other._pint_qty
                return UWQuantity(result.magnitude, result.units)
            elif self._pint_qty is not None:
                result = self._pint_qty / other._value
                return UWQuantity(result.magnitude, result.units)
            elif other._pint_qty is not None:
                result = self._value / other._pint_qty
                return UWQuantity(result.magnitude, result.units)
            else:
                return UWQuantity(self._value / other._value)
        else:
            # Check if other is a UWexpression - handle specially to preserve units
            from .expressions import UWexpression
            if isinstance(other, UWexpression):
                other_units = other.units
                if self._pint_qty is not None and other_units is not None:
                    # Both have units - compute combined units (self / other)
                    from ..scaling import units as ureg
                    combined_units = (1 * self._pint_unit / other_units).units
                    sympy_quotient = self._value / other.value
                    return UWexpression(
                        f"({self}/{other.name})",
                        UWQuantity(sympy_quotient, combined_units),
                        _unique_name_generation=True
                    )
                elif self._pint_qty is not None:
                    # Only self has units
                    sympy_quotient = self._value / other.value
                    return UWexpression(
                        f"({self}/{other.name})",
                        UWQuantity(sympy_quotient, self._pint_unit),
                        _unique_name_generation=True
                    )
                elif other_units is not None:
                    # Only other has units - result has 1/other_units
                    from ..scaling import units as ureg
                    combined_units = (1 / other_units).units
                    sympy_quotient = self._value / other.value
                    return UWexpression(
                        f"({self}/{other.name})",
                        UWQuantity(sympy_quotient, combined_units),
                        _unique_name_generation=True
                    )
                else:
                    # Neither has units - just delegate to SymPy
                    return NotImplemented

            # Handle SymPy expressions - wrap in UWexpression to preserve units
            if isinstance(other, sympy.Basic):
                if self._pint_qty is not None:
                    sympy_quotient = self._value / other
                    # If result is a number, convert to Python float for UWQuantity
                    if sympy_quotient.is_number:
                        return UWexpression(
                            f"({self}/sympy)",
                            UWQuantity(float(sympy_quotient), self._pint_unit),
                            _unique_name_generation=True
                        )
                    else:
                        # Symbolic result - store sympy expr with units
                        return UWexpression(
                            f"({self}/sympy)",
                            sympy_quotient,
                            _unique_name_generation=True,
                            units=self._pint_unit
                        )
                else:
                    # No units - return plain SymPy result
                    return self._value / other

            if self._pint_qty is not None:
                result = self._pint_qty / other
                return UWQuantity(result.magnitude, result.units)
            else:
                return UWQuantity(self._value / other)

    def __rtruediv__(self, other):
        """Right division: other / self."""
        from .expressions import UWexpression

        # Handle SymPy types - return UWexpression with combined units
        # other / self → units = units(other) / units(self)
        if isinstance(other, sympy.Basic):
            if self._pint_qty is not None:
                from ..units import get_units
                from ..scaling import units as ureg
                inverted_units = 1 / self._pint_unit  # Unit / Unit = Unit

                # Compute combined units: get_units(other) / self.units
                other_units = get_units(other)
                if other_units is not None:
                    # Unit / Unit = Unit (no .units needed)
                    combined_units = other_units / self._pint_unit
                else:
                    combined_units = inverted_units

                sympy_quotient = other / self._value
                if sympy_quotient.is_number:
                    return UWexpression(
                        f"(sympy/{self})",
                        UWQuantity(float(sympy_quotient), combined_units),
                        _unique_name_generation=True
                    )
                else:
                    return UWexpression(
                        f"(sympy/{self})",
                        sympy_quotient,
                        _unique_name_generation=True,
                        units=combined_units
                    )
            else:
                return other / self._value

        # Python scalars with Pint
        if self._pint_qty is not None:
            result = other / self._pint_qty
            return UWQuantity(result.magnitude, result.units)
        else:
            return UWQuantity(other / self._value)

    def __pow__(self, exponent: Union[float, int]) -> 'UWQuantity':
        """Exponentiation via Pint."""
        if self._pint_qty is not None:
            result = self._pint_qty ** exponent
            return UWQuantity(result.magnitude, result.units)
        else:
            return UWQuantity(self._value ** exponent)

    def __neg__(self) -> 'UWQuantity':
        """Negation."""
        if self._pint_qty is not None:
            result = -self._pint_qty
            return UWQuantity(result.magnitude, result.units)
        else:
            return UWQuantity(-self._value)

    # =========================================================================
    # Comparisons - Via Pint
    # =========================================================================

    def __lt__(self, other):
        if isinstance(other, UWQuantity):
            if self._pint_qty is not None and other._pint_qty is not None:
                return self._pint_qty < other._pint_qty
            return self._value < other._value
        return self._value < other

    def __le__(self, other):
        if isinstance(other, UWQuantity):
            if self._pint_qty is not None and other._pint_qty is not None:
                return self._pint_qty <= other._pint_qty
            return self._value <= other._value
        return self._value <= other

    def __gt__(self, other):
        if isinstance(other, UWQuantity):
            if self._pint_qty is not None and other._pint_qty is not None:
                return self._pint_qty > other._pint_qty
            return self._value > other._value
        return self._value > other

    def __ge__(self, other):
        if isinstance(other, UWQuantity):
            if self._pint_qty is not None and other._pint_qty is not None:
                return self._pint_qty >= other._pint_qty
            return self._value >= other._value
        return self._value >= other

    def __eq__(self, other):
        if isinstance(other, UWQuantity):
            if self._pint_qty is not None and other._pint_qty is not None:
                return self._pint_qty == other._pint_qty
            return self._value == other._value
        return self._value == other

    def __ne__(self, other):
        return not self.__eq__(other)

    # =========================================================================
    # SymPy Compatibility
    # =========================================================================

    def _sympy_(self):
        """
        SymPy protocol - controls how SymPy converts this object.

        For quantities WITH units: raise SympifyError to force SymPy to
        return NotImplemented, which triggers our __rmul__/__radd__ etc.

        For quantities WITHOUT units: return the numeric value.
        """
        from sympy.core.sympify import SympifyError

        # If we have units, don't let SymPy consume us silently
        # This forces SymPy to return NotImplemented so our __rmul__ gets called
        if self._pint_unit is not None:
            raise SympifyError(self)

        # No units - safe to return numeric value
        if isinstance(self._value, np.ndarray):
            return sympy.Matrix(self._value.tolist())
        try:
            return sympy.Float(float(self._value))
        except (TypeError, ValueError):
            return sympy.sympify(self._value)

    def __float__(self):
        """Convert to float."""
        return float(self._value)

    def diff(self, *args, **kwargs):
        """Derivative of a constant is zero."""
        return 0

    # =========================================================================
    # Display
    # =========================================================================

    def __str__(self) -> str:
        """String representation matching UWexpression style: value [units]."""
        if self._pint_unit is not None:
            return f"{self._value} [{self._pint_unit}]"
        return str(self._value)

    def __repr__(self) -> str:
        """User-friendly representation matching UWexpression style: value [units]."""
        if self._pint_unit is not None:
            return f"{self._value} [{self._pint_unit}]"
        return str(self._value)

    def __format__(self, format_spec: str) -> str:
        """Formatted representation matching UWexpression style."""
        if format_spec:
            formatted = format(self._value, format_spec)
        else:
            formatted = str(self._value)

        if self._pint_unit is not None:
            return f"{formatted} [{self._pint_unit}]"
        return formatted

    # =========================================================================
    # Jupyter Display Methods
    # =========================================================================

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        value = self._value

        # Format value for LaTeX
        if isinstance(value, float):
            # Use scientific notation for very small/large numbers
            if value != 0 and (abs(value) < 0.01 or abs(value) >= 10000):
                value_latex = f"{value:.2e}".replace('e', r' \times 10^{') + '}'
            else:
                value_latex = str(value)
        else:
            value_latex = str(value)

        # Format units for LaTeX
        if self._pint_unit is not None:
            units_str = str(self._pint_unit).replace('**', '^').replace('*', r' \cdot ')
            return f"${value_latex} \\; \\mathrm{{{units_str}}}$"
        else:
            return f"${value_latex}$"

    def _repr_mimebundle_(self, **kwargs):
        """
        MIME bundle for Jupyter display - highest priority representation.

        This method has ABSOLUTE HIGHEST PRIORITY in Jupyter's display system.
        """
        return {
            'text/latex': self._repr_latex_(),
            'text/plain': repr(self),
        }

    def _ipython_display_(self):
        """
        IPython/Jupyter display hook - ABSOLUTE highest priority.

        Shows the quantity with units in LaTeX format.
        """
        try:
            from IPython.display import display, Latex

            latex_str = self._repr_latex_()
            display(Latex(latex_str))
        except ImportError:
            # IPython not available - silent fallback
            pass


def quantity(
    value: Union[float, int, np.ndarray],
    units: Optional[str] = None
) -> UWQuantity:
    """
    Create a unit-aware quantity.

    Parameters
    ----------
    value : float, int, or array-like
        The numerical value
    units : str, optional
        Units specification (e.g., "Pa*s", "cm/year", "K")

    Returns
    -------
    UWQuantity
        Unit-aware quantity

    Examples
    --------
    >>> viscosity = uw.quantity(1e21, "Pa*s")
    >>> velocity = uw.quantity(5, "cm/year")
    >>> dT = uw.quantity(1000, "K") - uw.quantity(273, "K")
    """
    return UWQuantity(value, units)
