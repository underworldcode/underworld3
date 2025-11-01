"""
UWQuantity - Lightweight unit-aware quantities for Underworld3

This module provides UWQuantity, a minimal unit-aware object that serves as the base
class for UWexpression. UWQuantity objects are ephemeral and lightweight, carrying
only the essential information needed for unit-aware operations:

- Numerical or symbolic value
- Units and scale factors (via UnitAwareMixin)
- Unit conversion methods
- Basic arithmetic operations

UWQuantity objects are intended for:
- Temporary calculations and intermediate results
- Parameter assignments to constitutive models
- Unit conversions between different unit systems
- Input to UWexpression constructor for promotion
"""

import sympy
from typing import Union, Optional, Any
from ..utilities.units_mixin import UnitAwareMixin
from ..utilities.dimensionality_mixin import DimensionalityMixin


class UWQuantity(DimensionalityMixin, UnitAwareMixin):
    """
    Lightweight unit-aware quantity.

    A minimal object that carries a value with units and scale factors,
    designed for ephemeral use and as a base class for UWexpression.

    Parameters
    ----------
    value : float, int, sympy.Expr
        The numerical or symbolic value
    units : str, optional
        Units specification (e.g., "Pa*s", "cm/year", "K")

    Examples
    --------
    >>> import underworld3 as uw
    >>> # Create quantity with units
    >>> viscosity = uw.quantity(1e21, "Pa*s")
    >>> velocity = uw.quantity(5, "cm/year")
    >>>
    >>> # Unit conversions
    >>> velocity_mps = velocity.to("m/s")
    >>> velocity_base = velocity.to_model_units()
    >>>
    >>> # Arithmetic operations preserve units
    >>> time_scale = distance / velocity  # Automatic unit calculation
    """

    def __init__(self, value: Union[float, int, sympy.Basic], units: Optional[str] = None, dimensionality: Optional[dict] = None, _custom_units: Optional[str] = None, _model_registry=None, _model_instance=None):
        """
        Initialize a UWQuantity.

        Parameters
        ----------
        value : float, int, sympy.Basic
            The value of the quantity
        units : str, optional
            Units specification
        dimensionality : dict, optional
            Pint dimensionality dictionary (e.g., {'[length]': 1, '[time]': -1} for velocity).
            Used to preserve original dimensionality for dimensionless quantities.
        _custom_units : str, optional
            Custom unit name that bypasses Pint validation (for model units)
        """
        # Initialize DimensionalityMixin
        DimensionalityMixin.__init__(self)

        # Initialize the symbolic value
        self._sym = sympy.sympify(value)

        # Handle custom model units that don't exist in Pint registry
        if _custom_units is not None and _model_registry is not None:
            # NEW: Native Pint approach using model's registry with _constants
            try:
                import pint
                self._pint_qty = value * getattr(_model_registry, _custom_units)
                self._has_pint_qty = True
                self._model_registry = _model_registry
                self._model_instance = _model_instance  # Store model instance for alias access
                self._custom_units = _custom_units
                self._has_custom_units = True
            except (AttributeError, ImportError):
                # Fallback to old approach if model registry doesn't have the constant
                self._custom_units = _custom_units
                self._has_custom_units = True
                self._has_pint_qty = False
        elif _custom_units is not None:
            # OLD: Custom units without model registry (legacy support)
            self._custom_units = _custom_units
            self._has_custom_units = True
            self._has_pint_qty = False
        elif units is not None:
            # Regular units: create standard Pint quantity
            try:
                # Use the scaling module's registry which includes planetary units
                from ..scaling import units as ureg
                self._pint_qty = value * ureg.parse_expression(units)
                self._has_pint_qty = True
                self._has_custom_units = False
            except ImportError:
                # Fallback to UnitAwareMixin if Pint not available
                self._has_custom_units = False
                self._has_pint_qty = False
                self.set_units(units)  # From UnitAwareMixin - sets up scale factors
        else:
            # Dimensionless
            self._has_custom_units = False
            self._has_pint_qty = False

        # Dimensionality tracking for non-dimensionalization/re-dimensionalization
        if dimensionality is not None:
            # Explicitly provided dimensionality (for dimensionless with memory)
            self._dimensionality = dimensionality
        elif self._has_pint_qty and hasattr(self._pint_qty, 'dimensionality'):
            # Extract from Pint quantity (normal case)
            self._dimensionality = dict(self._pint_qty.dimensionality)
        else:
            # No dimensionality information
            self._dimensionality = {}

    @property
    def value(self) -> Union[float, sympy.Basic]:
        """Get the value in the quantity's specified units."""
        if hasattr(self._sym, 'evalf'):
            # Try to get numerical value, fall back to symbolic
            try:
                return float(self._sym.evalf())
            except (TypeError, ValueError):
                return self._sym
        else:
            return self._sym

    @property
    def has_units(self) -> bool:
        """
        Override UnitAwareMixin.has_units to handle custom model units and
        cases where UnitAwareMixin wasn't properly initialized.
        """
        if hasattr(self, '_has_pint_qty') and self._has_pint_qty:
            return True
        if hasattr(self, '_has_custom_units') and self._has_custom_units:
            return True
        return hasattr(self, '_dimensional_quantity') and self._dimensional_quantity is not None

    @property
    def units(self) -> str:
        """Get the units string for this quantity."""
        if hasattr(self, '_has_pint_qty') and self._has_pint_qty:
            return str(self._pint_qty.units)
        elif hasattr(self, '_has_custom_units') and self._has_custom_units:
            return self._custom_units
        elif hasattr(self, '_units_backend') and hasattr(self, '_dimensional_quantity') and self._dimensional_quantity is not None:
            return str(self._units_backend.get_units(self._dimensional_quantity))
        else:
            return None

    @property
    def sym(self) -> sympy.Basic:
        """Get the symbolic representation (in model base units if scaled)."""
        return self._sym

    @property
    def dimensionality(self) -> dict:
        """
        Get dimensionality information (Pint format).

        Returns the dimensionality dictionary, e.g., {'[length]': 1, '[time]': -1} for velocity.
        This allows re-dimensionalization of dimensionless quantities by preserving their
        original dimensional nature.

        Returns
        -------
        dict
            Pint dimensionality dictionary, empty dict {} if dimensionless

        Examples
        --------
        >>> velocity = uw.quantity(5, "cm/year")
        >>> velocity.dimensionality
        {'[length]': 1, '[time]': -1}

        >>> nondim_velocity = uw.quantity(1.5, "dimensionless", dimensionality={'[length]': 1, '[time]': -1})
        >>> nondim_velocity.dimensionality
        {'[length]': 1, '[time]': -1}
        """
        return self._dimensionality

    @classmethod
    def _from_pint(cls, pint_qty, model_registry=None):
        """Create UWQuantity from Pint quantity."""
        value = pint_qty.magnitude

        # Try to extract model registry from pint quantity if it contains _constants
        if model_registry is None and hasattr(pint_qty, 'units') and hasattr(pint_qty.units, '_REGISTRY'):
            # Check if this looks like a model registry (has _constants)
            registry = pint_qty.units._REGISTRY
            if hasattr(registry, '_definitions') and any(name.startswith('_') for name in registry._definitions):
                model_registry = registry

        units_str = str(pint_qty.units)

        # Check if units contain model constants (names starting with _)
        has_model_constants = any(part.strip().startswith('_') for part in units_str.replace('*', ' ').replace('/', ' ').replace('**', ' ').split())

        if model_registry is not None and has_model_constants:
            # This contains model constants - treat as custom units with model registry
            return cls(value, _custom_units=units_str, _model_registry=model_registry)
        elif has_model_constants:
            # Has model constants but no registry - store as custom units (fallback)
            return cls(value, _custom_units=units_str)
        else:
            # Regular Pint units - for offset units, create a simple value-only UWQuantity
            # This handles Celsius, Fahrenheit, etc. that can't be multiplied by their units
            try:
                # Try normal construction first
                return cls(value, units=units_str)
            except:
                # For offset units, create without Pint reconstruction
                instance = cls.__new__(cls)
                instance._value = value
                instance.units = units_str
                instance._has_pint_qty = True
                instance._pint_qty = pint_qty
                return instance

    def copy(self, other: 'UWQuantity') -> None:
        """
        Copy the symbolic value from another UWQuantity.

        This transfers the symbolic representation (in model base units)
        from the source quantity to this quantity, preserving the
        target object identity while updating its internal value.

        Parameters
        ----------
        other : UWQuantity
            Source quantity to copy from
        """
        if not isinstance(other, UWQuantity):
            raise TypeError(f"Can only copy from UWQuantity objects, got {type(other)}")

        # Copy the symbolic value (which should be in model base units if scaling is active)
        self._sym = other._sym

    def to(self, target_units: str) -> 'UWQuantity':
        """
        Convert to different units, returning a new UWQuantity.

        Parameters
        ----------
        target_units : str
            Target units specification

        Returns
        -------
        UWQuantity
            New quantity with converted value and target units

        Raises
        ------
        ValueError
            If this quantity has no units
        pint.DimensionalityError
            If units are incompatible
        """
        if not self.has_units:
            raise ValueError("Cannot convert quantity without units - create with units parameter")

        # NEW: Try Pint-native conversion first for model units
        if hasattr(self, '_has_pint_qty') and self._has_pint_qty:
            try:
                # Use the model registry if available, otherwise standard Pint
                if hasattr(self, '_model_registry') and self._model_registry:
                    target_qty = self._model_registry.parse_expression(target_units)
                    converted_pint = self._pint_qty.to(target_qty)
                else:
                    import pint
                    ureg = pint.UnitRegistry()
                    target_qty = ureg.parse_expression(target_units)
                    converted_pint = self._pint_qty.to(target_qty)

                # Try _from_pint, but handle offset unit issues
                try:
                    return UWQuantity._from_pint(converted_pint)
                except Exception as inner_e:
                    # Handle offset units and setter issues
                    from pint.errors import OffsetUnitCalculusError

                    # For any error in _from_pint, return a simple value-only result
                    # This handles both offset units and property setter issues
                    return type('ConvertedQuantity', (), {
                        'value': converted_pint.magnitude,
                        'units': str(converted_pint.units),
                        '__str__': lambda self: f"{self.value} {self.units}",
                        '__repr__': lambda self: self.__str__(),
                        '__float__': lambda self: float(self.value)
                    })()

            except Exception as e:
                # Check for invalid unit names first
                if "is not defined" in str(e) or "Unknown unit" in str(e):
                    raise ValueError(
                        f"Invalid unit name '{target_units}'. "
                        f"For temperature conversions, use 'degC' (not 'Celsius') or 'degF'. "
                        f"Original error: {str(e)}"
                    ) from e
                # Fall back to legacy approach if Pint conversion fails
                elif "Cannot convert from" in str(e) or "_constants" in str(e):
                    raise ValueError(
                        f"Cannot convert model units '{self.units}' to '{target_units}'. "
                        f"Model units use special scaling constants that are not convertible. "
                        f"Original error: {str(e)}"
                    ) from e

        # Legacy approach for non-Pint quantities
        if not hasattr(self, '_units_backend'):
            raise AttributeError(
                f"UWQuantity missing _units_backend - cannot convert units. "
                f"For model quantities, ensure you're using valid Pint unit names. "
                f"For temperature: use 'degC' or 'degF', not 'Celsius' or 'Fahrenheit'."
            )

        current_qty = self._units_backend.create_quantity(self.value, self.units)

        try:
            converted_qty = self._units_backend.convert_units(current_qty, target_units)
        except Exception as e:
            # Enhance error message for common cases
            raise type(e)(
                f"Cannot convert {self.units} to {target_units}. "
                f"Units have incompatible dimensionalities. "
                f"Original error: {str(e)}"
            ) from e

        # Extract magnitude and create new quantity
        magnitude = self._units_backend.get_magnitude(converted_qty)
        return UWQuantity(magnitude, target_units)

    def to_units(self, target_units: str) -> 'UWQuantity':
        """
        Convert to different units using Pint-native approach.

        This method overrides UnitAwareMixin.to_units() to provide proper
        Pint-native unit conversion for UWQuantity objects.

        Parameters
        ----------
        target_units : str
            Target units specification

        Returns
        -------
        UWQuantity
            New quantity with converted value and target units
        """
        return self.to(target_units)

    def to_compact(self) -> 'UWQuantity':
        """
        Convert to compact representation with best automatic units.

        This uses Pint's to_compact() method to automatically select the most
        readable unit representation. For example, 1e-9 GPa becomes 1.0 Pa,
        and 1000000 mm becomes 1.0 km.

        Returns
        -------
        UWQuantity
            New quantity with automatically selected compact units

        Raises
        ------
        ValueError
            If this quantity has no units or doesn't have Pint quantity

        Examples
        --------
        >>> import underworld3 as uw
        >>> awkward = uw.quantity(1e-9, "GPa")
        >>> nice = awkward.to_compact()
        >>> print(nice)  # 1.0 pascal

        >>> distance = uw.quantity(1000000, "mm")
        >>> compact = distance.to_compact()
        >>> print(compact)  # 1.0 kilometer
        """
        if not self.has_units:
            raise ValueError("Cannot compact quantity without units")

        if not (hasattr(self, '_has_pint_qty') and self._has_pint_qty):
            raise ValueError("to_compact() requires Pint quantity (not available for model units)")

        try:
            # Use Pint's compact functionality
            compact_pint = self._pint_qty.to_compact()
            return UWQuantity._from_pint(compact_pint)
        except AttributeError:
            # Pint version doesn't have to_compact() (requires Pint >= 0.17)
            raise AttributeError(
                "to_compact() requires Pint >= 0.17. "
                "Upgrade with: pip install --upgrade pint"
            )

    def to_nice_units(self) -> 'UWQuantity':
        """
        Convert to 'nice' representation using automatic compact units.

        Alias for to_compact() - finds the most readable unit representation.

        Returns
        -------
        UWQuantity
            New quantity with nice, readable units

        Examples
        --------
        >>> import underworld3 as uw
        >>> pressure = uw.quantity(0.001, "GPa")
        >>> nice = pressure.to_nice_units()
        >>> print(nice)  # 1.0 megapascal
        """
        return self.to_compact()

    def _to_model_units_(self, model) -> 'UWQuantity':
        """
        Hidden method for model.to_model_units() protocol.

        This follows the _sympify_() pattern - called by model.to_model_units()
        when available, allowing UWQuantity objects to handle their own conversion.

        Parameters
        ----------
        model : Model
            The model instance requesting conversion

        Returns
        -------
        UWQuantity or None
            Converted quantity, or None if cannot convert
        """
        if not self.has_units:
            return None  # Dimensionless - let model handle gracefully

        # For UWQuantity objects, delegate to the model's general conversion method
        # This ensures consistent behavior and avoids code duplication
        return None  # Let model handle the conversion with its scaling system

    # Arithmetic operations that preserve units
    def __add__(self, other: Union['UWQuantity', float, int]) -> 'UWQuantity':
        """Addition with automatic unit handling."""
        if isinstance(other, UWQuantity):
            if self.has_units and other.has_units:
                # NEW: Try Pint-native arithmetic first
                if (hasattr(self, '_has_pint_qty') and self._has_pint_qty and
                    hasattr(other, '_has_pint_qty') and other._has_pint_qty):
                    try:
                        result_pint = self._pint_qty + other._pint_qty
                        return self._from_pint(result_pint,
                                             getattr(self, '_model_registry', None))
                    except Exception:
                        # Fall back to value arithmetic if Pint fails
                        pass

                # Legacy approach with unit conversion
                try:
                    other_converted = other.to(self.units)
                    result_value = self.value + other_converted.value
                    return UWQuantity(result_value, self.units)
                except (AttributeError, ValueError):
                    # If conversion fails, check if units are the same and do direct addition
                    if self.units == other.units:
                        result_value = self.value + other.value
                        return UWQuantity(result_value, self.units)
                    else:
                        raise
            elif self.has_units:
                # Self has units, other doesn't - assume other matches self's units
                result_value = self.value + other.value
                return UWQuantity(result_value, self.units)
            elif other.has_units:
                # Other has units, self doesn't - assume self matches other's units
                result_value = self.value + other.value
                return UWQuantity(result_value, other.units)
            else:
                # Neither has units
                return UWQuantity(self.value + other.value)
        else:
            # Scalar addition - preserve units
            return UWQuantity(self.value + other, self.units if self.has_units else None)

    def __radd__(self, other: Union[float, int]) -> 'UWQuantity':
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Union['UWQuantity', float, int]) -> 'UWQuantity':
        """Subtraction with automatic unit handling."""
        if isinstance(other, UWQuantity):
            if self.has_units and other.has_units:
                # NEW: Try Pint-native arithmetic first
                if (hasattr(self, '_has_pint_qty') and self._has_pint_qty and
                    hasattr(other, '_has_pint_qty') and other._has_pint_qty):
                    try:
                        result_pint = self._pint_qty - other._pint_qty
                        return self._from_pint(result_pint,
                                             getattr(self, '_model_registry', None))
                    except Exception:
                        # Fall back to value arithmetic if Pint fails
                        pass

                # Legacy approach with unit conversion
                try:
                    other_converted = other.to(self.units)
                    result_value = self.value - other_converted.value
                    return UWQuantity(result_value, self.units)
                except (AttributeError, ValueError):
                    # Handle cases where .to() fails (missing _units_backend)
                    # For model units, assume same units if unit strings match
                    if str(self.units) == str(other.units):
                        result_value = self.value - other.value
                        return UWQuantity(result_value, self.units)
                    else:
                        raise ValueError(f"Cannot subtract {other.units} from {self.units} - units incompatible")
            elif self.has_units:
                result_value = self.value - other.value
                return UWQuantity(result_value, self.units)
            elif other.has_units:
                result_value = self.value - other.value
                return UWQuantity(result_value, other.units)
            else:
                return UWQuantity(self.value - other.value)
        else:
            return UWQuantity(self.value - other, self.units if self.has_units else None)

    def __rsub__(self, other: Union[float, int]) -> 'UWQuantity':
        """Right subtraction."""
        return UWQuantity(other - self.value, self.units if self.has_units else None)

    def __mul__(self, other: Union['UWQuantity', float, int]) -> 'UWQuantity':
        """Multiplication with unit combination - NEW: Pint-native approach."""
        if isinstance(other, UWQuantity):
            # NEW: Try Pint-native arithmetic first
            self_has_pint = hasattr(self, '_has_pint_qty') and self._has_pint_qty
            other_has_pint = hasattr(other, '_has_pint_qty') and other._has_pint_qty

            if self_has_pint and other_has_pint:
                # Both have Pint quantities - use native Pint arithmetic
                result_pint = self._pint_qty * other._pint_qty
                # Pass model registry if available
                model_registry = getattr(self, '_model_registry', None) or getattr(other, '_model_registry', None)
                return self._from_pint(result_pint, model_registry)
            elif self_has_pint:
                # Self has Pint, other doesn't - multiply by scalar
                result_pint = self._pint_qty * other.value
                model_registry = getattr(self, '_model_registry', None)
                return self._from_pint(result_pint, model_registry)
            elif other_has_pint:
                # Other has Pint, self doesn't - multiply by scalar
                result_pint = self.value * other._pint_qty
                model_registry = getattr(other, '_model_registry', None)
                return self._from_pint(result_pint, model_registry)

            # FALLBACK: Old approach for non-Pint quantities
            if self.has_units and other.has_units:
                # Check if either operand has custom model units
                self_has_custom = hasattr(self, '_has_custom_units') and self._has_custom_units
                other_has_custom = hasattr(other, '_has_custom_units') and other._has_custom_units

                if self_has_custom or other_has_custom:
                    # Model units fallback: just multiply values and return dimensionless
                    # This allows model unit arithmetic to work without complex unit tracking
                    result_value = self.value * other.value
                    return UWQuantity(result_value, units=None)
                elif hasattr(self, '_units_backend') and self._units_backend is not None:
                    # Let the units backend handle unit multiplication for regular Pint units
                    self_qty = self._units_backend.create_quantity(self.value, self.units)
                    other_qty = other._units_backend.create_quantity(other.value, other.units)
                    result_qty = self_qty * other_qty

                    result_magnitude = self._units_backend.get_magnitude(result_qty)
                    result_units = str(self._units_backend.get_units(result_qty))
                    return UWQuantity(result_magnitude, result_units)
                else:
                    # No units backend available, use fallback
                    result_value = self.value * other.value
                    return UWQuantity(result_value, units=None)
            elif self.has_units:
                # Check if self has custom model units
                if hasattr(self, '_has_custom_units') and self._has_custom_units:
                    # Don't try to preserve custom model units, return dimensionless
                    return UWQuantity(self.value * other.value, units=None)
                else:
                    return UWQuantity(self.value * other.value, self.units)
            elif other.has_units:
                # Check if other has custom model units
                if hasattr(other, '_has_custom_units') and other._has_custom_units:
                    # Don't try to preserve custom model units, return dimensionless
                    return UWQuantity(self.value * other.value, units=None)
                else:
                    return UWQuantity(self.value * other.value, other.units)
            else:
                return UWQuantity(self.value * other.value)
        else:
            # Scalar multiplication
            if hasattr(self, '_has_pint_qty') and self._has_pint_qty:
                # Use Pint arithmetic for scalar multiplication
                result_pint = self._pint_qty * other
                model_registry = getattr(self, '_model_registry', None)
                return self._from_pint(result_pint, model_registry)
            else:
                # Fallback for non-Pint quantities
                return UWQuantity(self.value * other, self.units if self.has_units else None)

    def _sympify_(self):
        """
        Return SymPy representation for symbolic mathematics.

        This enables UWQuantity objects to work in mathematical operations
        with symbolic variables (mesh variables, expressions, etc.).
        Returns the scalar value for symbolic computation.
        """
        import sympy
        # Handle cases where value is already a SymPy expression
        if hasattr(self.value, '_sympify_') or isinstance(self.value, (sympy.Basic,)):
            return self.value
        # For scalar numeric values, convert to SymPy Float
        try:
            return sympy.Float(self.value)
        except (TypeError, ValueError):
            # If conversion fails, return value as-is
            return self.value

    def diff(self, *args, **kwargs):
        """
        Derivative of a UWQuantity (constant) is always zero.

        This enables UWQuantity objects to work in SymPy derivative operations
        that the solver performs when setting up equations.

        Returns
        -------
        int
            Always returns 0 since UWQuantity represents a constant
        """
        return 0

    def __float__(self):
        """Convert to float for SymPy compatibility."""
        return float(self.value)

    def is_number(self):
        """UWQuantity represents a number/constant."""
        return True

    def atoms(self, *types):
        """
        Return atomic parts of UWQuantity for SymPy compatibility.

        UWQuantity represents a constant, so it returns itself if it matches
        the requested types, otherwise delegates to the SymPy representation.
        """
        import sympy
        if not types:
            # Default to all atomic types
            types = (sympy.Atom,)

        # Get SymPy representation
        sympy_repr = self._sympify_()

        # If the SymPy representation has its own atoms method, use it
        if hasattr(sympy_repr, 'atoms'):
            return sympy_repr.atoms(*types)

        # Otherwise, check if this UWQuantity matches the requested types
        if any(isinstance(sympy_repr, t) for t in types):
            return {sympy_repr}
        return set()

    def __format__(self, format_spec: str) -> str:
        """
        Format the UWQuantity using the format specification.

        This applies the format specification to the numerical value.
        For model units with no format spec, includes human-readable interpretation.

        Examples
        --------
        >>> qty = UWQuantity(3.14159, "m")
        >>> f"{qty:.2f}"  # "3.14 m" (with units)
        >>> f"{qty:e}"    # "3.141590e+00 m"

        >>> qty_model = model.to_model_units(uw.quantity(5, "cm/year"))
        >>> f"{qty_model}"  # "3.16e9 (≈ 5.000 cm/year)"
        """
        try:
            # Format the numerical value
            if format_spec:
                formatted_value = format(self.value, format_spec)
            else:
                formatted_value = str(self.value)

            # Add units or interpretation
            if self.has_units:
                # Check if model units can be interpreted
                interpretation = self._interpret_model_units()
                if interpretation:
                    # Model units: show interpretation
                    return f"{formatted_value} ({interpretation})"
                else:
                    # Regular units: show units
                    units_str = self.units
                    if units_str:
                        display_units = self._get_display_units(units_str)
                        return f"{formatted_value} {display_units}"

            return formatted_value

        except (ValueError, TypeError):
            # If formatting fails, fall back to string representation
            return str(self)

    def __rmul__(self, other: Union[float, int]) -> 'UWQuantity':
        """Right multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Union['UWQuantity', float, int]) -> 'UWQuantity':
        """Division with unit combination."""
        if isinstance(other, UWQuantity):
            if self.has_units and other.has_units:
                # Check if either operand has custom model units
                self_has_custom = hasattr(self, '_has_custom_units') and self._has_custom_units
                other_has_custom = hasattr(other, '_has_custom_units') and other._has_custom_units

                if self_has_custom or other_has_custom:
                    # Model units fallback: just divide values and return dimensionless
                    result_value = self.value / other.value
                    return UWQuantity(result_value, units=None)
                elif hasattr(self, '_units_backend') and self._units_backend is not None:
                    # Let the units backend handle unit division for regular Pint units
                    self_qty = self._units_backend.create_quantity(self.value, self.units)
                    other_qty = other._units_backend.create_quantity(other.value, other.units)
                    result_qty = self_qty / other_qty

                    result_magnitude = self._units_backend.get_magnitude(result_qty)
                    result_units = str(self._units_backend.get_units(result_qty))
                    return UWQuantity(result_magnitude, result_units)
                else:
                    # No units backend available, use fallback
                    result_value = self.value / other.value
                    return UWQuantity(result_value, units=None)
            elif self.has_units:
                # Check if self has custom model units
                if hasattr(self, '_has_custom_units') and self._has_custom_units:
                    # Don't try to preserve custom model units, return dimensionless
                    return UWQuantity(self.value / other.value, units=None)
                else:
                    return UWQuantity(self.value / other.value, self.units)
            elif other.has_units:
                # Check if other has custom model units
                if hasattr(other, '_has_custom_units') and other._has_custom_units:
                    # Don't try to create inverse of custom model units, return dimensionless
                    return UWQuantity(self.value / other.value, units=None)
                else:
                    # Result has inverse units for regular units
                    inv_units = f"1/({other.units})"
                    return UWQuantity(self.value / other.value, inv_units)
            else:
                return UWQuantity(self.value / other.value)
        else:
            # Scalar division - check for custom model units
            if self.has_units and hasattr(self, '_has_custom_units') and self._has_custom_units:
                # Don't try to preserve custom model units, return dimensionless
                return UWQuantity(self.value / other, units=None)
            else:
                # Scalar division preserves units for regular units
                return UWQuantity(self.value / other, self.units if self.has_units else None)

    def __rtruediv__(self, other: Union[float, int]) -> 'UWQuantity':
        """Right division."""
        if self.has_units:
            # Check if self has custom model units
            if hasattr(self, '_has_custom_units') and self._has_custom_units:
                # Don't try to create inverse of custom model units, return dimensionless
                return UWQuantity(other / self.value, units=None)
            else:
                # Create inverse units for regular units
                inv_units = f"1/({self.units})"
                return UWQuantity(other / self.value, inv_units)
        else:
            return UWQuantity(other / self.value)

    def __pow__(self, exponent: Union[float, int]) -> 'UWQuantity':
        """Power with unit exponentiation."""
        if self.has_units:
            # Check if self has custom model units
            if hasattr(self, '_has_custom_units') and self._has_custom_units:
                # Don't try to exponentiate custom model units, return dimensionless
                return UWQuantity(self.value ** exponent, units=None)
            elif hasattr(self, '_units_backend') and self._units_backend is not None:
                # Let units backend handle unit exponentiation for regular units
                self_qty = self._units_backend.create_quantity(self.value, self.units)
                result_qty = self_qty ** exponent

                result_magnitude = self._units_backend.get_magnitude(result_qty)
                result_units = str(self._units_backend.get_units(result_qty))
                return UWQuantity(result_magnitude, result_units)
            else:
                # No units backend, preserve units as-is (may not be mathematically correct)
                return UWQuantity(self.value ** exponent, self.units)
        else:
            return UWQuantity(self.value ** exponent)

    def __neg__(self) -> 'UWQuantity':
        """Negation preserves units."""
        return UWQuantity(-self.value, self.units if self.has_units else None)

    def __str__(self) -> str:
        """
        String representation with human-readable units.

        For regular units: "5.0 cm/year"
        For model units: "1.0 (≈ 0.05 cm/year)" - shows interpretation prominently
        """
        if self.has_units:
            units_str = self.units
            if units_str:
                # Check if we have model units that can be interpreted
                interpretation = self._interpret_model_units()
                if interpretation:
                    # For model units, show the human-readable interpretation prominently
                    return f"{self.value} ({interpretation})"
                else:
                    # For regular units, use elegant aliases if available
                    display_units = self._get_display_units(units_str)
                    return f"{self.value} {display_units}"
            else:
                return str(self.value)
        else:
            return str(self.value)

    def _get_display_units(self, units_str: str) -> str:
        """
        Get display-friendly unit string using aliases if available.

        Converts technical constants like '_6p31e41kg' to elegant aliases like 'ℳ'.
        """
        # Check if we have a model instance with alias substitution
        if hasattr(self, '_model_instance') and self._model_instance:
            if hasattr(self._model_instance, '_substitute_display_aliases'):
                return self._model_instance._substitute_display_aliases(units_str)

        # Fallback: Check model registry (older approach)
        if hasattr(self, '_model_registry') and self._model_registry:
            if hasattr(self._model_registry, '_substitute_display_aliases'):
                return self._model_registry._substitute_display_aliases(units_str)

        # Fallback: return original units
        return units_str

    def _interpret_model_units(self) -> str:
        """
        Interpret model units in human-readable form.

        For model units like '_2900000m / _1p83E15s', this:
        1. Combines the Pint constants (e.g., 2.9e6 m / 1.83e15 s = 1.58e-9 m/s)
        2. Converts to user-friendly units (e.g., ≈ 0.05 cm/year)

        Returns
        -------
        str
            Human-readable interpretation like "≈ 0.05 cm/year", or None if not interpretable
        """
        # Only interpret if we have model units
        if not (hasattr(self, '_has_custom_units') and self._has_custom_units):
            return None

        # Need Pint quantity to work with
        if not hasattr(self, '_pint_qty'):
            return None

        try:
            # STEP 1: Convert the actual value (not just 1.0) to base units
            # This combines the numerical value with all the Pint constants
            # e.g., 1.584e-24 * (_2900000m / _1p83E15s) → 2.5e-33 m/s → 5 cm/year

            # Convert to SI base units using the actual quantity value
            base_qty = self._pint_qty.to_base_units()

            # STEP 2: Try to express in user-friendly units
            from ..scaling import units as u

            # Common user-friendly unit combinations
            # Ordered by priority within each category
            friendly_conversions = [
                # Velocity (geological first)
                ('cm/year', u.cm / u.year),
                ('km/Myr', u.km / (1e6 * u.year)),
                ('mm/year', u.mm / u.year),
                ('m/Myr', u.m / (1e6 * u.year)),
                ('m/s', u.m / u.s),
                ('km/s', u.km / u.s),
                # Length
                ('km', u.km),
                ('m', u.m),
                ('cm', u.cm),
                ('mm', u.mm),
                # Time (geological first)
                ('Myr', 1e6 * u.year),
                ('kyr', 1e3 * u.year),
                ('year', u.year),
                ('day', u.day),
                ('hour', u.hour),
                ('s', u.s),
                # Pressure/stress (geological scales first)
                ('GPa', 1e9 * u.Pa),
                ('MPa', 1e6 * u.Pa),
                ('kPa', 1e3 * u.Pa),
                ('Pa', u.Pa),
                # Temperature
                ('K', u.K),
                ('degC', u.degC),
                # Viscosity
                ('Pa*s', u.Pa * u.s),
                # Density
                ('kg/m**3', u.kg / u.m**3),
                ('g/cm**3', u.g / u.cm**3),
            ]

            # Try each friendly unit to find the best representation
            best_conversion = None
            best_magnitude = None
            best_score = float('inf')

            for name, friendly_unit in friendly_conversions:
                try:
                    # Check if dimensionally compatible
                    # If friendly_unit is a Quantity (e.g., 1e6 * year), we need to:
                    # 1. Convert to its base units (e.g., year)
                    # 2. Divide by its magnitude (e.g., 1e6) to get the scaled value
                    if hasattr(friendly_unit, 'magnitude') and hasattr(friendly_unit, 'units'):
                        # It's a Quantity with a scale factor
                        converted = base_qty.to(friendly_unit.units)
                        magnitude = abs(converted.magnitude / friendly_unit.magnitude)
                    else:
                        # It's a pure Unit (no scale factor)
                        converted = base_qty.to(friendly_unit)
                        magnitude = abs(converted.magnitude)

                    # Skip zero or invalid magnitudes
                    if magnitude == 0 or not (0 < magnitude < 1e100):
                        continue

                    import math
                    log_mag = math.log10(magnitude)

                    # SCORING: Prefer magnitudes in range [0.001, 1000]
                    # with sweet spot around 0.01-100
                    if -3 <= log_mag <= 3:
                        # In good range - score by distance from ideal ~1
                        score = abs(log_mag)
                    elif log_mag < -3:
                        # Too small - penalize heavily
                        score = 100 + abs(log_mag + 3)
                    else:
                        # Too large - penalize heavily
                        score = 100 + (log_mag - 3)

                    if score < best_score:
                        best_score = score
                        best_magnitude = magnitude
                        best_conversion = name

                except:
                    # Dimensionally incompatible or conversion error, skip
                    continue

            if best_conversion and best_magnitude is not None:
                # Format the magnitude nicely based on size
                if abs(best_magnitude) >= 1000:
                    mag_str = f"{best_magnitude:.2e}"
                elif abs(best_magnitude) >= 100:
                    mag_str = f"{best_magnitude:.1f}"
                elif abs(best_magnitude) >= 10:
                    mag_str = f"{best_magnitude:.2f}"
                elif abs(best_magnitude) >= 1:
                    mag_str = f"{best_magnitude:.3f}"
                elif abs(best_magnitude) >= 0.01:
                    mag_str = f"{best_magnitude:.4f}"
                else:
                    mag_str = f"{best_magnitude:.3g}"

                return f"≈ {mag_str} {best_conversion}"

            # If no good conversion found, at least show the base units with magnitude
            base_mag = base_qty.magnitude
            base_units = str(base_qty.units)
            if abs(base_mag - 1.0) > 1e-10:  # Not unity
                if abs(base_mag) >= 1000 or abs(base_mag) < 0.001:
                    return f"≈ {base_mag:.2e} {base_units}"
                else:
                    return f"≈ {base_mag:.4g} {base_units}"
            else:
                return f"model units"

        except Exception as e:
            # If interpretation fails, return None silently
            return None

    def __repr__(self) -> str:
        """
        Detailed representation with human-readable interpretation for model units.

        For regular units: UWQuantity(5.0, 'cm/year')
        For model units: UWQuantity(0.9999..., '_2900000m / _1p83E15s')  [≈ 0.05 cm/year]
        """
        if self.has_units:
            units_str = self.units
            if units_str:
                # Check if we have model units and can interpret them
                interpretation = self._interpret_model_units()
                if interpretation:
                    # Show both technical units and human-readable interpretation
                    return f"UWQuantity({self.value}, '{units_str}')  [{interpretation}]"
                else:
                    # Regular units or cannot interpret
                    return f"UWQuantity({self.value}, '{units_str}')"
            else:
                return f"UWQuantity({self.value})"
        else:
            return f"UWQuantity({self.value})"



def quantity(value: Union[float, int, sympy.Basic], units: Optional[Union[str, Any]] = None, _custom_units: Optional[str] = None, _model_registry=None, _model_instance=None, _is_model_units: bool = False) -> UWQuantity:
    """
    Create a lightweight unit-aware quantity.

    Parameters
    ----------
    value : float, int, sympy.Basic
        The numerical or symbolic value
    units : str or Pint Unit, optional
        Units specification (e.g., "Pa*s", "cm/year", "K", uw.units.earth_mass)
    _custom_units : str, optional
        Custom unit name that bypasses Pint validation (for model units)

    Returns
    -------
    UWQuantity
        Unit-aware quantity object

    Examples
    --------
    >>> import underworld3 as uw
    >>> viscosity = uw.quantity(1e21, "Pa*s")
    >>> velocity = uw.quantity(5, "cm/year")
    >>> mass = uw.quantity(1, "earth_mass")  # Planetary units
    >>> mass2 = uw.quantity(1, uw.units.earth_mass)  # Or use Unit object
    >>> time_scale = distance / velocity  # Units calculated automatically
    """
    # Handle Pint Unit objects
    if units is not None and hasattr(units, 'dimensionality'):
        # This is a Pint Unit object - convert to string
        units = str(units)

    qty = UWQuantity(value, units, _custom_units=_custom_units, _model_registry=_model_registry, _model_instance=_model_instance)

    # Set model units flag if specified
    if _is_model_units:
        qty._is_model_units = True

    return qty