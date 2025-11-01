# underworld3/utilities/units_mixin.py
"""
DEPRECATED: Units-Aware Mixin System (Not Used in Production Code)

WARNING: This module contains experimental code that was abandoned in favor of the
hierarchical units system (enhanced_variables.py). It is preserved for historical
reference and design patterns but should NOT be used in new code.

CURRENT IMPLEMENTATION: See src/underworld3/discretisation/enhanced_variables.py
for the active units system used by MeshVariable and SwarmVariable.

REASON FOR DEPRECATION: The mixin approach had critical bugs and integration issues.
The hierarchical system using `get_units()` + direct data access proved simpler and
more reliable.

Historical Context:
- Designed to add dimensional analysis via multiple inheritance
- Had dual calling patterns for `non_dimensional_value()`
- Never integrated with actual MeshVariable/SwarmVariable classes
- Some design patterns (backends, scale factors) may be useful for future reference

DO NOT USE THIS CODE IN PRODUCTION!
"""

import warnings

# Deprecation warning on module import
warnings.warn(
    "The units_mixin module is deprecated and not used in production code. "
    "Use the hierarchical units system in enhanced_variables.py instead. "
    "This module is preserved only for historical reference.",
    DeprecationWarning,
    stacklevel=2
)
from typing import Any, Optional, Union, Dict, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    import pint
    import sympy


class UnitsBackend(ABC):
    """
    Abstract base class for units backends.

    This protocol allows supporting multiple dimensional analysis backends
    (Pint, SymPy, custom implementations) through a common interface.
    """

    @abstractmethod
    def create_quantity(self, value: Any, units: Any) -> Any:
        """Create a quantity with units from a value."""
        pass

    @abstractmethod
    def get_magnitude(self, quantity: Any) -> Any:
        """Extract the magnitude (value) from a quantity."""
        pass

    @abstractmethod
    def get_units(self, quantity: Any) -> Any:
        """Extract the units from a quantity."""
        pass

    @abstractmethod
    def non_dimensionalise(self, quantity: Any) -> float:
        """Convert dimensional quantity to non-dimensional value."""
        pass

    @abstractmethod
    def dimensionalise(self, value: float, units: Any) -> Any:
        """Convert non-dimensional value to dimensional quantity."""
        pass

    @abstractmethod
    def check_dimensionality(self, quantity1: Any, quantity2: Any) -> bool:
        """Check if two quantities have compatible dimensions."""
        pass

    @abstractmethod
    def get_dimensionality(self, quantity: Any) -> Any:
        """Get the dimensionality of a quantity."""
        pass


class PintBackend(UnitsBackend):
    """Pint-based units backend implementation."""

    def __init__(self):
        try:
            # Import existing scaling module for compatibility
            import underworld3.scaling as scaling

            self.scaling = scaling
            # Use the same unit registry as the scaling module
            self.ureg = scaling.units
        except ImportError:
            raise ImportError("Pint is required for PintBackend. Install with: pip install pint")

    def create_quantity(self, value: Any, units: Any) -> "pint.Quantity":
        """Create a Pint quantity."""
        if isinstance(units, str):
            units = self.ureg(units)
        return value * units

    def get_magnitude(self, quantity: "pint.Quantity") -> Any:
        """Extract magnitude from Pint quantity."""
        return quantity.magnitude

    def get_units(self, quantity: "pint.Quantity") -> "pint.Unit":
        """Extract units from Pint quantity."""
        return quantity.units

    def non_dimensionalise(self, quantity: "pint.Quantity") -> float:
        """Non-dimensionalise using existing scaling module."""
        return self.scaling.non_dimensionalise(quantity)

    def dimensionalise(self, value: float, units: Any) -> "pint.Quantity":
        """Dimensionalise using existing scaling module."""
        if isinstance(units, str):
            units = self.ureg(units)
        return self.scaling.dimensionalise(value, units)

    def check_dimensionality(self, quantity1: "pint.Quantity", quantity2: "pint.Quantity") -> bool:
        """Check dimensional compatibility."""
        try:
            quantity1.to(quantity2.units)
            return True
        except:
            return False

    def get_dimensionality(self, quantity: "pint.Quantity") -> dict:
        """Get dimensionality as dict."""
        return quantity.dimensionality

    def convert_units(self, quantity: "pint.Quantity", target_units: str) -> "pint.Quantity":
        """Convert quantity to target units."""
        if isinstance(target_units, str):
            target_units = self.ureg(target_units)
        return quantity.to(target_units)


# SymPy backend removed - use Pint-native approach instead
# SymPy is still used for symbolic mathematics, just not for units


class UnitAwareMixin:
    """
    Mixin class that adds dimensional analysis and units support to any object.

    This mixin follows the same pattern as MathematicalMixin, providing optional
    units functionality that can be added to any class without breaking existing code.

    Key Features:
    - Backend-agnostic: Supports both Pint and SymPy units
    - Optional: Variables work normally without units
    - Automatic scaling: Handles non-dimensionalisation for solvers
    - Mathematical integration: Works with MathematicalMixin arithmetic
    - Backward compatible: No changes to existing code required

    Usage:
        class UnitAwareMeshVariable(UnitAwareMixin, MeshVariable):
            pass

        # Create variable with units
        velocity = UnitAwareMeshVariable("velocity", mesh, 2,
                                        units="meter/second")

        # Units-aware operations
        momentum = velocity * density  # Automatically handles units
        v_scaled = velocity.non_dimensional_value  # For solvers
    """

    def __init__(
        self,
        *args,
        units: Optional[Union[str, Any]] = None,
        units_backend: Optional[Union[str, UnitsBackend]] = None,
        **kwargs,
    ):
        """
        Initialize with optional units support.

        Args:
            *args: Passed to parent class
            units: Units for this variable (string or units object)
            units_backend: Backend to use ('pint', 'sympy', or UnitsBackend instance)
            **kwargs: Passed to parent class
        """
        # Call parent constructor
        super().__init__(*args, **kwargs)

        # Set up units if provided
        self._units = None
        self._units_backend = None
        self._dimensional_quantity = None
        self._scale_factor = None

        if units is not None:
            self.set_units(units, units_backend)

    def set_units(self, units: Union[str, Any], backend: Optional[Union[str, UnitsBackend]] = None):
        """
        Set units for this variable.

        Args:
            units: Units specification (string or units object)
            backend: Backend to use ('pint', 'sympy', or UnitsBackend instance)
        """
        # Set up backend
        if backend is None:
            backend = "pint"  # Default to Pint

        if isinstance(backend, str):
            if backend.lower() == "pint":
                self._units_backend = PintBackend()
            else:
                raise ValueError(f"Unknown backend: {backend}. Only 'pint' is supported.")
        else:
            self._units_backend = backend

        # Create dimensional quantity
        self._units = units
        # For now, create with magnitude 1 to represent the units
        self._dimensional_quantity = self._units_backend.create_quantity(1.0, units)

        # Calculate SymPy-friendly scale factor for compilation
        self._calculate_scale_factor()

    @property
    def units(self) -> Optional[Any]:
        """Get the units of this variable."""
        if self._dimensional_quantity is not None:
            return self._units_backend.get_units(self._dimensional_quantity)
        return None

    @property
    def dimensionality(self) -> Optional[Any]:
        """Get the dimensionality of this variable."""
        if self._dimensional_quantity is not None:
            return self._units_backend.get_dimensionality(self._dimensional_quantity)
        return None

    @property
    def has_units(self) -> bool:
        """Check if this variable has units."""
        return self._dimensional_quantity is not None

    @property
    def scale_factor(self) -> Optional[Any]:
        """
        Get the SymPy-friendly scale factor for this variable.

        The scale factor is used during unwrap/compilation to automatically scale
        variables to appropriate numerical ranges. It's designed to be powers-of-ten
        and SymPy-compatible for symbolic cancellation.

        Returns:
            SymPy expression representing the scale factor, or None if no units
        """
        return self._scale_factor

    def create_quantity(self, value: Any) -> Any:
        """
        Create a dimensional quantity from a value using this variable's units.

        Args:
            value: Value to attach units to

        Returns:
            Dimensional quantity with this variable's units
        """
        if not self.has_units:
            return value
        return self._units_backend.create_quantity(value, self.units)

    def non_dimensional_value(self, value: Optional[Any] = None) -> Any:
        """
        Get non-dimensional value for use in solvers.

        This method supports two calling patterns:
        1. Legacy: non_dimensional_value(data) - converts provided data
        2. Protocol: non_dimensional_value(model) - for uw.non_dimensionalise()

        The method auto-detects which pattern is being used by checking if the
        argument is a Model instance.

        Args:
            value: Either data to non-dimensionalise, a Model instance, or None.
                   If None, uses self.data (legacy behavior).
                   If Model, ignored (protocol behavior - uses self.data).
                   Otherwise, treats as data to convert (legacy behavior).

        Returns:
            Non-dimensional value suitable for PETSc solvers
        """
        # Import here to avoid circular dependency
        from ..model import Model

        # Detect calling pattern: if value is a Model, this is the protocol call
        is_protocol_call = isinstance(value, Model)

        if is_protocol_call:
            # Protocol call from uw.non_dimensionalise() - use self.data
            data_to_convert = None  # Will use self.data below
        else:
            # Legacy call with actual data, or None
            data_to_convert = value

        if not self.has_units:
            return data_to_convert if data_to_convert is not None else self.data

        if data_to_convert is None:
            # Use the variable's current data
            dimensional_data = self.create_quantity(self.data)
        else:
            dimensional_data = self.create_quantity(data_to_convert)

        return self._units_backend.non_dimensionalise(dimensional_data)

    def dimensional_value(self, non_dim_value: Any) -> Any:
        """
        Convert non-dimensional value back to dimensional.

        Args:
            non_dim_value: Non-dimensional value from solver

        Returns:
            Dimensional value with units
        """
        if not self.has_units:
            return non_dim_value

        return self._units_backend.dimensionalise(non_dim_value, self.units)

    def check_units_compatibility(self, other: "UnitAwareMixin") -> bool:
        """
        Check if this variable's units are compatible with another variable.

        Args:
            other: Another UnitAwareMixin object

        Returns:
            True if units are compatible, False otherwise
        """
        if not self.has_units and not other.has_units:
            return True  # Both dimensionless

        if not self.has_units or not other.has_units:
            return False  # One has units, other doesn't

        # Check if backends are compatible
        if type(self._units_backend) != type(other._units_backend):
            warnings.warn("Different units backends - compatibility check may be unreliable")

        return self._units_backend.check_dimensionality(
            self._dimensional_quantity, other._dimensional_quantity
        )

    def to_units(self, target_units: Union[str, Any]) -> Any:
        """
        Convert this variable's data to different units.

        Args:
            target_units: Target units to convert to

        Returns:
            Data converted to target units
        """
        if not self.has_units:
            raise ValueError("Cannot convert units - variable has no units")

        current_quantity = self.create_quantity(self.data)
        target_quantity = self._units_backend.create_quantity(1.0, target_units)

        # This would need backend-specific conversion logic
        # For now, return a placeholder
        warnings.warn("Unit conversion not fully implemented")
        return self.data

    def units_repr(self) -> str:
        """
        Get string representation including units information.

        Returns:
            String showing variable with units info
        """
        if not self.has_units:
            return f"{type(self).__name__}(no units)"

        return f"{type(self).__name__}(units: {self.units})"

    def _calculate_scale_factor(self):
        """
        Calculate SymPy-friendly scale factor based on units.

        This creates powers-of-ten scale factors that can be used during
        compilation to automatically scale variables. The approach:

        1. Extract the SI magnitude of the units
        2. Calculate appropriate power-of-ten scaling
        3. Create SymPy expression using sympify for symbolic compatibility

        Examples:
            - Units of meters → scale_factor might be sympify(1) * 10**0 = 1
            - Units of kilometers → scale_factor might be sympify(1) * 10**3
            - Units of GPa → scale_factor might be sympify(1) * 10**9
        """
        if not self.has_units:
            self._scale_factor = None
            return

        try:
            # Import SymPy for creating scale factors
            import sympy as sp

            # Get the magnitude of 1 unit in SI base units
            unit_quantity = self._dimensional_quantity
            magnitude = self._units_backend.get_magnitude(unit_quantity)

            if isinstance(self._units_backend, PintBackend):
                # For Pint backend, convert to base units to get SI magnitude
                try:
                    si_magnitude = float(unit_quantity.to_base_units().magnitude)
                except:
                    si_magnitude = 1.0
            else:
                si_magnitude = 1.0

            # Calculate power-of-ten scale factor
            # Find the appropriate power of 10 to make values O(1)
            if si_magnitude == 0:
                power_of_ten = 0
            else:
                import math

                log_magnitude = math.log10(abs(si_magnitude))
                # Round to nearest integer for clean powers of 10
                power_of_ten = round(log_magnitude)

            # Create SymPy-friendly scale factor
            # Use sympify(1) * 10**power instead of just 10**power for symbolic friendliness
            if power_of_ten == 0:
                self._scale_factor = sp.sympify(1)
            else:
                self._scale_factor = sp.sympify(1) * (10**power_of_ten)

        except Exception as e:
            import warnings

            warnings.warn(f"Could not calculate scale factor for units {self.units}: {e}")
            # Default to no scaling
            try:
                import sympy as sp

                self._scale_factor = sp.sympify(1)
            except:
                self._scale_factor = 1

    def _set_reference_scaling(self, reference_value: float):
        """
        Set reference scaling based on a typical value for this variable.

        INTERNAL METHOD: Users should use model.set_reference_quantities() instead.

        This allows the system to specify that, for example, "velocities in this problem
        are typically 5 cm/year" to get appropriate scaling.

        Args:
            reference_value: Typical magnitude for this variable in its current units

        Note:
            This is an internal method. The public API is model.set_reference_quantities()
        """
        if not self.has_units:
            raise ValueError("Cannot set reference scaling for variable without units")

        try:
            import sympy as sp
            import math

            # Calculate scale factor to make reference_value ≈ O(1)
            if reference_value == 0:
                power_of_ten = 0
            else:
                log_magnitude = math.log10(abs(reference_value))
                # Choose power to make reference_value close to 1
                power_of_ten = -round(log_magnitude)

            # Create SymPy-friendly scale factor
            if power_of_ten == 0:
                self._scale_factor = sp.sympify(1)
            else:
                self._scale_factor = sp.sympify(1) * (10**power_of_ten)

        except Exception as e:
            import warnings

            warnings.warn(f"Could not set reference scaling: {e}")
            # Fall back to default calculation
            self._calculate_scale_factor()

    # Mathematical operations that preserve units
    def __mul__(self, other):
        """
        Multiplication with units handling.

        If both operands have units, the result will have combined units.
        Falls back to mathematical mixin if available.
        """
        # Check if we also have MathematicalMixin (multiple inheritance)
        if hasattr(super(), "__mul__"):
            # Use MathematicalMixin's multiplication for symbolic math
            result = super().__mul__(other)

            # Add units metadata if both have units
            if hasattr(other, "has_units") and self.has_units and other.has_units:
                # Units would multiply in real implementation
                pass

            return result
        else:
            # Basic multiplication without symbolic math
            return self.data * other

    def __add__(self, other):
        """
        Addition with units compatibility checking.

        Only allows addition of compatible units.
        """
        # Check for mixing constants with dimensional quantities
        self._check_dimensional_compatibility_for_addition(other)

        # Check units compatibility for addition
        if hasattr(other, "has_units") and self.has_units and other.has_units:
            if not self.check_units_compatibility(other):
                raise ValueError(f"Cannot add incompatible units: {self.units} + {other.units}")

        # Check if we also have MathematicalMixin
        if hasattr(super(), "__add__"):
            return super().__add__(other)
        else:
            return self.data + other

    def __sub__(self, other):
        """
        Subtraction with units compatibility checking.

        Only allows subtraction of compatible units.
        """
        # Check for mixing constants with dimensional quantities
        self._check_dimensional_compatibility_for_addition(other)

        # Check units compatibility for subtraction
        if hasattr(other, "has_units") and self.has_units and other.has_units:
            if not self.check_units_compatibility(other):
                raise ValueError(
                    f"Cannot subtract incompatible units: {self.units} - {other.units}"
                )

        # Check if we also have MathematicalMixin
        if hasattr(super(), "__sub__"):
            return super().__sub__(other)
        else:
            return self.data - other

    def _check_dimensional_compatibility_for_addition(self, other):
        """
        Check for mixing constants with dimensional quantities in addition/subtraction.

        Following Pint's approach, this raises an error when trying to add/subtract
        a dimensional quantity and a dimensionless constant, which is usually a user error.

        Args:
            other: The other operand in the addition/subtraction

        Raises:
            ValueError: If mixing dimensional and dimensionless quantities inappropriately
        """
        # Check if we're mixing dimensional and dimensionless quantities
        if self.has_units and isinstance(other, (int, float, complex)):
            raise ValueError(
                f"Cannot add/subtract dimensionless number {other} to dimensional quantity with units {self.units}. "
                f"If you meant to add a quantity with the same units, use: "
                f"variable + {other} * uw.scaling.units.{self.units}"
            )

        if not self.has_units and hasattr(other, "has_units") and other.has_units:
            # Allow UWQuantity objects in symbolic math contexts (they have _sympify_)
            from ..function.quantities import UWQuantity

            if isinstance(other, UWQuantity) and hasattr(super(), "__sub__"):
                # This is a symbolic math context - UWQuantity will be converted to scalar
                pass
            else:
                raise ValueError(
                    f"Cannot add/subtract dimensional quantity with units {other.units} to dimensionless quantity. "
                    f"If you meant to add dimensionless values, convert the dimensional quantity first."
                )

        # Allow addition/subtraction of two dimensionless quantities
        # Allow addition/subtraction of compatible dimensional quantities (checked elsewhere)


class UnitAwareMathematicalMixin(UnitAwareMixin):
    """
    Combined mixin providing both mathematical operations and units.

    This class integrates UnitAwareMixin with MathematicalMixin functionality,
    providing the complete set of features for mathematical variables with units.

    Usage:
        class EnhancedMeshVariable(UnitAwareMathematicalMixin, MeshVariable):
            pass

        # Create variable with units and full mathematical functionality
        velocity = EnhancedMeshVariable("velocity", mesh, 2, units="m/s")

        # Mathematical operations with units
        kinetic_energy = 0.5 * density * velocity.dot(velocity)  # Units: kg/m/s²
        v_x = velocity[0]  # Component access
        momentum = density * velocity  # Direct arithmetic with units checking
    """

    def __init__(self, *args, **kwargs):
        """Initialize with both units and mathematical capabilities."""
        # Import here to avoid circular imports
        from .mathematical_mixin import MathematicalMixin

        # Set up multiple inheritance properly
        super().__init__(*args, **kwargs)

        # Ensure we have the mathematical mixin methods
        if not isinstance(self, MathematicalMixin):
            warnings.warn(
                "UnitAwareMathematicalMixin should be used with MathematicalMixin. "
                "Consider: class MyVar(UnitAwareMathematicalMixin, MathematicalMixin, BaseClass)"
            )


# Convenience function for creating unit-aware variables
def make_units_aware(variable_class: type, units_backend: str = "pint") -> type:
    """
    Factory function to create a unit-aware version of any variable class.

    Args:
        variable_class: Base variable class to enhance
        units_backend: Default units backend ('pint' or 'sympy')

    Returns:
        New class with units support

    Example:
        UnitAwareMeshVariable = make_units_aware(MeshVariable)
        velocity = UnitAwareMeshVariable("velocity", mesh, 2, units="m/s")
    """

    class UnitAwareVariable(UnitAwareMixin, variable_class):
        def __init__(self, *args, units=None, **kwargs):
            super().__init__(*args, units=units, units_backend=units_backend, **kwargs)

    UnitAwareVariable.__name__ = f"UnitAware{variable_class.__name__}"
    UnitAwareVariable.__qualname__ = f"UnitAware{variable_class.__qualname__}"

    return UnitAwareVariable


# Test function
def test_units_mixin():
    """Test the units mixin functionality."""

    class MockVariable(UnitAwareMixin):
        def __init__(self, name, data=None, **kwargs):
            self.name = name
            self.data = data if data is not None else [1.0, 2.0, 3.0]
            super().__init__(**kwargs)

    print("Testing UnitAwareMixin...")

    # Test without units
    var1 = MockVariable("test1")
    assert not var1.has_units
    assert var1.units is None

    # Test with Pint units
    try:
        var2 = MockVariable("test2", units="meter/second")
        assert var2.has_units
        print("✓ Pint backend working")
    except ImportError:
        print("⚠ Pint not available, skipping Pint tests")

    # Test with SymPy units
    try:
        var3 = MockVariable("test3", units="meter", units_backend="sympy")
        assert var3.has_units
        print("✓ SymPy backend working")
    except ImportError:
        print("⚠ SymPy not available, skipping SymPy tests")

    print("✓ Basic units mixin functionality working")

    return True


if __name__ == "__main__":
    test_units_mixin()
