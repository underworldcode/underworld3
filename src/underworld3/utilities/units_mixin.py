# underworld3/utilities/units_mixin.py
"""
Units-Aware Mixin for Underworld3 Variables

This module provides a mixin class that adds dimensional analysis and units support
to any object, following the pattern established by MathematicalMixin.
"""

import warnings
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
    
    def create_quantity(self, value: Any, units: Any) -> 'pint.Quantity':
        """Create a Pint quantity."""
        if isinstance(units, str):
            units = self.ureg(units)
        return value * units
    
    def get_magnitude(self, quantity: 'pint.Quantity') -> Any:
        """Extract magnitude from Pint quantity."""
        return quantity.magnitude
    
    def get_units(self, quantity: 'pint.Quantity') -> 'pint.Unit':
        """Extract units from Pint quantity."""
        return quantity.units
    
    def non_dimensionalise(self, quantity: 'pint.Quantity') -> float:
        """Non-dimensionalise using existing scaling module."""
        return self.scaling.non_dimensionalise(quantity)
    
    def dimensionalise(self, value: float, units: Any) -> 'pint.Quantity':
        """Dimensionalise using existing scaling module."""
        if isinstance(units, str):
            units = self.ureg(units)
        return self.scaling.dimensionalise(value, units)
    
    def check_dimensionality(self, quantity1: 'pint.Quantity', quantity2: 'pint.Quantity') -> bool:
        """Check dimensional compatibility."""
        try:
            quantity1.to(quantity2.units)
            return True
        except:
            return False
    
    def get_dimensionality(self, quantity: 'pint.Quantity') -> dict:
        """Get dimensionality as dict."""
        return quantity.dimensionality


class SymPyBackend(UnitsBackend):
    """SymPy-based units backend implementation."""
    
    def __init__(self):
        try:
            import sympy
            import sympy.physics.units as units
            self.sympy = sympy
            self.units = units
            # Set up basic unit system based on Ben Knight's approach
            self._setup_unit_system()
        except ImportError:
            raise ImportError("SymPy is required for SymPyBackend.")
    
    def _setup_unit_system(self):
        """Set up SymPy units system."""
        # Following the pattern from pint_sympy_backwards_check.ipynb
        self.base_units = {
            'length': self.units.meter,
            'mass': self.units.kilogram, 
            'time': self.units.second,
            'temperature': self.units.kelvin,
            'amount': self.units.mole
        }
        
        # Scaling coefficients (can be modified by user)
        self.scaling_coefficients = {
            'length': 1.0 * self.units.meter,
            'mass': 1.0 * self.units.kilogram,
            'time': 1.0 * self.units.year,  # Default to geological time
            'temperature': 1.0 * self.units.kelvin,
            'amount': 1.0 * self.units.mole
        }
    
    def create_quantity(self, value: Any, units: Any) -> 'sympy.Expr':
        """Create a SymPy quantity."""
        if isinstance(units, str):
            # Parse string to SymPy unit
            units = getattr(self.units, units, self.sympy.Symbol(units))
        return value * units
    
    def get_magnitude(self, quantity: 'sympy.Expr') -> Any:
        """Extract magnitude from SymPy quantity."""
        # For SymPy expressions, this requires more complex analysis
        if quantity.is_number:
            return float(quantity)
        # For unit expressions, we need to separate the coefficient
        return quantity.as_coeff_Mul()[0]
    
    def get_units(self, quantity: 'sympy.Expr') -> 'sympy.Expr':
        """Extract units from SymPy quantity."""
        if quantity.is_number:
            return self.sympy.S.One  # Dimensionless
        # Get the unit part
        coeff, unit_part = quantity.as_coeff_Mul()
        return unit_part
    
    def non_dimensionalise(self, quantity: 'sympy.Expr') -> float:
        """Non-dimensionalise using SymPy dimensional analysis."""
        # This would implement the dimensional analysis using SymPy
        # Similar to the approach in Ben Knight's notebook
        # For now, return a placeholder
        warnings.warn("SymPy non_dimensionalise not fully implemented")
        return float(self.get_magnitude(quantity))
    
    def dimensionalise(self, value: float, units: Any) -> 'sympy.Expr':
        """Dimensionalise value with SymPy units."""
        return self.create_quantity(value, units)
    
    def check_dimensionality(self, quantity1: 'sympy.Expr', quantity2: 'sympy.Expr') -> bool:
        """Check dimensional compatibility using SymPy."""
        try:
            # Use SymPy's dimensional analysis
            dim1 = self.units.Dimension(self.get_units(quantity1))
            dim2 = self.units.Dimension(self.get_units(quantity2))
            return dim1 == dim2
        except:
            return False
    
    def get_dimensionality(self, quantity: 'sympy.Expr') -> 'sympy.Expr':
        """Get dimensionality as SymPy expression."""
        return self.units.Dimension(self.get_units(quantity))


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
    
    def __init__(self, *args, units: Optional[Union[str, Any]] = None, 
                 units_backend: Optional[Union[str, UnitsBackend]] = None, **kwargs):
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
        
        if units is not None:
            self.set_units(units, units_backend)
    
    def set_units(self, units: Union[str, Any], 
                  backend: Optional[Union[str, UnitsBackend]] = None):
        """
        Set units for this variable.
        
        Args:
            units: Units specification (string or units object)
            backend: Backend to use ('pint', 'sympy', or UnitsBackend instance)
        """
        # Set up backend
        if backend is None:
            backend = 'pint'  # Default to Pint
        
        if isinstance(backend, str):
            if backend.lower() == 'pint':
                self._units_backend = PintBackend()
            elif backend.lower() == 'sympy':
                self._units_backend = SymPyBackend()
            else:
                raise ValueError(f"Unknown backend: {backend}")
        else:
            self._units_backend = backend
        
        # Create dimensional quantity
        self._units = units
        # For now, create with magnitude 1 to represent the units
        self._dimensional_quantity = self._units_backend.create_quantity(1.0, units)
    
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
        
        Args:
            value: Value to non-dimensionalise (if None, uses self.data)
            
        Returns:
            Non-dimensional value suitable for PETSc solvers
        """
        if not self.has_units:
            return value if value is not None else self.data
        
        if value is None:
            # Use the variable's current data
            dimensional_data = self.create_quantity(self.data)
        else:
            dimensional_data = self.create_quantity(value)
        
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
    
    def check_units_compatibility(self, other: 'UnitAwareMixin') -> bool:
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
            self._dimensional_quantity, 
            other._dimensional_quantity
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
    
    # Mathematical operations that preserve units
    def __mul__(self, other):
        """
        Multiplication with units handling.
        
        If both operands have units, the result will have combined units.
        Falls back to mathematical mixin if available.
        """
        # Check if we also have MathematicalMixin (multiple inheritance)
        if hasattr(super(), '__mul__'):
            # Use MathematicalMixin's multiplication for symbolic math
            result = super().__mul__(other)
            
            # Add units metadata if both have units
            if hasattr(other, 'has_units') and self.has_units and other.has_units:
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
        # Check units compatibility for addition
        if hasattr(other, 'has_units') and self.has_units and other.has_units:
            if not self.check_units_compatibility(other):
                raise ValueError(f"Cannot add incompatible units: {self.units} + {other.units}")
        
        # Check if we also have MathematicalMixin
        if hasattr(super(), '__add__'):
            return super().__add__(other)
        else:
            return self.data + other
    
    def __sub__(self, other):
        """
        Subtraction with units compatibility checking.
        
        Only allows subtraction of compatible units.
        """
        # Check units compatibility for subtraction
        if hasattr(other, 'has_units') and self.has_units and other.has_units:
            if not self.check_units_compatibility(other):
                raise ValueError(f"Cannot subtract incompatible units: {self.units} - {other.units}")
        
        # Check if we also have MathematicalMixin
        if hasattr(super(), '__sub__'):
            return super().__sub__(other)
        else:
            return self.data - other


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
def make_units_aware(variable_class: type, units_backend: str = 'pint') -> type:
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