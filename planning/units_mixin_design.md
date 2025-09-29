# Units System: Mixin-Based Architecture Design

## Overview

This document outlines a **mixin-based architecture** for adding dimensional analysis and units to Underworld3, following the successful pattern established by `MathematicalMixin`. This approach provides:

- **Optional functionality**: Units can be added to any object without changing core architecture
- **Clean separation**: Units logic separate from computational logic  
- **Flexible backends**: Support for both Pint and SymPy approaches
- **Backward compatibility**: Non-unit-aware objects continue to work unchanged

## Architecture Design

### **Core Mixin Classes**

```python
# underworld3/utilities/units_mixin.py
from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Protocol
import numpy as np

class UnitsBackend(Protocol):
    """Protocol defining the interface for units backends (Pint or SymPy)"""
    
    def create_quantity(self, value: Any, units: str) -> Any:
        """Create a quantity with units"""
        ...
    
    def get_units(self, quantity: Any) -> Any:
        """Extract units from quantity"""
        ...
        
    def get_magnitude(self, quantity: Any) -> Any:
        """Extract numerical value from quantity"""
        ...
        
    def check_dimensionality(self, quantity1: Any, quantity2: Any) -> bool:
        """Check if two quantities have compatible dimensions"""
        ...
        
    def non_dimensionalise(self, quantity: Any) -> float:
        """Convert to dimensionless value using scaling coefficients"""
        ...
        
    def dimensionalise(self, value: float, target_units: str) -> Any:
        """Convert dimensionless value to quantity with units"""
        ...

class UnitAwareMixin:
    """
    Mixin class that adds dimensional analysis and units support to any object.
    
    Provides automatic unit tracking, dimensional validation, and scaling operations
    while maintaining compatibility with dimensionless computational workflows.
    
    Usage:
        class MyUnitAwareVariable(MeshVariable, UnitAwareMixin):
            pass
    """
    
    def __init__(self, *args, units: Optional[Union[str, Any]] = None, 
                 units_backend: Optional[str] = None, **kwargs):
        # Initialize parent classes
        super().__init__(*args, **kwargs)
        
        # Units configuration
        self._units_backend = self._get_backend(units_backend)
        self._units = self._parse_units(units) if units else None
        self._coordinate_units = None  # Set by mesh if applicable
        
    def _get_backend(self, backend_name: Optional[str]) -> UnitsBackend:
        """Get the specified units backend (Pint or SymPy)"""
        if backend_name is None:
            backend_name = _get_default_backend()
            
        if backend_name.lower() == 'pint':
            return _get_pint_backend()
        elif backend_name.lower() == 'sympy':
            return _get_sympy_backend()
        else:
            raise ValueError(f"Unknown units backend: {backend_name}")
    
    def _parse_units(self, units: Union[str, Any]) -> Any:
        """Parse units string or object into backend-specific format"""
        if isinstance(units, str):
            return self._units_backend.create_quantity(1.0, units).units
        else:
            return units
    
    # === Core Units Properties ===
    
    @property
    def units(self) -> Optional[Any]:
        """Physical units of this object"""
        return self._units
    
    @units.setter  
    def units(self, value: Union[str, Any]):
        """Set units for this object"""
        self._units = self._parse_units(value)
    
    @property
    def has_units(self) -> bool:
        """Check if this object has units defined"""
        return self._units is not None
    
    @property
    def is_dimensionless(self) -> bool:
        """Check if this object is dimensionless"""
        if not self.has_units:
            return True
        return self._units_backend.is_dimensionless(self._units)
    
    # === Unit-Aware Data Access ===
    
    def set_dimensional_values(self, values: Union[np.ndarray, Any]):
        """
        Set values with automatic unit conversion and validation.
        
        Parameters
        ----------
        values : array or quantity
            Values to set. If a quantity with units, automatically validates
            compatibility and converts for computation.
        """
        if not hasattr(self, 'array'):
            raise AttributeError("Object must have 'array' property for unit-aware data access")
            
        if self._units_backend.is_quantity(values):
            # Validate unit compatibility
            if self.has_units:
                if not self._units_backend.check_dimensionality(values, self._units):
                    raise ValueError(
                        f"Cannot assign {self._units_backend.get_units(values)} "
                        f"to object with units {self._units}"
                    )
            
            # Convert to dimensionless for computation
            dimensionless_values = self._units_backend.non_dimensionalise(values)
        else:
            # Assume already dimensionless or compatible
            dimensionless_values = values
            
        # Set computational array
        self.array[...] = dimensionless_values
    
    def get_dimensional_values(self) -> Any:
        """
        Get values with proper units attached.
        
        Returns
        -------
        quantity
            Array values with appropriate units
        """
        if not hasattr(self, 'array'):
            raise AttributeError("Object must have 'array' property for unit-aware data access")
            
        dimensionless_array = self.array[...]
        
        if self.has_units:
            return self._units_backend.dimensionalise(dimensionless_array, self._units)
        else:
            return dimensionless_array
    
    # === Unit-Aware Mathematical Operations ===
    
    def _unit_aware_add(self, other):
        """Addition with unit checking"""
        if isinstance(other, UnitAwareMixin):
            # Both have units - check compatibility
            if self.has_units and other.has_units:
                if not self._units_backend.check_dimensionality(self._units, other._units):
                    raise ValueError(
                        f"Cannot add {self._units} to {other._units} - incompatible dimensions"
                    )
                result_units = self._units  # Left operand units
            elif self.has_units:
                result_units = self._units
            elif other.has_units:
                result_units = other._units
            else:
                result_units = None
                
            # Create result (assumes underlying mathematical operation exists)
            result = super().__add__(other)
            if hasattr(result, '_units'):
                result._units = result_units
            return result
        else:
            # Adding dimensionless value
            if self.has_units and other != 0:
                raise ValueError(
                    f"Cannot add dimensionless value {other} to dimensional object with units {self._units}"
                )
            result = super().__add__(other)
            if hasattr(result, '_units'):
                result._units = self._units
            return result
    
    def _unit_aware_mul(self, other):
        """Multiplication with unit algebra"""
        if isinstance(other, UnitAwareMixin):
            # Unit algebra: combine units
            if self.has_units and other.has_units:
                result_units = self._units_backend.multiply_units(self._units, other._units)
            elif self.has_units:
                result_units = self._units
            elif other.has_units:
                result_units = other._units
            else:
                result_units = None
        else:
            # Multiplying by scalar preserves units
            result_units = self._units
            
        result = super().__mul__(other)
        if hasattr(result, '_units'):
            result._units = result_units
        return result
    
    def _unit_aware_truediv(self, other):
        """Division with unit algebra"""
        if isinstance(other, UnitAwareMixin):
            # Unit algebra: divide units
            if self.has_units and other.has_units:
                result_units = self._units_backend.divide_units(self._units, other._units)
            elif self.has_units:
                result_units = self._units
            elif other.has_units:
                result_units = self._units_backend.invert_units(other._units)
            else:
                result_units = None
        else:
            # Dividing by scalar preserves units
            result_units = self._units
            
        result = super().__truediv__(other)
        if hasattr(result, '_units'):
            result._units = result_units
        return result
    
    def _unit_aware_pow(self, exponent):
        """Power operation with unit exponentiation"""
        if self.has_units:
            result_units = self._units_backend.power_units(self._units, exponent)
        else:
            result_units = None
            
        result = super().__pow__(exponent)
        if hasattr(result, '_units'):
            result._units = result_units
        return result
    
    # === Calculus Operations with Units ===
    
    def _unit_aware_grad(self):
        """Gradient with proper unit handling"""
        if not hasattr(super(), 'grad'):
            raise AttributeError("Object must support grad() for unit-aware gradients")
            
        # Gradient changes units: [field_units] / [coordinate_units]
        if self.has_units:
            coord_units = getattr(self, '_coordinate_units', self._units_backend.meter)
            result_units = self._units_backend.divide_units(self._units, coord_units)
        else:
            result_units = None
            
        result = super().grad()
        if hasattr(result, '_units'):
            result._units = result_units
        return result
    
    def _unit_aware_div(self):
        """Divergence with unit handling"""
        if not hasattr(super(), 'div'):
            raise AttributeError("Object must support div() for unit-aware divergence")
            
        # Divergence: [field_units] / [coordinate_units] for vector fields
        if self.has_units:
            coord_units = getattr(self, '_coordinate_units', self._units_backend.meter)
            result_units = self._units_backend.divide_units(self._units, coord_units)
        else:
            result_units = None
            
        result = super().div()
        if hasattr(result, '_units'):
            result._units = result_units
        return result
    
    # === Boundary Conditions with Unit Validation ===
    
    def set_bc_with_units(self, boundary: str, value: Union[Any, str], **kwargs):
        """Set boundary condition with automatic unit validation"""
        
        if self._units_backend.is_quantity(value):
            # Validate unit compatibility
            if self.has_units:
                if not self._units_backend.check_dimensionality(value, self._units):
                    raise ValueError(
                        f"Boundary condition units {self._units_backend.get_units(value)} "
                        f"don't match variable units {self._units}"
                    )
            
            # Convert to dimensionless for solver
            bc_value = self._units_backend.non_dimensionalise(value)
        else:
            # Assume already appropriate
            bc_value = value
            
        # Set boundary condition using parent method
        if hasattr(super(), 'add_dirichlet_bc'):
            return super().add_dirichlet_bc(bc_value, boundary, **kwargs)
        else:
            raise AttributeError("Object must support boundary conditions for unit-aware BCs")
    
    # === Integration with Scaling System ===
    
    def auto_detect_scaling_contribution(self) -> dict:
        """
        Automatically detect what scaling coefficients this object contributes.
        Used by automatic problem scaling detection.
        """
        contributions = {}
        
        if self.has_units:
            # Analyze units to determine scaling contributions
            unit_dimensions = self._units_backend.get_dimensionality(self._units)
            
            # Map to scaling coefficient names
            dimension_mapping = {
                '[length]': 'length_scale',
                '[time]': 'time_scale', 
                '[mass]': 'mass_scale',
                '[temperature]': 'temperature_scale',
                '[velocity]': 'velocity_scale',
                '[pressure]': 'pressure_scale'
            }
            
            for dimension, coeff_name in dimension_mapping.items():
                if dimension in unit_dimensions and unit_dimensions[dimension] == 1:
                    # Simple dimension - could contribute to scaling
                    if hasattr(self, 'array') and self.array.size > 0:
                        # Use representative value (e.g., RMS, max, etc.)
                        representative_value = self._get_representative_value()
                        if representative_value > 0:
                            contributions[coeff_name] = self._units_backend.create_quantity(
                                representative_value, self._units
                            )
        
        return contributions
    
    def _get_representative_value(self) -> float:
        """Get representative value for scaling (RMS, max, etc.)"""
        if hasattr(self, 'array'):
            data = self.array[...]
            if data.size > 0:
                # Use RMS value as representative
                return float(np.sqrt(np.mean(data**2)))
        return 0.0
    
    # === Backend Integration Methods ===
    
    def convert_to_backend(self, backend_name: str):
        """Convert this object to use a different units backend"""
        if backend_name == self._units_backend.name:
            return  # Already using this backend
            
        # Convert units to new backend
        if self.has_units:
            old_quantity = self._units_backend.create_quantity(1.0, self._units)
            new_backend = self._get_backend(backend_name)
            new_units = new_backend.convert_from_other(old_quantity)
            
            self._units_backend = new_backend
            self._units = new_units
    
    # === Utility Methods ===
    
    def units_info(self) -> dict:
        """Get comprehensive information about units for this object"""
        info = {
            'has_units': self.has_units,
            'units': str(self._units) if self.has_units else None,
            'is_dimensionless': self.is_dimensionless,
            'backend': self._units_backend.name,
        }
        
        if self.has_units:
            info['dimensionality'] = self._units_backend.get_dimensionality(self._units)
            info['base_units'] = self._units_backend.to_base_units(self._units)
            
        return info
    
    def __repr_units__(self) -> str:
        """Unit-aware representation (used by enhanced __repr__)"""
        if self.has_units:
            return f" [{self._units}]"
        return ""

# === Integration with MathematicalMixin ===

class UnitAwareMathematicalMixin(UnitAwareMixin):
    """
    Combined mixin that provides both mathematical operations and units.
    Integrates with the existing MathematicalMixin system.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Override mathematical operations to include unit handling
    def __add__(self, other):
        return self._unit_aware_add(other)
    
    def __mul__(self, other): 
        return self._unit_aware_mul(other)
        
    def __truediv__(self, other):
        return self._unit_aware_truediv(other)
        
    def __pow__(self, exponent):
        return self._unit_aware_pow(exponent)
    
    # Override calculus operations
    def grad(self):
        return self._unit_aware_grad()
        
    def div(self):
        return self._unit_aware_div()
    
    # Enhanced representation
    def __repr__(self):
        base_repr = super().__repr__()
        units_suffix = self.__repr_units__()
        return base_repr.rstrip() + units_suffix

# === Backend Implementations ===

def _get_default_backend() -> str:
    """Get default units backend from configuration"""
    import underworld3 as uw
    return getattr(uw.config, 'default_units_backend', 'pint')

def _get_pint_backend() -> UnitsBackend:
    """Get Pint backend implementation"""
    from .backends.pint_backend import PintBackend
    return PintBackend()

def _get_sympy_backend() -> UnitsBackend:
    """Get SymPy backend implementation"""  
    from .backends.sympy_backend import SymPyBackend
    return SymPyBackend()
```

## **Backend Implementations**

### **Pint Backend**
```python
# underworld3/utilities/backends/pint_backend.py
import pint
import numpy as np
from typing import Any, Union

class PintBackend:
    """Pint implementation of units backend"""
    
    def __init__(self):
        self.name = 'pint'
        self.registry = pint.UnitRegistry()
        
        # Add geophysical units
        self._setup_geophysical_units()
    
    def _setup_geophysical_units(self):
        """Add common geophysical units"""
        r = self.registry
        r.define('year = 365.25 * day = a = yr')
        r.define('Ma = 1e6 * year = Myr')
        r.define('Ga = 1e9 * year = Gyr') 
        r.define('cm_per_year = centimeter / year')
        r.define('GPa = 1e9 * pascal')
    
    def create_quantity(self, value: Any, units: Union[str, Any]) -> pint.Quantity:
        """Create Pint quantity"""
        return self.registry.Quantity(value, units)
    
    def get_units(self, quantity: pint.Quantity) -> pint.Unit:
        """Extract units from Pint quantity"""
        return quantity.units
    
    def get_magnitude(self, quantity: pint.Quantity) -> Any:
        """Extract numerical value from Pint quantity"""
        return quantity.magnitude
    
    def is_quantity(self, obj: Any) -> bool:
        """Check if object is a Pint quantity"""
        return isinstance(obj, pint.Quantity)
    
    def is_dimensionless(self, units: pint.Unit) -> bool:
        """Check if units are dimensionless"""
        return units.dimensionless
    
    def check_dimensionality(self, units1: pint.Unit, units2: pint.Unit) -> bool:
        """Check dimensional compatibility"""
        return units1.dimensionality == units2.dimensionality
    
    def multiply_units(self, units1: pint.Unit, units2: pint.Unit) -> pint.Unit:
        """Multiply units"""
        return units1 * units2
    
    def divide_units(self, units1: pint.Unit, units2: pint.Unit) -> pint.Unit:
        """Divide units"""
        return units1 / units2
    
    def power_units(self, units: pint.Unit, exponent: float) -> pint.Unit:
        """Raise units to power"""
        return units ** exponent
    
    def invert_units(self, units: pint.Unit) -> pint.Unit:
        """Invert units"""
        return 1 / units
    
    def get_dimensionality(self, units: pint.Unit) -> dict:
        """Get dimensionality dictionary"""
        return dict(units.dimensionality)
    
    def to_base_units(self, units: pint.Unit) -> pint.Unit:
        """Convert to base units"""
        return units.to_base_units()
    
    def non_dimensionalise(self, quantity: pint.Quantity) -> float:
        """Non-dimensionalise using UW3 scaling system"""
        import underworld3.scaling as scaling
        return scaling.non_dimensionalise(quantity)
    
    def dimensionalise(self, value: float, units: pint.Unit) -> pint.Quantity:
        """Dimensionalise using UW3 scaling system"""
        import underworld3.scaling as scaling
        return scaling.dimensionalise(value, units)

    @property
    def meter(self) -> pint.Unit:
        """Standard length unit"""
        return self.registry.meter
```

### **SymPy Backend**  
```python
# underworld3/utilities/backends/sympy_backend.py
import sympy as sp
from sympy.physics.units import meter, second, kilogram, kelvin
import numpy as np
from typing import Any, Union

class SymPyBackend:
    """SymPy implementation of units backend"""
    
    def __init__(self):
        self.name = 'sympy'
        self._setup_units()
    
    def _setup_units(self):
        """Setup SymPy units"""
        # Basic units namespace  
        from types import SimpleNamespace
        self.units = SimpleNamespace()
        
        self.units.meter = meter
        self.units.second = second
        self.units.kilogram = kilogram  
        self.units.kelvin = kelvin
        
        # Derived units
        self.units.centimeter = meter / 100
        self.units.year = 365.25 * 24 * 3600 * second
        # ... more units as needed
    
    def create_quantity(self, value: Any, units: Union[str, sp.Expr]) -> sp.Expr:
        """Create SymPy quantity"""
        if isinstance(units, str):
            units_expr = getattr(self.units, units, None)
            if units_expr is None:
                raise ValueError(f"Unknown SymPy unit: {units}")
        else:
            units_expr = units
        return value * units_expr
    
    def get_units(self, quantity: sp.Expr) -> sp.Expr:
        """Extract units from SymPy expression (simplified)"""
        # This is a simplified implementation
        # Real version would need sophisticated dimensional analysis
        return quantity / self._get_numerical_value(quantity)
    
    def get_magnitude(self, quantity: sp.Expr) -> float:
        """Extract numerical value"""
        return self._get_numerical_value(quantity)
    
    def is_quantity(self, obj: Any) -> bool:
        """Check if object is SymPy expression with units"""
        return isinstance(obj, sp.Expr) and self._has_units(obj)
    
    def is_dimensionless(self, units: sp.Expr) -> bool:
        """Check if expression is dimensionless"""
        return not self._has_units(units)
    
    def check_dimensionality(self, units1: sp.Expr, units2: sp.Expr) -> bool:
        """Check dimensional compatibility (simplified)"""
        # Simplified implementation - real version needs proper dimensional analysis
        dims1 = self._extract_dimensions(units1)
        dims2 = self._extract_dimensions(units2)
        return dims1 == dims2
    
    def multiply_units(self, units1: sp.Expr, units2: sp.Expr) -> sp.Expr:
        """Multiply units"""
        return units1 * units2
    
    def divide_units(self, units1: sp.Expr, units2: sp.Expr) -> sp.Expr:
        """Divide units"""
        return units1 / units2
    
    def power_units(self, units: sp.Expr, exponent: float) -> sp.Expr:
        """Raise units to power"""
        return units ** exponent
    
    def invert_units(self, units: sp.Expr) -> sp.Expr:
        """Invert units"""
        return 1 / units
    
    def non_dimensionalise(self, quantity: sp.Expr) -> float:
        """Non-dimensionalise using SymPy scaling"""
        # Would need implementation of Ben Knight's sympy scaling functions
        from underworld3.scaling.sympy_scaling import non_dimensionalise_sympy
        return non_dimensionalise_sympy(quantity)
    
    def dimensionalise(self, value: float, units: sp.Expr) -> sp.Expr:
        """Dimensionalise using SymPy scaling"""
        from underworld3.scaling.sympy_scaling import dimensionalise_sympy
        return dimensionalise_sympy(value, units)
    
    # Helper methods (simplified implementations)
    def _has_units(self, expr: sp.Expr) -> bool:
        """Check if expression has physical units"""
        unit_symbols = {meter, second, kilogram, kelvin}
        return bool(unit_symbols.intersection(expr.free_symbols))
    
    def _get_numerical_value(self, expr: sp.Expr) -> float:
        """Extract numerical coefficient"""
        # Simplified - real implementation needs proper analysis
        return float(expr.subs({meter: 1, second: 1, kilogram: 1, kelvin: 1}))
    
    def _extract_dimensions(self, expr: sp.Expr) -> dict:
        """Extract dimensional powers (simplified)"""
        # This would need a real dimensional analysis implementation
        return {}
    
    @property  
    def meter(self) -> sp.Expr:
        """Standard length unit"""
        return self.units.meter
```

## **Usage Examples**

### **Basic Usage with Variables**
```python
# Create unit-aware mesh variable
class UnitAwareMeshVariable(MeshVariable, UnitAwareMathematicalMixin):
    pass

# Usage
mesh = uw.meshing.StructuredQuadBox(elementRes=(10, 10))

# Temperature field with units
temperature = UnitAwareMeshVariable(
    "Temperature", mesh, 1, 
    units="kelvin",
    units_backend="pint"  # or "sympy"
)

# Set values with automatic unit conversion
temperature.set_dimensional_values(1300 * uw.units.celsius)  # Converts automatically

# Mathematical operations preserve units
temp_gradient = temperature.grad()  # Units: kelvin/meter
print(temp_gradient.units)  # kelvin/meter

# Boundary conditions with unit validation
temperature.set_bc_with_units("Top", 273.15 * uw.units.kelvin)  # Validates compatibility
```

### **Velocity Field Example**
```python
# Velocity field with units
velocity = UnitAwareMeshVariable(
    "Velocity", mesh, 2,
    units="cm/year",
    units_backend="pint"
)

# Set plate motion velocities
plate_velocity = 3.2 * uw.units.cm_per_year
velocity.set_dimensional_values(plate_velocity)

# Strain rate calculation with automatic unit algebra
strain_rate = velocity.grad()  # Units: (cm/year)/meter = 1/second (automatically simplified)
```

### **Backend Switching**
```python
# Start with Pint backend
temperature_pint = UnitAwareMeshVariable("T", mesh, 1, units="kelvin", units_backend="pint")

# Convert to SymPy backend if needed
temperature_pint.convert_to_backend("sympy")

# Now uses SymPy units internally
print(temperature_pint.units_info())
```

## **Integration with Existing Classes**

### **Minimal Changes to Current Architecture**
```python
# underworld3/discretisation/discretisation_mesh_variables.py
from ..utilities.units_mixin import UnitAwareMathematicalMixin

class _MeshVariable(MathematicalMixin, Stateful, uw_object):
    # Existing implementation unchanged
    pass

# New optional unit-aware version
class UnitAwareMeshVariable(_MeshVariable, UnitAwareMathematicalMixin):
    """
    Mesh variable with optional units support.
    Backward compatible - can be used exactly like _MeshVariable.
    """
    pass

# Factory function for easy creation
def MeshVariable(name, mesh, num_components, *, units=None, **kwargs):
    """
    Create mesh variable, optionally with units.
    
    If units are specified, returns UnitAwareMeshVariable.
    Otherwise returns standard _MeshVariable.
    """
    if units is not None:
        return UnitAwareMeshVariable(name, mesh, num_components, units=units, **kwargs)
    else:
        return _MeshVariable(name, mesh, num_components, **kwargs)
```

## **Benefits of Mixin Architecture**

### **1. Clean Separation** ✅
- Units logic completely separate from computational logic
- Core mesh/swarm variables unchanged
- Optional functionality doesn't complicate base classes

### **2. Backward Compatibility** ✅
- Existing code continues to work unchanged
- No breaking changes to current API
- Units are purely additive functionality

### **3. Flexible Backend Support** ✅
- Same interface works with Pint or SymPy backends
- Easy to switch between backends
- Can add new backends without changing user code

### **4. Consistent with UW3 Patterns** ✅
- Follows successful MathematicalMixin pattern
- Maintains architectural consistency
- Easy for developers to understand and extend

### **5. Gradual Adoption** ✅
- Users can adopt units incrementally
- No forced migration required
- Clear upgrade path from dimensionless to unit-aware

The mixin architecture provides the perfect solution for adding units to Underworld3 while maintaining the clean, modular design principles that make the codebase maintainable and extensible.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Complete units system plan revision", "status": "completed", "activeForm": "Completing units system plan revision"}, {"content": "Design units mixin class architecture", "status": "completed", "activeForm": "Designing units mixin class architecture"}, {"content": "Implement UnitAwareMixin for mathematical objects", "status": "pending", "activeForm": "Implementing UnitAwareMixin for mathematical objects"}, {"content": "Integrate units mixin with existing classes", "status": "pending", "activeForm": "Integrating units mixin with existing classes"}]