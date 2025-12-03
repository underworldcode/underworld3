# Units System Plan: Physical Dimensions Throughout Underworld3

## Project Overview

This plan outlines the implementation of a comprehensive units system using Pint that provides physical dimensional analysis throughout Underworld3, from user input to solver output, while maintaining numerical accuracy through automatic scaling.

## Current State Analysis

### Existing Units Infrastructure

Underworld3 already has basic Pint integration in the scaling system:
- `underworld3.scaling` uses `pint.UnitRegistry`
- Basic `non_dimensionalise()` and `dimensionalise()` functions
- Scaling coefficients for `[length]`, `[mass]`, `[time]`, `[temperature]`, `[substance]`

### Current Limitations

1. **Limited Usage**: Units only used in scaling, not throughout codebase
2. **No Variable Units**: MeshVariable and SwarmVariable have no unit awareness
3. **No Expression Units**: Mathematical expressions don't track units
4. **No Validation**: No unit consistency checking
5. **Poor User Experience**: Users must manually handle scaling

### Problems to Solve

1. **Unit Consistency**: Ensure dimensional consistency across operations
2. **Automatic Scaling**: Convert between physical and computational units
3. **User Experience**: Natural unit specification and validation
4. **Performance**: Maintain solver accuracy with dimensionless computation
5. **Integration**: Units work seamlessly with mathematical objects

## Proposed Architecture

### 1. Enhanced Units Registry

```python
# underworld3/units/registry.py
import pint
from typing import Optional, Dict, Any, Union
import numpy as np

# Global unit registry - single source of truth
u = pint.UnitRegistry()

# Add common geophysical units
u.define('year = 365.25 * day = a = yr')
u.define('Ma = 1e6 * year = Myr')
u.define('Ga = 1e9 * year = Gyr')
u.define('km = 1000 * meter')
u.define('cm_per_year = centimeter / year')

class UnitsRegistry:
    """
    Enhanced units registry for Underworld3 with geophysical extensions
    and problem-specific scaling management.
    """
    
    def __init__(self):
        self.registry = u
        self._problem_scales = {}
        
    def Quantity(self, value, units):
        """Create a Pint Quantity"""
        return self.registry.Quantity(value, units)
    
    def parse_units(self, units_str: str):
        """Parse units string into Pint Unit"""
        return self.registry.parse_units(units_str)
    
    def check_dimensionality(self, quantity: pint.Quantity, expected_dimension: str):
        """Check if quantity has expected dimensionality"""
        expected = self.registry.parse_units(expected_dimension)
        return quantity.dimensionality == expected.dimensionality
    
    def set_problem_scales(self, length_scale=None, time_scale=None, 
                          mass_scale=None, temperature_scale=None):
        """Set characteristic scales for current problem"""
        if length_scale:
            self._problem_scales['[length]'] = self.registry.Quantity(length_scale)
        if time_scale:
            self._problem_scales['[time]'] = self.registry.Quantity(time_scale) 
        if mass_scale:
            self._problem_scales['[mass]'] = self.registry.Quantity(mass_scale)
        if temperature_scale:
            self._problem_scales['[temperature]'] = self.registry.Quantity(temperature_scale)
    
    def non_dimensionalise(self, quantity: pint.Quantity) -> float:
        """Convert dimensional quantity to dimensionless using problem scales"""
        dimension = quantity.dimensionality
        
        # Look for matching problem scale
        for scale_dim, scale_value in self._problem_scales.items():
            scale_dimensionality = scale_value.dimensionality
            if dimension == scale_dimensionality:
                return (quantity / scale_value).to_reduced_units().magnitude
        
        # Fallback to default scaling
        return self._default_non_dimensionalise(quantity)
    
    def dimensionalise(self, value: float, target_units: Union[str, pint.Unit]) -> pint.Quantity:
        """Convert dimensionless value to dimensional quantity"""
        if isinstance(target_units, str):
            target_units = self.registry.parse_units(target_units)
        
        dimension = target_units.dimensionality
        
        # Find matching problem scale
        for scale_dim, scale_value in self._problem_scales.items():
            if dimension == scale_value.dimensionality:
                return (value * scale_value).to(target_units)
        
        # Fallback to default scaling
        return self._default_dimensionalise(value, target_units)

# Global instance
units_registry = UnitsRegistry()
units = units_registry.registry  # Shortcut for creating quantities
```

### 2. Unit-Aware Mathematical Objects

```python
# underworld3/mathematical_objects/unit_aware.py
from typing import Optional, Union
import pint
from .mathematical_objects import MathematicalObject

class UnitAwareMathematicalObject(MathematicalObject):
    """
    Mathematical object that tracks physical units through operations.
    Integrates with the mathematical objects system.
    """
    
    def __init__(self, name: str, value=0, units: Optional[Union[str, pint.Unit]] = None,
                 description: str = ""):
        super().__init__(name, description)
        
        # Unit tracking
        if units is not None:
            if isinstance(units, str):
                self._units = units_registry.parse_units(units)
            else:
                self._units = units
        else:
            self._units = units_registry.registry.dimensionless
            
        self._dimensional_value = value
        
    @property
    def units(self) -> pint.Unit:
        """Physical units of this object"""
        return self._units
    
    @property
    def dimensional_value(self) -> pint.Quantity:
        """Value with units"""
        return units_registry.Quantity(self._dimensional_value, self._units)
    
    @property 
    def dimensionless_value(self) -> float:
        """Dimensionless value for computation"""
        if self._units == units_registry.registry.dimensionless:
            return self._dimensional_value
        return units_registry.non_dimensionalise(self.dimensional_value)
    
    def __add__(self, other):
        """Addition with unit checking"""
        if isinstance(other, UnitAwareMathematicalObject):
            # Check unit compatibility
            if self._units.dimensionality != other._units.dimensionality:
                raise pint.DimensionalityError(
                    self._units, other._units,
                    extra_msg=f"Cannot add {self.name} ({self._units}) + {other.name} ({other._units})"
                )
            
            # Result has same units as left operand
            result_name = f"({self.name} + {other.name})"
            result_expr = self.mathematical_form + other.mathematical_form
            result = UnitAwareMathematicalExpression(result_name, result_expr, self._units)
            return result
        else:
            # Adding dimensionless constant
            if not isinstance(other, (int, float)) or other != 0:
                # Only allow adding zero to dimensional quantities
                if self._units != units_registry.registry.dimensionless:
                    raise pint.DimensionalityError(
                        self._units, units_registry.registry.dimensionless,
                        extra_msg=f"Cannot add dimensional quantity {self.name} to dimensionless {other}"
                    )
            
            result_name = f"({self.name} + {other})"
            result_expr = self.mathematical_form + other
            return UnitAwareMathematicalExpression(result_name, result_expr, self._units)
    
    def __mul__(self, other):
        """Multiplication with unit combination"""
        if isinstance(other, UnitAwareMathematicalObject):
            result_units = self._units * other._units
            result_name = f"{self.name}⋅{other.name}"
            result_expr = self.mathematical_form * other.mathematical_form
            return UnitAwareMathematicalExpression(result_name, result_expr, result_units)
        else:
            # Multiplying by dimensionless scalar
            result_name = f"{other}⋅{self.name}"
            result_expr = other * self.mathematical_form
            return UnitAwareMathematicalExpression(result_name, result_expr, self._units)
    
    def __truediv__(self, other):
        """Division with unit combination"""
        if isinstance(other, UnitAwareMathematicalObject):
            result_units = self._units / other._units
            result_name = f"{self.name}/{other.name}"
            result_expr = self.mathematical_form / other.mathematical_form
            return UnitAwareMathematicalExpression(result_name, result_expr, result_units)
        else:
            result_name = f"{self.name}/{other}"
            result_expr = self.mathematical_form / other
            return UnitAwareMathematicalExpression(result_name, result_expr, self._units)
    
    def __pow__(self, exponent):
        """Power operation with unit exponentiation"""
        result_units = self._units ** exponent
        result_name = f"{self.name}^{exponent}"
        result_expr = self.mathematical_form ** exponent
        return UnitAwareMathematicalExpression(result_name, result_expr, result_units)
    
    def diff(self, variable):
        """Differentiation changes units"""
        if isinstance(variable, UnitAwareMathematicalObject):
            result_units = self._units / variable._units
        else:
            # Assume spatial coordinate with length units
            coordinate_units = getattr(variable, 'coordinate_units', units_registry.registry.meter)
            result_units = self._units / coordinate_units
            
        result_name = f"∂{self.name}/∂{variable.name if hasattr(variable, 'name') else variable}"
        result_expr = self.mathematical_form.diff(variable)
        return UnitAwareMathematicalExpression(result_name, result_expr, result_units)

class UnitAwareMathematicalExpression(UnitAwareMathematicalObject):
    """Mathematical expression with unit tracking"""
    
    def __init__(self, name: str, sympy_expr, units: pint.Unit):
        super().__init__(name, 0, units)
        self._sympy_expr = sympy_expr
    
    def _create_mathematical_form(self):
        return self._sympy_expr
```

### 3. Unit-Aware Variables

```python
# underworld3/discretisation/unit_aware_variables.py
from ..mathematical_objects.unit_aware import UnitAwareMathematicalObject
from ..mathematical_objects.mathematical_objects import MathematicalField

class UnitAwareMeshVariable(MathematicalField, UnitAwareMathematicalObject):
    """
    Mesh variable with full unit support.
    Mathematical operations preserve units, solver uses dimensionless values.
    """
    
    def __init__(self, name: str, mesh, components: int, 
                 units: Optional[Union[str, pint.Unit]] = None, **kwargs):
        
        # Initialize both parent classes
        MathematicalField.__init__(self, name, mesh, components)
        UnitAwareMathematicalObject.__init__(self, name, 0, units)
        
        # Mesh variable specifics
        self.mesh = mesh
        self.vtype = kwargs.get('vtype')
        
        # Storage for computational arrays (always dimensionless)
        self._array_with_callback = None
        self.vec = None
        
        # Coordinate units from mesh
        self.coordinate_units = getattr(mesh, 'coordinate_units', units_registry.registry.dimensionless)
        
    def set_dimensional_values(self, values: Union[np.ndarray, pint.Quantity]):
        """Set values with units, automatically converting to dimensionless"""
        
        if isinstance(values, pint.Quantity):
            # Check unit compatibility
            if self._units.dimensionality != values.dimensionality:
                raise pint.DimensionalityError(
                    self._units, values.units,
                    extra_msg=f"Cannot assign {values.units} to variable with {self._units}"
                )
            
            # Convert to dimensionless
            dimensionless_values = units_registry.non_dimensionalise(values.to(self._units))
        else:
            # Assume values are already in correct units
            dimensionless_values = values
        
        # Set computational array
        self.array[...] = dimensionless_values
    
    def get_dimensional_values(self) -> pint.Quantity:
        """Get values with units"""
        dimensionless_array = self.array[...]
        return units_registry.dimensionalise(dimensionless_array, self._units)
    
    def grad(self):
        """Gradient with proper unit handling"""
        # Gradient changes units: [quantity]/[length]  
        if self.coordinate_units != units_registry.registry.dimensionless:
            result_units = self._units / self.coordinate_units
        else:
            result_units = self._units  # Dimensionless coordinates
            
        # Create mathematical gradient
        grad_expr = super().grad()
        
        # Return unit-aware result
        return UnitAwareMathematicalExpression(
            grad_expr.name, grad_expr.mathematical_form, result_units
        )
    
    def div(self):
        """Divergence with unit handling"""
        if self.num_components == 1:
            raise ValueError("Cannot take divergence of scalar field")
            
        # Divergence: [quantity/length] for each component
        if self.coordinate_units != units_registry.registry.dimensionless:
            result_units = self._units / self.coordinate_units
        else:
            result_units = self._units
            
        div_expr = super().div()
        return UnitAwareMathematicalExpression(
            div_expr.name, div_expr.mathematical_form, result_units
        )
    
    def set_bc_dirichlet(self, boundary: str, value):
        """Set boundary condition with unit validation"""
        
        if isinstance(value, UnitAwareMathematicalObject):
            # Check unit compatibility
            if self._units.dimensionality != value._units.dimensionality:
                raise pint.DimensionalityError(
                    self._units, value._units,
                    extra_msg=f"Boundary condition units {value._units} don't match variable units {self._units}"
                )
            bc_value = value.mathematical_form
        elif isinstance(value, pint.Quantity):
            # Convert to correct units
            if self._units.dimensionality != value.dimensionality:
                raise pint.DimensionalityError(self._units, value.units)
            bc_value = value.to(self._units).magnitude
        else:
            # Assume dimensionless or correct units
            bc_value = value
            
        # Store boundary condition
        super().set_bc_dirichlet(boundary, bc_value)

class UnitAwareSwarmVariable(UnitAwareMeshVariable):
    """Swarm variable with unit support"""
    
    def __init__(self, name: str, swarm, components: int, 
                 units: Optional[Union[str, pint.Unit]] = None, **kwargs):
        # Initialize as unit-aware mesh variable if has proxy
        proxy_degree = kwargs.get('proxy_degree', 0)
        if proxy_degree > 0:
            super().__init__(name, swarm.mesh, components, units, **kwargs)
        else:
            # Pure swarm variable - discrete units
            UnitAwareMathematicalObject.__init__(self, name, 0, units)
            self.swarm = swarm
            self.num_components = components
```

### 4. Problem-Specific Scaling

```python
# underworld3/scaling/problem_scaling.py
from typing import List, Optional, Dict
import pint
from ..materials import Material

class ProblemScaling:
    """
    Automatic problem-specific scaling for numerical accuracy.
    Determines characteristic scales from problem setup.
    """
    
    def __init__(self, mesh=None):
        self.mesh = mesh
        self._length_scale = None
        self._time_scale = None
        self._velocity_scale = None
        self._temperature_scale = None
        self._pressure_scale = None
        
    def auto_detect_scales(self, materials: List[Material] = None, 
                          boundary_conditions: Dict = None):
        """Automatically detect characteristic scales from problem"""
        
        # Length scale from mesh
        if self.mesh and hasattr(self.mesh, 'coordinate_units'):
            if hasattr(self.mesh, 'bounding_box'):
                domain_size = max(self.mesh.bounding_box.max() - self.mesh.bounding_box.min())
                self._length_scale = units.Quantity(domain_size, self.mesh.coordinate_units)
        
        # Material-based scales
        if materials:
            self._extract_material_scales(materials)
            
        # Boundary condition scales  
        if boundary_conditions:
            self._extract_bc_scales(boundary_conditions)
            
        # Derive dependent scales
        self._compute_derived_scales()
        
        # Set in global registry
        units_registry.set_problem_scales(
            length_scale=self._length_scale,
            time_scale=self._time_scale,
            temperature_scale=self._temperature_scale
        )
    
    def _extract_material_scales(self, materials: List[Material]):
        """Extract characteristic scales from material properties"""
        
        # Collect material properties
        viscosities = []
        densities = []
        diffusivities = []
        
        for material in materials:
            if hasattr(material, 'viscosity') and material.viscosity:
                viscosities.append(material.viscosity.dimensional_value)
            if hasattr(material, 'density') and material.density:
                densities.append(material.density.dimensional_value)
            if hasattr(material, 'thermal_diffusivity') and material.thermal_diffusivity:
                diffusivities.append(material.thermal_diffusivity.dimensional_value)
        
        # Compute representative scales (geometric mean)
        if viscosities:
            visc_values = [v.magnitude for v in viscosities]
            mean_visc = (np.prod(visc_values)) ** (1/len(visc_values))
            self._viscosity_scale = units.Quantity(mean_visc, viscosities[0].units)
            
        if densities:
            dens_values = [d.magnitude for d in densities] 
            mean_dens = np.mean(dens_values)
            self._density_scale = units.Quantity(mean_dens, densities[0].units)
            
        if diffusivities:
            diff_values = [d.magnitude for d in diffusivities]
            mean_diff = (np.prod(diff_values)) ** (1/len(diff_values))
            self._diffusivity_scale = units.Quantity(mean_diff, diffusivities[0].units)
    
    def _extract_bc_scales(self, boundary_conditions: Dict):
        """Extract scales from boundary conditions"""
        
        velocity_bcs = []
        temperature_bcs = []
        
        for var_name, var_bcs in boundary_conditions.items():
            if 'velocity' in var_name.lower():
                for bc in var_bcs.values():
                    if isinstance(bc.get('value'), pint.Quantity):
                        velocity_bcs.append(bc['value'])
            elif 'temperature' in var_name.lower():
                for bc in var_bcs.values():
                    if isinstance(bc.get('value'), pint.Quantity):
                        temperature_bcs.append(bc['value'])
        
        # Set velocity scale
        if velocity_bcs:
            max_vel = max(v.magnitude for v in velocity_bcs)
            self._velocity_scale = units.Quantity(max_vel, velocity_bcs[0].units)
        
        # Set temperature scale  
        if temperature_bcs:
            temp_range = max(t.magnitude for t in temperature_bcs) - min(t.magnitude for t in temperature_bcs)
            self._temperature_scale = units.Quantity(temp_range, temperature_bcs[0].units)
    
    def _compute_derived_scales(self):
        """Compute derived scales from fundamental scales"""
        
        # Time scale from length and velocity
        if self._length_scale and self._velocity_scale:
            self._time_scale = (self._length_scale / self._velocity_scale).to_base_units()
        
        # Time scale from thermal diffusion
        elif self._length_scale and hasattr(self, '_diffusivity_scale'):
            thermal_time = (self._length_scale**2 / self._diffusivity_scale).to_base_units()
            if self._time_scale is None:
                self._time_scale = thermal_time
        
        # Pressure scale from viscosity and velocity
        if hasattr(self, '_viscosity_scale') and self._velocity_scale and self._length_scale:
            self._pressure_scale = (self._viscosity_scale * self._velocity_scale / self._length_scale).to_base_units()
```

## Implementation Strategy

### Phase 1: Core Units Infrastructure

**Deliverables:**
1. Enhanced `UnitsRegistry` with geophysical units
2. `UnitAwareMathematicalObject` base class
3. Unit validation and error handling
4. Integration with existing scaling system
5. Basic mathematical operations with units

**Key Files:**
- `units/registry.py`: Enhanced units registry
- `mathematical_objects/unit_aware.py`: Unit-aware mathematical objects
- `test_units_system.py`: Comprehensive unit tests

### Phase 2: Variable Integration

**Deliverables:**
1. `UnitAwareMeshVariable` implementation
2. `UnitAwareSwarmVariable` implementation  
3. Boundary condition unit validation
4. Data I/O with unit preservation
5. Automatic dimensionless conversion for solvers

**Key Files:**
- `discretisation/unit_aware_variables.py`
- Integration with existing variable classes

### Phase 3: Problem Scaling

**Deliverables:**
1. `ProblemScaling` class with auto-detection
2. Material-based scale extraction
3. Boundary condition scale analysis
4. Integration with mesh and solver setup
5. Scaling optimization for numerical accuracy

### Phase 4: Integration and Validation

**Deliverables:**
1. Integration with mathematical objects system
2. Integration with material properties system
3. Examples conversion to use units throughout
4. Performance benchmarking and optimization
5. User documentation and migration guide

## Cross-References to Other Plans

### Mathematical Objects Integration
**See: `mathematical_objects_plan.md`**
- Unit-aware mathematical objects extend mathematical object system
- Mathematical operations must preserve units
- SymPy integration with unit validation

### Material Properties Integration
**See: `material_properties_plan.md`**
- Material properties become unit-aware expressions
- Parameter registry includes units and validation
- Multi-material systems use consistent units

### Multi-Material System  
**See: `MultiMaterial_ConstitutiveModel_Plan.md`**
- Level-set averaging preserves units
- Flux computation maintains dimensional consistency
- Constitutive models validate parameter units

## Benefits

### For Users
1. **Physical Intuition**: Work with real physical units throughout
2. **Error Prevention**: Automatic dimensional analysis catches mistakes
3. **Natural Interface**: Set boundary conditions with intuitive units
4. **Automatic Conversion**: System handles scaling for numerical accuracy

### For Developers
1. **Dimensional Safety**: Unit checking prevents dimensional errors
2. **Clear Interfaces**: Function signatures specify expected units
3. **Automatic Documentation**: Units provide self-documenting code
4. **Testing**: Unit consistency can be automatically tested

### For Science
1. **Physical Correctness**: Ensures dimensional consistency in models
2. **Reproducibility**: Clear specification of all dimensional parameters
3. **Model Validation**: Unit analysis helps validate physical correctness
4. **Cross-System Compatibility**: Standard units enable data exchange

## Success Criteria

### Functional Requirements
1. **Complete Coverage**: All variables, parameters, and expressions have units
2. **Automatic Scaling**: Transparent conversion between physical and computational units
3. **Error Detection**: Dimensional errors caught at compile/runtime
4. **Performance**: No significant impact on solver performance
5. **User Experience**: Natural unit specification and conversion

### Integration Requirements  
1. **Mathematical Objects**: Seamless integration with mathematical object system
2. **Material Properties**: Units work with material property system
3. **Solver Integration**: Automatic dimensionless conversion for solvers
4. **I/O Consistency**: Units preserved in save/load operations

### Validation Requirements
1. **Physical Correctness**: All examples demonstrate correct dimensional analysis
2. **Numerical Accuracy**: Automatic scaling maintains solver conditioning
3. **User Feedback**: Domain scientists find units system intuitive
4. **Performance Benchmarks**: Units system adds <10% computational overhead

---

*Document Version: 1.0*  
*Created: 2025-01-19*  
*Cross-References: mathematical_objects_plan.md, material_properties_plan.md*  
*Status: Planning Phase*