# Units System Plan: Physical Dimensions Throughout Underworld3
## Comprehensive Analysis of Pint vs SymPy Approaches

---

## Project Overview

This plan evaluates and outlines the implementation of a comprehensive units system for Underworld3, providing physical dimensional analysis throughout the codebase while maintaining numerical accuracy. **Two primary approaches are analyzed**: using **Pint** (current) vs **SymPy-based dimensional analysis** (proposed by Ben Knight).

## Current State Analysis

### Existing Units Infrastructure

Underworld3 has basic Pint integration in the scaling system:
- `underworld3.scaling` uses `pint.UnitRegistry`
- Basic `non_dimensionalise()` and `dimensionalise()` functions
- Scaling coefficients for `[length]`, `[mass]`, `[time]`, `[temperature]`, `[substance]`

### Current Limitations

1. **Limited Usage**: Units only used in scaling, not throughout codebase
2. **No Variable Units**: MeshVariable and SwarmVariable have no unit awareness  
3. **No Expression Units**: Mathematical expressions don't track units
4. **No Validation**: No unit consistency checking
5. **Poor User Experience**: Users must manually handle scaling

---

## SymPy vs Pint: Technical Comparison

### **Approach 1: Pint-Based System (Current/Enhanced)**

#### Architecture Overview
```python
# Enhanced Pint approach
import pint
u = pint.UnitRegistry()

# Define geophysical units
u.define('year = 365.25 * day = a = yr')
u.define('Ma = 1e6 * year = Myr') 
u.define('Ga = 1e9 * year = Gyr')

# Usage
velocity = 1.0 * u.centimeter / u.year
model_height = 660.0 * u.kilometer
viscosity = 1e22 * u.pascal * u.second

# Automatic dimensional analysis
time_scale = model_height / velocity  # Automatically gets correct units
nd_velocity = non_dimensionalise(velocity)  # Type-safe conversion
```

#### Pint Advantages ✅

1. **Comprehensive Unit Database**: 
   - 1000+ predefined units and prefixes
   - Automatic unit conversions (m/s ↔ km/hr ↔ cm/year)
   - Support for complex derived units

2. **Automatic Dimensional Analysis**:
   - Operations automatically generate correct units: `length / time → velocity`
   - Catches dimensional errors at runtime: `length + time → DimensionalityError`
   - Type-safe unit arithmetic without manual bookkeeping

3. **Robust Error Detection**:
   - Prevents adding incompatible quantities: `meters + seconds → Error`
   - Validates function arguments: `@pint.check('[length]')` decorator
   - Clear error messages for debugging

4. **User-Friendly Interface**:
   - Natural syntax: `5 * u.meter + 3 * u.kilometer → 5.003 km`
   - Intelligent unit simplification: `kg⋅m²/s² → Joule`
   - Context-aware conversions

5. **Industry Standard**:
   - Mature, well-tested library (10+ years development)
   - Used by SciPy ecosystem (astropy, uncertainties, etc.)
   - Comprehensive documentation and community support

6. **Temperature Handling**:
   - Proper temperature scales (Celsius, Kelvin, Fahrenheit)
   - Automatic temperature difference vs absolute temperature
   - Critical for geophysical applications

#### Pint Disadvantages ❌

1. **External Dependency**: Additional package requirement
2. **Runtime Overhead**: Unit checking adds computational cost
3. **Learning Curve**: Users must understand unit registry concepts
4. **Complexity**: Can be "wordy" for simple scaling operations

---

### **Approach 2: SymPy-Based System (Ben Knight's Alternative)**

#### Architecture Overview
```python
# SymPy-based approach
import sympy as sp
from sympy.physics.units import meter, second, kilogram, kelvin

# Manual unit definitions using SymPy
u_sp = SimpleNamespace()
u_sp.meter = meter
u_sp.second = second  
u_sp.centimeter = meter / 100
u_sp.year = 365.25 * 24 * 3600 * second
u_sp.kilometer = 1000 * meter
u_sp.pascal = kilogram / (meter * second**2)

# Usage
velocity = 1.0 * u_sp.centimeter / u_sp.year
model_height = 660.0 * u_sp.kilometer
viscosity = 1e22 * u_sp.pascal * u_sp.second

# Manual dimensional analysis
time_scale = model_height / velocity  # Returns SymPy expression with units
nd_velocity = non_dimensionalise_sympy(velocity)  # Custom function needed
```

#### SymPy Advantages ✅

1. **No External Dependencies**: 
   - Uses only SymPy (already required by UW3)
   - Reduces package requirements and potential conflicts
   - Self-contained within existing infrastructure

2. **Symbolic Integration**:
   - Units work naturally with SymPy mathematical expressions
   - Seamless integration with existing mathematical objects
   - Unit algebra follows SymPy expression rules

3. **Lightweight Implementation**:
   - Minimal overhead for simple scaling operations
   - Direct control over unit behavior
   - No hidden complexity or "magic" operations

4. **Customizable**:
   - Full control over unit definitions and behavior
   - Can implement domain-specific unit systems
   - No dependency on external unit registry decisions

5. **Familiar to UW3 Users**:
   - Users already working with SymPy expressions
   - Consistent API patterns with mathematical objects
   - Natural extension of existing symbolic workflow

#### SymPy Disadvantages ❌

1. **Limited Unit Database**:
   - SymPy.physics.units has ~50 units vs Pint's 1000+
   - Missing many geophysical units (requires manual definition)
   - No automatic unit prefix handling (milli-, kilo-, mega-, etc.)

2. **Manual Dimensional Analysis**:
   - No automatic error checking for dimensional consistency
   - Users must manually track unit compatibility
   - Easy to make mistakes without validation

3. **Verbose Implementation**:
   - Requires custom implementation of unit operations
   - More boilerplate code for common operations
   - Manual handling of unit conversions

4. **Limited Conversion Support**:
   - No automatic unit conversion system
   - Manual implementation needed for complex conversions
   - No context-aware unit simplification

5. **Temperature Limitations**:
   - SymPy units don't handle temperature scales properly
   - No distinction between absolute temperature and temperature differences
   - Critical limitation for thermal applications

6. **Error-Prone**:
   - Easy to forget unit tracking in complex expressions
   - No runtime validation of dimensional correctness
   - Debugging dimensional errors more difficult

---

## Detailed Feature Comparison

| Feature | Pint | SymPy | Impact on UW3 |
|---------|------|-------|---------------|
| **Unit Database** | 1000+ units | ~50 units | **Critical**: Geophysics needs diverse units |
| **Auto Conversions** | Full support | Manual only | **High**: User convenience, error reduction |
| **Error Detection** | Automatic | Manual | **Critical**: Prevents physics errors |
| **Temperature Scales** | Full support | Limited | **High**: Essential for thermal problems |
| **External Dependencies** | +1 package | None | **Low**: Pint is stable, well-maintained |
| **Performance** | ~10% overhead | Minimal | **Medium**: Acceptable for most applications |
| **Learning Curve** | Moderate | Low (for UW3 users) | **Medium**: Documentation can address |
| **Maintenance Burden** | Low | High | **High**: Custom code requires ongoing support |
| **Integration Complexity** | High initial | Low initial | **Medium**: One-time implementation cost |
| **Scientific Correctness** | Validated | User-dependent | **Critical**: Prevents published errors |

---

## Implementation Feasibility Analysis

### **What CAN be implemented with SymPy instead of Pint:**

#### Core Functionality ✅
1. **Basic Scaling Operations**:
   - Dimensional and non-dimensional conversions
   - Simple unit arithmetic (multiply, divide, power)
   - Integration with scaling coefficients

2. **Mathematical Object Integration**:
   - Unit-aware variables and expressions  
   - Basic unit tracking through operations
   - SymPy expression compatibility

3. **Simple Unit Systems**:
   - Custom geophysical unit definitions
   - Problem-specific scaling setups
   - Basic dimensional analysis

#### Example Implementation:
```python
# SymPy approach can handle basic cases
def create_scaling_sympy():
    velocity = 1 * u_sp.centimeter / u_sp.year
    model_height = 660 * u_sp.kilometer
    time_scale = model_height / velocity
    return {"length": model_height, "time": time_scale}

# Works for simple dimensional analysis
def check_dimensions_sympy(expr, expected_dims):
    # Manual checking - possible but tedious
    actual_dims = extract_dimensions(expr)
    return actual_dims == expected_dims
```

### **What CANNOT be implemented with SymPy instead of Pint:**

#### Critical Missing Features ❌

1. **Comprehensive Unit Database**:
   ```python
   # Pint: Built-in
   pressure_psi = 1000 * u.psi  # Automatic
   pressure_pa = pressure_psi.to('pascal')  # Automatic conversion
   
   # SymPy: Manual implementation required
   u_sp.psi = 6894.757 * u_sp.pascal  # Must define manually
   pressure_pa = pressure_psi * 6894.757  # Manual conversion
   ```

2. **Automatic Error Detection**:
   ```python
   # Pint: Automatic validation
   length = 5 * u.meter
   time = 3 * u.second
   try:
       wrong = length + time  # Raises DimensionalityError automatically
   except pint.DimensionalityError as e:
       print(f"Cannot add {e.units1} to {e.units2}")
   
   # SymPy: No automatic validation
   length = 5 * u_sp.meter  
   time = 3 * u_sp.second
   wrong = length + time  # Silently creates invalid expression!
   ```

3. **Temperature Scale Handling**:
   ```python
   # Pint: Proper temperature handling
   temp_c = 20 * u.celsius
   temp_k = temp_c.to('kelvin')  # Handles offset correctly (293.15 K)
   temp_diff = 10 * u.celsius    # Temperature difference (10 K)
   
   # SymPy: Cannot distinguish temperature vs temperature difference
   # This is a fundamental physics error that SymPy units cannot prevent
   ```

4. **Complex Unit Conversions**:
   ```python
   # Pint: Handles complex derived units
   viscosity = 1e22 * u.pascal * u.second
   viscosity_poise = viscosity.to('poise')  # Automatic conversion
   
   # SymPy: Requires manual conversion factors for every combination
   # Becomes unmaintainable for complex geophysical applications
   ```

5. **Unit Prefix Handling**:
   ```python
   # Pint: Automatic prefix handling
   distance = 5.2 * u.kilometer
   distance_mm = distance.to('millimeter')  # = 5,200,000 mm
   
   # SymPy: Each prefix must be manually defined
   u_sp.millimeter = u_sp.meter / 1000    # Manual definition required
   u_sp.kilometer = 1000 * u_sp.meter     # For every prefix
   ```

6. **Robust Mathematical Operations**:
   ```python
   # Pint: Handles complex dimensional analysis
   @u.check('[length]', '[time]', '[mass]')
   def stokes_law(diameter, velocity, density):
       # Automatic unit validation and conversion
       drag = 6 * np.pi * viscosity * diameter * velocity
       return drag.to_base_units()
   
   # SymPy: Cannot provide automatic validation
   # Users must manually ensure dimensional correctness
   ```

---

## **Architecture Decision: Hybrid Approach Recommendation**

### **Recommended Strategy: Enhanced Pint with SymPy Integration**

Given the analysis, the optimal approach is to **enhance the existing Pint system** while providing **optional SymPy compatibility** for specialized use cases:

#### **Phase 1: Enhanced Pint System** (Recommended Primary)
```python
# Primary interface using enhanced Pint
import underworld3 as uw

# Enhanced Pint registry with geophysical extensions
u = uw.units  # Enhanced pint registry
velocity = 1.0 * u.cm_per_year  # Geophysical units built-in
temperature = 1300 * u.celsius  # Proper temperature handling
```

#### **Phase 2: Optional SymPy Compatibility** (For Specialized Cases)
```python
# Optional SymPy interface for dependency-sensitive environments
import underworld3.scaling.sympy_units as u_sp

# For users who specifically need to avoid Pint
velocity_sp = 1.0 * u_sp.centimeter / u_sp.year
nd_velocity = uw.scaling.non_dimensionalise_sympy(velocity_sp)
```

### **Implementation Plan: Dual Approach**

#### Core Pint Enhancement (Phase 1)
1. **Enhanced Units Registry**:
   - Pre-defined geophysical units (cm/year, Ma, GPa, etc.)
   - Earth science contexts (mantle viscosity scales, etc.)
   - Automatic problem-specific scaling detection

2. **Unit-Aware Mathematical Objects**:
   - Variables track units through operations
   - Automatic dimensional validation
   - Seamless solver integration with auto-scaling

3. **Advanced Error Prevention**:
   - Function argument validation decorators
   - Boundary condition unit checking
   - Expression dimensional analysis

#### SymPy Compatibility Layer (Phase 2)
1. **Basic SymPy Units Module**:
   - Common geophysical unit definitions
   - Manual scaling functions (nd_sp, dim_sp)
   - Limited dimensional checking utilities

2. **Conversion Utilities**:
   - Pint ↔ SymPy unit conversion
   - Expression migration tools
   - Compatibility documentation

#### **Rationale for Hybrid Approach:**

1. **Best of Both Worlds**: 
   - Pint provides robust, validated dimensional analysis
   - SymPy option available for dependency-constrained environments

2. **Migration Path**:
   - Users can start with SymPy approach if preferred
   - Natural upgrade path to full Pint functionality
   - Existing SymPy users not forced to change immediately

3. **Scientific Integrity**:
   - Primary system (Pint) provides maximum error protection
   - SymPy option clearly documented as "basic/manual" approach
   - No compromise on scientific correctness for main use cases

---

## **Detailed Implementation: Enhanced Pint System**

### 1. Enhanced Units Registry
```python
# underworld3/units/registry.py
import pint
from typing import Optional, Dict, Any, Union

class GeophysicalUnitsRegistry:
    """
    Enhanced Pint registry with geophysical extensions
    and automatic problem scaling detection.
    """
    
    def __init__(self):
        self.registry = pint.UnitRegistry()
        
        # Add geophysical units
        self._add_geophysical_units()
        self._add_earth_science_contexts()
        
    def _add_geophysical_units(self):
        """Add common geophysical units to registry"""
        r = self.registry
        
        # Time scales
        r.define('year = 365.25 * day = a = yr')
        r.define('Ma = 1e6 * year = Myr = megayear') 
        r.define('Ga = 1e9 * year = Gyr = gigayear')
        r.define('ka = 1e3 * year = kyr = kiloyear')
        
        # Velocity scales  
        r.define('cm_per_year = centimeter / year')
        r.define('mm_per_year = millimeter / year')
        r.define('km_per_Myr = kilometer / megayear')
        
        # Pressure scales
        r.define('GPa = 1e9 * pascal = gigapascal')
        r.define('kbar = 1e8 * pascal')
        r.define('bar = 1e5 * pascal')
        
        # Viscosity scales
        r.define('Pa_s = pascal * second')
        r.define('poise = 0.1 * Pa_s')
        
        # Temperature differences (important distinction!)
        r.define('delta_celsius = kelvin')
        r.define('delta_kelvin = kelvin')

    def _add_earth_science_contexts(self):
        """Add context-specific unit systems"""
        r = self.registry
        
        # Mantle convection context
        mantle_ctx = pint.Context('mantle_convection')
        mantle_ctx.add_transformation('[viscosity]', '[length] * [time]',
                                    lambda r, v: v * r)
        r.add_context(mantle_ctx)
        
        # Plate tectonics context  
        plate_ctx = pint.Context('plate_tectonics')
        plate_ctx.add_transformation('[velocity]', '[length] / [time]', 
                                   lambda r, l, t: l / t)
        r.add_context(plate_ctx)

    def create_problem_scaling(self, **characteristic_scales):
        """Create problem-specific scaling from characteristic scales"""
        scaling = {}
        
        # Length scale
        if 'length' in characteristic_scales:
            scaling['[length]'] = self.registry.Quantity(characteristic_scales['length'])
        elif 'model_height' in characteristic_scales:
            scaling['[length]'] = self.registry.Quantity(characteristic_scales['model_height'])
            
        # Velocity scale  
        if 'velocity' in characteristic_scales:
            scaling['[velocity]'] = self.registry.Quantity(characteristic_scales['velocity'])
            
        # Derive time scale from length/velocity if not provided
        if 'time' not in characteristic_scales and '[length]' in scaling and '[velocity]' in scaling:
            scaling['[time]'] = scaling['[length]'] / scaling['[velocity]']
            
        return scaling

# Global enhanced registry
geo_units = GeophysicalUnitsRegistry()
u = geo_units.registry  # Standard interface
```

### 2. Unit-Aware Mathematical Objects
```python
# underworld3/mathematical_objects/unit_aware.py
from typing import Optional, Union
import pint
from .mathematical_objects import MathematicalField

class UnitAwareMathematicalField(MathematicalField):
    """
    Mathematical field with comprehensive unit support.
    Provides automatic dimensional analysis and error detection.
    """
    
    def __init__(self, name: str, mesh, num_components: int, 
                 units: Union[str, pint.Unit, pint.Quantity],
                 description: str = ""):
        
        super().__init__(name, mesh, num_components, description)
        
        # Unit handling
        if isinstance(units, str):
            self._units = geo_units.registry.parse_units(units)
        elif isinstance(units, pint.Quantity):
            self._units = units.units
        else:
            self._units = units
            
        # Coordinate units from mesh
        self._coordinate_units = getattr(mesh, 'coordinate_units', geo_units.registry.meter)
    
    @property 
    def units(self) -> pint.Unit:
        """Physical units of this field"""
        return self._units
        
    def set_values(self, values: Union[np.ndarray, pint.Quantity]):
        """Set field values with automatic unit validation and conversion"""
        
        if isinstance(values, pint.Quantity):
            # Validate dimensional compatibility
            try:
                converted_values = values.to(self._units)
            except pint.DimensionalityError as e:
                raise ValueError(
                    f"Cannot assign {values.units} to field '{self.name}' "
                    f"with units {self._units}. {e}"
                )
                
            # Store dimensionless values for computation
            scaling_factor = converted_values.units
            dimensionless_values = converted_values.magnitude
        else:
            # Assume values are already in correct units/dimensionless
            dimensionless_values = values
            
        # Set computational array (always dimensionless)
        self.data[...] = dimensionless_values
    
    def get_values(self) -> pint.Quantity:
        """Get field values with proper units"""
        return geo_units.registry.Quantity(self.data, self._units)
    
    def __add__(self, other):
        """Addition with automatic unit checking"""
        if isinstance(other, UnitAwareMathematicalField):
            # Check dimensional compatibility
            try:
                # This will raise DimensionalityError if incompatible
                test_addition = (1 * self._units) + (1 * other._units)
            except pint.DimensionalityError as e:
                raise ValueError(
                    f"Cannot add field '{self.name}' ({self._units}) "
                    f"to field '{other.name}' ({other._units}). {e}"
                )
            
            # Create result with same units as left operand
            result_name = f"({self.name} + {other.name})"
            result = UnitAwareMathematicalExpression(
                result_name, self.sym + other.sym, self._units
            )
            return result
        else:
            # Adding dimensionless constant - only allow if self is dimensionless or other is zero
            if not (self._units.dimensionless or other == 0):
                raise ValueError(
                    f"Cannot add dimensionless constant {other} "
                    f"to dimensional field '{self.name}' ({self._units})"
                )
            
            result_name = f"({self.name} + {other})"
            result = UnitAwareMathematicalExpression(
                result_name, self.sym + other, self._units
            )
            return result
    
    def __mul__(self, other):
        """Multiplication with unit algebra"""
        if isinstance(other, UnitAwareMathematicalField):
            result_units = self._units * other._units
            result_name = f"{self.name}⋅{other.name}"
            result_expr = self.sym * other.sym
        else:
            # Multiplication by dimensionless scalar
            result_units = self._units
            result_name = f"{other}⋅{self.name}"
            result_expr = other * self.sym
            
        return UnitAwareMathematicalExpression(result_name, result_expr, result_units)
    
    def grad(self):
        """Gradient with proper unit handling: [field_units]/[length_units]"""
        result_units = self._units / self._coordinate_units
        result_name = f"∇{self.name}"
        result_expr = super().grad()
        
        return UnitAwareMathematicalExpression(
            result_name, result_expr.sym, result_units
        )
    
    @pint.check('[length]', '[velocity]', '[pressure]')  
    def set_stokes_bc(self, boundary: str, velocity_bc=None, pressure_bc=None):
        """Example: Type-safe boundary condition setting with unit validation"""
        
        if velocity_bc is not None:
            # Pint automatically validates that velocity_bc has velocity dimensions
            velocity_dimensionless = geo_units.registry.non_dimensionalise(velocity_bc)
            self.add_dirichlet_bc(velocity_dimensionless, boundary, component=[0,1])
            
        if pressure_bc is not None:
            # Pint automatically validates that pressure_bc has pressure dimensions  
            pressure_dimensionless = geo_units.registry.non_dimensionalise(pressure_bc)
            self.add_dirichlet_bc(pressure_dimensionless, boundary, component=2)

class UnitAwareMathematicalExpression:
    """Mathematical expression that carries units through operations"""
    
    def __init__(self, name: str, sympy_expr, units: pint.Unit):
        self.name = name
        self.sym = sympy_expr
        self._units = units
    
    @property
    def units(self) -> pint.Unit:
        return self._units
        
    def subs(self, substitutions):
        """Substitution preserving units"""
        new_expr = self.sym.subs(substitutions)
        return UnitAwareMathematicalExpression(f"{self.name}_subs", new_expr, self._units)
    
    def diff(self, variable):
        """Differentiation changing units appropriately"""
        if hasattr(variable, 'units'):
            result_units = self._units / variable.units
        else:
            # Assume spatial coordinate with length units
            result_units = self._units / geo_units.registry.meter
            
        result_expr = self.sym.diff(variable)
        return UnitAwareMathematicalExpression(
            f"∂{self.name}/∂{variable}", result_expr, result_units
        )
```

### 3. Enhanced Problem Scaling
```python
# underworld3/scaling/enhanced_scaling.py
from typing import Dict, List, Optional, Union
import pint
import numpy as np

class AutomaticProblemScaling:
    """
    Automatically detect and set up problem-specific scaling
    based on mesh, materials, and boundary conditions.
    """
    
    def __init__(self, mesh=None):
        self.mesh = mesh
        self._detected_scales = {}
        
    def detect_scales_from_setup(self, 
                                materials: Optional[List] = None,
                                boundary_conditions: Optional[Dict] = None,
                                **explicit_scales):
        """
        Automatically detect characteristic scales from problem setup
        """
        
        # Explicit scales take priority
        self._detected_scales.update(explicit_scales)
        
        # Mesh-based length scale
        if self.mesh and not self._detected_scales.get('length'):
            self._detect_length_scale()
            
        # Material-based scales
        if materials:
            self._detect_material_scales(materials)
            
        # Boundary condition scales  
        if boundary_conditions:
            self._detect_bc_scales(boundary_conditions)
            
        # Derive missing scales
        self._derive_dependent_scales()
        
        # Set global scaling coefficients
        self._apply_scaling()
        
        return self._detected_scales
    
    def _detect_length_scale(self):
        """Detect characteristic length from mesh geometry"""
        if hasattr(self.mesh, 'bounding_box'):
            # Use largest dimension of domain
            bbox = self.mesh.bounding_box
            domain_dims = bbox.max - bbox.min
            char_length = np.max(domain_dims)
            
            # Get mesh coordinate units if available
            coord_units = getattr(self.mesh, 'coordinate_units', geo_units.registry.meter)
            
            self._detected_scales['length'] = char_length * coord_units
        
    def _detect_material_scales(self, materials: List):
        """Extract characteristic scales from material properties"""
        
        viscosities = []
        densities = []
        thermal_conductivities = []
        
        for material in materials:
            # Extract material properties (assuming they have units)
            if hasattr(material, 'viscosity') and material.viscosity:
                if isinstance(material.viscosity, pint.Quantity):
                    viscosities.append(material.viscosity)
                    
            if hasattr(material, 'density') and material.density:  
                if isinstance(material.density, pint.Quantity):
                    densities.append(material.density)
                    
            if hasattr(material, 'thermal_conductivity') and material.thermal_conductivity:
                if isinstance(material.thermal_conductivity, pint.Quantity):
                    thermal_conductivities.append(material.thermal_conductivity)
        
        # Compute representative scales (geometric mean for viscosity, arithmetic mean for density)
        if viscosities and not self._detected_scales.get('viscosity'):
            visc_values = [v.to_base_units().magnitude for v in viscosities]
            mean_visc_value = np.exp(np.mean(np.log(visc_values)))  # Geometric mean
            mean_visc_units = viscosities[0].to_base_units().units
            self._detected_scales['viscosity'] = mean_visc_value * mean_visc_units
            
        if densities and not self._detected_scales.get('density'):
            dens_values = [d.to_base_units().magnitude for d in densities]
            mean_dens_value = np.mean(dens_values)  # Arithmetic mean
            mean_dens_units = densities[0].to_base_units().units
            self._detected_scales['density'] = mean_dens_value * mean_dens_units
    
    def _detect_bc_scales(self, boundary_conditions: Dict):
        """Extract scales from boundary conditions"""
        
        velocity_bcs = []
        temperature_bcs = []
        pressure_bcs = []
        
        # Extract boundary condition values
        for field_name, field_bcs in boundary_conditions.items():
            field_type = field_name.lower()
            
            for boundary, bc_info in field_bcs.items():
                bc_value = bc_info.get('value')
                
                if isinstance(bc_value, pint.Quantity):
                    if 'velocity' in field_type or 'flow' in field_type:
                        velocity_bcs.append(bc_value)
                    elif 'temperature' in field_type or 'thermal' in field_type:
                        temperature_bcs.append(bc_value)
                    elif 'pressure' in field_type or 'stress' in field_type:
                        pressure_bcs.append(bc_value)
        
        # Set velocity scale from maximum boundary velocity
        if velocity_bcs and not self._detected_scales.get('velocity'):
            max_vel_magnitude = max(v.to_base_units().magnitude for v in velocity_bcs)
            vel_units = velocity_bcs[0].to_base_units().units
            self._detected_scales['velocity'] = max_vel_magnitude * vel_units
        
        # Set temperature scale from temperature range  
        if temperature_bcs and not self._detected_scales.get('temperature'):
            temp_magnitudes = [t.to_base_units().magnitude for t in temperature_bcs]
            temp_range = max(temp_magnitudes) - min(temp_magnitudes)
            temp_units = temperature_bcs[0].to_base_units().units
            self._detected_scales['temperature'] = temp_range * temp_units
            
        # Set pressure scale
        if pressure_bcs and not self._detected_scales.get('pressure'):
            max_pressure_magnitude = max(p.to_base_units().magnitude for p in pressure_bcs)
            pressure_units = pressure_bcs[0].to_base_units().units
            self._detected_scales['pressure'] = max_pressure_magnitude * pressure_units
    
    def _derive_dependent_scales(self):
        """Compute derived scales from fundamental scales"""
        
        # Time scale from advection: length/velocity
        if ('length' in self._detected_scales and 
            'velocity' in self._detected_scales and 
            not self._detected_scales.get('time')):
            
            time_scale = (self._detected_scales['length'] / 
                         self._detected_scales['velocity']).to_base_units()
            self._detected_scales['time'] = time_scale
        
        # Mass scale from viscous scaling: viscosity * length * time  
        if ('viscosity' in self._detected_scales and
            'length' in self._detected_scales and
            'time' in self._detected_scales and
            not self._detected_scales.get('mass')):
            
            mass_scale = (self._detected_scales['viscosity'] * 
                         self._detected_scales['length'] * 
                         self._detected_scales['time']).to_base_units()
            self._detected_scales['mass'] = mass_scale
            
        # Pressure scale from viscous flow: viscosity * velocity / length
        if ('viscosity' in self._detected_scales and
            'velocity' in self._detected_scales and  
            'length' in self._detected_scales and
            not self._detected_scales.get('pressure')):
            
            pressure_scale = (self._detected_scales['viscosity'] * 
                             self._detected_scales['velocity'] / 
                             self._detected_scales['length']).to_base_units()
            self._detected_scales['pressure'] = pressure_scale
    
    def _apply_scaling(self):
        """Apply detected scales to global scaling system"""
        
        scaling_coefficients = uw.scaling.get_coefficients()
        
        # Map detected scales to scaling coefficients  
        scale_mapping = {
            'length': '[length]',
            'time': '[time]', 
            'mass': '[mass]',
            'temperature': '[temperature]'
        }
        
        for scale_name, coeff_name in scale_mapping.items():
            if scale_name in self._detected_scales:
                scaling_coefficients[coeff_name] = self._detected_scales[scale_name]
                
        print("Applied automatic scaling:")
        for name, value in self._detected_scales.items():
            print(f"  {name}: {value}")
```

---

## **Optional SymPy Compatibility Implementation**

### Basic SymPy Units Module
```python
# underworld3/scaling/sympy_units.py (Optional alternative)
"""
Basic SymPy-based units for dependency-constrained environments.

WARNING: This module provides basic dimensional analysis using SymPy
but lacks the comprehensive error checking and unit database of Pint.
Use only when Pint cannot be used. For full functionality, use the
enhanced Pint system in underworld3.units.

Credits: Developed by Ben Knight as an alternative approach.
"""

import sympy as sp
from sympy.physics.units import (
    meter, second, kilogram, kelvin, # Basic SI units
    length, time, mass, temperature  # Dimension objects
)
from types import SimpleNamespace
import numpy as np
from typing import Union, Dict

class SymPyUnitsRegistry:
    """
    Basic units registry using SymPy physics units.
    Provides minimal dimensional analysis without external dependencies.
    """
    
    def __init__(self):
        # Create unit namespace
        self.units = SimpleNamespace()
        
        # Basic SI units
        self.units.meter = meter
        self.units.second = second  
        self.units.kilogram = kilogram
        self.units.kelvin = kelvin
        
        # Common derived units (manual definitions)
        self.units.centimeter = meter / 100
        self.units.kilometer = 1000 * meter
        self.units.millimeter = meter / 1000
        
        # Time units
        self.units.minute = 60 * second
        self.units.hour = 3600 * second
        self.units.day = 86400 * second
        self.units.year = 365.25 * 24 * 3600 * second
        
        # Geophysical time units
        self.units.Ma = 1e6 * self.units.year  # Megayear
        self.units.ka = 1e3 * self.units.year  # Kiloyear
        self.units.Ga = 1e9 * self.units.year  # Gigayear
        
        # Pressure units
        self.units.pascal = kilogram / (meter * second**2)
        self.units.bar = 1e5 * self.units.pascal
        self.units.GPa = 1e9 * self.units.pascal
        
        # Velocity units (geophysical)
        self.units.cm_per_year = self.units.centimeter / self.units.year
        self.units.mm_per_year = self.units.millimeter / self.units.year
        
        # Scaling coefficients storage  
        self._coefficients = {}
        
    def get_coefficients(self) -> Dict[str, sp.Expr]:
        """Get scaling coefficients (SymPy expressions with units)"""
        if not self._coefficients:
            # Default scaling coefficients
            self._coefficients = {
                '[length]': 1.0 * self.units.meter,
                '[time]': 1.0 * self.units.year,
                '[mass]': 1.0 * self.units.kilogram,
                '[temperature]': 1.0 * self.units.kelvin,
            }
        return self._coefficients
    
    def set_coefficients(self, **scales):
        """Set scaling coefficients"""
        for name, value in scales.items():
            if not name.startswith('['):
                name = f'[{name}]'
            self._coefficients[name] = value

def non_dimensionalise_sympy(quantity: sp.Expr) -> float:
    """
    Non-dimensionalise a SymPy quantity using scaling coefficients.
    
    WARNING: This function provides basic scaling but does not validate
    dimensional consistency like Pint. Users must manually ensure correct units.
    
    Parameters
    ----------
    quantity : sympy expression with units
        Quantity to non-dimensionalise
        
    Returns  
    -------
    float
        Non-dimensional value
        
    Example
    -------
    >>> velocity = 1.0 * u_sp.centimeter / u_sp.year  
    >>> nd_velocity = non_dimensionalise_sympy(velocity)
    """
    
    # Get scaling coefficients
    coefficients = _sympy_registry.get_coefficients()
    
    # Extract dimensions from quantity (basic implementation)
    quantity_dims = _extract_dimensions_sympy(quantity)
    
    # Build scaling factor 
    scaling_factor = sp.S(1)
    for dim_name, power in quantity_dims.items():
        if dim_name in coefficients:
            scaling_factor *= coefficients[dim_name]**power
    
    # Perform scaling (extract numerical value)
    if scaling_factor != 1:
        scaled_quantity = quantity / scaling_factor
        # Convert to float (assumes all units cancel out)
        return float(scaled_quantity.subs(_unit_substitutions()))
    else:
        # No scaling needed
        return float(quantity.subs(_unit_substitutions()))

def dimensionalise_sympy(value: float, target_units: sp.Expr) -> sp.Expr:
    """
    Dimensionalise a value with target units.
    
    WARNING: No automatic unit validation - user must ensure correctness.
    
    Parameters
    ----------  
    value : float
        Non-dimensional value
    target_units : sympy expression
        Units to apply
        
    Returns
    -------
    sympy expression
        Dimensional quantity
    """
    
    coefficients = _sympy_registry.get_coefficients()
    
    # Extract dimensions from target units
    target_dims = _extract_dimensions_sympy(target_units)
    
    # Build scaling factor
    scaling_factor = sp.S(1)
    for dim_name, power in target_dims.items():
        if dim_name in coefficients:
            scaling_factor *= coefficients[dim_name]**power
    
    return value * scaling_factor

def _extract_dimensions_sympy(expr: sp.Expr) -> Dict[str, int]:
    """
    Extract dimensions from SymPy expression (basic implementation).
    
    WARNING: This is a simplified implementation that may not handle
    all cases correctly. Use with caution.
    """
    
    # This is a simplified implementation - real version would need
    # more sophisticated dimensional analysis
    dims = {}
    
    # Check for basic dimensions in expression
    if meter in expr.free_symbols:
        dims['[length]'] = _get_power_of_symbol(expr, meter)
    if second in expr.free_symbols:
        dims['[time]'] = _get_power_of_symbol(expr, second)  
    if kilogram in expr.free_symbols:
        dims['[mass]'] = _get_power_of_symbol(expr, kilogram)
    if kelvin in expr.free_symbols:
        dims['[temperature]'] = _get_power_of_symbol(expr, kelvin)
        
    return dims

def _get_power_of_symbol(expr: sp.Expr, symbol: sp.Symbol) -> int:
    """Get the power of a symbol in expression"""
    # Simplified - real implementation needs more sophisticated analysis
    if symbol in expr.free_symbols:
        # Basic case: assume linear occurrence
        return 1  
    return 0

def _unit_substitutions() -> Dict[sp.Symbol, float]:
    """Substitutions to convert units to numerical values for computation"""
    return {
        meter: 1.0,
        second: 1.0, 
        kilogram: 1.0,
        kelvin: 1.0
    }

# Global registry
_sympy_registry = SymPyUnitsRegistry()
u_sp = _sympy_registry.units  # User interface

# Example usage functions
def create_sympy_scaling_example():
    """
    Example of setting up problem scaling with SymPy units.
    
    WARNING: This is a basic example. Users must manually ensure
    dimensional correctness throughout.
    """
    
    # Define problem characteristics  
    velocity = 1.0 * u_sp.cm_per_year
    model_height = 660.0 * u_sp.kilometer
    viscosity = 1e22 * u_sp.pascal * u_sp.second
    
    # Compute derived scales
    time_scale = model_height / velocity
    mass_scale = viscosity * model_height * time_scale
    
    # Set scaling coefficients
    _sympy_registry.set_coefficients(
        length=model_height,
        time=time_scale, 
        mass=mass_scale
    )
    
    # Example non-dimensionalisation
    density = 3.3e3 * u_sp.kilogram / u_sp.meter**3
    nd_density = non_dimensionalise_sympy(density)
    
    return {
        'dimensional_density': density,
        'non_dimensional_density': nd_density,
        'scaling_coefficients': _sympy_registry.get_coefficients()
    }

def sympy_units_limitations():
    """
    Documentation of SymPy units limitations compared to Pint.
    
    This function exists to clearly document what cannot be done
    with the SymPy approach, helping users make informed decisions.
    """
    
    limitations = {
        'no_automatic_error_checking': """
        SymPy units do not automatically check dimensional consistency.
        Example that would fail with Pint but succeeds (incorrectly) with SymPy:
        
        length = 5 * u_sp.meter
        time = 3 * u_sp.second  
        wrong = length + time  # Creates invalid expression silently!
        """,
        
        'no_unit_conversion_database': """
        SymPy requires manual definition of all unit conversions.
        Example that works with Pint but fails with SymPy:
        
        # Pint: automatic
        pressure_psi = 1000 * u.psi
        pressure_pa = pressure_psi.to('pascal')  
        
        # SymPy: manual conversion factor required
        u_sp.psi = 6894.757 * u_sp.pascal  # Must define manually!
        pressure_pa = pressure_psi * 6894.757  
        """,
        
        'no_temperature_handling': """
        SymPy units cannot distinguish between temperature and temperature difference.
        This is a critical physics error that can invalidate results:
        
        # Pint: handles correctly
        temp_c = 20 * u.celsius
        temp_k = temp_c.to('kelvin')  # = 293.15 K (correct)
        
        # SymPy: cannot handle temperature scales
        # This leads to physics errors in thermal calculations
        """,
        
        'limited_dimensional_analysis': """
        SymPy dimensional analysis is basic and error-prone.
        Complex derived units require manual handling:
        
        # Pint: automatic handling of complex units
        viscosity = 1e22 * u.pascal * u.second
        shear_rate = velocity_gradient  # [1/time] 
        stress = viscosity * shear_rate  # Automatic unit algebra
        
        # SymPy: manual tracking required for each operation
        # Easy to lose track and introduce errors
        """,
        
        'no_function_validation': """
        SymPy cannot validate function arguments have correct dimensions:
        
        # Pint: automatic validation
        @u.check('[length]', '[velocity]', '[pressure]')
        def compute_reynolds_number(length, velocity, pressure):
            # Automatic validation of argument units
            pass
            
        # SymPy: no validation possible
        # Functions can be called with wrong units silently
        """
    }
    
    return limitations
```

---

## **Final Recommendations**

### **Primary Recommendation: Enhanced Pint System**

Based on comprehensive analysis, **Enhanced Pint should be the primary units system** for Underworld3 because:

1. **Scientific Correctness**: Automatic dimensional validation prevents physics errors
2. **User Safety**: Comprehensive error checking catches mistakes before they propagate
3. **Maintenance**: Mature, well-tested library reduces long-term maintenance burden
4. **Completeness**: Full feature set supports complex geophysical applications
5. **Industry Standard**: Consistent with scientific Python ecosystem

### **Secondary Option: Basic SymPy Compatibility**

Provide **optional SymPy units** for specialized cases:
- Dependency-constrained environments
- Users requiring lightweight scaling only
- Educational/demonstration purposes  
- Transition path for existing SymPy-heavy workflows

### **Implementation Priority**

**Phase 1** (High Priority): Enhanced Pint System
- Complete unit-aware mathematical objects
- Automatic problem scaling detection
- Comprehensive error validation
- Integration with existing UW3 systems

**Phase 2** (Medium Priority): SymPy Compatibility Layer  
- Basic SymPy units module
- Manual scaling functions
- Clear documentation of limitations
- Migration utilities between systems

**Phase 3** (Low Priority): Advanced Features
- Context-specific unit systems
- Advanced scaling optimization
- Performance optimization
- Comprehensive examples and tutorials

### **Key Decision Factors for Ben's Review:**

1. **Dependency Philosophy**: Is adding Pint acceptable for the enhanced functionality it provides?

2. **User Safety vs Simplicity**: Do we prioritize preventing dimensional errors (Pint) or simplicity (SymPy)?

3. **Maintenance Resources**: Do we have resources to maintain custom SymPy dimensional analysis vs using mature Pint?

4. **Scientific Correctness**: How important is automatic validation of physical correctness in user models?

5. **Ecosystem Integration**: Should UW3 follow SciPy ecosystem standards (Pint) or remain maximally self-contained?

The analysis shows that **both approaches are technically feasible**, but they make very different tradeoffs between **functionality and simplicity**. The recommendation favors **scientific robustness and user safety** while providing an **optional lightweight alternative** for specialized needs.

---

*Document Version: 2.0 - Comprehensive Pint vs SymPy Analysis*  
*Updated: 2025-09-26*  
*Contributors: Original plan + Ben Knight's SymPy approach analysis*  
*Status: Ready for Technical Review*