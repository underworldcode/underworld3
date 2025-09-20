# Material Properties Plan: Physical Constants Handbook System

## Project Overview

This plan outlines the implementation of a comprehensive material properties system that acts as a "Physical Constants Handbook" for Underworld3. The system provides a registry of material parameters with units, symbols, validation, and seamless integration with constitutive models.

## Current State Analysis

### Problems with Current Material Handling

1. **No Central Material Concept**: Materials are just collections of isolated parameters
2. **Parameter Duplication**: Each constitutive model redefines its own parameters  
3. **No Unit Validation**: Parameters can have inconsistent or missing units
4. **Spelling Errors**: String-based parameter names prone to typos
5. **No Physical Validation**: No checking against typical ranges
6. **Poor Discoverability**: No way to explore available material properties

### Current Parameter Patterns

```python
# Current awkward patterns
viscous_model = ViscousFlowModel(unknowns)
viscous_model.material_properties.viscosity = 1e21  # No units, no validation

# Different models have different parameter names
plastic_model = ViscoPlasticFlowModel(unknowns)
plastic_model.material_properties.yield_stress = 100e6  # Inconsistent interface

# No connection between related models
thermal_model = DiffusionModel(unknowns) 
thermal_model.material_properties.diffusivity = 1e-6  # Separate property definition
```

## Proposed Architecture

### 1. Parameter Registry - The "Physical Constants Handbook"

```python
# underworld3/materials/parameter_registry.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Union
import pint
from ..units import units_registry

class ParameterType(Enum):
    """
    Master enumeration of all material parameters in Underworld3.
    Acts as the vocabulary for material properties across all physics systems.
    """
    
    # Core mechanical properties
    DENSITY = "density"
    VISCOSITY = "viscosity"
    YIELD_STRESS = "yield_stress"
    ELASTIC_MODULUS = "elastic_modulus"
    BULK_MODULUS = "bulk_modulus"
    SHEAR_MODULUS = "shear_modulus"
    POISSON_RATIO = "poisson_ratio"
    
    # Plastic behavior
    COHESION = "cohesion" 
    FRICTION_ANGLE = "friction_angle"
    FRICTION_COEFFICIENT = "friction_coefficient"
    DILATANCY_ANGLE = "dilatancy_angle"
    
    # Thermal properties
    THERMAL_DIFFUSIVITY = "thermal_diffusivity"
    THERMAL_CONDUCTIVITY = "thermal_conductivity" 
    THERMAL_EXPANSION = "thermal_expansion"
    SPECIFIC_HEAT = "specific_heat"
    HEAT_CAPACITY = "heat_capacity"
    LATENT_HEAT_FUSION = "latent_heat_fusion"
    
    # Chemical/compositional
    CHEMICAL_DIFFUSIVITY = "chemical_diffusivity"
    PARTITION_COEFFICIENT = "partition_coefficient"
    REACTION_RATE = "reaction_rate"
    SOLUBILITY = "solubility"
    
    # Phase transitions
    MELTING_TEMPERATURE = "melting_temperature"
    SOLIDUS_TEMPERATURE = "solidus_temperature" 
    LIQUIDUS_TEMPERATURE = "liquidus_temperature"
    CLAPEYRON_SLOPE = "clapeyron_slope"
    
    # Electromagnetic (future)
    ELECTRICAL_CONDUCTIVITY = "electrical_conductivity"
    MAGNETIC_PERMEABILITY = "magnetic_permeability"
    DIELECTRIC_CONSTANT = "dielectric_constant"

@dataclass
class ParameterDefinition:
    """
    Complete definition of a material parameter - like an entry in a
    physical constants handbook.
    """
    name: str                          # Human-readable name
    symbol: str                        # LaTeX mathematical symbol
    units: pint.Unit                   # Canonical SI units
    description: str                   # Physical meaning and context
    typical_range: Optional[Tuple[float, float]] = None  # (min, max) typical values  
    dimensionality: str = ""           # Physical dimension string
    aliases: List[str] = None          # Alternative names
    references: List[str] = None       # Literature references
    related_parameters: List[ParameterType] = None  # Physically related parameters
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.references is None:
            self.references = []
        if self.related_parameters is None:
            self.related_parameters = []

class MaterialParameterRegistry:
    """
    Registry of all material parameters - the 'Physical Constants Handbook'.
    Central source of truth for parameter definitions, units, and validation.
    """
    
    def __init__(self):
        self._parameters = {}
        self._aliases = {}  # Map aliases to primary parameter types
        self._initialize_standard_parameters()
    
    def register_parameter(self, param_type: ParameterType, definition: ParameterDefinition):
        """Register a parameter definition"""
        self._parameters[param_type] = definition
        
        # Register aliases
        for alias in definition.aliases:
            self._aliases[alias] = param_type
    
    def get_definition(self, param_key: Union[ParameterType, str]) -> ParameterDefinition:
        """Get parameter definition by type or alias"""
        if isinstance(param_key, str):
            # Look up by alias first
            if param_key in self._aliases:
                param_type = self._aliases[param_key]
            else:
                # Try to find by enum value
                try:
                    param_type = ParameterType(param_key)
                except ValueError:
                    raise ValueError(f"Unknown parameter: {param_key}")
        else:
            param_type = param_key
            
        if param_type not in self._parameters:
            raise ValueError(f"Parameter {param_type} not registered")
            
        return self._parameters[param_type]
    
    def validate_value(self, param_type: ParameterType, value: pint.Quantity) -> bool:
        """Validate parameter value against definition"""
        definition = self.get_definition(param_type)
        
        # Check units
        if value.dimensionality != definition.units.dimensionality:
            raise pint.DimensionalityError(
                value.units, definition.units,
                extra_msg=f"Parameter {param_type.value} expects {definition.units.dimensionality}"
            )
        
        # Check typical range
        if definition.typical_range:
            magnitude = value.to(definition.units).magnitude
            min_val, max_val = definition.typical_range
            if not (min_val <= magnitude <= max_val):
                import warnings
                warnings.warn(
                    f"{param_type.value} = {value} outside typical range "
                    f"[{min_val}, {max_val}] {definition.units}",
                    UserWarning
                )
        
        return True
    
    def _initialize_standard_parameters(self):
        """Initialize registry with standard geophysical parameters"""
        
        # Mechanical properties
        self.register_parameter(ParameterType.DENSITY, ParameterDefinition(
            name="Density",
            symbol=r"\\rho",
            units=units_registry.parse_units("kg/m^3"),
            description="Mass density - fundamental property determining inertia and gravitational effects",
            typical_range=(1000, 15000),
            aliases=["rho", "mass_density"],
            references=["Turcotte & Schubert (2002)", "Dziewonski & Anderson (1981)"]
        ))
        
        self.register_parameter(ParameterType.VISCOSITY, ParameterDefinition(
            name="Dynamic Viscosity",
            symbol=r"\\eta",
            units=units_registry.parse_units("Pa*s"),
            description="Dynamic viscosity - resistance to shear deformation in viscous flow",
            typical_range=(1e18, 1e25),
            aliases=["eta", "dynamic_viscosity", "shear_viscosity"],
            references=["Ranalli (1995)", "Karato (2008)"]
        ))
        
        self.register_parameter(ParameterType.YIELD_STRESS, ParameterDefinition(
            name="Yield Stress",
            symbol=r"\\tau_y",
            units=units_registry.parse_units("Pa"),
            description="Yield stress - critical stress for onset of plastic deformation",
            typical_range=(1e6, 1e9),
            aliases=["tau_y", "yield_strength", "cohesion_plastic"],
            references=["Byerlee (1978)", "Kohlstedt et al. (1995)"]
        ))
        
        self.register_parameter(ParameterType.ELASTIC_MODULUS, ParameterDefinition(
            name="Shear Modulus",
            symbol=r"G",
            units=units_registry.parse_units("Pa"),
            description="Shear modulus - elastic resistance to shear deformation",
            typical_range=(1e9, 1e12),
            aliases=["G", "shear_modulus", "rigidity_modulus"],
            references=["Anderson (1989)", "Dziewonski & Anderson (1981)"]
        ))
        
        # Thermal properties
        self.register_parameter(ParameterType.THERMAL_DIFFUSIVITY, ParameterDefinition(
            name="Thermal Diffusivity",
            symbol=r"\\kappa",
            units=units_registry.parse_units("m^2/s"),
            description="Thermal diffusivity - rate of temperature equilibration by conduction",
            typical_range=(1e-7, 1e-5),
            aliases=["kappa", "thermal_diff"],
            references=["Hofmeister (1999)", "Clauser & Huenges (1995)"],
            related_parameters=[ParameterType.THERMAL_CONDUCTIVITY, ParameterType.HEAT_CAPACITY, ParameterType.DENSITY]
        ))
        
        self.register_parameter(ParameterType.THERMAL_CONDUCTIVITY, ParameterDefinition(
            name="Thermal Conductivity",
            symbol=r"k",
            units=units_registry.parse_units("W/(m*K)"),
            description="Thermal conductivity - ability to conduct heat",
            typical_range=(1, 10),
            aliases=["k_thermal", "conductivity"],
            references=["Hofmeister (1999)", "Clauser & Huenges (1995)"]
        ))
        
        self.register_parameter(ParameterType.THERMAL_EXPANSION, ParameterDefinition(
            name="Thermal Expansion Coefficient",
            symbol=r"\\alpha",
            units=units_registry.parse_units("1/K"),
            description="Coefficient of thermal expansion - volume change per temperature change",
            typical_range=(1e-6, 5e-5),
            aliases=["alpha", "expansion_coeff", "thermal_exp"],
            references=["Boehler (2000)", "Anderson (1995)"]
        ))

# Global registry instance
PARAMETER_REGISTRY = MaterialParameterRegistry()
```

### 2. Material Class with Expression Integration

```python
# underworld3/materials/material.py
from typing import Dict, Optional, Union, Any
import pint
from ..mathematical_objects.unit_aware import UnitAwareMathematicalObject
from .parameter_registry import PARAMETER_REGISTRY, ParameterType

class MaterialProperty(UnitAwareMathematicalObject):
    """
    Material property that integrates with mathematical expression system.
    Combines parameter definition, value, and mathematical representation.
    """
    
    def __init__(self, material_name: str, param_type: ParameterType, 
                 value: Union[float, pint.Quantity], 
                 expression=None, description: str = ""):
        
        param_def = PARAMETER_REGISTRY.get_definition(param_type)
        
        # Create symbol with material subscript
        symbol = f"{param_def.symbol}_{{{material_name}}}"
        full_description = description or f"{param_def.description} of {material_name}"
        
        # Handle value and units
        if isinstance(value, pint.Quantity):
            units = value.units
            dimensional_value = value.magnitude
        else:
            units = param_def.units
            dimensional_value = value
        
        # Initialize as mathematical object
        super().__init__(symbol, dimensional_value, units, full_description)
        
        self.material_name = material_name
        self.param_type = param_type
        self.param_definition = param_def
        
        # Validate value
        quantity_value = units_registry.Quantity(dimensional_value, units)
        PARAMETER_REGISTRY.validate_value(param_type, quantity_value)
        
        # Store expression if provided (for temperature-dependent properties etc.)
        self._expression = expression
    
    @property
    def references(self) -> List[str]:
        """Literature references for this parameter"""
        return self.param_definition.references
    
    @property
    def typical_range(self) -> Optional[Tuple[float, float]]:
        """Typical range for this parameter"""
        return self.param_definition.typical_range
    
    def set_expression(self, expression):
        """Set mathematical expression for temperature/pressure dependence"""
        self._expression = expression
        # Update mathematical form to include expression
        self._mathematical_form = None  # Force recreation
    
    def _create_mathematical_form(self):
        """Create mathematical form, including expression if present"""
        if self._expression:
            return self._expression
        else:
            return super()._create_mathematical_form()

class Material:
    """
    Material definition using the expression-centric parameter system.
    Acts as a collection of MaterialProperty objects.
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._properties = {}  # ParameterType -> MaterialProperty
        
        # For convenience, create property accessors
        self._create_property_accessors()
    
    def set_property(self, param_type: ParameterType, 
                    value: Union[float, pint.Quantity], 
                    expression=None, **kwargs) -> MaterialProperty:
        """Set material property with validation"""
        
        prop = MaterialProperty(
            material_name=self.name,
            param_type=param_type,
            value=value,
            expression=expression,
            **kwargs
        )
        
        self._properties[param_type] = prop
        
        # Set as attribute for convenience
        attr_name = param_type.value
        setattr(self, attr_name, prop)
        
        return prop
    
    def get_property(self, param_type: ParameterType) -> Optional[MaterialProperty]:
        """Get material property"""
        return self._properties.get(param_type)
    
    def has_property(self, param_type: ParameterType) -> bool:
        """Check if property is defined"""
        return param_type in self._properties
    
    def get_defined_properties(self) -> Dict[ParameterType, MaterialProperty]:
        """Get all defined properties"""
        return self._properties.copy()
    
    def create_constitutive_model(self, model_class, unknowns, **overrides):
        """
        Factory method to create constitutive model using this material.
        Validates that material provides required parameters.
        """
        # Get model requirements
        requirements = model_class.get_parameter_requirements()
        
        # Check for missing required parameters
        missing = []
        for param_type, spec in requirements.items():
            if spec.required and not self.has_property(param_type):
                if spec.default_value is None:
                    missing.append(param_type.value)
        
        if missing:
            raise ValueError(
                f"Material '{self.name}' missing required parameters "
                f"for {model_class.__name__}: {missing}"
            )
        
        # Create model
        model = model_class(unknowns)
        
        # Bind material properties
        for param_type, spec in requirements.items():
            if self.has_property(param_type):
                # Use material property
                prop = self.get_property(param_type)
                model.bind_parameter(param_type, prop)
            elif not spec.required and spec.default_value is not None:
                # Use default value
                default_prop = MaterialProperty(
                    self.name, param_type, spec.default_value
                )
                model.bind_parameter(param_type, default_prop)
        
        # Apply overrides
        for param_name, value in overrides.items():
            try:
                param_type = ParameterType(param_name)
                override_prop = MaterialProperty(self.name, param_type, value)
                model.bind_parameter(param_type, override_prop)
            except ValueError:
                raise ValueError(f"Unknown parameter override: {param_name}")
        
        return model
    
    def _create_property_accessors(self):
        """Create convenience properties for all parameter types"""
        for param_type in ParameterType:
            attr_name = param_type.value
            
            # Create getter
            def make_getter(pt):
                def getter(self):
                    return self._properties.get(pt)
                return getter
            
            # Create setter  
            def make_setter(pt):
                def setter(self, value):
                    self.set_property(pt, value)
                return setter
            
            # Set property
            setattr(self.__class__, attr_name, 
                   property(make_getter(param_type), make_setter(param_type)))

class MaterialDatabase:
    """
    Database of standard geophysical materials.
    Like reference tables in a physical constants handbook.
    """
    
    def __init__(self):
        self._materials = {}
        self._initialize_standard_materials()
    
    def register_material(self, material: Material):
        """Add material to database"""
        self._materials[material.name.lower()] = material
    
    def get_material(self, name: str) -> Material:
        """Get material by name (returns copy)"""
        key = name.lower()
        if key not in self._materials:
            available = list(self._materials.keys())
            raise ValueError(f"Unknown material: {name}. Available: {available}")
        
        # Return copy to prevent modification of database entries
        return self._copy_material(self._materials[key])
    
    def list_materials(self) -> List[str]:
        """List all available materials"""
        return [mat.name for mat in self._materials.values()]
    
    def search_materials(self, **criteria) -> List[Material]:
        """Search materials by property criteria"""
        results = []
        for material in self._materials.values():
            match = True
            for param_name, value_range in criteria.items():
                try:
                    param_type = ParameterType(param_name)
                    if not material.has_property(param_type):
                        match = False
                        break
                    
                    prop = material.get_property(param_type)
                    value = prop.dimensional_value.to(prop.param_definition.units).magnitude
                    
                    if isinstance(value_range, tuple):
                        min_val, max_val = value_range
                        if not (min_val <= value <= max_val):
                            match = False
                            break
                except ValueError:
                    match = False
                    break
            
            if match:
                results.append(self._copy_material(material))
        
        return results
    
    def _copy_material(self, material: Material) -> Material:
        """Create independent copy of material"""
        copy = Material(material.name, material.description)
        for param_type, prop in material._properties.items():
            copy.set_property(param_type, prop.dimensional_value, prop._expression)
        return copy
    
    def _initialize_standard_materials(self):
        """Initialize database with standard geophysical materials"""
        
        # Upper mantle olivine
        olivine = Material("Olivine", "Upper mantle olivine-dominated rock")
        olivine.set_property(ParameterType.DENSITY, 3300, units="kg/m^3")
        olivine.set_property(ParameterType.VISCOSITY, 1e21, units="Pa*s")
        olivine.set_property(ParameterType.THERMAL_DIFFUSIVITY, 1e-6, units="m^2/s")
        olivine.set_property(ParameterType.THERMAL_CONDUCTIVITY, 4.0, units="W/(m*K)")
        olivine.set_property(ParameterType.THERMAL_EXPANSION, 3e-5, units="1/K")
        olivine.set_property(ParameterType.ELASTIC_MODULUS, 80e9, units="Pa")
        self.register_material(olivine)
        
        # Oceanic basalt
        basalt = Material("Basalt", "Oceanic crustal basalt")
        basalt.set_property(ParameterType.DENSITY, 2900, units="kg/m^3")
        basalt.set_property(ParameterType.VISCOSITY, 1e22, units="Pa*s")
        basalt.set_property(ParameterType.YIELD_STRESS, 100e6, units="Pa")
        basalt.set_property(ParameterType.THERMAL_DIFFUSIVITY, 0.7e-6, units="m^2/s")
        basalt.set_property(ParameterType.THERMAL_CONDUCTIVITY, 2.5, units="W/(m*K)")
        self.register_material(basalt)
        
        # Continental granite
        granite = Material("Granite", "Continental crustal granite")
        granite.set_property(ParameterType.DENSITY, 2700, units="kg/m^3")
        granite.set_property(ParameterType.VISCOSITY, 1e23, units="Pa*s")
        granite.set_property(ParameterType.YIELD_STRESS, 200e6, units="Pa")
        granite.set_property(ParameterType.THERMAL_DIFFUSIVITY, 1.2e-6, units="m^2/s")
        granite.set_property(ParameterType.THERMAL_CONDUCTIVITY, 3.0, units="W/(m*K)")
        self.register_material(granite)

# Global database instance
MATERIAL_DATABASE = MaterialDatabase()
```

### 3. Enhanced Constitutive Model Integration

```python
# underworld3/constitutive_models/parameter_aware.py
from dataclasses import dataclass
from typing import Dict, Optional, Any
from ..materials.parameter_registry import ParameterType
from ..materials.material import MaterialProperty

@dataclass
class ParameterSpec:
    """Specification for a parameter required by a constitutive model"""
    param_type: ParameterType
    required: bool = True
    default_value: Any = None
    description: str = ""

class ParameterAwareConstitutiveModel:
    """
    Base class for constitutive models that advertise their parameter requirements
    and bind to Material objects.
    """
    
    @classmethod
    def get_parameter_requirements(cls) -> Dict[ParameterType, ParameterSpec]:
        """Each model advertises its parameter requirements"""
        return {}
    
    def __init__(self, unknowns):
        self.unknowns = unknowns
        self._bound_parameters = {}
        
    def bind_parameter(self, param_type: ParameterType, property_obj: MaterialProperty):
        """Bind a material property to this model"""
        # Store the property object for mathematical expressions
        self._bound_parameters[param_type] = property_obj
        
        # Set as attribute for direct access
        attr_name = param_type.value
        setattr(self, attr_name, property_obj)
    
    def get_bound_parameter(self, param_type: ParameterType) -> Optional[MaterialProperty]:
        """Get bound parameter"""
        return self._bound_parameters.get(param_type)

class ViscousFlowModel(ParameterAwareConstitutiveModel):
    """Viscous flow model with parameter requirements"""
    
    @classmethod
    def get_parameter_requirements(cls) -> Dict[ParameterType, ParameterSpec]:
        return {
            ParameterType.VISCOSITY: ParameterSpec(
                ParameterType.VISCOSITY,
                required=True,
                description="Dynamic viscosity for viscous flow"
            )
        }
    
    @property
    def flux(self):
        """Compute viscous stress using bound viscosity"""
        eta = self.get_bound_parameter(ParameterType.VISCOSITY)
        if eta is None:
            raise RuntimeError("Viscosity not bound to model")
        
        # Use eta directly in mathematical expression (it's a mathematical object)
        strain_rate = self._compute_strain_rate()  # Implementation detail
        return 2 * eta * strain_rate

class ViscoPlasticFlowModel(ViscousFlowModel):
    """Viscoplastic flow with inherited viscosity plus yield stress"""
    
    @classmethod
    def get_parameter_requirements(cls) -> Dict[ParameterType, ParameterSpec]:
        requirements = super().get_parameter_requirements()
        requirements.update({
            ParameterType.YIELD_STRESS: ParameterSpec(
                ParameterType.YIELD_STRESS,
                required=False,
                default_value=float('inf'),  # Default to purely viscous
                description="Yield stress for plastic behavior"
            )
        })
        return requirements
    
    @property
    def flux(self):
        """Compute viscoplastic stress"""
        eta = self.get_bound_parameter(ParameterType.VISCOSITY)
        tau_y = self.get_bound_parameter(ParameterType.YIELD_STRESS)
        
        strain_rate = self._compute_strain_rate()
        strain_rate_magnitude = self._compute_strain_rate_magnitude(strain_rate)
        
        # Effective viscosity with yield stress cutoff
        eta_eff = sympy.Min(eta, tau_y / (2 * strain_rate_magnitude))
        
        return 2 * eta_eff * strain_rate
```

## Implementation Strategy

### Phase 1: Parameter Registry

**Deliverables:**
1. `ParameterType` enum with comprehensive geophysical parameters
2. `ParameterDefinition` dataclass with full metadata
3. `MaterialParameterRegistry` with validation and lookup
4. Standard parameter definitions with references and ranges
5. Unit validation and error handling

**Key Files:**
- `materials/parameter_registry.py`: Core registry system
- `test_parameter_registry.py`: Comprehensive tests

### Phase 2: Material System 

**Deliverables:**
1. `MaterialProperty` class integrating with mathematical objects
2. `Material` class with property management and validation
3. `MaterialDatabase` with standard geophysical materials  
4. Integration with units system
5. Factory methods for constitutive model creation

**Key Files:**
- `materials/material.py`: Core material system
- `materials/database.py`: Standard materials database

### Phase 3: Constitutive Model Integration

**Deliverables:**
1. `ParameterAwareConstitutiveModel` base class
2. Enhanced constitutive models with parameter advertisement
3. Automatic parameter binding from materials
4. Validation of model-material compatibility
5. Integration with mathematical expression system

### Phase 4: Examples and Validation

**Deliverables:**
1. Convert all examples to use Material objects
2. Demonstrate parameter validation and error handling
3. Show cross-physics parameter consistency
4. Performance benchmarking
5. User documentation and handbook

## Cross-References to Other Plans

### Mathematical Objects Integration
**See: `mathematical_objects_plan.md`**
- `MaterialProperty` extends `UnitAwareMathematicalObject`
- Material properties work directly in mathematical expressions
- Seamless integration with variable and expression systems

### Units System Integration  
**See: `units_system_plan.md`**
- All material properties have units and validation
- Automatic unit conversion and checking
- Integration with problem-specific scaling

### Multi-Material System
**See: `MultiMaterial_ConstitutiveModel_Plan.md`**
- Materials provide properties for multi-material constitutive models
- Level-set averaging works with material property expressions
- Consistent parameter interfaces across all materials

### Parameter System Integration
**See: `parameter_system_plan.md`**
- Material properties can reference global parameters
- PETSc command-line integration for material property overrides
- Consistent parameter management across systems

## Benefits

### For Users
1. **Physical Constants Reference**: Like having a geophysics handbook built-in
2. **Error Prevention**: Automatic validation prevents common mistakes
3. **Discoverability**: Easy to explore available materials and properties
4. **Consistency**: Same material properties used across all physics
5. **Literature Integration**: Built-in references to source literature

### For Developers
1. **Type Safety**: Parameter types prevent spelling errors
2. **Clear Interfaces**: Models advertise exactly what they need
3. **Extensibility**: Easy to add new parameters and materials
4. **Validation**: Automatic checking of parameter compatibility
5. **Documentation**: Self-documenting parameter requirements

### for Science
1. **Reproducibility**: Clear specification of all material parameters
2. **Physical Correctness**: Validation against literature ranges
3. **Cross-System Consistency**: Same parameters for thermal, mechanical, chemical
4. **Literature Traceability**: References to original data sources
5. **Community Standards**: Common vocabulary for material properties

## Success Criteria

### Functional Requirements
1. **Complete Parameter Coverage**: All common geophysical parameters defined
2. **Automatic Validation**: Units, ranges, and compatibility checking
3. **Error Prevention**: Clear error messages for parameter mismatches
4. **Cross-Physics Integration**: Materials work with all constitutive models
5. **Database Integration**: Comprehensive library of standard materials

### User Experience Requirements
1. **Intuitive API**: Natural material definition and usage patterns
2. **Discoverability**: Easy exploration of available parameters and materials
3. **Migration Path**: Clear upgrade from current parameter patterns
4. **Error Messages**: Helpful guidance for parameter issues
5. **Documentation**: Complete handbook-style documentation

### Integration Requirements
1. **Mathematical Objects**: Seamless integration with mathematical expression system
2. **Units System**: Full unit awareness and validation
3. **Constitutive Models**: Automatic parameter binding and validation
4. **Multi-Material**: Consistent interfaces for multi-material systems

---

*Document Version: 1.0*  
*Created: 2025-01-19*  
*Cross-References: mathematical_objects_plan.md, units_system_plan.md, parameter_system_plan.md*  
*Status: Planning Phase*