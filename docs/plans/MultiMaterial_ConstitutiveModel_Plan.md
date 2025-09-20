# Multi-Material Constitutive Model Implementation Plan

## Project Overview

This document outlines the implementation of a multi-material constitutive model system for Underworld3. The goal is to create a robust framework that allows multiple constitutive models to coexist and be averaged using level-set methods through the IndexSwarmVariable approach.

## Current Architecture Analysis

### Existing Constitutive Model Framework

**Base Class: `Constitutive_Model`**
- **Core Properties**:
  - `self.dim`: Spatial dimensions (2D/3D)
  - `self.u_dim`: Number of components in unknowns (1 for scalar, `dim` for vector problems)
  - `self.Unknowns`: Contains mesh variables `u`, `DFDt`, `DuDt`
  - `self._c`: Constitutive tensor (rank 2 for scalar, rank 4 for vector problems)

**Key Method: `flux` Property**
```python
@property
def flux(self):
    """Returns flux = c * gradient(u)"""
    # Scalar: flux_i = c_ij * du/dx_j  
    # Vector: flux_ij = c_ijkl * du_k/dx_l
```

**Available Models**:
- `ViscousFlowModel`: η-dependent viscous flow
- `ViscoPlasticFlowModel`: Plasticity with yield stress
- `ViscoElasticPlasticFlowModel`: Full viscoelastoplastic behavior
- `DiffusionModel`: Heat/chemical diffusion
- `DarcyFlowModel`: Porous media flow
- `TransverseIsotropicFlowModel`: Anisotropic flow

### IndexSwarmVariable Implementation

**Purpose**: Track material indices on particles and create level-set representations on mesh

**Key Features**:
- **Material Indexing**: Each particle carries an integer material index
- **Level-Set Creation**: For `N` materials, creates `N` level-set functions on mesh
- **RBF Interpolation**: Uses radial basis functions to interpolate from particles to mesh nodes
- **Mask Generation**: Each level-set represents the spatial extent of one material (0-1 values)

**Core Methods**:
```python
def _update(self):
    """Creates level-set masks for each material on mesh nodes"""
    for material_index in range(self.indices):
        # Create mask: 1.0 where particles have this index, 0.0 elsewhere
        # Interpolate to mesh nodes using RBF
```

**Indexing Access**:
```python
material_var = IndexSwarmVariable("material", swarm, indices=3)  # 3 materials
material_var[0].sym  # Level-set for material 0 (sympy expression for mesh integration)
material_var[1].sym  # Level-set for material 1  
material_var[2].sym  # Level-set for material 2
```

### Existing Multi-Material Attempt Analysis

**Class: `MultiMaterial_ViscoElasticPlastic`**

**Current Implementation Issues**:
1. **Incomplete flux method**: References undefined variables (`ddu`, `ddu_dt`, `u`, `u_dt`)
2. **No dimension compatibility checking**: Missing validation that all models have same `u_dim`
3. **No proper initialization**: Doesn't call parent `__init__` or set up proper unknowns
4. **Averaging method unclear**: Simple multiplication may not be mathematically correct

**Positive Aspects**:
- Correct overall approach: `combined_flux += model[i].flux * mask[i]`
- Uses IndexSwarmVariable for material tracking
- Recognizes need for model list management

## Proposed Architecture

### 1. Material Definition System

**New Core Concept**: `Material` class as central repository for all material properties

```python
class Material(uw_object):
    """
    Central definition of material properties used across multiple physics systems.
    
    This class acts as a single source of truth for material properties that are
    used by constitutive models, density fields, thermal diffusion, etc.
    """
    
    def __init__(self, name: str):
        self.name = name
        
        # Mechanical properties
        self.density = expression(r"\rho", 1000, f"Density of {name}")
        self.viscosity = expression(r"\eta", 1e21, f"Viscosity of {name}")
        self.yield_stress = expression(r"\tau_y", sympy.oo, f"Yield stress of {name}")
        self.elastic_modulus = expression(r"G", 0, f"Elastic modulus of {name}")
        self.bulk_modulus = expression(r"K", 0, f"Bulk modulus of {name}")
        
        # Thermal properties  
        self.thermal_diffusivity = expression(r"\kappa", 1e-6, f"Thermal diffusivity of {name}")
        self.thermal_conductivity = expression(r"k", 3.0, f"Thermal conductivity of {name}")
        self.heat_capacity = expression(r"C_p", 1000, f"Heat capacity of {name}")
        
        # Chemical properties (for future extension)
        self.chemical_diffusivity = expression(r"D", 1e-9, f"Chemical diffusivity of {name}")
        
    def create_constitutive_model(self, model_type: str, unknowns):
        """Factory method to create constitutive models using this material's properties"""
        if model_type == "viscous":
            model = ViscousFlowModel(unknowns)
            model.material_properties.viscosity = self.viscosity
        elif model_type == "viscoplastic":
            model = ViscoPlasticFlowModel(unknowns)
            model.material_properties.viscosity = self.viscosity
            model.material_properties.yield_stress = self.yield_stress
        elif model_type == "viscoelastoplastic":
            model = ViscoElasticPlasticFlowModel(unknowns)
            model.material_properties.viscosity = self.viscosity
            model.material_properties.yield_stress = self.yield_stress
            model.material_properties.elastic_modulus = self.elastic_modulus
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return model

### 2. Material-Based Multi-Material System

```
class MultiMaterialConstitutiveModel(Constitutive_Model):
    """
    Multi-material constitutive model that references Material objects rather than
    directly managing parameters.
    """
    
    def __init__(self, 
                 unknowns,
                 material_swarmVariable: IndexSwarmVariable,
                 materials: List[Material],
                 model_type: str = "viscous"):
        """
        Parameters:
        -----------
        unknowns : UnknownSet
            The mesh variables (u, DFDt, DuDt)
        material_swarmVariable : IndexSwarmVariable  
            Index variable tracking material on particles
        materials : List[Material]
            List of Material objects defining properties
        model_type : str
            Type of constitutive model to create for each material
        """
        self._materials = materials
        self._material_var = material_swarmVariable
        
        # Create constitutive models from material definitions
        self._constitutive_models = [
            material.create_constitutive_model(model_type, unknowns)
            for material in materials
        ]
        
        super().__init__(unknowns)
```

### 2. Compatibility Validation System

**Requirement**: All constituent models must have compatible dimensions

```python
def _validate_model_compatibility(self, models: List[Constitutive_Model]) -> bool:
    """
    Ensure all models are compatible for averaging:
    - Same u_dim (scalar vs vector problem)
    - Same spatial dimension
    - Compatible flux tensor shapes
    """
    reference_model = models[0]
    reference_u_dim = reference_model.u_dim
    reference_dim = reference_model.dim
    
    for i, model in enumerate(models[1:], 1):
        if model.u_dim != reference_u_dim:
            raise ValueError(f"Model {i} has u_dim={model.u_dim}, expected {reference_u_dim}")
        if model.dim != reference_dim:
            raise ValueError(f"Model {i} has dim={model.dim}, expected {reference_dim}")
    
    return True
```

### 3. Level-Set Flux Averaging

**Mathematical Foundation**:
For N materials with level-set functions φᵢ(x) and constitutive models Cᵢ:

$$\text{flux}_{composite}(x) = \sum_{i=1}^{N} \phi_i(x) \cdot \text{flux}_i(x)$$

where $\sum_{i=1}^{N} \phi_i(x) \approx 1$ (partition of unity)

**Implementation**:
```python
@property  
def flux(self):
    """
    Compute level-set weighted average of constituent model fluxes
    """
    combined_flux = sympy.Matrix.zeros(*self._flux_shape)
    
    for i in range(self._material_var.indices):
        # Get level-set function for material i
        material_mask = self._material_var[i].sym
        
        # Get flux from constituent model i  
        model_flux = self._constitutive_models[i].flux
        
        # Add weighted contribution
        combined_flux += material_mask * model_flux
    
    return combined_flux
```

### 4. Model Management System

**Features**:
- **Type safety**: Ensure model list matches material count
- **Parameter propagation**: Handle parameter updates across models  
- **Lazy evaluation**: Only compute needed model fluxes

```python
def add_material_model(self, material_index: int, model: Constitutive_Model):
    """Add or replace constitutive model for specific material"""
    
def set_material_parameters(self, material_index: int, **parameters):
    """Set parameters for specific material model"""
    
def get_material_model(self, material_index: int) -> Constitutive_Model:
    """Get constitutive model for specific material"""
```

## Implementation Phases

### Phase 1: Core Infrastructure

**Deliverables**:
1. **Enhanced base class** with proper initialization
2. **Compatibility validation** system  
3. **Basic flux averaging** implementation
4. **Unit tests** for dimension checking and model management

**Key Files to Modify**:
- `constitutive_models.py`: Add new `MultiMaterialConstitutiveModel` class
- Create `test_multi_material_constitutive.py`: Comprehensive test suite

### Phase 2: Advanced Features  

**Deliverables**:
1. **Parameter management** system
2. **Model factory methods** for common multi-material scenarios
3. **Performance optimization** for level-set evaluation
4. **Integration tests** with existing solvers

**Example Factory Methods**:
```python
@classmethod
def create_viscous_materials(cls, unknowns, material_var, viscosities):
    """Create multi-material model with different viscosities"""
    
@classmethod  
def create_viscoplastic_materials(cls, unknowns, material_var, 
                                  viscosities, yield_stresses):
    """Create multi-material viscoplastic model"""
```

### Phase 3: Integration & Validation

**Deliverables**:
1. **Solver integration** with Stokes/diffusion systems
2. **Benchmark examples** comparing with analytical solutions
3. **Performance profiling** and optimization
4. **User documentation** and examples

## Technical Challenges & Solutions

### Challenge 1: Level-Set Consistency

**Issue**: Level-set functions may not sum to 1.0 everywhere, especially near material boundaries

**Solution**: Implement normalization in flux computation:
```python
# Normalize level-sets to ensure partition of unity
total_levelset = sum(self._material_var[i].sym for i in range(self.indices))
normalized_flux = sum((self._material_var[i].sym / total_levelset) * flux_i 
                     for i, flux_i in enumerate(model_fluxes))
```

### Challenge 2: Performance Optimization

**Issue**: Multiple model evaluations and level-set computations may be expensive

**Solutions**:
1. **Lazy evaluation**: Only compute fluxes where level-sets are non-zero
2. **Caching**: Cache level-set evaluations between solver iterations
3. **Vectorization**: Evaluate all materials simultaneously where possible

### Challenge 3: Parameter Updates

**Issue**: Changing parameters in constituent models should trigger proper updates

**Solution**: Implement observer pattern:
```python
def _on_material_parameter_change(self, material_index: int, parameter: str, value):
    """Handle parameter changes in constituent models"""
    # Invalidate cached flux computations
    # Trigger solver re-setup if needed
```

## Testing Strategy

### Unit Tests
- **Dimension compatibility** checking
- **Model list management** operations
- **Parameter propagation** correctness
- **Level-set averaging** mathematics

### Integration Tests  
- **Two-material viscous flow** with analytical solution
- **Multi-material diffusion** with known temperature profiles
- **Viscoplastic boundaries** with yield stress contrasts

### Performance Tests
- **Scaling with material count** (2, 4, 8, 16 materials)
- **Memory usage profiling** for large problems
- **Solver convergence rates** compared to single-material cases

## API Design Examples

### Basic Usage with Material Objects
```python
# Define materials with their properties
weak_mantle = Material("Weak Mantle")
weak_mantle.density = 3200
weak_mantle.viscosity = 1e20

strong_mantle = Material("Strong Mantle")  
strong_mantle.density = 3300
strong_mantle.viscosity = 1e23

oceanic_crust = Material("Oceanic Crust")
oceanic_crust.density = 2900
oceanic_crust.viscosity = 1e22
oceanic_crust.yield_stress = 100e6

# Create material index variable
material_var = IndexSwarmVariable("material", swarm, indices=3)

# Create multi-material constitutive model
multi_model = MultiMaterialConstitutiveModel(
    unknowns=unknowns,
    material_swarmVariable=material_var,
    materials=[weak_mantle, strong_mantle, oceanic_crust],
    model_type="viscoplastic"  # All materials get viscoplastic behavior
)

# Use in solver
stokes_solver.constitutive_model = multi_model
```

### Cross-Physics Material Usage
```python
# Same materials used across multiple physics
# 1. Mechanical behavior
multi_mechanical = MultiMaterialConstitutiveModel(
    unknowns=stokes_unknowns,
    material_swarmVariable=material_var,
    materials=[weak_mantle, strong_mantle, oceanic_crust],
    model_type="viscoplastic"
)

# 2. Thermal diffusion using same materials
thermal_diffusion = MultiMaterialThermalModel(
    unknowns=thermal_unknowns,
    material_swarmVariable=material_var,
    materials=[weak_mantle, strong_mantle, oceanic_crust]
    # Automatically uses thermal_diffusivity from each material
)

# 3. Density field for buoyancy
density_field = MaterialDensityField(
    material_swarmVariable=material_var,
    materials=[weak_mantle, strong_mantle, oceanic_crust]
    # Automatically uses density from each material
)
```

### Material Property Updates
```python
# Update material properties directly
oceanic_crust.yield_stress = 200e6  # Affects all models using this material

# Temperature-dependent viscosity
weak_mantle.viscosity = expression(r"\eta", 1e20 * sympy.exp(5000/T), "Temperature-dependent viscosity")

# Material database approach
material_db = MaterialDatabase()
material_db.add_material("olivine", density=3300, viscosity=1e22)
material_db.add_material("basalt", density=2900, yield_stress=100e6)

olivine = material_db.get_material("olivine")
```

## Success Criteria

### Functional Requirements
1. **✅ Multi-material support**: Handle 2+ different constitutive models simultaneously
2. **✅ Level-set averaging**: Smooth transitions between materials using IndexSwarmVariable
3. **✅ Dimension compatibility**: Enforce that all models have compatible flux dimensions
4. **✅ Parameter management**: Easy access and modification of individual material parameters

### Performance Requirements  
1. **✅ Solver integration**: Work seamlessly with existing Stokes/diffusion solvers
2. **✅ Memory efficiency**: <20% overhead compared to single-material models
3. **✅ Convergence**: Maintain solver convergence rates within 2x of single-material cases

### Usability Requirements
1. **✅ Simple API**: Intuitive creation and management of multi-material models
2. **✅ Error handling**: Clear error messages for dimension mismatches and invalid configurations  
3. **✅ Documentation**: Complete examples for common multi-material scenarios

## Future Extensions

### Constitutive Model Parameter Inheritance Refactoring

**Problem**: Current inheritance pattern fights against mathematical reality where complex models contain simpler ones as special cases.

**Proposed Solution**: Invert inheritance hierarchy to match physics:

```python
class ViscoElasticPlasticFlowModel(Constitutive_Model):
    """Most complete model - contains all physics"""
    
    class UnifiedParameters:
        """Single parameter class supporting all constitutive behaviors"""
        def __init__(self, _owning_model):
            # Viscous parameters (always present)
            self._viscosity = expression(r"\eta", 1, "Shear viscosity")
            
            # Plastic parameters (default to "inactive" values)
            self._yield_stress = expression(r"\tau_y", sympy.oo, "Yield stress")
            
            # Elastic parameters (default to "inactive" values)  
            self._elastic_modulus = expression(r"G", 0, "Elastic modulus")
            
        @property
        def effective_viscosity(self):
            """SymPy automatically simplifies based on parameter values"""
            strain_rate = self._owning_model.strain_rate_invariant
            plastic_viscosity = self.yield_stress / (2 * strain_rate)
            return sympy.Min(self.viscosity, plastic_viscosity)
            # When yield_stress = ∞, this becomes: viscosity (purely viscous)
    
    @classmethod
    def create_viscous(cls, unknowns, viscosity=1):
        """Factory: Create purely viscous model"""
        model = cls(unknowns)
        model.material_properties.yield_stress = sympy.oo  # Disable plasticity
        return model

# Inheritance follows physical reality
class ViscoPlasticFlowModel(ViscoElasticPlasticFlowModel):
    """Specialization: elastic_modulus → 0"""
    
class ViscousFlowModel(ViscoPlasticFlowModel):
    """Specialization: yield_stress → ∞"""
```

**Benefits**:
- **Mathematical consistency**: Inheritance follows physical reality
- **SymPy power**: Automatic simplification when parameters are "inactive" 
- **Code reuse**: Single flux implementation handles all cases
- **Multi-material compatibility**: All models share same parameter interface

**Critical Issue: History Term Management**

**Problem**: Different constitutive models require different history terms:
- **Viscous/Viscoplastic**: No stress history needed (instantaneous response)
- **Elastic/Viscoelastic**: Requires stress history tracking via DDT objects

**Risk**: Unified parameter system might force all models to track unnecessary history terms, adding computational overhead and memory usage where not needed.

**Proposed Solution**: Conditional history term activation:

```python
class UnifiedParameters:
    def __init__(self, _owning_model):
        # Always present parameters
        self._viscosity = expression(r"\eta", 1, "Shear viscosity")
        self._yield_stress = expression(r"\tau_y", sympy.oo, "Yield stress")
        
        # Elastic parameters with history tracking flags
        self._elastic_modulus = expression(r"G", 0, "Elastic modulus")
        self._requires_stress_history = False  # Flag for DDT object creation
        
    @elastic_modulus.setter
    def elastic_modulus(self, value):
        self._elastic_modulus = value
        # Automatically enable history tracking when elasticity is activated
        if value != 0 and not sympy.simplify(value).is_zero:
            self._requires_stress_history = True
            self._owning_model._enable_stress_history()
        else:
            self._requires_stress_history = False
            self._owning_model._disable_stress_history()

class ViscoElasticPlasticFlowModel(Constitutive_Model):
    def _enable_stress_history(self):
        """Create DDT objects only when needed"""
        if not hasattr(self, '_stress_ddt'):
            self._stress_ddt = SemiLagrangian_DDt(self.stress_field, ...)
            
    def _disable_stress_history(self):
        """Clean up DDT objects when not needed"""
        if hasattr(self, '_stress_ddt'):
            del self._stress_ddt
```

**Performance Requirements**:
- **Zero overhead**: Viscous models should have no DDT objects or history tracking
- **Lazy activation**: History terms only created when elastic parameters become non-zero
- **Memory efficiency**: Automatic cleanup when models revert to non-elastic behavior

**Implementation Priority**: High - must be addressed during refactoring to avoid performance regression

### Advanced Averaging Methods
- **Volume-weighted averaging** for better mass conservation
- **Harmonic averaging** for specific physical properties
- **Custom averaging functions** for specialized applications

### Dynamic Material Properties
- **Temperature-dependent** viscosity variations
- **Strain-rate dependent** plasticity parameters  
- **Time-evolution** of material properties

### GPU Acceleration
- **CUDA kernels** for level-set evaluation
- **Vectorized flux** computation on GPU
- **Memory-optimized** data structures for GPU execution

---

*Document Version: 1.0*  
*Created: 2025-01-19*  
*Author: Claude Code Assistant*  
*Status: Planning Phase*
