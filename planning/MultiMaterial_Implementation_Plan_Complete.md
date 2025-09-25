# Multi-Material Constitutive Model Implementation Plan (Phase 1)

## Executive Summary

This plan outlines the implementation of robust multi-material constitutive models for Underworld3. Based on analysis of the existing incomplete implementations and the broader UW3 architecture, this plan provides a complete roadmap for creating a production-ready multi-material system using level-set averaging via IndexSwarmVariable.

## Current State Analysis

### Existing Infrastructure ✅
- **IndexSwarmVariable**: Mature implementation for material tracking with RBF interpolation to mesh
- **Base Constitutive_Model class**: Well-established with proper `flux` property interface  
- **Materials system**: New comprehensive material properties framework (`materials.py`)
- **Level-set methodology**: Proven approach with partition of unity guarantees

### Critical Issues in Current Implementation ❌
- **`MultiMaterial_ViscoElasticPlastic`** (constitutive_models.py:4070-4090):
  - References undefined variables: `ddu`, `ddu_dt`, `u`, `u_dt` 
  - Incorrect flux signature: expects parameters but calls property-style `.flux`
  - Missing proper initialization and parent class setup
  - No dimension compatibility validation

- **`MultiMaterial`** (constitutive_models_new.py:850-870):
  - Basic shell only, no actual implementation
  - Missing flux averaging mathematics

## Architecture Design

### 🎯 Critical Architectural Insight: Solver-Authoritative State

**Key Discovery:** The solver holds authoritative copies of all unknowns and their histories. Individual constitutive models do **NOT** maintain independent field histories - they read from shared solver state via $D\mathbf{F}/Dt.\psi^*[0]$.

**Implications for Multi-Material Models:**
- All constituent models must share the **same `Unknowns` object**
- Stress history is the **composite flux**, not individual model contributions
- This ensures physical correctness: all materials experience the same stress history

### 1. Multi-Material Base Class (Revised)

```python
class MultiMaterialConstitutiveModel(Constitutive_Model):
    """
    Multi-material constitutive model using level-set weighted flux averaging.
    
    Mathematical Foundation:
    $\mathbf{f}_{\text{composite}}(\mathbf{x}) = \sum_{i=1}^{N} \phi_i(\mathbf{x}) \cdot \mathbf{f}_i(\mathbf{x})$
    
    Critical Architecture:
    - Solver owns Unknowns (including $D\mathbf{F}/Dt$ stress history)
    - All constituent models share solver's Unknowns
    - Composite flux becomes stress history for all materials
    """
    
    def __init__(self, 
                 unknowns,
                 material_swarmVariable: IndexSwarmVariable,
                 constitutive_models: List[Constitutive_Model]):
        """
        Parameters:
        -----------
        unknowns : UnknownSet
            The solver's authoritative unknowns ($\mathbf{u}$, $D\mathbf{F}/Dt$, $D\mathbf{u}/Dt$)
        material_swarmVariable : IndexSwarmVariable
            Index variable tracking material distribution on particles
        constitutive_models : List[Constitutive_Model]  
            Pre-configured constitutive models for each material
        """
        # Validate compatibility before initialization
        self._validate_model_compatibility(constitutive_models)
        
        self._material_var = material_swarmVariable
        self._constitutive_models = constitutive_models
        
        # Ensure model count matches material indices
        if len(constitutive_models) != material_swarmVariable.indices:
            raise ValueError(
                f"Model count ({len(constitutive_models)}) must match "
                f"material indices ({material_swarmVariable.indices})"
            )
        
        # CRITICAL: Share solver's unknowns with all constituent models
        self._setup_shared_unknowns(constitutive_models, unknowns)
        
        super().__init__(unknowns)
    
    def _setup_shared_unknowns(self, constitutive_models, unknowns):
        """
        Ensure all constituent models share the solver's authoritative unknowns.
        This is critical for proper stress history management.
        """
        for i, model in enumerate(constitutive_models):
            # Share solver's unknowns - this gives access to composite $D\mathbf{F}/Dt$ history
            model.Unknowns = unknowns
            
            # Validation: Ensure sharing worked correctly
            assert model.Unknowns is unknowns, \
                f"Model {i} failed to share unknowns - memory issue?"
            
            # For elastic models, verify DFDt access
            if hasattr(model, '_stress_star'):
                assert hasattr(unknowns, 'DFDt'), \
                    f"Model {i} needs stress history but $D\mathbf{{F}}/Dt$ not available"
```

### 2. Compatibility Validation System

```python
def _validate_model_compatibility(self, models: List[Constitutive_Model]) -> bool:
    """
    Ensure all constituent models are compatible for flux averaging.
    
    Checks:
    - Same u_dim (scalar vs vector problem compatibility)
    - Same spatial dimension (2D/3D consistency)  
    - Compatible flux tensor shapes
    - All models properly initialized
    """
    if not models:
        raise ValueError("At least one constitutive model required")
    
    reference_model = models[0]
    reference_u_dim = reference_model.u_dim
    reference_dim = reference_model.dim
    
    for i, model in enumerate(models):
        if model.u_dim != reference_u_dim:
            raise ValueError(
                f"Model {i} has u_dim={model.u_dim}, expected {reference_u_dim}"
            )
        if model.dim != reference_dim:
            raise ValueError(
                f"Model {i} has dim={model.dim}, expected {reference_dim}"
            )
        # Validate model is properly initialized
        if not hasattr(model, 'Unknowns') or model.Unknowns is None:
            raise ValueError(f"Model {i} is not properly initialized")
    
    return True
```

### 3. Stress History Flow in Multi-Material Systems

**Critical Understanding:** The stress history that elastic materials experience is the **composite flux history**, not individual model histories.

**History Update Sequence:**
```
Time Step n:
1. Solver Setup    : $D\mathbf{F}/Dt.\psi_{\text{fn}} = \mathbf{f}_{\text{composite}}$  (composite)
2. Pre-Solve       : $D\mathbf{F}/Dt.\text{update\_pre\_solve}() → \psi^*[0] = \text{history}$
3. Model Access    : $\text{elastic\_model.stress\_star} → \text{reads } \psi^*[0]$ (composite)
4. Flux Evaluation : $\mathbf{f}_{\text{composite}} = \sum_i \phi_i(\mathbf{x}) \mathbf{f}_i(\mathbf{x})$
5. Solve System    : Uses composite flux in PDE
6. Post-Solve      : $D\mathbf{F}/Dt.\text{update\_post\_solve}() → \text{current flux becomes history}$

Time Step n+1:
- $\psi^*[0]$ now contains composite flux from step n
- All elastic materials read SAME composite history
- Physically correct: materials respond to total stress environment
```

**Why This Matters - Physical Example:**
```
Mantle convection with elastic lithosphere:
- Viscous mantle: responds to current strain rate (no history)
- Elastic lithosphere: responds to composite stress history (mantle + lithosphere)
- Result: Lithospheric stress influenced by entire stress field, not just local elastic stress
```

### 4. Level-Set Flux Averaging Mathematics

**Mathematical Foundation:**

For $N$ materials with level-set functions $\phi_i(\mathbf{x}) \in [0,1]$:

$$\mathbf{f}_{\text{composite}}(\mathbf{x}) = \sum_{i=1}^{N} \phi_i(\mathbf{x}) \cdot \mathbf{f}_i(\mathbf{x})$$

where:
- $\phi_i(\mathbf{x})$ represents the volume fraction of material $i$ at point $\mathbf{x}$
- $\sum_{i=1}^{N} \phi_i(\mathbf{x}) \approx 1$ (partition of unity, enforced by RBF normalization)
- $\mathbf{f}_i(\mathbf{x}) = \mathbf{C}_i : \nabla \mathbf{u}$ (constitutive tensor applied to velocity gradients)
- $\mathbf{f}_{\text{composite}}$ becomes stress history for ALL materials

**Implementation (Revised):**
```python
@property
def flux(self):
    """
    Compute level-set weighted average of constituent model fluxes.
    
    CRITICAL: This composite flux becomes the stress history that
    all constituent models (including elastic ones) will read via
    $D\mathbf{F}/Dt.\psi^*[0]$ in the next time step.
    """
    # Get reference flux shape from first model
    reference_flux = self._constitutive_models[0].flux
    combined_flux = sympy.Matrix.zeros(*reference_flux.shape)
    
    # Compute normalization factor to ensure partition of unity
    total_levelset = sum(
        self._material_var[i].sym 
        for i in range(self._material_var.indices)
    )
    
    for i in range(self._material_var.indices):
        # Get normalized level-set function for material i
        material_fraction = self._material_var[i].sym / total_levelset
        
        # Get flux contribution from constituent model i
        # Note: If model i is elastic, it reads composite history from $D\mathbf{F}/Dt.\psi^*[0]$
        model_flux = self._constitutive_models[i].flux
        
        # Add weighted contribution to composite flux
        combined_flux += material_fraction * model_flux
    
    # This combined_flux will become the stress history for ALL materials
    return combined_flux
```

### 4. Robust Error Handling and Validation

```python
def _validate_runtime_state(self):
    """Validate system state before flux computation"""
    # Check IndexSwarmVariable is updated
    if not self._material_var._proxy_updated:
        raise RuntimeError(
            "Material index variable proxy not updated. "
            "Call material_var._update() after particle changes."
        )
    
    # Check for degenerate level-sets (all zero)
    for i in range(self._material_var.indices):
        levelset = self._material_var[i].sym
        if levelset == 0:
            warnings.warn(
                f"Material {i} level-set is zero everywhere. "
                f"This material will not contribute to flux."
            )

@property  
def flux(self):
    """Flux computation with comprehensive validation"""
    self._validate_runtime_state()
    
    # Main flux computation (as above)
    # ... implementation ...
    
    return combined_flux
```

## Testing Strategy

### Unit Tests (Level 1)

**Compatibility Validation:**
```python
def test_model_compatibility_validation():
    """Test dimension compatibility checking"""
    # Create models with mismatched dimensions
    model_2d = ViscousFlowModel(unknowns_2d)  # u_dim=2
    model_3d = ViscousFlowModel(unknowns_3d)  # u_dim=3
    
    with pytest.raises(ValueError, match="u_dim=3, expected 2"):
        MultiMaterialConstitutiveModel(
            unknowns_2d, material_var, [model_2d, model_3d]
        )

def test_flux_shape_consistency():
    """Test flux tensor shapes are consistent"""
    multi_model = create_test_multi_material_model(2)
    flux = multi_model.flux
    
    # Compare with single material flux shape
    reference_flux = multi_model._constitutive_models[0].flux  
    assert flux.shape == reference_flux.shape
```

**Level-Set Mathematics:**
```python
def test_partition_of_unity():
    """Test level-set functions sum to approximately 1"""
    material_var = IndexSwarmVariable("test", swarm, indices=3)
    # Set up particles with known material distribution
    material_var.data[0:100] = 0  # Material 0
    material_var.data[100:200] = 1  # Material 1  
    material_var.data[200:300] = 2  # Material 2
    material_var._update()
    
    # Sum level-sets should be close to 1 everywhere
    total = sum(material_var[i].sym for i in range(3))
    # Test at sample points using sympy evaluation
    
def test_flux_averaging_mathematics():
    """Test weighted averaging produces expected results"""
    # Create materials with known viscosities: η₁=1, η₂=2
    model1 = ViscousFlowModel(unknowns) 
    model1.Parameters.viscosity = 1
    model2 = ViscousFlowModel(unknowns)
    model2.Parameters.viscosity = 2
    
    # Create material distribution: 50% material 0, 50% material 1
    # Expected effective viscosity: 0.5*1 + 0.5*2 = 1.5
    
    multi_model = MultiMaterialConstitutiveModel(unknowns, material_var, [model1, model2])
    # Test flux evaluation gives expected weighted average
```

### Integration Tests (Level 2)

**Solver Integration:**
```python
def test_stokes_solver_integration():
    """Test multi-material model works with Stokes solver"""
    # Create two-material viscous flow problem
    multi_model = create_viscous_multi_material(viscosities=[1e20, 1e23])
    
    stokes_solver = uw.systems.Stokes(
        mesh, 
        velocityField=v, 
        pressureField=p,
        constitutive_model=multi_model
    )
    
    # Test solver setup and basic solve
    stokes_solver.solve()
    assert stokes_solver.converged

def test_parameter_update_propagation():
    """Test parameter changes propagate correctly"""
    multi_model = create_test_multi_material_model(2)
    
    # Change viscosity in one material
    multi_model._constitutive_models[0].Parameters.viscosity = 100
    
    # Verify flux computation reflects the change
    flux_updated = multi_model.flux
    # Compare with expected result
```

### Validation Tests (Level 3)

**Analytical Benchmarks:**
```python 
def test_two_layer_analytical_solution():
    """
    Compare with analytical solution for two-layer viscous flow.
    
    Setup: Two horizontal layers with different viscosities
    Boundary conditions: No-slip bottom, free-slip top, pressure gradient
    Analytical solution: Known velocity profile
    """
    # Implementation comparing numerical vs analytical velocity profiles
    
def test_inclusion_benchmark():
    """
    Circular inclusion in matrix - compare with analytical/literature solutions.
    
    Setup: Circular high-viscosity inclusion in low-viscosity matrix
    Loading: Simple shear or pure shear
    Expected: Known flow patterns around inclusion
    """
```

**Performance Benchmarks:**
```python
def test_scaling_with_material_count():
    """Test computational cost scales reasonably with material count"""
    for n_materials in [2, 4, 8, 16]:
        multi_model = create_multi_material_model(n_materials)
        
        # Time flux evaluation
        start_time = time.time()
        for _ in range(100):
            flux = multi_model.flux
        elapsed = time.time() - start_time
        
        # Cost should scale roughly linearly
        assert elapsed < n_materials * baseline_time * 1.5

def test_memory_usage():
    """Test memory usage is reasonable"""
    import psutil
    process = psutil.Process()
    
    baseline_memory = process.memory_info().rss
    multi_model = create_large_multi_material_model()
    peak_memory = process.memory_info().rss
    
    memory_overhead = (peak_memory - baseline_memory) / baseline_memory
    assert memory_overhead < 0.2  # <20% overhead
```

## Integration with Existing Systems

### 1. Materials Registry Integration

The new materials system provides a perfect foundation for multi-material models:

```python
def create_from_materials_registry(
    unknowns, 
    material_swarmVariable: IndexSwarmVariable,
    materials: List[MaterialDefinition],
    model_type: str = "viscous"
) -> MultiMaterialConstitutiveModel:
    """
    Factory method to create multi-material model from materials registry.
    
    Parameters:
    -----------
    materials : List[MaterialDefinition]
        Materials from MaterialRegistry with properties defined
    model_type : str
        Type of constitutive model ("viscous", "viscoplastic", etc.)
    """
    # Create constitutive models from material definitions
    constitutive_models = []
    
    for material in materials:
        if model_type == "viscous":
            model = ViscousFlowModel(unknowns)
            model.Parameters.viscosity = material.get_property('viscosity')
            
        elif model_type == "viscoplastic":
            model = ViscoPlasticFlowModel(unknowns)
            model.Parameters.viscosity = material.get_property('viscosity')
            model.Parameters.yield_stress = material.get_property('yield_stress')
            
        # Add other model types as needed
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        constitutive_models.append(model)
    
    return MultiMaterialConstitutiveModel(
        unknowns, material_swarmVariable, constitutive_models
    )

# Usage example:
registry = MaterialRegistry()
mantle = registry.create_material('mantle')
mantle.set_property('viscosity', 1e21)
mantle.set_property('density', 3300)

crust = registry.create_material('crust') 
crust.set_property('viscosity', 1e22)
crust.set_property('density', 2700)

multi_model = MultiMaterialConstitutiveModel.create_from_materials_registry(
    unknowns, material_var, [mantle, crust], "viscous"
)
```

### 2. Solver Compatibility

**Stokes Solver Integration:**
```python
# The multi-material model seamlessly integrates with existing solvers
stokes = uw.systems.Stokes(
    mesh, 
    velocityField=v,
    pressureField=p, 
    constitutive_model=multi_model,  # Drop-in replacement
    bodyforce=buoyancy_force
)

# No changes needed to solver interface or solution methods
stokes.solve()
```

**Advection-Diffusion Integration:**
```python
# Works with thermal diffusion, chemical diffusion, etc.
thermal_model = MultiMaterialThermalModel(
    thermal_unknowns,
    material_swarmVariable, 
    thermal_diffusivities=[1e-6, 2e-6, 5e-7]
)

thermal_solver = uw.systems.AdvDiffusion(
    mesh,
    temperatureField=T,
    constitutive_model=thermal_model
)
```

### 3. Example Integration

**Complete Multi-Physics Setup:**
```python
# 1. Create materials registry
registry = MaterialRegistry()

weak_mantle = create_standard_mantle_material(registry)
strong_mantle = registry.create_material('strong_mantle')
strong_mantle.set_property('viscosity', 1e23)  # 100x stronger

oceanic_crust = create_standard_crust_material(registry)
oceanic_crust.set_property('yield_stress', 100e6)  # Add plasticity

# 2. Assign materials to regions
registry.assign_to_region('weak_mantle', region_id=0)
registry.assign_to_region('strong_mantle', region_id=1) 
registry.assign_to_region('oceanic_crust', region_id=2)

# 3. Create material tracking variable
material_var = IndexSwarmVariable("material", swarm, indices=3)

# Initialize particle materials based on position/region
# ... particle initialization code ...

# 4. Create multi-material constitutive models
mechanical_model = MultiMaterialConstitutiveModel.create_from_materials_registry(
    stokes_unknowns, material_var, 
    [weak_mantle, strong_mantle, oceanic_crust],
    "viscoplastic"
)

thermal_model = MultiMaterialConstitutiveModel.create_from_materials_registry(
    thermal_unknowns, material_var,
    [weak_mantle, strong_mantle, oceanic_crust], 
    "diffusion"
)

# 5. Create solvers
stokes = uw.systems.Stokes(
    mesh, v, p,
    constitutive_model=mechanical_model,
    bodyforce=registry.evaluate_property_field('density', material_var.data) * gravity
)

advdiff = uw.systems.AdvDiffusion(
    mesh, T,
    constitutive_model=thermal_model,
    velocity_field=v
)

# 6. Solve coupled system
for step in range(num_steps):
    stokes.solve()
    advdiff.solve(dt=dt) 
    swarm.advection(v, dt)  # This triggers material_var._update()
```

## Implementation Phases

### Phase 1: Core Multi-Material Infrastructure (Week 1-2)
**Files to Create/Modify:**
- Modify `constitutive_models.py`: Remove broken implementation, add new `MultiMaterialConstitutiveModel`
- Create `test_multi_material_constitutive.py`: Comprehensive test suite
- Update `__init__.py`: Export new multi-material classes

**Deliverables:**
1. ✅ Working `MultiMaterialConstitutiveModel` class with proper initialization
2. ✅ Compatibility validation system preventing dimension mismatches  
3. ✅ Level-set flux averaging implementation with partition of unity normalization
4. ✅ Unit tests for all core functionality

### Phase 2: Materials Registry Integration (Week 3)
**Files to Modify:**
- Add factory methods to `MultiMaterialConstitutiveModel`
- Enhance integration with `materials.py` MaterialRegistry
- Create example notebooks demonstrating materials workflow

**Deliverables:**
1. ✅ `create_from_materials_registry()` factory method
2. ✅ Integration tests with MaterialRegistry system
3. ✅ Example notebooks showing complete materials workflow
4. ✅ Performance benchmarking suite

### Phase 3: Validation & Optimization (Week 4)  
**Files to Create:**
- `examples/multi_material_benchmarks.py`: Analytical validation cases
- Performance profiling and optimization
- Documentation updates

**Deliverables:**
1. ✅ Analytical benchmark validation (two-layer flow, inclusion problems)
2. ✅ Performance optimization for large material counts
3. ✅ Complete user documentation with examples
4. ✅ Integration with existing solver test suites

## Success Criteria

### Functional Requirements ✅
- [ ] **Multi-material support**: Handle 2+ different constitutive models simultaneously
- [ ] **Level-set averaging**: Smooth material transitions using IndexSwarmVariable RBF interpolation  
- [ ] **Dimension compatibility**: Enforce all models have compatible flux tensor dimensions
- [ ] **Materials integration**: Seamless integration with MaterialRegistry system
- [ ] **Solver compatibility**: Drop-in replacement for single-material models in all solvers

### Performance Requirements ✅  
- [ ] **Memory efficiency**: <20% overhead compared to single-material models
- [ ] **Computational efficiency**: Linear scaling with material count (O(N) not O(N²))
- [ ] **Solver convergence**: Maintain convergence rates within 2x of single-material cases
- [ ] **Large problem scaling**: Handle problems with 10+ materials efficiently

### Quality Requirements ✅
- [ ] **Mathematical rigor**: Proper partition of unity enforcement and flux averaging
- [ ] **Error handling**: Clear error messages for all failure modes
- [ ] **Test coverage**: >95% code coverage with unit, integration, and validation tests
- [ ] **Documentation**: Complete API documentation with working examples

## Risk Mitigation

### Technical Risks
1. **Level-set normalization numerical issues**
   - *Mitigation*: Add numerical regularization (small epsilon) to prevent division by zero
   - *Fallback*: Graceful degradation to single-material behavior in degenerate cases

2. **Performance degradation with many materials** 
   - *Mitigation*: Lazy evaluation - only compute fluxes where level-sets > threshold
   - *Monitoring*: Continuous performance benchmarking in test suite

3. **Solver convergence issues**
   - *Mitigation*: Extensive validation against analytical solutions  
   - *Fallback*: Option to disable multi-material averaging for debugging

### Integration Risks
1. **Breaking changes to existing API**
   - *Mitigation*: Multi-material models are purely additive - no changes to existing single-material interface
   - *Validation*: All existing tests must continue to pass

2. **IndexSwarmVariable dependency issues**
   - *Mitigation*: Comprehensive testing of IndexSwarmVariable edge cases
   - *Monitoring*: Tests for particle migration, swarm recreation scenarios

## Future Extensions

### Advanced Averaging Methods
- **Volume-weighted averaging**: More accurate for heterogeneous material distributions  
- **Harmonic averaging**: Better for properties like conductivity/permeability
- **Custom averaging functions**: User-defined averaging schemes for specialized physics

### Dynamic Material Evolution
- **Phase transitions**: Materials changing type based on P-T conditions
- **Chemical reactions**: Multi-component chemical diffusion with reaction kinetics  
- **Damage/healing**: Evolution of material properties based on deformation history

### GPU Acceleration  
- **CUDA kernels**: Vectorized level-set evaluation and flux computation
- **Memory optimization**: GPU-friendly data structures for multi-material problems
- **Scaling studies**: Performance on large GPU clusters

---

**Document Status**: Implementation Ready  
**Approval Required**: Architecture review before implementation begins  
**Estimated Implementation Time**: 4 weeks (1 developer)  
**Dependencies**: IndexSwarmVariable (✅ complete), MaterialRegistry (✅ complete)