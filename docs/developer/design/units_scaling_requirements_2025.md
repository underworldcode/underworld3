# Enhanced Units and Scaling Requirements (2025)

## Core Philosophy

### General Principles

1. **Flexible Input**: Users should be able to specify data and problems in dimensional form with maximum flexibility about the units they use
2. **Numerical Optimization**: Solvers and preconditioners work best when stiffness matrices and unknowns are both close to O(1)
3. **Simple Mechanism**: Provide simple mechanism for specifying reference lengths, times, etc. that achieve "non-dimensionalisation" and automatically apply them

### Two Approaches to Non-Dimensionalisation

#### 1. Formal Dimensional Analysis (Mathematical Approach)
- Rigorous dimensional analysis using fundamental dimensions
- Mathematically pure but can be complex for users
- Results in true dimensionless numbers

#### 2. Practical Units Selection (Human Approach)
- Choose appropriate units that make every quantity a single-digit number
- This is why different domains have different units: "Microns to Parsecs, we choose the best units for the job"
- **Not necessarily SI units** - easier to think about than formal non-dimensionalisation
- Achieves roughly the same numerical benefits as formal approach

### Implementation Strategies

#### Option A: Automatic Solver-Level Non-Dimensionalisation
- **When possible**: Auto non-dimensionalise at solver level if units are available
- **Challenge**: Difficult if we only have numbers without unit information
- **Benefit**: Transparent to user, optimal numerical conditioning

#### Option B: Good Units from the Start
- **Approach**: Choose units at outset which work well numerically
- **Requirement**: Needs user cooperation and guidance
- **Benefit**: Simpler implementation, user maintains control

## Current State Analysis

### Existing Pint Implementation
- Basic Pint registry in `uw.scaling.units`
- `non_dimensionalise()` function using scaling coefficients
- Manual scaling coefficient setup

### Current `non_dimensionalise()` Implementation
```python
def non_dimensionalise(dimValue):
    # Convert to base units
    dimValue = dimValue.to_base_units()

    # Get scaling coefficients
    scaling_coefficients = get_coefficients()
    length = scaling_coefficients["[length]"]   # Default: 1.0 * u.meter
    time = scaling_coefficients["[time]"]       # Default: 1.0 * u.year
    mass = scaling_coefficients["[mass]"]       # Default: 1.0 * u.kilogram
    temperature = scaling_coefficients["[temperature]"]  # Default: 1.0 * u.degK
    substance = scaling_coefficients["[substance]"]      # Default: 1.0 * u.mole

    # Extract dimensionality from input
    dlength = dimValue.dimensionality['[length]']
    dtime = dimValue.dimensionality['[time]']
    dmass = dimValue.dimensionality['[mass]']
    dtemp = dimValue.dimensionality['[temperature]']
    dsubstance = dimValue.dimensionality['[substance]']

    # Build scaling factor: input_units / (scaling_units^dimensions)
    factor = (length**(-dlength) *
              time**(-dtime) *
              mass**(-dmass) *
              temperature**(-dtemp) *
              substance**(-dsubstance))

    # Apply scaling
    dimValue *= factor

    # Return dimensionless magnitude
    return dimValue.magnitude
```

### Key Insight: Scaling Coefficient Pattern
The current system uses **dimensional scaling coefficients**:
- `[length]` → characteristic length (e.g., 1000 km)
- `[time]` → characteristic time (e.g., 1 Myr)
- `[mass]` → characteristic mass (derived from viscosity scaling)

This allows any quantity to be non-dimensionalised by: `value / (L^a * T^b * M^c * ...)^dimensions`

## Enhanced Requirements

### 1. Automatic Scale Discovery
**Goal**: Automatically discover appropriate dimensional scalings from problem setup

**Inputs for Discovery**:
- Mesh geometry (characteristic lengths)
- Material properties (viscosities, densities, conductivities)
- Boundary conditions (velocities, temperatures, pressures)
- Physical constants (gravity, etc.)

**Discovery Algorithm**:
```python
def discover_scaling_automatically(mesh, materials, boundary_conditions):
    """
    Analyze problem setup and suggest optimal scaling coefficients
    """
    scales = {}

    # Length scale from mesh
    if mesh:
        scales['length'] = max_dimension(mesh.bounding_box)

    # Velocity scale from BCs
    if boundary_conditions.get('velocity'):
        scales['velocity'] = max_velocity(boundary_conditions['velocity'])

    # Material property scales
    if materials:
        scales['viscosity'] = geometric_mean([m.viscosity for m in materials])
        scales['density'] = arithmetic_mean([m.density for m in materials])

    # Derive dependent scales
    if 'length' in scales and 'velocity' in scales:
        scales['time'] = scales['length'] / scales['velocity']

    if 'viscosity' in scales and 'velocity' in scales and 'length' in scales:
        scales['pressure'] = scales['viscosity'] * scales['velocity'] / scales['length']

    return scales
```

### 2. Multiple Scaling Strategies
Support different scaling approaches for different use cases:

#### Strategy 1: "Geological Units" (Practical Approach)
```python
# Use units that make everything O(1) for geological problems
setup_geological_scaling(
    length_unit="1000*km",      # Continental scale
    time_unit="1*Myr",          # Geological time
    velocity_unit="cm/year",    # Plate motion
    pressure_unit="GPa",        # Mantle pressure
    temperature_unit="1000*K"   # Mantle temperature range
)
```

#### Strategy 2: "Problem-Specific" (Auto-Discovery)
```python
# Automatically detect from problem setup
auto_scaling = discover_scaling_from_problem(mesh, materials, bcs)
apply_scaling(auto_scaling)
```

#### Strategy 3: "Rayleigh Number" (Dimensionless Groups)
```python
# Base scaling on dimensionless parameters
setup_dimensionless_scaling(
    rayleigh_number=1e6,
    prandtl_number=1e24,
    aspect_ratio=2.0
)
```

### 3. Flexible Units Specification
Support multiple ways to specify units:

```python
# Method 1: Direct unit specification
velocity = uw.discretisation.MeshVariable("v", mesh, 2, units="cm/year")

# Method 2: Physical quantity with conversion
plate_velocity = 5 * uw.scaling.units.cm / uw.scaling.units.year
velocity.set_from_physical(plate_velocity)

# Method 3: Scale-relative specification
velocity.set_from_scale(0.1)  # 0.1 times the velocity scale

# Method 4: Dimensionless with context
velocity.set_dimensionless(0.1, context="typical_plate_motion")
```

### 4. Unit-Aware Problem Specification
```python
# High-level problem specification with units
problem = uw.ThermalConvectionProblem(
    domain=uw.Box(width=2900*uw.units.km, height=660*uw.units.km),

    materials=[
        uw.Material("upper_mantle",
                   viscosity=1e21*uw.units.Pa*uw.units.s,
                   density=3300*uw.units.kg/uw.units.m**3),
        uw.Material("lower_mantle",
                   viscosity=1e22*uw.units.Pa*uw.units.s,
                   density=3500*uw.units.kg/uw.units.m**3)
    ],

    boundary_conditions={
        'top': {'temperature': 300*uw.units.K, 'velocity': [5*uw.units.cm/uw.units.year, 0]},
        'bottom': {'temperature': 1600*uw.units.K, 'velocity': [0, 0]}
    }
)

# Auto-discover optimal scaling
scaling = problem.suggest_scaling()
# Apply scaling and solve with optimal conditioning
problem.solve()
```

## Prototype Exploration Plan

### Phase 1: Scale Discovery Experiments
**Notebook**: `units_scaling_discovery.ipynb`

1. **Manual Scale Setting**: Replicate current scaling workflow
2. **Geometric Scale Detection**: Extract scales from mesh geometry
3. **Material Scale Detection**: Extract scales from material properties
4. **Boundary Condition Scale Detection**: Extract scales from BCs
5. **Combined Auto-Detection**: Full automatic scale discovery

### Phase 2: Alternative Scaling Strategies
**Notebook**: `scaling_strategies_comparison.ipynb`

1. **SI Units Approach**: Everything in SI, let solvers handle conditioning
2. **Geological Units Approach**: Use domain-appropriate units throughout
3. **Hybrid Approach**: Input in natural units, solve in optimal units
4. **Performance Comparison**: Solver conditioning and convergence rates

### Phase 3: Units Conversion Patterns
**Notebook**: `units_conversion_patterns.ipynb`

1. **Input Flexibility**: Multiple ways to specify the same quantity
2. **Conversion Validation**: Ensure physical consistency
3. **Error Detection**: Catch common dimensional mistakes
4. **User Experience**: Most intuitive patterns for typical workflows

## Implementation Roadmap

### Phase 1: Discovery Mechanisms (Current Need)
- [ ] Prototype automatic scale discovery from problem setup
- [ ] Compare different scaling strategies numerically
- [ ] Validate that "good units" ≈ "non-dimensionalised" for solver performance
- [ ] Document best practices for scale selection

### Phase 2: Enhanced User Interface
- [ ] Flexible units specification for variables
- [ ] Automatic scale suggestion and application
- [ ] Unit-aware boundary conditions and material properties
- [ ] Validation and error checking for dimensional consistency

### Phase 3: Solver Integration
- [ ] Automatic solver-level non-dimensionalisation when units available
- [ ] Transparent scaling without user code changes
- [ ] Performance optimization for unit-aware operations
- [ ] Advanced scaling strategies for complex problems

## Success Criteria

1. **Usability**: Users can specify problems in their preferred units
2. **Performance**: Solvers achieve optimal conditioning automatically
3. **Robustness**: Dimensional errors caught before causing physics mistakes
4. **Flexibility**: Support both formal and practical scaling approaches
5. **Transparency**: Users understand what scaling is applied and why

## Open Questions for Exploration

1. **Scale Detection Reliability**: How robust is automatic scale detection across different problem types?
2. **Performance Impact**: What's the computational cost of unit tracking vs benefits of optimal scaling?
3. **User Preferences**: Do users prefer explicit control or automatic scaling?
4. **Error Handling**: What dimensional errors are most common and how to prevent them?
5. **Integration Complexity**: How to integrate with existing solver infrastructure?

---

*Updated: 2025-10-01*
*Status: Requirements defined, prototype development in progress*
*Next: Create exploration notebook to validate concepts*