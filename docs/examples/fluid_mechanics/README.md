# Fluid Mechanics Examples

Fluid mechanics forms the foundation for understanding mantle convection, magma flow, and many other geophysical processes. These examples focus on viscous flow solutions using Stokes and Navier-Stokes equations.

## 🎯 Learning Progression

### 📚 Basic Examples (`basic/`)

**Essential concepts for geophysical fluid mechanics.**

1. **Driven Cavity Flow** - `stokes_driven_cavity.py`
   - Classic benchmark problem with moving lid
   - Velocity-pressure coupling (saddle-point systems)
   - Introduces: Stokes equations, velocity boundary conditions

2. **Flow Around Objects** - `stokes_sphere_drag.py`
   - Viscous flow past a falling sphere
   - Drag force calculation and Stokes law validation
   - Introduces: object boundaries, force integration

3. **Channel Flow** - `poiseuille_flow.py`
   - Pressure-driven flow between parallel plates
   - Analytical vs numerical solution comparison
   - Introduces: pressure gradients, no-slip conditions

### 🔬 Intermediate Examples (`intermediate/`)

**More complex rheology and flow configurations.**

4. **Variable Viscosity Flow** - `temperature_dependent_viscosity.py`
   - Exponential viscosity variation with temperature
   - Strongly non-linear rheology
   - Introduces: iterative solvers, viscosity constitutive laws

5. **Non-Newtonian Rheology** - `power_law_rheology.py`
   - Shear-rate dependent viscosity
   - Yield stress and plasticity effects  
   - Introduces: stress-dependent rheology, yielding

6. **Free Surface Flow** - `surface_gravity_waves.py`
   - Deformable upper boundary under gravity
   - Surface tension effects
   - Introduces: free boundaries, interface tracking

### 🎓 Advanced Examples (`advanced/`)

**Research-level fluid mechanics problems.**

7. **Inertial Effects** - `navier_stokes_cylinder.py`
   - Flow past cylinder with Reynolds number effects
   - Vortex shedding and unsteady flow
   - Introduces: Navier-Stokes equations, time dependence

8. **Turbulent Flow Models** - `turbulent_channel_flow.py`
   - Large Eddy Simulation approach
   - Subgrid-scale models
   - Applications: atmospheric/oceanic flows

9. **Multi-Phase Flow** - `two_phase_flow.py`
   - Solid-liquid or liquid-gas systems
   - Interface dynamics and surface tension
   - Applications: magma-crystal systems, air-water flows

## 🧮 Mathematical Background

### Governing Equations

**Stokes Flow (Creeping Flow, Re << 1):**
```
∇·σ = 0          (momentum balance)
∇·v = 0          (incompressibility)
σ = -pI + η(∇v + ∇vᵀ)   (constitutive law)
```

**Navier-Stokes Flow (Finite Reynolds Number):**
```
ρ(∂v/∂t + v·∇v) = ∇·σ + f    (momentum balance)
∇·v = 0                        (incompressibility)
```

### Key Dimensionless Numbers
- **Reynolds Number**: `Re = ρvL/η` (inertia vs viscous forces)
- **Rayleigh Number**: `Ra = ρgαΔTL³/(ηκ)` (buoyancy vs diffusion)
- **Prandtl Number**: `Pr = η/(ρκ)` (momentum vs thermal diffusion)

## 🌍 Geophysical Applications

### Mantle Dynamics
- **Mantle convection**: Large-scale flow in planetary interiors
- **Subduction zones**: Slab descent and corner flow
- **Plume dynamics**: Rising hot material from deep mantle

### Crustal Processes
- **Magma transport**: Flow through dikes and conduits
- **Salt tectonics**: Viscous flow of evaporite layers
- **Glacial isostasy**: Viscoelastic response to ice loading

### Surface and Near-Surface
- **Glacier flow**: Ice as a non-Newtonian fluid
- **Lava flows**: Temperature-dependent rheology
- **Sediment transport**: Particle-laden flows

### Planetary Applications
- **Atmospheric dynamics**: Large-scale circulation patterns  
- **Ocean circulation**: Thermohaline and wind-driven flows
- **Planetary cores**: Liquid metal convection and dynamos

## 🔧 Common Parameters

### Viscosity Values
```python
# Viscosity ranges in Earth materials (Pa·s)
WATER_VISCOSITY = 1e-3          # Reference fluid
MAGMA_VISCOSITY = 1e3           # Basaltic magma
CRUSTAL_ROCK_VISCOSITY = 1e20   # Lower crust (high T)
MANTLE_VISCOSITY = 1e21         # Upper mantle  
ICE_VISCOSITY = 1e14            # Glacier ice
```

### Typical Reynolds Numbers
```python
# Reynolds numbers in geophysical flows
MANTLE_CONVECTION_RE = 1e-20    # Extremely low Re (Stokes flow)
MAGMA_FLOW_RE = 1e-6            # Very low Re (Stokes flow)
LAVA_FLOW_RE = 1e-3             # Low Re (transitional)
ATMOSPHERIC_FLOW_RE = 1e6       # High Re (turbulent)
```

## 🚀 Getting Started

1. **New to fluid mechanics?** Start with `basic/stokes_driven_cavity.py`
2. **Familiar with Stokes flow?** Try `intermediate/variable_viscosity_flow.py`
3. **Research applications?** Check `advanced/` examples for your domain
4. **Specific rheology?** Look for non-Newtonian examples

## 📚 Prerequisites

### Mathematical Background
- Vector calculus (divergence, curl, gradients)
- Partial differential equations
- Linear algebra (matrix systems)
- Numerical methods (finite elements)

### Physics Background  
- Continuum mechanics fundamentals
- Stress and strain concepts
- Conservation laws (mass, momentum)
- Constitutive relationships

### Programming Skills
- Python fundamentals
- NumPy and SciPy for numerical operations
- Understanding of iterative solvers

## 🎛️ Common Modifications

### Boundary Conditions
- **No-slip**: `v = 0` at solid boundaries
- **Free-slip**: `v·n = 0, σ·n×n = 0` at free boundaries  
- **Inflow/outflow**: Prescribed velocities or tractions

### Rheological Models
- **Newtonian**: `η = constant`
- **Temperature-dependent**: `η = η₀exp(E/RT)`
- **Stress-dependent**: `η = η₀(ε̇/ε̇₀)^((n-1)/n)`
- **Composite**: Combined temperature and stress dependence

## 🔗 Related Examples

- **Heat Transfer**: For thermal convection problems
- **Convection**: For buoyancy-driven flows  
- **Multi-Physics**: For fully coupled systems
- **Solid Mechanics**: For viscoelastic behaviors

---

*From Stokes flow fundamentals to complex rheology - these examples build the foundation for understanding geophysical fluid systems.*