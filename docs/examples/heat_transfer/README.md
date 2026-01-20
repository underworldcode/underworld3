# Heat Transfer & Diffusion Examples

Heat transfer is fundamental to many geophysical processes. These examples demonstrate solving diffusion equations using finite element methods.

## 🎯 Learning Progression

### 📚 Basic Examples (`basic/`)

**Start here if you're new to computational heat transfer or finite elements.**

1. **Steady-State Heat Conduction** - `steady_heat_2d.py`
   - Simple temperature distribution in a square domain
   - Fixed temperature boundaries
   - Introduces: mesh creation, boundary conditions, basic solving

2. **Heat Conduction with Sources** - `heat_with_sources.py`
   - Internal heat generation (radioactive decay, viscous heating)
   - Mixed boundary conditions
   - Introduces: source terms, natural boundary conditions

3. **Temperature-Dependent Properties** - `variable_conductivity.py`
   - Conductivity that varies with temperature
   - Non-linear material behavior
   - Introduces: constitutive relationships, iteration

### 🔬 Intermediate Examples (`intermediate/`)

**Build on basic concepts with more complex physics.**

4. **Transient Heat Diffusion** - `time_dependent_heating.py`
   - Time-stepping for cooling/heating processes
   - Temperature evolution over time
   - Introduces: time derivatives, implicit methods

5. **Anisotropic Heat Conduction** - `anisotropic_diffusion.py`
   - Directional thermal properties (layered materials)
   - Tensor-valued diffusivity
   - Introduces: tensor constitutive models

6. **Phase Change Problems** - `melting_solidification.py`
   - Solid-liquid transitions with latent heat
   - Stefan problems
   - Introduces: phase boundaries, enthalpy methods

### 🎓 Advanced Examples (`advanced/`)

**Research-level applications and complex physics.**

7. **Coupled Thermal-Chemical Diffusion** - `thermal_reactive_transport.py`
   - Multiple diffusing species with reactions
   - Temperature-dependent reaction rates
   - Applications: metamorphic processes, hydrothermal systems

8. **Radiative Heat Transfer** - `thermal_radiation.py`
   - Non-linear boundary conditions for radiation
   - High-temperature applications
   - Applications: planetary atmospheres, magma bodies

## 🧮 Mathematical Background

### Fundamental Equation
The heat equation in its most general form:
```
ρc ∂T/∂t = ∇·(k∇T) + H
```

Where:
- `T`: Temperature
- `ρ`: Density  
- `c`: Specific heat capacity
- `k`: Thermal conductivity (tensor)
- `H`: Heat source/sink term

### Common Boundary Conditions
- **Dirichlet**: Fixed temperature `T = T₀`
- **Neumann**: Fixed heat flux `q = -k∇T·n = q₀`
- **Robin**: Heat transfer `q = h(T - T_ambient)`

## 🌍 Geophysical Applications

### Planetary Sciences
- **Thermal evolution**: Planetary cooling over geological time
- **Geotherms**: Temperature profiles in planetary interiors
- **Impact heating**: Temperature rise from meteorite impacts

### Crustal Processes  
- **Geothermal systems**: Underground heat transport
- **Magma chambers**: Cooling and crystallization
- **Metamorphism**: Temperature-driven mineral reactions

### Surface Processes
- **Permafrost dynamics**: Seasonal freeze-thaw cycles
- **Ice sheet thermal structure**: Temperature in glaciers
- **Soil thermal regimes**: Agricultural and ecological applications

## 🔧 Common Parameters

### Material Properties
```python
# Typical values for common materials
ROCK_DENSITY = 2700.0           # kg/m³
ROCK_HEAT_CAPACITY = 1000.0     # J/(kg·K)  
ROCK_CONDUCTIVITY = 2.5         # W/(m·K)

WATER_DENSITY = 1000.0          # kg/m³
WATER_HEAT_CAPACITY = 4184.0    # J/(kg·K)
WATER_CONDUCTIVITY = 0.6        # W/(m·K)
```

### Dimensionless Numbers
- **Péclet Number**: `Pe = vL/κ` (advection vs diffusion)
- **Fourier Number**: `Fo = κt/L²` (time scale ratio)

## 🚀 Getting Started

1. **Complete beginner?** Start with `basic/steady_heat_2d.py`
2. **Familiar with heat transfer?** Jump to `intermediate/time_dependent_heating.py`  
3. **Research applications?** Check `advanced/` examples
4. **Looking for specific physics?** Use the application index above

## 📚 Prerequisites

### Mathematical Background
- Partial differential equations (basic)
- Vector calculus (gradients, divergence)
- Basic numerical methods concepts

### Programming Skills
- Python fundamentals
- NumPy for array operations
- Basic matplotlib for visualization

### Physics Background  
- Undergraduate thermodynamics
- Heat transfer mechanisms (conduction, convection, radiation)

## 🔗 Related Examples

- **Fluid Mechanics**: For convective heat transfer
- **Multi-Physics**: For fully coupled thermal-mechanical systems
- **Porous Flow**: For heat transport in groundwater systems

---

*These examples progress from fundamental concepts to research applications. Each builds skills needed for more complex geophysical modeling.*