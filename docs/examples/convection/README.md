# Thermal Convection Examples

Thermal convection combines heat transfer and fluid mechanics to model buoyancy-driven flows. This is fundamental to understanding mantle convection, atmospheric circulation, and many other geophysical processes.

## 🎯 Learning Progression

### 📚 Basic Examples (`basic/`)

**Foundation concepts for thermal convection.**

1. **Rayleigh-Benard Convection** - `rayleigh_benard_2d.py`
   - Classical onset of convection in a heated layer
   - Critical Rayleigh number and instability
   - Introduces: coupled heat-flow, buoyancy forces

2. **Convection Onset Analysis** - `linear_stability_analysis.py`
   - Theoretical vs numerical critical Rayleigh numbers
   - Mode shapes and growth rates
   - Introduces: eigenvalue problems, instability theory

3. **Steady Convection Cells** - `steady_convection_rolls.py`
   - Finite amplitude convection above critical Ra
   - Heat transfer efficiency (Nusselt number)
   - Introduces: non-linear solutions, heat flux calculations

### 🔬 Intermediate Examples (`intermediate/`)

**More realistic convection scenarios.**

4. **Temperature-Dependent Viscosity** - `variable_viscosity_convection.py`
   - Exponentially varying viscosity with temperature
   - Boundary layer formation and focusing
   - Introduces: strongly non-linear rheology effects

5. **Internal Heating** - `internally_heated_convection.py`
   - Uniform volumetric heat sources (radioactive decay)
   - Different flow patterns vs bottom heating
   - Introduces: volumetric forcing, modified scaling laws

6. **Aspect Ratio Effects** - `convection_aspect_ratio.py`
   - Wide vs narrow domains
   - Multiple convection cells and interactions
   - Introduces: domain geometry effects, cell interactions

### 🎓 Advanced Examples (`advanced/`)

**Complex convection systems.**

7. **3D Convection** - `three_dimensional_convection.py`
   - Transition from 2D rolls to 3D cells
   - Plume formation and dynamics
   - Applications: mantle convection, laboratory experiments

8. **Compositional Convection** - `double_diffusive_convection.py`
   - Chemical density variations with thermal effects
   - Fingering vs layering instabilities
   - Applications: magma chambers, ocean stratification

9. **Phase Change Convection** - `phase_change_convection.py`
   - Solid-liquid transitions in convecting systems
   - Latent heat effects on flow patterns
   - Applications: core formation, magma ocean crystallization

## 🧮 Mathematical Background

### Coupled Equations
**Momentum Balance (Boussinesq Approximation):**
```
0 = -∇p + η∇²v + ρ₀gα(T-T₀)ẑ
∇·v = 0
```

**Energy Balance:**
```
∂T/∂t + v·∇T = κ∇²T + H/ρcp
```

### Key Dimensionless Parameters

**Rayleigh Number:**
```
Ra = ρ₀gαΔTL³/(ηκ)
```
- Controls convection vigor
- Critical value Ra_c ≈ 1708 for onset

**Prandtl Number:**
```
Pr = η/(ρκ) = ν/κ
```
- Momentum vs thermal diffusion
- Earth's mantle: Pr >> 1 (thermal boundary layers)

**Nusselt Number:**
```
Nu = qL/(kΔT)
```
- Heat transfer efficiency
- Nu = 1 for pure conduction, Nu > 1 for convection

## 🌍 Geophysical Applications

### Planetary Interiors
- **Mantle convection**: Heat transfer from core to surface
- **Core convection**: Liquid iron convection driving geodynamo
- **Magma ocean convection**: Early planetary differentiation

### Crustal Systems
- **Magma chamber convection**: Crystal settling and differentiation
- **Hydrothermal circulation**: Fluid flow in heated crust
- **Salt dome convection**: Buoyant rise of evaporites

### Atmospheric & Oceanic
- **Atmospheric convection**: Thunderstorm formation, general circulation
- **Ocean convection**: Deep water formation, thermohaline circulation
- **Lake turnover**: Seasonal density-driven mixing

### Laboratory Analogues
- **Tank experiments**: Scaled laboratory convection studies
- **Centrifuge modeling**: Enhanced gravity experiments
- **Numerical benchmarks**: Code validation studies

## 🔧 Characteristic Values

### Earth's Mantle
```python
# Mantle convection parameters
MANTLE_RAYLEIGH = 1e6           # Vigorous convection
MANTLE_PRANDTL = 1e24           # Thermal boundary layer dominated
THERMAL_DIFFUSIVITY = 1e-6      # m²/s
VISCOSITY = 1e21                # Pa·s
THERMAL_EXPANSION = 3e-5        # 1/K
```

### Laboratory Experiments
```python
# Typical lab convection (water, oil)
LAB_RAYLEIGH = 1e4 - 1e8        # Accessible range
LAB_PRANDTL = 1 - 1000          # Fluid dependent
ASPECT_RATIO = 1 - 10           # Width/height ratio
```

## 🚀 Getting Started

1. **New to convection?** Start with `basic/rayleigh_benard_2d.py`
2. **Understanding onset?** Try `basic/linear_stability_analysis.py`
3. **Mantle applications?** Check `intermediate/variable_viscosity_convection.py`
4. **Advanced physics?** Explore `advanced/` examples

## 📚 Prerequisites

### Mathematical Background
- Partial differential equations (heat equation, Stokes flow)
- Dimensionless analysis and scaling
- Linear stability theory (basic)
- Vector calculus

### Physics Background
- Fluid mechanics fundamentals
- Heat transfer mechanisms
- Buoyancy and density effects
- Thermal expansion concepts

### Computational Skills
- Understanding of coupled PDEs
- Non-linear solver concepts
- Boundary value problems

## 🎛️ Parameter Exploration

### Critical Phenomena
- **Onset of convection**: Ra_c determination
- **Mode selection**: Preferred wavelengths
- **Bifurcations**: Transitions between flow states

### Scaling Laws
- **Heat transfer**: Nu vs Ra relationships
- **Boundary layer thickness**: δ ~ Ra^(-1/4)
- **Convection velocity**: v ~ Ra^(2/3)

### Rheological Effects
- **Temperature-dependent viscosity**: Stagnant lid formation
- **Stress-dependent rheology**: Focusing and localization
- **Composite rheology**: Multiple deformation mechanisms

## 🔗 Related Examples

- **Heat Transfer**: For thermal diffusion fundamentals
- **Fluid Mechanics**: For flow solver techniques
- **Multi-Physics**: For additional coupling (chemical, magnetic)
- **Solid Mechanics**: For viscoelastic convection

---

*These examples progress from fundamental convection concepts to complex multi-physics systems relevant to planetary science and geophysics.*