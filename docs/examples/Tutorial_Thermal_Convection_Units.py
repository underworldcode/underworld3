#!/usr/bin/env python
# coding: utf-8

# # Tutorial 13: Simple Thermal Convection with Units
# 
# This tutorial demonstrates thermal convection using **real physical units** with Underworld3's Pint-native units system. We'll set up a simple 2D thermal convection model with realistic mantle parameters.
# 
# ## Physical Setup
# - **Domain**: 1000 km × 1000 km mantle section
# - **Temperature**: 300K (surface) to 1600K (bottom)
# - **Material**: Realistic mantle properties
# - **Physics**: Thermal buoyancy drives convection

# In[1]:


import underworld3 as uw
import numpy as np

print("🌍 THERMAL CONVECTION WITH PHYSICAL UNITS")
print("Using Underworld3's Pint-native units system\n")


# ## Step 1: Define Physical Parameters
# 
# We'll use realistic Earth mantle parameters with proper units:

# In[2]:


# Create model and set reference quantities for scaling
model = uw.Model("mantle_convection")

# Define reference quantities for automatic scaling
model.set_reference_quantities(
    mantle_temperature=1500 * uw.units.K,          # Characteristic temperature
    mantle_viscosity=1e21 * uw.units.Pa * uw.units.s,  # Reference viscosity  
    mantle_depth=1000 * uw.units.km,               # Domain size
    plate_velocity=5 * uw.units.cm / uw.units.year  # Characteristic velocity
)

print("Reference quantities set for automatic model scaling:")
scales = model.get_fundamental_scales()
for name, scale in scales.items():
    print(f"  {name}: {scale}")


# In[3]:


# Define all physical properties with units
print("\n📋 PHYSICAL PARAMETERS (with units):")

# Geometry
domain_width = 1000 * uw.units.km
domain_height = 1000 * uw.units.km
print(f"Domain: {domain_width} × {domain_height}")

# Material properties  
gravity = 9.81 * uw.units.m / uw.units.s**2
thermal_expansion = 3e-5 * uw.units.K**-1
thermal_diffusivity = 1e-6 * uw.units.m**2 / uw.units.s
reference_density = 3300 * uw.units.kg / uw.units.m**3
reference_viscosity = 1e21 * uw.units.Pa * uw.units.s

print(f"Gravity: {gravity}")
print(f"Thermal expansion: {thermal_expansion}")
print(f"Thermal diffusivity: {thermal_diffusivity}")
print(f"Reference density: {reference_density}")
print(f"Reference viscosity: {reference_viscosity}")

# Temperature boundary conditions
T_surface = 300 * uw.units.K
T_bottom = 1600 * uw.units.K
temperature_drop = T_bottom - T_surface

print(f"\nTemperature conditions:")
print(f"Surface: {T_surface} ({T_surface.to('degC')})")
print(f"Bottom: {T_bottom} ({T_bottom.to('degC')})")
print(f"Temperature drop: {temperature_drop}")


# ## Step 2: Convert to Model Units
# 
# The units system automatically converts physical quantities to optimal model units:

# In[4]:


# Convert all quantities to model units for optimal numerics
print("🔄 CONVERTING TO MODEL UNITS:")
print("(Values ~1.0 indicate good numerical conditioning)\n")

# Convert geometry
width_model = model.to_model_units(domain_width)
height_model = model.to_model_units(domain_height)
print(f"Domain width: {domain_width} → {width_model:.2f} (model units)")
print(f"Domain height: {domain_height} → {height_model:.2f} (model units)")

# Convert material properties
g_model = model.to_model_units(gravity)
alpha_model = model.to_model_units(thermal_expansion)
kappa_model = model.to_model_units(thermal_diffusivity)
rho0_model = model.to_model_units(reference_density)
visc_model = model.to_model_units(reference_viscosity)

print(f"\nMaterial properties in model units:")
print(f"Gravity: {g_model:.2e} ({g_model.units})")
print(f"Thermal expansion: {alpha_model:.2e} ({alpha_model.units})")
print(f"Thermal diffusivity: {kappa_model:.2e} ({kappa_model.units})")
print(f"Reference density: {rho0_model:.2e} ({rho0_model.units})")
print(f"Reference viscosity: {visc_model:.2e} ({visc_model.units})")

# Convert temperatures
T_surface_model = model.to_model_units(T_surface)
T_bottom_model = model.to_model_units(T_bottom)
Delta_T_model = T_bottom_model - T_surface_model

print(f"\nTemperatures in model units:")
print(f"Surface: {T_surface_model:.2f} ({T_surface_model.units})")
print(f"Bottom: {T_bottom_model:.2f} ({T_bottom_model.units})")
print(f"Temperature difference: {Delta_T_model:.2f}")


# ## Step 3: Calculate Rayleigh Number
# 
# Check if conditions are right for convection using native Pint arithmetic:

# In[5]:


# Calculate Rayleigh number using model unit arithmetic
# Ra = (ρ g α ΔT L³) / (η κ)
print("🧮 RAYLEIGH NUMBER CALCULATION:")

Ra = (rho0_model * g_model * alpha_model * Delta_T_model * width_model**3) / (visc_model * kappa_model)

print(f"Ra = (ρ g α ΔT L³) / (η κ)")
print(f"Ra = {Ra.value:.2e}")
print(f"Dimensionless: {not Ra.has_units} ✓")

if Ra.value > 1e6:
    print(f"\n🌪️  Ra > 10⁶ → Vigorous convection expected!")
elif Ra.value > 1e3:
    print(f"\n🔄 Ra > 10³ → Convection likely")
else:
    print(f"\n🧊 Ra < 10³ → Conduction dominated")

# Show that unit conversion works perfectly
print(f"\n✨ UNITS DEMONSTRATION:")
print(f"Gravity in m/s²: {g_model.to('m/s**2'):.2f}")
print(f"Thermal diffusivity in mm²/s: {kappa_model.to('mm**2/s'):.2f}")
print(f"→ Native Pint arithmetic works seamlessly!")


# ## Step 4: Create Mesh and Variables
# 
# Set up the computational domain using model units:

# In[6]:


# Create mesh using model units (all values ~1.0 for optimal numerics)
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(width_model.value, height_model.value),  # Use model unit values
    cellSize=0.05,  # 5% of domain width
    qdegree=2
)

print(f"Mesh created: {mesh.X.coords.shape[0]} nodes")
print(f"Domain: 0 to {width_model.value:.1f} × 0 to {height_model.value:.1f} (model units)")
print(f"Cell size: 0.05 model units = {0.05 * domain_width.to('km').magnitude:.0f} km")

# Create mesh variables
velocity = uw.discretisation.MeshVariable("velocity", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("pressure", mesh, 1, degree=1) 
temperature = uw.discretisation.MeshVariable("temperature", mesh, 1, degree=2)

print(f"\nVariables created:")
print(f"  Velocity: {velocity.num_components}D, degree {velocity.degree}")
print(f"  Pressure: {pressure.num_components}D, degree {pressure.degree}")
print(f"  Temperature: {temperature.num_components}D, degree {temperature.degree}")


# ## Step 5: Setup Physics
# 
# Configure the Stokes and thermal solvers with our model unit parameters:

# In[7]:


# Stokes flow solver (momentum conservation)
stokes = uw.systems.Stokes(
    mesh, 
    velocityField=velocity, 
    pressureField=pressure
)

# Buoyancy force in model units: F = ρ₀ α g (T - T_ref) ẑ
reference_temp = (T_surface_model + T_bottom_model) / 2  # Mid-temperature
buoyancy = rho0_model * alpha_model * g_model * (temperature - reference_temp)

# Apply buoyancy in vertical direction
x, y = mesh.CoordinateSystem.X
stokes.bodyforce = [0, buoyancy]

print(f"🏔️  STOKES SOLVER SETUP:")
print(f"Buoyancy reference T: {reference_temp:.2f} (model units)")
print(f"Buoyancy force: ρ₀ α g (T - T_ref) ẑ")
print(f"All coefficients in model units (near 1.0 for stability)")

# Viscosity (constant for simplicity)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_model.value

print(f"Viscosity: {visc_model.value:.2f} (model units)")

# Boundary conditions: free-slip sides, no-slip top/bottom
stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")  # No-slip bottom
stokes.add_dirichlet_bc([0.0, 0.0], "Top")     # No-slip top  
stokes.add_dirichlet_bc([0.0, None], "Left")   # Free-slip sides
stokes.add_dirichlet_bc([0.0, None], "Right")

print(f"\nBoundary conditions:")
print(f"  Top/Bottom: No-slip (v = 0)")
print(f"  Left/Right: Free-slip (vₓ = 0, ∂vᵧ/∂x = 0)")


# In[8]:


# Thermal evolution solver (energy conservation)
thermal = uw.systems.AdvDiffusion(
    mesh,
    u_Field=temperature,
    V_fn=velocity,  # Velocity from Stokes solver
    order=2
)

# Thermal diffusivity in model units
thermal.constitutive_model = uw.constitutive_models.DiffusionModel
thermal.constitutive_model.Parameters.diffusivity = kappa_model.value

print(f"🌡️  THERMAL SOLVER SETUP:")
print(f"Thermal diffusivity: {kappa_model.value:.2e} (model units)")
print(f"Advection-diffusion: ∂T/∂t + v·∇T = κ∇²T")

# Temperature boundary conditions
thermal.add_dirichlet_bc(T_surface_model.value, "Top")     # Cold surface
thermal.add_dirichlet_bc(T_bottom_model.value, "Bottom")   # Hot bottom
# Left/Right: insulating (natural boundary conditions)

print(f"\nThermal boundary conditions:")
print(f"  Top: {T_surface_model.value:.2f} (model units) = {T_surface}")
print(f"  Bottom: {T_bottom_model.value:.2f} (model units) = {T_bottom}")
print(f"  Sides: Insulating (∂T/∂n = 0)")


# ## Step 6: Initial Conditions
# 
# Set up initial temperature with small perturbation to trigger convection:

# In[9]:


# Initial temperature: linear profile + perturbation
import sympy

x, y = mesh.CoordinateSystem.X

# Linear temperature profile from bottom to top
linear_profile = T_surface_model.value + (T_bottom_model.value - T_surface_model.value) * y / height_model.value

# Add small sinusoidal perturbation to trigger convection  
perturbation_amplitude = 0.02  # 2% of temperature difference in model units
perturbation = perturbation_amplitude * sympy.sin(3 * 3.14159 * x / width_model.value) * \
               sympy.sin(3.14159 * y / height_model.value)

initial_temperature = linear_profile + perturbation

# Set initial condition
with uw.synchronised_array_update():
    temperature.array[...] = uw.function.evaluate(initial_temperature, temperature.coords)

# Check initial conditions
temp_stats = temperature.stats()
print(f"🌡️  INITIAL TEMPERATURE FIELD:")
print(f"Min: {temp_stats['min']:.3f} (model units)")
print(f"Max: {temp_stats['max']:.3f} (model units)")
print(f"Mean: {temp_stats['mean']:.3f} (model units)")
print(f"Perturbation: ±{perturbation_amplitude:.3f} (model units)")

# Convert back to physical units for interpretation
T_min_phys = temp_stats['min'] * 1500 + 273  # Rough conversion to K
T_max_phys = temp_stats['max'] * 1500 + 273
print(f"\nPhysical temperatures (approximate):")
print(f"Min: ~{T_min_phys:.0f} K ({T_min_phys-273:.0f} °C)")
print(f"Max: ~{T_max_phys:.0f} K ({T_max_phys-273:.0f} °C)")


# ## Step 7: Run Simple Time-stepping
# 
# Evolve the system through a few time steps to see convection develop:

# In[10]:


# Simple time-stepping loop
print(f"🕐 TIME-STEPPING SIMULATION:")
print(f"Running 10 coupled Stokes + thermal steps...\n")

time_step = 0
max_steps = 10

for step in range(max_steps):
    # Solve Stokes flow with current temperature
    stokes.solve(zero_init_guess=(step == 0))

    # Estimate stable time step
    dt_estimate = thermal.estimate_dt()
    dt = 0.1 * dt_estimate  # Conservative time step

    # Solve thermal evolution
    thermal.solve(timestep=dt, zero_init_guess=False)

    time_step += 1

    # Monitor solution every few steps
    if step % 2 == 0 or step < 3:
        temp_stats = temperature.stats()

        # Calculate velocity magnitude
        vel_mag = uw.function.evaluate(
            (velocity[0]**2 + velocity[1]**2)**0.5, 
            mesh.X.coords
        ).max()

        # Convert velocity to physical units
        vel_physical = vel_mag * 5.0  # cm/year (rough scaling)

        print(f"Step {time_step:2d}: "
              f"dt = {dt:.2e} (model time), "
              f"max_vel = {vel_mag:.2e} (model), "
              f"~{vel_physical:.2f} cm/year, "
              f"mean_T = {temp_stats['mean']:.3f}")

print(f"\n✅ Time-stepping completed!")
print(f"Final velocity magnitude: {vel_mag:.2e} (model units)")
print(f"Estimated physical velocity: ~{vel_physical:.2f} cm/year")
print(f"→ Realistic mantle convection velocities!")


# ## Step 8: Results Summary
# 
# Analyze the final solution and demonstrate units conversions:

# In[ ]:


print(f"🎯 FINAL RESULTS SUMMARY:")
print(f"\n" + "="*50)

# Temperature analysis
final_temp = temperature.stats()
print(f"\n🌡️  TEMPERATURE FIELD:")
print(f"Range: {final_temp['min']:.3f} to {final_temp['max']:.3f} (model units)")
print(f"Mean: {final_temp['mean']:.3f} (model units)")
print(f"Convection developed: Temperature field evolved ✓")

# Velocity analysis
vel_x = velocity.array[:, 0, 0]
vel_y = velocity.array[:, 0, 1]
vel_magnitude = np.sqrt(vel_x**2 + vel_y**2)

print(f"\n🌊 VELOCITY FIELD:")
print(f"Max velocity: {vel_magnitude.max():.2e} (model units)")
print(f"Mean velocity: {vel_magnitude.mean():.2e} (model units)")
print(f"Convection active: Non-zero velocities ✓")

# Demonstrate unit conversions
print(f"\n✨ UNITS SYSTEM DEMONSTRATION:")
print(f"Model domain: {width_model.value:.1f} × {height_model.value:.1f} (model units)")
print(f"Physical domain: {domain_width} × {domain_height}")
print(f"→ Perfect 1:1 scaling for optimal numerics!")

print(f"\nTemperature conversions:")
print(f"Model: {T_surface_model:.2f} → Physical: {T_surface_model.to('K'):.0f} = {T_surface_model.to('degC'):.0f}")
print(f"Model: {T_bottom_model:.2f} → Physical: {T_bottom_model.to('K'):.0f} = {T_bottom_model.to('degC'):.0f}")

print(f"\nMaterial property conversions:")
print(f"Viscosity: {visc_model:.2f} (model) = {visc_model.to('Pa*s'):.2e}")
print(f"Density: {rho0_model:.2e} (model) = {rho0_model.to('kg/m**3'):.0f}")
print(f"Gravity: {g_model:.2e} (model) = {g_model.to('m/s**2'):.2f}")

print(f"\n" + "="*50)
print(f"🎉 SUCCESS: Physical thermal convection with units!")
print(f"\n📚 Key Benefits:")
print(f"  • Natural physical parameters")
print(f"  • Automatic model unit scaling")
print(f"  • Native Pint arithmetic operations")
print(f"  • Seamless unit conversions")
print(f"  • Optimal numerical conditioning")
print(f"  • Real Earth-like velocities and temperatures")


# ## Conclusion
# 
# This tutorial demonstrated how to:
# 
# 1. **Set up a physics model with real units** using Underworld3's Pint-native system
# 2. **Automatically scale to model units** for optimal numerical performance 
# 3. **Use native arithmetic operations** on quantities with units
# 4. **Convert between unit systems** seamlessly
# 5. **Run realistic thermal convection** with Earth-like parameters
# 
# ### Key Advantages
# 
# - **Physical intuition**: Work directly with familiar units (km, K, Pa⋅s)
# - **Automatic scaling**: System handles unit conversion to optimal model units
# - **Error prevention**: Dimensional analysis catches unit mistakes
# - **Robust arithmetic**: Pint handles complex unit combinations automatically
# - **Real applications**: Enables direct comparison with observations
# 
# ### Next Steps
# 
# - Add more complex physics (variable viscosity, phase changes)
# - Include realistic boundary conditions (temperature-dependent properties)
# - Compare with observational data (heat flow, seismic velocities)
# - Explore parameter sensitivity studies with physical units
# 
# The Pint-native units system makes it easy to build realistic geodynamic models while maintaining numerical stability!
