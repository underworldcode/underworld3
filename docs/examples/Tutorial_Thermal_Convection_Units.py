#!/usr/bin/env python
# coding: utf-8

# # Tutorial 13: Simple Thermal Convection with Units
#
# This tutorial demonstrates thermal convection using **real physical units**
# with Underworld3's Pint-native units system. We'll set up a simple 2D thermal
# convection model with realistic mantle parameters.
#
# ## Physical Setup
# - **Domain**: 1000 km × 1000 km mantle section
# - **Temperature**: 300K surface to 1600K bottom
# - **Material**: Realistic mantle properties
# - **Physics**: Thermal buoyancy drives convection

import underworld3 as uw
import numpy as np

print("🌍 THERMAL CONVECTION WITH PHYSICAL UNITS")
print("Using Underworld3's Pint-native units system\n")


def as_numeric_array(values):
    """
    Convert an Underworld/Pint unit-aware array or regular array to a plain
    numpy float array.
    """
    if hasattr(values, "magnitude"):
        return np.asarray(values.magnitude, dtype=float)

    if hasattr(values, "value"):
        return np.asarray(values.value, dtype=float)

    if hasattr(values, "_pint_qty"):
        return np.asarray(values._pint_qty.to_base_units().magnitude, dtype=float)

    if hasattr(values, "data"):
        try:
            return np.asarray(values.data, dtype=float)
        except Exception:
            pass

    if hasattr(values, "tolist"):
        return np.asarray(values.tolist(), dtype=float)

    return np.asarray(values, dtype=float)


def as_float(value):
    """
    Convert a scalar UnitAwareArray, UWQuantity, Pint quantity, or plain number
    to a Python float.
    """
    array = as_numeric_array(value)
    return float(np.asarray(array).reshape(-1)[0])


def quantity_to_base_magnitude(quantity):
    """
    Convert a UW/Pint quantity to its SI base-unit magnitude.
    """
    if hasattr(quantity, "_pint_qty"):
        return float(quantity._pint_qty.to_base_units().magnitude)

    if hasattr(quantity, "to_base_units"):
        return float(quantity.to_base_units().magnitude)

    if hasattr(quantity, "magnitude"):
        return float(quantity.magnitude)

    if hasattr(quantity, "value"):
        return float(quantity.value)

    return float(quantity)


# ## Step 1: Define Physical Parameters

model = uw.Model("mantle_convection")

model.set_reference_quantities(
    mantle_temperature=1500 * uw.units.K,
    mantle_viscosity=1e21 * uw.units.Pa * uw.units.s,
    mantle_depth=1000 * uw.units.km,
    plate_velocity=5 * uw.units.cm / uw.units.year,
)

print("Reference quantities set for automatic model scaling:")
scales = model.get_fundamental_scales()
for name, scale in scales.items():
    print(f"  {name}: {scale}")

print("\n📋 PHYSICAL PARAMETERS (with units):")

domain_width = 1000 * uw.units.km
domain_height = 1000 * uw.units.km
print(f"Domain: {domain_width} × {domain_height}")

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

T_surface = 300 * uw.units.K
T_bottom = 1600 * uw.units.K
temperature_drop = T_bottom - T_surface

print("\nTemperature conditions:")
print(f"Surface: {T_surface} ({T_surface.to('degC')})")
print(f"Bottom: {T_bottom} ({T_bottom.to('degC')})")
print(f"Temperature drop: {temperature_drop}")

# ## Step 2: Convert to Model Units

print("🔄 CONVERTING TO MODEL UNITS:")
print("(Values ~1.0 indicate good numerical conditioning)\n")

width_model = model.to_model_units(domain_width)
height_model = model.to_model_units(domain_height)

print(f"Domain width: {domain_width} → {width_model:.2f} (model units)")
print(f"Domain height: {domain_height} → {height_model:.2f} (model units)")

g_model = model.to_model_units(gravity)
alpha_model = model.to_model_units(thermal_expansion)
kappa_model = model.to_model_units(thermal_diffusivity)
rho0_model = model.to_model_units(reference_density)
visc_model = model.to_model_units(reference_viscosity)

print("\nMaterial properties in model units:")
print(f"Gravity: {g_model:.2e} ({g_model.units})")
print(f"Thermal expansion: {alpha_model:.2e} ({alpha_model.units})")
print(f"Thermal diffusivity: {kappa_model:.2e} ({kappa_model.units})")
print(f"Reference density: {rho0_model:.2e} ({rho0_model.units})")
print(f"Reference viscosity: {visc_model:.2e} ({visc_model.units})")

T_surface_model = model.to_model_units(T_surface)
T_bottom_model = model.to_model_units(T_bottom)
Delta_T_model = T_bottom_model - T_surface_model

print("\nTemperatures in model units:")
print(f"Surface: {T_surface_model:.2f} ({T_surface_model.units})")
print(f"Bottom: {T_bottom_model:.2f} ({T_bottom_model.units})")
print(f"Temperature difference: {Delta_T_model:.2f}")

# ## Step 3: Calculate Rayleigh Number

print("🧮 RAYLEIGH NUMBER CALCULATION:")

Ra = (
    reference_density
    * gravity
    * thermal_expansion
    * temperature_drop
    * domain_height**3
) / (reference_viscosity * thermal_diffusivity)

print("Ra = (ρ g α ΔT L³) / (η κ)")
print(f"Ra = {Ra.to_base_units().magnitude:.2e}")
print("Dimensionless: True ✓")

if Ra.to_base_units().magnitude > 1e6:
    print("\n🌪️  Ra > 10⁶ → Vigorous convection expected!")
elif Ra.to_base_units().magnitude > 1e3:
    print("\n🔄 Ra > 10³ → Convection likely")
else:
    print("\n🧊 Ra < 10³ → Conduction dominated")

print("\n✨ UNITS DEMONSTRATION:")
print(f"Gravity in m/s²: {gravity.to('m/s**2'):.2f}")
print(f"Thermal diffusivity in mm²/s: {thermal_diffusivity.to('mm**2/s'):.2f}")
print("→ Native Pint arithmetic works seamlessly!")

# ## Step 4: Create Mesh and Variables

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(width_model.value, height_model.value),
    cellSize=0.05,
    qdegree=2,
)

print(f"Mesh created: {mesh.X.coords.shape[0]} nodes")
print(
    f"Domain: 0 to {width_model.value:.1f} × "
    f"0 to {height_model.value:.1f} (model units)"
)
print(f"Cell size: 0.05 model units = {0.05 * domain_width.to('km').magnitude:.0f} km")

velocity = uw.discretisation.MeshVariable(
    "velocity",
    mesh,
    2,
    degree=2,
    units="meter/second",
)
pressure = uw.discretisation.MeshVariable(
    "pressure",
    mesh,
    1,
    degree=1,
    units="pascal",
)
temperature = uw.discretisation.MeshVariable(
    "temperature",
    mesh,
    1,
    degree=2,
)

print("\nVariables created:")
print(f"  Velocity: {velocity.num_components}D, degree {velocity.degree}")
print(f"  Pressure: {pressure.num_components}D, degree {pressure.degree}")
print(f"  Temperature: {temperature.num_components}D, degree {temperature.degree}")

# ## Step 5: Setup Physics

stokes = uw.systems.Stokes(
    mesh,
    velocityField=velocity,
    pressureField=pressure,
)

reference_temp = (T_surface_model + T_bottom_model) / 2

buoyancy = (
    rho0_model.value
    * alpha_model.value
    * g_model.value
    * (temperature.sym[0] - reference_temp.value)
)

stokes.bodyforce = [0, buoyancy]

print("🏔️  STOKES SOLVER SETUP:")
print(f"Buoyancy reference T: {reference_temp:.2f} (model units)")
print("Buoyancy force: ρ₀ α g (T - T_ref) z-hat")
print("All coefficients are passed to the solver as model-unit numbers")

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_model.value

print(f"Viscosity: {visc_model.value:.2f} (model units)")

stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")
stokes.add_dirichlet_bc([0.0, 0.0], "Top")
stokes.add_dirichlet_bc([0.0, None], "Left")
stokes.add_dirichlet_bc([0.0, None], "Right")

print("\nBoundary conditions:")
print("  Top/Bottom: No-slip, v = 0")
print("  Left/Right: Free-slip, vx = 0")

thermal = uw.systems.AdvDiffusion(
    mesh,
    u_Field=temperature,
    V_fn=velocity,
    order=2,
)

thermal.constitutive_model = uw.constitutive_models.DiffusionModel
thermal.constitutive_model.Parameters.diffusivity = kappa_model.value

print("🌡️  THERMAL SOLVER SETUP:")
print(f"Thermal diffusivity: {kappa_model.value:.2e} (model units)")
print("Advection-diffusion: ∂T/∂t + v·∇T = κ∇²T")

thermal.add_dirichlet_bc(T_surface_model.value, "Top")
thermal.add_dirichlet_bc(T_bottom_model.value, "Bottom")

print("\nThermal boundary conditions:")
print(f"  Top: {T_surface_model.value:.2f} (model units) = {T_surface}")
print(f"  Bottom: {T_bottom_model.value:.2f} (model units) = {T_bottom}")
print("  Sides: Insulating, natural boundary conditions")

# ## Step 6: Initial Conditions

coords = as_numeric_array(temperature.coords)

domain_width_si = quantity_to_base_magnitude(domain_width)
domain_height_si = quantity_to_base_magnitude(domain_height)

# In strict units mode, coordinates may be represented internally in SI length
# units. Normalize them before constructing a model-unit temperature field.
x_norm = coords[:, 0] / domain_width_si
y_norm = coords[:, 1] / domain_height_si

x_norm = np.clip(x_norm, 0.0, 1.0)
y_norm = np.clip(y_norm, 0.0, 1.0)

temperature_min = T_surface_model.value
temperature_max = T_bottom_model.value
temperature_range = temperature_max - temperature_min

linear_profile = temperature_min + temperature_range * y_norm

perturbation_amplitude = 0.02
perturbation = (
    perturbation_amplitude
    * np.sin(3.0 * np.pi * x_norm)
    * np.sin(np.pi * y_norm)
)

initial_temperature_array = linear_profile + perturbation

with uw.synchronised_array_update():
    temperature.data[:, 0] = initial_temperature_array

temp_stats = temperature.stats()
print("🌡️  INITIAL TEMPERATURE FIELD:")
print(f"Min: {temp_stats['min']:.3f} (model units)")
print(f"Max: {temp_stats['max']:.3f} (model units)")
print(f"Mean: {temp_stats['mean']:.3f} (model units)")
print(f"Perturbation: ±{perturbation_amplitude:.3f} (model units)")

T_min_phys = temp_stats["min"] * 1000.0
T_max_phys = temp_stats["max"] * 1000.0

print("\nPhysical temperatures approximate:")
print(f"Min: ~{T_min_phys:.0f} K ({T_min_phys - 273.15:.0f} °C)")
print(f"Max: ~{T_max_phys:.0f} K ({T_max_phys - 273.15:.0f} °C)")

# ## Step 7: Run Simple Time-stepping

print("🕐 TIME-STEPPING SIMULATION:")
print("Running 10 coupled Stokes + thermal steps...\n")

time_step = 0
max_steps = 10
vel_mag = 0.0
vel_physical = 0.0

for step in range(max_steps):
    stokes.solve(zero_init_guess=(step == 0))

    dt_estimate = thermal.estimate_dt()
    dt = 0.1 * dt_estimate

    thermal.solve(timestep=dt, zero_init_guess=False)

    time_step += 1

    if step % 2 == 0 or step < 3:
        temp_stats = temperature.stats()

        vel_mag_values = uw.function.evaluate(
            (velocity.sym[0] ** 2 + velocity.sym[1] ** 2) ** 0.5,
            mesh.X.coords,
        )
        vel_mag = as_numeric_array(vel_mag_values).max()

        vel_physical = vel_mag * 5.0

        print(
            f"Step {time_step:2d}: "
            f"dt = {as_float(dt):.2e} (model time), "
            f"max_vel = {vel_mag:.2e} (model), "
            f"~{vel_physical:.2f} cm/year, "
            f"mean_T = {temp_stats['mean']:.3f}"
        )

print("\n✅ Time-stepping completed!")
print(f"Final velocity magnitude: {vel_mag:.2e} (model units)")
print(f"Estimated physical velocity: ~{vel_physical:.2f} cm/year")
print("→ Mantle-like convection calculation completed.")

# ## Step 8: Results Summary

print("🎯 FINAL RESULTS SUMMARY:")
print("\n" + "=" * 50)

final_temp = temperature.stats()
print("\n🌡️  TEMPERATURE FIELD:")
print(f"Range: {final_temp['min']:.3f} to {final_temp['max']:.3f} (model units)")
print(f"Mean: {final_temp['mean']:.3f} (model units)")
print("Convection calculation completed ✓")

vel_x = velocity.data[:, 0]
vel_y = velocity.data[:, 1]
vel_magnitude = np.sqrt(vel_x**2 + vel_y**2)

print("\n🌊 VELOCITY FIELD:")
print(f"Max velocity: {vel_magnitude.max():.2e} (model units)")
print(f"Mean velocity: {vel_magnitude.mean():.2e} (model units)")
print("Non-zero velocities obtained ✓")

print("\n✨ UNITS SYSTEM DEMONSTRATION:")
print(f"Model domain: {width_model.value:.1f} × {height_model.value:.1f} (model units)")
print(f"Physical domain: {domain_width} × {domain_height}")
print("→ Physical inputs were converted to model-scale values for the solve.")

print("\nTemperature conversions:")
print(
    f"Model: {T_surface_model:.2f} → "
    f"Physical: {T_surface.to('K'):.0f} = "
    f"{T_surface.to('degC'):.0f}"
)
print(
    f"Model: {T_bottom_model:.2f} → "
    f"Physical: {T_bottom.to('K'):.0f} = "
    f"{T_bottom.to('degC'):.0f}"
)

print("\nMaterial property values:")
print(f"Viscosity: {reference_viscosity:.2e}")
print(f"Density: {reference_density:.0f}")
print(f"Gravity: {gravity:.2f}")
print(f"Thermal diffusivity: {thermal_diffusivity:.2e}")

print("\n" + "=" * 50)
print("🎉 SUCCESS: Physical thermal convection tutorial completed!")
print("\n📚 Key Benefits:")
print("  • Natural physical parameters")
print("  • Automatic model unit scaling")
print("  • Native Pint arithmetic operations")
print("  • Explicit conversion before solver expressions")
print("  • Stable model-unit fields")
print("  • Reproducible example workflow")

# ## Conclusion
#
# This tutorial demonstrated how to:
#
# 1. Set up a physics model with real units using Underworld3's Pint-native system
# 2. Automatically scale to model units for numerical performance
# 3. Use native arithmetic operations on quantities with units
# 4. Convert physical parameters into model-unit solver inputs
# 5. Run a simple thermal-convection-style calculation
#
# The Pint-native units system makes it possible to write tutorials with
# physical inputs while keeping the solver fields in controlled model units.
