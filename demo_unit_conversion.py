#!/usr/bin/env python3
"""
Demonstration of accessing rounding modes and converting quantities to better units.
"""

import underworld3 as uw

print("="*70)
print("UNIT ROUNDING MODE AND CONVERSION DEMO")
print("="*70)
print()

# ============================================================================
# ACCESSING THE ROUNDING MODE
# ============================================================================
print("1. Accessing the Rounding Mode")
print("-" * 70)

# Reset and get model
uw.reset_default_model()
model = uw.get_default_model()

# Check current mode
print(f"Current rounding mode: '{model.unit_rounding_mode}'")
print()

# Change to engineering mode
model.unit_rounding_mode = "engineering"
print(f"Changed to: '{model.unit_rounding_mode}'")
print()

# Change back to powers_of_10
model.unit_rounding_mode = "powers_of_10"
print(f"Changed back to: '{model.unit_rounding_mode}'")
print()

# ============================================================================
# CONVERTING QUANTITIES TO BETTER UNITS
# ============================================================================
print("="*70)
print("2. Converting Quantities to Better Units")
print("-" * 70)
print()

# Create quantities with units
velocity = uw.quantity(5, "cm/year")
print(f"Original velocity: {velocity}")
print(f"  Type: {type(velocity)}")
print()

# Convert to different units using .to() method
velocity_mps = velocity.to("m/s")
velocity_mmyr = velocity.to("mm/year")
velocity_kmmyr = velocity.to("km/Myr")  # kilometers per million years

print("Conversions:")
print(f"  {velocity} → {velocity_mps}")
print(f"  {velocity} → {velocity_mmyr}")
print(f"  {velocity} → {velocity_kmmyr}")
print()

# ============================================================================
# EXPLORING PINT'S COMPACT UNITS
# ============================================================================
print("="*70)
print("3. Using Pint's Built-in Compact Units")
print("-" * 70)
print()

# Create a quantity with awkward units
distance = uw.quantity(1000000, "mm")  # 1 million millimeters
print(f"Awkward units: {distance}")

# Convert to better units
distance_m = distance.to("m")
distance_km = distance.to("km")

print(f"Better representation: {distance_m}")
print(f"Even better: {distance_km}")
print()

# ============================================================================
# WORKING WITH MODEL UNITS
# ============================================================================
print("="*70)
print("4. Model Units and Human-Readable Interpretation")
print("-" * 70)
print()

# Set up model with reference quantities
uw.reset_default_model()
model = uw.get_default_model()

model.set_reference_quantities(
    reference_viscosity=uw.quantity(1, "ZPa.s"),
    reference_density=uw.quantity(3000, "kg/(m^3)"),
    domain_depth=uw.quantity(500, "m"),
    mantle_temperature=uw.quantity(1000, "K"),
)

# Create a physical quantity
L_x = uw.quantity(1000, "m")

# Convert to model units
L_x_model = model.to_model_units(L_x)

print(f"Physical quantity: {L_x}")
print(f"In model units: {L_x_model}")
print()

# The model units automatically show human-readable interpretation
print("Note: Model units automatically display with human-readable interpretation!")
print(f"  Technical: {L_x_model.units}")
print(f"  Display: {L_x_model}")
print()

# ============================================================================
# CHECKING IF QUANTITIES HAVE UNITS
# ============================================================================
print("="*70)
print("5. Checking Quantity Properties")
print("-" * 70)
print()

# Create different quantities
q1 = uw.quantity(5, "cm/year")
q2 = uw.quantity(100)  # Dimensionless
q3 = model.to_model_units(uw.quantity(500, "m"))

print(f"q1 = {q1}")
print(f"  Has units? {q1.has_units}")
print(f"  Units string: '{q1.units}'")
print()

print(f"q2 = {q2}")
print(f"  Has units? {q2.has_units}")
print(f"  Units string: '{q2.units}'")
print()

print(f"q3 = {q3}")
print(f"  Has units? {q3.has_units}")
print(f"  Units string: '{q3.units}'")
print()

# ============================================================================
# ACCESSING PINT FUNCTIONALITY DIRECTLY
# ============================================================================
print("="*70)
print("6. Direct Pint Access for Advanced Operations")
print("-" * 70)
print()

# UWQuantity wraps Pint quantities - you can access them directly
velocity = uw.quantity(5, "cm/year")

if hasattr(velocity, '_pint_qty'):
    pint_qty = velocity._pint_qty
    print(f"Original: {velocity}")
    print(f"Pint quantity: {pint_qty}")
    print()

    # Use Pint's compact() method to automatically choose best units
    try:
        compact = pint_qty.to_compact()
        print(f"Compact form: {compact}")
    except AttributeError:
        print("Note: to_compact() requires Pint >= 0.17")
    print()

    # Use Pint's to_base_units() for SI base units
    base_units = pint_qty.to_base_units()
    print(f"SI base units: {base_units}")
    print()
else:
    print("This quantity doesn't have a Pint quantity attached.")
    print()

# ============================================================================
# PRACTICAL WORKFLOW
# ============================================================================
print("="*70)
print("7. Practical Workflow Example")
print("-" * 70)
print()

print("Scenario: You have a velocity from a simulation in model units")
print("          and want to display it in geological units (cm/year)")
print()

# Setup
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    reference_viscosity=uw.quantity(1, "ZPa.s"),
    reference_density=uw.quantity(3000, "kg/(m^3)"),
    domain_depth=uw.quantity(2900, "km"),
    mantle_temperature=uw.quantity(1600, "K"),
)

# Simulate: computational result in model units (dimensionless number)
simulation_result = 1.58e-9  # Some computed velocity

# Step 1: Create a physical quantity from the simulation result
# You need to know what the base units are for velocity in your model
print(f"Step 1: Simulation result (dimensionless): {simulation_result}")

# For this demo, let's say we know velocity should be in m/s in base units
velocity_physical = uw.quantity(simulation_result, "m/s")
print(f"Step 2: Interpret as physical quantity: {velocity_physical}")

# Step 3: Convert to geological units for readability
velocity_geol = velocity_physical.to("cm/year")
print(f"Step 3: Convert to geological units: {velocity_geol}")
print()

print("="*70)
print("SUMMARY")
print("="*70)
print()
print("Key Methods:")
print("  - model.unit_rounding_mode          : Get/set rounding mode")
print("  - quantity.to(target_units)         : Convert to different units")
print("  - quantity.has_units                : Check if has units")
print("  - quantity.units                    : Get units string")
print("  - quantity._pint_qty                : Access underlying Pint quantity")
print("  - pint_qty.to_compact()             : Auto-select best units (Pint>=0.17)")
print("  - pint_qty.to_base_units()          : Convert to SI base units")
print()
