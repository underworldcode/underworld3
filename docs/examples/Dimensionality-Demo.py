# %% [markdown]
# # Dimensionality Tracking and Non-Dimensionalization Demo
#
# This notebook demonstrates the new dimensionality tracking system that enables
# reference scaling for improved numerical conditioning.

# %%
import underworld3 as uw
import numpy as np

# %% [markdown]
# ## 1. Basic Dimensionality Tracking
#
# All variables now track their dimensionality automatically from units.

# %%
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(3000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    density=uw.quantity(3.3, "g/cm**3"),
    temperature_diff=uw.quantity(1000, "kelvin"),
)

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(3000.0, 3000.0),
    cellSize=500.0
)

T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
v = uw.discretisation.MeshVariable('v', mesh, mesh.dim, units='meter/second')
p = uw.discretisation.MeshVariable('p', mesh, 1, units='pascal')

print(f"Temperature: {T.dimensionality}")
print(f"Velocity:    {v.dimensionality}")
print(f"Pressure:    {p.dimensionality}")

# %% [markdown]
# ## 2. Setting Reference Scales
#
# Reference scales can be set manually for non-dimensionalization.

# %%
# Set characteristic scales
T.set_reference_scale(1000.0)  # 1000 K temperature difference
v.set_reference_scale(0.05)     # example velocity scale
p.set_reference_scale(1e9)      # GPa pressure scale

print(f"Scaling coefficients set:")
print(f"  T: {T.scaling_coefficient:.0f} K")
print(f"  v: {v.scaling_coefficient:.2e} m/s")
print(f"  p: {p.scaling_coefficient:.0e} Pa")

# %% [markdown]
# ## 3. Non-Dimensional Conversion
#
# Current strict-units mode stores non-dimensional values in `.data`.
# Dimensional values can be converted manually using the scaling coefficient:
#
# `non_dimensional_value = dimensional_value / scaling_coefficient`

# %%
# Set some dimensional values
T_dim_value = 1300.0  # K
v_dim_value = 0.03    # m/s
p_dim_value = 2e9     # Pa

T_nd_value = T_dim_value / T.scaling_coefficient
v_nd_value = v_dim_value / v.scaling_coefficient
p_nd_value = p_dim_value / p.scaling_coefficient

with uw.synchronised_array_update():
    T.data[...] = T_nd_value
    v.data[...] = v_nd_value
    p.data[...] = p_nd_value

print(f"Dimensional values:")
print(f"  T = {T_dim_value:.1f} K")
print(f"  v = {v_dim_value:.3f} m/s")
print(f"  p = {p_dim_value:.2e} Pa")
print()
print(f"Non-dimensional stored values:")
print(f"  T* = {T_nd_value:.2f}")
print(f"  v* = {v_nd_value:.2f}")
print(f"  p* = {p_nd_value:.2f}")

# %%
# For symbolic/JIT use, use the variable symbol together with the scaling coefficient.
T_nd_expr = T.sym / T.scaling_coefficient

print(f"Symbolic non-dimensional form:")
print(f"  {T_nd_expr}")
print(f"\nThis preserves the original function symbol for JIT:")
print(f"  {uw.unwrap(T_nd_expr)}")

# %% [markdown]
# ## 4. UWQuantity with Dimensionality

# %%
viscosity = uw.quantity(1e21, "Pa*s")
velocity = uw.quantity(5, "cm/year")

print(f"Viscosity dimensionality: {viscosity.dimensionality}")
print(f"Velocity dimensionality:  {velocity.dimensionality}")

# Manual reference scaling for UWQuantity values
viscosity_reference = 1e21
velocity_reference = 5.0

visc_nd = 1e21 / viscosity_reference
vel_nd = 5.0 / velocity_reference

print(f"\nNon-dimensional values:")
print(f"  η* = {visc_nd}")
print(f"  v* = {vel_nd}")

# %% [markdown]
# ## 5. Automatic Scale Derivation from Model
#
# When you set reference quantities on a model, scaling coefficients
# are automatically derived for all variables via dimensional analysis.
#
# **Important**: Set reference quantities BEFORE creating variables
# so the auto-derivation can find them in the model's registry.

# %%
# Create model and set reference quantities FIRST
model = uw.Model()

model.set_reference_quantities(
    domain_depth=uw.quantity(3000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    density=uw.quantity(3.3, "g/cm**3"),
    temperature_diff=uw.quantity(1000, "kelvin"),
    verbose=True
)

# NOW create variables - they register with the model that has reference quantities
mesh2 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(3000.0, 3000.0),
    cellSize=500.0
)

T2 = uw.discretisation.MeshVariable('Temperature', mesh2, 1, units='kelvin')
v2 = uw.discretisation.MeshVariable('velocity', mesh2, mesh2.dim, units='meter/second')

print("\nAuto-derived scaling coefficients:")
print(f"  T2: scale = {T2.scaling_coefficient}")
print(f"  v2: scale = {v2.scaling_coefficient}")

# Demonstrate conversion
T2_dim_value = 1500.0
T2_nd_value = T2_dim_value / T2.scaling_coefficient

with uw.synchronised_array_update():
    T2.data[...] = T2_nd_value

print(f"\nExample: T = {T2_dim_value:.0f} K → T* = {T2_nd_value:.2f}")

# %%
T2_nd_expr = T2.sym / T2.scaling_coefficient
uw.unwrap(T2_nd_expr)

# %%
T2_nd_expr

# %% [markdown]
# ## 6. Round-Trip Conversion
#
# Non-dimensional values can be converted back to dimensional values
# using the scaling coefficient:
#
# `dimensional_value = non_dimensional_value * scaling_coefficient`

# %%
# Get non-dimensional value
T_star = T2_nd_value
print(f"Non-dimensional: T* = {T_star:.2f}")

# Convert back to dimensional
T_dim = T_star * T2.scaling_coefficient
print(f"Dimensional: T = {T_dim:.0f} K")

# Works with arrays too
nd_values = np.array([0.5, 1.0, 1.5, 2.0])
dim_values = nd_values * T2.scaling_coefficient
print(f"\nArray conversion:")
print(f"  Non-dimensional: {nd_values}")
print(f"  Dimensional: {dim_values} K")

# %% [markdown]
# ## Summary
#
# The dimensionality tracking system provides:
#
# - **Dimensionality as first-class property** - automatically derived from units
# - **Reference scaling coefficients** - characteristic scales for each variable
# - **Non-dimensional storage** - via `.data`
# - **Manual dimensional conversion** - using the scaling coefficient
# - **Array-based conversion** - through direct scale multiplication/division
# - **Automatic scale derivation** - from model reference quantities
# - **Strict unit safety** - unit-bearing variables reject ambiguous plain assignments
#
# This infrastructure enables proper non-dimensionalization for solving
# stiff systems while maintaining the units system.

# %%
