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
v.set_reference_scale(0.05)     # 5 cm/year = 1.58e-9 m/s ≈ 0.05 m/Myr
p.set_reference_scale(1e9)      # GPa pressure scale

print(f"Scaling coefficients set:")
print(f"  T: {T.scaling_coefficient:.0f} K")
print(f"  v: {v.scaling_coefficient:.2e} m/s")
print(f"  p: {p.scaling_coefficient:.0e} Pa")

# %% [markdown]
# ## 3. Non-Dimensional Conversion
#
# Two ways to access non-dimensional values:
# - `.to_nd()` returns a SymPy expression (for symbolic/JIT use)
# - `.nd_array` property returns non-dimensional array values

# %%
# Set some dimensional values
with uw.synchronised_array_update():
    T.array[...] = 1300.0  # K
    v.array[...] = 0.03    # m/s
    p.array[...] = 2e9     # Pa

print(f"Dimensional values:")
print(f"  T = {T.array[0,0,0]:.1f} K")
print(f"  v = {v.array[0,0,0]:.3f} m/s")
print(f"  p = {p.array[0,0,0]:.2e} Pa")
print()
print(f"Non-dimensional array values:")
print(f"  T* = {T.nd_array[0,0,0]:.2f}")
print(f"  v* = {v.nd_array[0,0,0]:.2f}")
print(f"  p* = {p.nd_array[0,0,0]:.2f}")

# %%
# For symbolic/JIT use, .to_nd() returns a SymPy expression
T_nd_expr = T.to_nd()
print(f"Symbolic non-dimensional form:")
print(f"  {T_nd_expr.sym}")
print(f"\nThis preserves the original function symbol for JIT:")
print(f"  {uw.unwrap(T_nd_expr)}")

# %% [markdown]
# ## 4. UWQuantity with Dimensionality

# %%
viscosity = uw.quantity(1e21, "Pa*s")
velocity = uw.quantity(5, "cm/year")

print(f"Viscosity dimensionality: {viscosity.dimensionality}")
print(f"Velocity dimensionality:  {velocity.dimensionality}")

# Set reference and convert
viscosity.set_reference_scale(1e21)
velocity.set_reference_scale(5.0)

visc_nd = viscosity.to_nd()
vel_nd = velocity.to_nd()

print(f"\nNon-dimensional values:")
print(f"  η* = {visc_nd.value}")
print(f"  v* = {vel_nd.value}")

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
    density=uw.quantity(3.3,"g/cm**3"),
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
T2.array[...] = 1500.0
print(f"\nExample: T = {T2.array[0,0,0]:.0f} K → T* = {T2.nd_array[0,0,0]:.2f}")

# %%
uw.unwrap(T2.to_nd())

# %%
T2.to_nd().sym

# %% [markdown]
# ## 6. Round-Trip Conversion
#
# The `.from_nd()` method converts non-dimensional values back to dimensional form.

# %%
# Get non-dimensional value
T_star = T2.nd_array[0,0,0]
print(f"Non-dimensional: T* = {T_star:.2f}")

# Convert back to dimensional
T_dim = T2.from_nd(T_star)
print(f"Dimensional: T = {T_dim:.0f} K")

# Works with arrays too
nd_values = np.array([0.5, 1.0, 1.5, 2.0])
dim_values = T2.from_nd(nd_values)
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
# - **Non-dimensional conversion** - via `.to_nd()` method for symbolic expressions
# - **Array-based access** - via `.nd_array` property for numerical values
# - **Round-trip conversion** - via `.from_nd()` method to restore dimensional values
# - **Automatic scale derivation** - from model reference quantities
# - **Zero side effects** - all existing code continues to work
#
# This infrastructure enables proper non-dimensionalization for solving
# stiff systems while maintaining the elegant units system.

# %%

# %%
