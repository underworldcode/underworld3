"""
Example: Using unit-aware coordinates with uw.function.evaluate()

This demonstrates the built-in unit-aware coordinate handling in Underworld3's
evaluate() function. As of the latest version, evaluate() automatically handles
unit conversion, so you can pass coordinates with units directly!
"""

import underworld3 as uw
import numpy as np

# Create a simple mesh
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.1,
    regular=False
)

# Create a temperature field
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
T.array[:] = mesh.X.coords[:, 0]  # Temperature increases with x

print("=" * 70)
print("Unit-Aware Coordinates with evaluate() - Now Built-in!")
print("=" * 70)
print("\nGOOD NEWS: evaluate() now accepts unit-aware coordinates directly!")
print("You can pass UWQuantity or Pint Quantity objects and they'll be")
print("automatically converted to SI base units. No manual conversion needed!")

# ============================================================================
# METHOD 1: Direct usage with list of tuples (RECOMMENDED)
# ============================================================================
print("\n1. Direct usage with unit-aware coordinates (RECOMMENDED):")
print("-" * 70)

# Create unit-aware x and y coordinates
x_val = uw.quantity(0.5, "m")
y_val = uw.quantity(0.25, "m")

print(f"   x_val = {x_val}")
print(f"   y_val = {y_val}")

# Just pass them directly as a list of tuples!
coords_1 = [(x_val, y_val)]
print(f"   coords: {coords_1}")

# Evaluate temperature - automatic unit conversion!
T_result = uw.function.evaluate(T.sym, coords_1)
print(f"   Temperature at point: {T_result[0, 0, 0]}")

# ============================================================================
# METHOD 2: Convert quantities to SI base units in bulk
# ============================================================================
print("\n2. Converting multiple coordinate points:")
print("-" * 70)

# Define multiple points with units
points_with_units = [
    (uw.quantity(0.1, "m"), uw.quantity(0.2, "m")),
    (uw.quantity(50, "cm"), uw.quantity(75, "cm")),
    (uw.quantity(0.9, "m"), uw.quantity(0.1, "m")),
]

# Helper function to convert unit-aware coordinates
def coords_to_si(coord_list):
    """
    Convert list of (x, y) or (x, y, z) tuples containing UWQuantity objects
    to numpy array in SI base units.

    Parameters
    ----------
    coord_list : list of tuples
        Each tuple contains UWQuantity objects for coordinates

    Returns
    -------
    np.ndarray
        Array of shape (n_points, n_dims) in SI base units
    """
    si_coords = []
    for coord_tuple in coord_list:
        si_point = []
        for coord in coord_tuple:
            if isinstance(coord, uw.function.quantities.UWQuantity) and hasattr(coord, '_pint_qty'):
                # Convert to SI base units
                si_value = coord._pint_qty.to_base_units().magnitude
            elif isinstance(coord, (float, int)):
                # Already dimensionless
                si_value = float(coord)
            else:
                raise TypeError(f"Unsupported coordinate type: {type(coord)}")
            si_point.append(si_value)
        si_coords.append(si_point)

    return np.array(si_coords, dtype=np.double)

# Convert to SI array
coords_2 = coords_to_si(points_with_units)
print(f"   Input: 3 points in m and cm")
print(f"   SI array shape: {coords_2.shape}")
print(f"   SI array:\n{coords_2}")

# Evaluate at multiple points
T_results = uw.function.evaluate(T.sym, coords_2)
print(f"   Temperatures: {T_results[:, 0, 0]}")

# ============================================================================
# METHOD 3: Using numpy arrays with unit conversion
# ============================================================================
print("\n3. Converting numpy arrays with units:")
print("-" * 70)

# Create arrays in user-friendly units (e.g., kilometers)
x_coords_km = np.array([0.0001, 0.0005, 0.0009])  # 0.1m, 0.5m, 0.9m in km
y_coords_km = np.array([0.0002, 0.0005, 0.0008])  # 0.2m, 0.5m, 0.8m in km

print(f"   x in km: {x_coords_km}")
print(f"   y in km: {y_coords_km}")

# Convert each coordinate array
x_qty = uw.quantity(1.0, "km")  # Unit quantity for conversion
y_qty = uw.quantity(1.0, "km")

# Get conversion factor to meters
km_to_m = x_qty._pint_qty.to_base_units().magnitude if hasattr(x_qty, '_pint_qty') else 1000.0

# Apply conversion
x_coords_m = x_coords_km * km_to_m
y_coords_m = y_coords_km * km_to_m

# Stack into coordinate array
coords_3 = np.column_stack([x_coords_m, y_coords_m])
print(f"   Converted to SI (m):\n{coords_3}")

T_results = uw.function.evaluate(T.sym, coords_3)
print(f"   Temperatures: {T_results[:, 0, 0]}")

# ============================================================================
# METHOD 4: Direct approach for model units (most common)
# ============================================================================
print("\n4. Best practice when mesh has no units (most common case):")
print("-" * 70)

# When mesh coordinates are dimensionless or in model units,
# just use regular numpy arrays directly
coords_simple = np.array([
    [0.1, 0.2],
    [0.5, 0.5],
    [0.9, 0.8],
])

print(f"   Direct numpy array:\n{coords_simple}")

T_results = uw.function.evaluate(T.sym, coords_simple)
print(f"   Temperatures: {T_results[:, 0, 0]}")

# ============================================================================
# KEY RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 70)
print("KEY RECOMMENDATIONS:")
print("=" * 70)
print("""
1. **For single points with units**: Use Method 1
   - Convert each UWQuantity to SI base units individually
   - Stack into numpy array

2. **For multiple points with units**: Use Method 2 with helper function
   - Cleaner and more maintainable
   - Handles mixed units automatically

3. **For arrays of same units**: Use Method 3
   - Most efficient for large coordinate sets
   - Single unit conversion applied to entire array

4. **For dimensionless models** (most common): Use Method 4
   - Just use numpy arrays directly
   - No unit conversion needed

5. **Key principle**: evaluate() needs np.array of doubles in SI base units
   - Convert UWQuantity → SI base units → extract magnitude
   - Use _pint_qty.to_base_units().magnitude for conversion
""")

print("\n" + "=" * 70)
print("Example complete!")
print("=" * 70)
