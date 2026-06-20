"""
Example: Using unit-aware coordinates with uw.function.evaluate()

This example demonstrates a safe pattern for evaluating an Underworld3 function
at coordinates that are originally specified with units.

In the current strict-units workflow, uw.function.evaluate() should receive a
numeric coordinate array. Unit-aware coordinates can be converted explicitly to
SI base units before evaluation.
"""

import underworld3 as uw
import numpy as np


def quantity_to_si_value(value):
    """
    Convert a UWQuantity, Pint-like quantity, or plain number to a float in
    SI base units.

    Parameters
    ----------
    value : UWQuantity, Pint quantity, float, or int
        Coordinate value to convert.

    Returns
    -------
    float
        Coordinate value as a plain float in SI base units.
    """
    if isinstance(value, uw.function.quantities.UWQuantity) and hasattr(value, "_pint_qty"):
        return float(value._pint_qty.to_base_units().magnitude)

    if hasattr(value, "to_base_units"):
        return float(value.to_base_units().magnitude)

    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)

    raise TypeError(f"Unsupported coordinate type: {type(value)}")


def coords_to_si(coord_list):
    """
    Convert a list of coordinate tuples to a numpy array in SI base units.

    Parameters
    ----------
    coord_list : list of tuples
        Each tuple contains coordinate values. Values may be UWQuantity objects,
        Pint quantities, or plain numbers.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, n_dims) in SI base units.
    """
    si_coords = []

    for coord_tuple in coord_list:
        si_point = [quantity_to_si_value(coord) for coord in coord_tuple]
        si_coords.append(si_point)

    return np.array(si_coords, dtype=np.double)


# Create a simple mesh
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.1,
    regular=False
)

# Create a temperature field.
#
# Since T is degree=2, its data live on T.coords, not mesh.X.coords.
# mesh.X.coords gives only the mesh vertices, which is too short for a
# degree=2 MeshVariable.
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

with uw.synchronised_array_update():
    T.data[:, 0] = T.coords[:, 0]  # Temperature increases with x

print("=" * 70)
print("Unit-Aware Coordinates with evaluate()")
print("=" * 70)
print("\nThis example converts unit-aware coordinates to SI base units")
print("before passing them to uw.function.evaluate().")

# ============================================================================
# METHOD 1: Single point with unit-aware coordinates
# ============================================================================
print("\n1. Single point with unit-aware coordinates:")
print("-" * 70)

x_val = uw.quantity(0.5, "m")
y_val = uw.quantity(0.25, "m")

print(f"   x_val = {x_val}")
print(f"   y_val = {y_val}")

coords_1 = coords_to_si([(x_val, y_val)])
print(f"   SI coordinate array:\n{coords_1}")

T_result = uw.function.evaluate(T.sym, coords_1)
print(f"   Temperature at point: {T_result[0, 0, 0]}")

# ============================================================================
# METHOD 2: Multiple points with mixed units
# ============================================================================
print("\n2. Multiple coordinate points with mixed units:")
print("-" * 70)

points_with_units = [
    (uw.quantity(0.1, "m"), uw.quantity(0.2, "m")),
    (uw.quantity(50, "cm"), uw.quantity(75, "cm")),
    (uw.quantity(0.9, "m"), uw.quantity(0.1, "m")),
]

coords_2 = coords_to_si(points_with_units)

print("   Input: 3 points in m and cm")
print(f"   SI array shape: {coords_2.shape}")
print(f"   SI array:\n{coords_2}")

T_results = uw.function.evaluate(T.sym, coords_2)
print(f"   Temperatures: {T_results[:, 0, 0]}")

# ============================================================================
# METHOD 3: Numpy arrays with a shared unit
# ============================================================================
print("\n3. Numpy arrays with a shared unit:")
print("-" * 70)

x_coords_km = np.array([0.0001, 0.0005, 0.0009])  # 0.1 m, 0.5 m, 0.9 m in km
y_coords_km = np.array([0.0002, 0.0005, 0.0008])  # 0.2 m, 0.5 m, 0.8 m in km

print(f"   x in km: {x_coords_km}")
print(f"   y in km: {y_coords_km}")

km_to_m = quantity_to_si_value(uw.quantity(1.0, "km"))

x_coords_m = x_coords_km * km_to_m
y_coords_m = y_coords_km * km_to_m

coords_3 = np.column_stack([x_coords_m, y_coords_m])
print(f"   Converted to SI (m):\n{coords_3}")

T_results = uw.function.evaluate(T.sym, coords_3)
print(f"   Temperatures: {T_results[:, 0, 0]}")

# ============================================================================
# METHOD 4: Direct model coordinates
# ============================================================================
print("\n4. Direct model coordinates:")
print("-" * 70)

coords_simple = np.array([
    [0.1, 0.2],
    [0.5, 0.5],
    [0.9, 0.8],
], dtype=np.double)

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
1. For single points with units:
   - Convert unit-aware coordinate tuples to a numeric SI array first.
   - Then pass the numeric array to uw.function.evaluate().

2. For multiple points with mixed units:
   - Use a helper function such as coords_to_si().
   - This keeps the conversion explicit and avoids object-array issues.

3. For arrays with one shared unit:
   - Convert the unit once.
   - Apply the conversion factor to the full numpy array.

4. For dimensionless model coordinates:
   - Use regular numpy arrays directly.

5. For MeshVariable initialization:
   - Use the variable's own coordinates, for example T.coords.
   - Do not assign mesh.X.coords to a degree > 1 MeshVariable.
""")

print("\n" + "=" * 70)
print("Example complete!")
print("=" * 70)
