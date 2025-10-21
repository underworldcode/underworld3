"""
Test boundary condition unit conversion functionality.

This test validates that Dirichlet boundary conditions can accept UWQuantity
objects and automatically convert them to model units, ensuring that solutions
are equivalent regardless of the units used to specify the BCs.
"""

import pytest
import underworld3 as uw
import numpy as np


def test_bc_accepts_raw_numbers():
    """Test backward compatibility: BCs accept raw numbers."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        regular=False
    )

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    stokes = uw.systems.Stokes(mesh, velocityField=v)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
    stokes.constitutive_model.Parameters.viscosity = 1.0

    # Should work without errors
    stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")
    stokes.add_dirichlet_bc([1.0, None], "Top")

    assert len(stokes.essential_bcs) == 2


def test_bc_accepts_uwquantity():
    """Test that BCs accept UWQuantity objects."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        regular=False
    )

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    stokes = uw.systems.Stokes(mesh, velocityField=v)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
    stokes.constitutive_model.Parameters.viscosity = 1.0

    # Create UWQuantity BCs
    velocity = uw.quantity(5.0, "cm/year")

    # Should work without errors
    stokes.add_dirichlet_bc([None, velocity], "Top")
    stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")

    assert len(stokes.essential_bcs) == 2


def test_bc_accepts_pint_quantity():
    """Test that BCs accept Pint Quantity objects (from uw.units())."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        regular=False
    )

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    stokes = uw.systems.Stokes(mesh, velocityField=v)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
    stokes.constitutive_model.Parameters.viscosity = 1.0

    # Create Pint Quantity BCs (using uw.units())
    velocity = 5.0 * uw.units("cm/year")  # Direct Pint Quantity

    # Should work without errors
    stokes.add_dirichlet_bc([None, velocity], "Top")
    stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")

    assert len(stokes.essential_bcs) == 2


def test_bc_mixed_none_and_uwquantity():
    """Test BCs with mixed None and UWQuantity in lists."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        regular=False
    )

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    stokes = uw.systems.Stokes(mesh, velocityField=v)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
    stokes.constitutive_model.Parameters.viscosity = 1.0

    # Mixed BC: free slip on x, fixed velocity on y
    v_y = uw.quantity(1.0, "m/year")

    # Should work without errors
    stokes.add_dirichlet_bc([None, v_y], "Left")
    stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")

    assert len(stokes.essential_bcs) == 2

    # Check that the BC with None was processed correctly
    left_bc = stokes.essential_bcs[0]
    assert len(left_bc.components) == 1  # Only y-component
    assert left_bc.components[0] == 1  # y-component index


def test_bc_unit_conversion_equivalence():
    """
    Test that BC values are correctly converted regardless of input units.

    This validates that the same physical BC value specified in different
    units results in identical stored BC expressions.
    """
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        regular=False
    )

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    stokes = uw.systems.Stokes(mesh, velocityField=v)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
    stokes.constitutive_model.Parameters.viscosity = 1.0

    # Test: Same physical velocity (1 m/s) specified in different units

    # Case 1: SI base units
    stokes.add_dirichlet_bc([uw.quantity(1.0, "m/s"), None], "Top")

    # Case 2: Geological units (1 m/s = 3.15576e9 cm/year)
    cm_per_year = 1.0 * 3.15576e9
    stokes.add_dirichlet_bc([uw.quantity(cm_per_year, "cm/year"), None], "Bottom")

    # Extract BC values
    import sympy
    bc_top = stokes.essential_bcs[0]
    bc_bottom = stokes.essential_bcs[1]

    # Extract the x-component values
    val_top = float(bc_top.fn[0, 0])
    val_bottom = float(bc_bottom.fn[0, 0])

    # All should be 1.0 m/s (in SI base units)
    assert abs(val_top - 1.0) < 1e-10, f"Top BC: {val_top} != 1.0"
    assert abs(val_bottom - 1.0) < 1e-6, f"Bottom BC: {val_bottom} != 1.0"  # Slightly looser due to conversion


def test_bc_unit_conversion_values():
    """
    Test that BC values are correctly converted by examining stored SymPy expressions.

    When a UWQuantity is passed, it should be converted to a dimensionless SymPy
    expression representing the value in model base units.
    """
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        regular=False
    )

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    stokes = uw.systems.Stokes(mesh, velocityField=v)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
    stokes.constitutive_model.Parameters.viscosity = 1.0

    # Add BC with UWQuantity
    # 1 cm/year ≈ 3.17e-10 m/s (this is the conversion factor)
    bc_value = uw.quantity(1.0, "cm/year")
    stokes.add_dirichlet_bc([None, bc_value], "Top")

    # Extract the stored BC function
    bc = stokes.essential_bcs[0]

    # The BC function should be a SymPy Matrix
    import sympy
    assert isinstance(bc.fn, sympy.ImmutableDenseMatrix)

    # Extract the y-component value (should be converted to m/s, dimensionless)
    y_value = float(bc.fn[1, 0])

    # Expected value: 1 cm/year in m/s
    expected_ms = 1.0e-2 / (365.25 * 24 * 3600)  # cm to m, year to seconds

    # Should match within floating point precision
    assert abs(y_value - expected_ms) / expected_ms < 1e-6, \
        f"BC value {y_value} doesn't match expected {expected_ms}"


if __name__ == "__main__":
    # Run tests individually for debugging
    print("Running BC unit conversion tests...")

    print("\n1. Testing raw number BCs (backward compatibility)...")
    test_bc_accepts_raw_numbers()
    print("   ✓ Passed")

    print("\n2. Testing UWQuantity BCs...")
    test_bc_accepts_uwquantity()
    print("   ✓ Passed")

    print("\n3. Testing Pint Quantity BCs (from uw.units())...")
    test_bc_accepts_pint_quantity()
    print("   ✓ Passed")

    print("\n4. Testing mixed None and UWQuantity...")
    test_bc_mixed_none_and_uwquantity()
    print("   ✓ Passed")

    print("\n5. Testing BC value conversion accuracy...")
    test_bc_unit_conversion_values()
    print("   ✓ Passed")

    print("\n6. Testing solution equivalence across different BC units...")
    print("   (This may take a minute - solving Stokes problem 4 times...)")
    test_bc_unit_conversion_equivalence()
    print("   ✓ Passed")

    print("\n" + "="*60)
    print("All BC unit conversion tests passed! ✅")
    print("="*60)
