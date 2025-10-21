#!/usr/bin/env python3
"""
Unit tests for dimensionality tracking and non-dimensionalization.

Tests the dimensionality tracking system that enables reference scaling
for improved numerical conditioning:
- Dimensionality property (shadow of units)
- Scaling coefficient storage and management
- Non-dimensional conversion via .to_nd()
- Unwrap compatibility for JIT compilation
- Array access via .nd_array property
"""

import os
import pytest
import numpy as np

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw
import sympy


def test_basic_dimensionality_tracking():
    """Test that variables track dimensionality from units."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
    v = uw.discretisation.MeshVariable('v', mesh, mesh.dim, units='meter/second')
    p = uw.discretisation.MeshVariable('p', mesh, 1, units='pascal')

    # Check dimensionality is derived from units
    assert T.dimensionality is not None
    assert 'temperature' in str(T.dimensionality).lower() or 'kelvin' in str(T.dimensionality).lower()
    assert v.dimensionality is not None
    assert p.dimensionality is not None

    # Check default scaling coefficients
    assert T.scaling_coefficient == 1.0
    assert v.scaling_coefficient == 1.0
    assert p.scaling_coefficient == 1.0

    # Check non-dimensional state
    assert T.is_nondimensional == False
    assert v.is_nondimensional == False


def test_manual_reference_scaling():
    """Test setting reference scales manually."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')

    # Set reference scale
    T.set_reference_scale(1000.0)
    assert T.scaling_coefficient == 1000.0

    # Set values and check non-dimensional access
    T.array[...] = 1300.0

    # Access non-dimensional array
    T_nd_values = T.nd_array
    assert np.allclose(T_nd_values, 1.3)


def test_scalar_unwrap_preserves_function():
    """Test that unwrap preserves function symbol for scalar variables."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
    T.set_reference_scale(1000.0)

    # Get non-dimensional expression
    T_nd = T.to_nd()

    # Unwrap should preserve the function symbol
    unwrapped = uw.unwrap(T_nd)

    # Extract from matrix if needed
    unwrapped_scalar = unwrapped[0, 0] if isinstance(unwrapped, sympy.Matrix) else unwrapped

    # Check that original function symbol is present
    funcs = unwrapped_scalar.atoms(sympy.Function)
    assert len(funcs) > 0, "No function symbols in unwrapped expression"
    assert "T" in str(unwrapped_scalar), "Original variable T not in unwrapped expression"

    # Check scaling coefficient is present
    assert "0.001" in str(unwrapped_scalar) or "1000" in str(unwrapped_scalar)


def test_vector_unwrap():
    """Test unwrap for vector variables."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    v = uw.discretisation.MeshVariable('v', mesh, mesh.dim, units='meter/second')
    v.set_reference_scale(0.05)

    # Get non-dimensional expression
    v_nd = v.to_nd()

    # Unwrap
    unwrapped = uw.unwrap(v_nd)

    # Check components
    v_nd_0 = v_nd.sym[0]
    unwrapped_0 = uw.unwrap(v_nd_0)

    # Should have scaling factor and function
    assert "v" in str(unwrapped_0)
    assert "20" in str(unwrapped_0) or "0.05" in str(unwrapped_0)


def test_derivative_unwrap():
    """Test that derivatives of non-dimensional variables unwrap correctly."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
    T.set_reference_scale(1000.0)

    # Create derivative of non-dimensional variable
    x, y = mesh.CoordinateSystem.N[0], mesh.CoordinateSystem.N[1]
    T_nd = T.to_nd()

    # Take derivative
    dT_nd_dx = T_nd.sym.diff(x)

    # Unwrap derivative
    unwrapped = uw.unwrap(dT_nd_dx)

    # Extract from matrix if needed
    unwrapped_scalar = unwrapped[0, 0] if isinstance(unwrapped, sympy.Matrix) else unwrapped

    # Check that derivative contains the original function
    assert "T" in str(unwrapped_scalar), "Function not preserved in derivative"


def test_multi_variable_expression():
    """Test expressions combining multiple non-dimensional variables."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
    p = uw.discretisation.MeshVariable('p', mesh, 1, units='pascal')

    T.set_reference_scale(1000.0)
    p.set_reference_scale(1e9)

    # Create expression with both
    T_nd = T.to_nd()
    p_nd = p.to_nd()

    expr = T_nd.sym * p_nd.sym

    # Unwrap
    unwrapped = uw.unwrap(expr)

    # Extract from matrix
    unwrapped_scalar = unwrapped[0, 0] if isinstance(unwrapped, sympy.Matrix) else unwrapped

    # Check both functions present
    unwrap_str = str(unwrapped_scalar)
    assert "T" in unwrap_str, "T not in multi-variable expression"
    assert "p" in unwrap_str, "p not in multi-variable expression"


def test_gradient_unwrap():
    """Test that gradients of non-dimensional variables unwrap correctly."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
    T.set_reference_scale(1000.0)

    T_nd = T.to_nd()

    # Create gradient using mesh utilities
    grad_T_nd = mesh.vector.gradient(T_nd.sym)

    # Unwrap gradient
    unwrapped = uw.unwrap(grad_T_nd)

    # Check function preserved in gradient
    unwrap_str = str(unwrapped)
    assert "T" in unwrap_str, "Function not preserved in gradient"


def test_mixed_dimensional_nondimensional():
    """Test expressions mixing dimensional and non-dimensional variables."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
    p = uw.discretisation.MeshVariable('p', mesh, 1, units='pascal')

    T.set_reference_scale(1000.0)
    p.set_reference_scale(1e9)

    # Mix dimensional and non-dimensional
    T_nd = T.to_nd()
    expr = T_nd.sym * p.sym  # T* × p (not p*)

    # Unwrap
    unwrapped = uw.unwrap(expr)

    # Extract from matrix
    unwrapped_scalar = unwrapped[0, 0] if isinstance(unwrapped, sympy.Matrix) else unwrapped

    # Both functions should be present
    unwrap_str = str(unwrapped_scalar)
    assert "T" in unwrap_str, "T not in mixed expression"
    assert "p" in unwrap_str, "p not in mixed expression"


def test_scaling_coefficient_visibility():
    """Test that scaling coefficients are visible in unwrapped expressions."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
    scale = 1000.0
    T.set_reference_scale(scale)

    T_nd = T.to_nd()

    # Unwrap and check for scale
    unwrapped = uw.unwrap(T_nd)

    # The unwrapped form should contain 1/scale as a factor
    unwrap_str = str(unwrapped)

    # Should contain the reciprocal of the scale
    assert "1000" in unwrap_str or "0.001" in unwrap_str, \
        "Scaling coefficient not visible in unwrapped expression"


def test_automatic_scale_derivation():
    """Test that variables automatically derive scales from model reference quantities."""
    uw.reset_default_model()
    model = uw.get_default_model()

    # Set reference quantities FIRST
    model.set_reference_quantities(
        domain_depth=uw.quantity(3000, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
        temperature_diff=uw.quantity(1000, "kelvin"),
    )

    # NOW create variables - they should auto-derive scales
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    T = uw.discretisation.MeshVariable('Temperature', mesh, 1, units='kelvin')
    v = uw.discretisation.MeshVariable('velocity', mesh, mesh.dim, units='meter/second')

    # Check that scales were auto-derived (should not be default 1.0)
    # Temperature should pick up the temperature_diff reference
    # Note: The exact values depend on the heuristic matching
    assert T.scaling_coefficient != 1.0 or v.scaling_coefficient != 1.0, \
        "Variables should have auto-derived scaling coefficients"


def test_uwquantity_dimensionality():
    """Test dimensionality tracking with UWQuantity objects."""
    uw.reset_default_model()

    viscosity = uw.quantity(1e21, "Pa*s")
    velocity = uw.quantity(5, "cm/year")

    # Check dimensionality
    assert viscosity.dimensionality is not None
    assert velocity.dimensionality is not None

    # Set reference scales
    viscosity.set_reference_scale(1e21)
    velocity.set_reference_scale(5.0)

    assert viscosity.scaling_coefficient == 1e21
    assert velocity.scaling_coefficient == 5.0

    # Convert to non-dimensional
    visc_nd = viscosity.to_nd()
    assert visc_nd.is_nondimensional == True
    assert float(visc_nd.value) == 1.0


def test_roundtrip_conversion():
    """Test that dimensional → non-dimensional → dimensional preserves values."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    # Test with MeshVariable
    T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
    T.set_reference_scale(1000.0)

    # Set dimensional values
    T.array[...] = 1300.0

    # Convert to non-dimensional
    T_nd_values = T.nd_array
    assert np.allclose(T_nd_values, 1.3)

    # Convert back to dimensional
    T_dimensional = T.from_nd(T_nd_values)
    assert np.allclose(T_dimensional, 1300.0), \
        "Round-trip conversion should preserve original values"

    # Test with UWQuantity
    pressure = uw.quantity(2e9, "pascal")
    pressure.set_reference_scale(1e9)

    # Convert to non-dimensional
    p_nd = pressure.to_nd()
    assert float(p_nd.value) == 2.0

    # Convert back to dimensional
    p_dimensional = pressure.from_nd(2.0)
    assert p_dimensional == 2e9, \
        "Round-trip conversion should preserve original quantity value"

    # Test with array of values
    test_values_nd = np.array([0.5, 1.0, 1.5, 2.0])
    test_values_dim = T.from_nd(test_values_nd)
    expected_dim = np.array([500.0, 1000.0, 1500.0, 2000.0])
    assert np.allclose(test_values_dim, expected_dim), \
        "from_nd should work with arrays of values"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
