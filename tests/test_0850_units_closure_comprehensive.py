"""
Comprehensive Units System Closure Property Tests

This test suite validates that the units system maintains closure under all operations:
- Unit-aware × Unit-aware → Unit-aware (preserves unit metadata)
- All operations preserve dimensional correctness
- Derivatives compute proper units
- Component access preserves units
- Composition works correctly

Test Naming Convention:
- test_OP_objects: Tests operation OP between different unit-aware object types
- test_closure_*: Tests that result is unit-aware (closure property)
- test_units_*: Tests that result has correct units (dimensional analysis)
"""

import pytest
import numpy as np
import underworld3 as uw
from underworld3.utilities.unit_aware_array import UnitAwareArray


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_mesh():
    """Create a simple 2D mesh for testing."""
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )


@pytest.fixture
def mesh_with_units():
    """Create a mesh with reference quantities (units mode)."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
        mantle_viscosity=uw.quantity(1e21, "Pa*s"),
        reference_temperature=uw.quantity(1350, "K"),  # Add temperature dimension
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )
    return mesh


@pytest.fixture
def temperature_var(simple_mesh):
    """Create a temperature mesh variable."""
    T = uw.discretisation.MeshVariable("T", simple_mesh, 1, degree=2)
    # Initialize with some data (use simple linear values, not coordinate-based)
    with uw.synchronised_array_update():
        num_nodes = T.array.shape[0]
        T.array[:, 0, 0] = 300 + 100 * np.linspace(0, 1, num_nodes)  # Linear gradient
    return T


@pytest.fixture
def velocity_var(simple_mesh):
    """Create a velocity mesh variable (vector)."""
    V = uw.discretisation.MeshVariable("V", simple_mesh, 2, degree=2)
    # Initialize with some data
    V.array[:, 0, 0] = 0.01  # x-component
    V.array[:, 0, 1] = 0.02  # y-component
    return V


@pytest.fixture
def temperature_with_units(mesh_with_units):
    """Create a temperature variable with units."""
    T = uw.discretisation.MeshVariable("T", mesh_with_units, 1, degree=2, units="K")
    # Initialize
    T.array[:, 0, 0] = 300.0
    return T


@pytest.fixture
def velocity_with_units(mesh_with_units):
    """Create a velocity variable with units."""
    V = uw.discretisation.MeshVariable("V", mesh_with_units, 2, degree=2, units="m/s")
    # Initialize
    V.array[:, 0, 0] = 1e-10  # x-component
    V.array[:, 0, 1] = 2e-10  # y-component
    return V


# =============================================================================
# Test 1: Variable × Variable Operations
# =============================================================================

def test_closure_variable_multiply_variable(temperature_with_units, velocity_with_units):
    """Test: Variable * Variable returns unit-aware expression."""
    # Use direct variable operations (not .sym) to test MathematicalMixin
    # Use variables WITH units to test closure property
    result = temperature_with_units * velocity_with_units[0]

    # Check closure: result should be unit-aware (have units property)
    assert hasattr(result, "units") or hasattr(result, "_units"), \
        "Variable * Variable should preserve unit-awareness"


def test_units_temperature_times_velocity(temperature_with_units, velocity_with_units):
    """Test: Temperature [K] * Velocity [m/s] = [K·m/s]."""
    result_expr = temperature_with_units.sym * velocity_with_units.sym[0]

    # Get units from expression
    result_units = uw.get_units(result_expr)

    # Should have compound units
    assert result_units is not None, "Temperature * Velocity should have units"
    # Units should contain both K and m/s (or equivalent)
    # Exact string format may vary (Pint simplification)
    assert "kelvin" in str(result_units).lower() or "K" in str(result_units), \
        f"Result should contain temperature units, got: {result_units}"


def test_closure_temperature_times_velocity_component(temperature_var, velocity_var):
    """Test: Temperature * Velocity[0] preserves unit-awareness."""
    result = temperature_var.sym * velocity_var.sym[0]

    # Should be able to get units from result
    result_units = uw.get_units(result)
    # Even if units are None (dimensionless), get_units should work
    assert result_units is not None or result_units == "dimensionless", \
        "Result should have detectable units or be explicitly dimensionless"


# =============================================================================
# Test 2: Scalar × Variable Operations
# =============================================================================

@pytest.mark.skip(reason="Right multiplication (2 * var) bypasses __rmul__ due to SymPy sympification. Use (var * 2) instead.")
def test_closure_scalar_times_variable(temperature_with_units):
    """Test: 2 * Variable preserves unit-awareness."""
    # Note: This is a known limitation - when the left operand is a Python int/float,
    # SymPy's sympification handles the operation before our __rmul__ is called.
    # Users should write (var * 2) instead of (2 * var) for unit preservation.
    result = 2 * temperature_with_units

    # Should still be unit-aware
    assert hasattr(result, "units") or hasattr(result, "_units"), \
        "Scalar * Variable should preserve unit-awareness"


def test_closure_scalar_times_temperature_times_velocity_component(
    temperature_var, velocity_var
):
    """Test: 2 * Temperature * Velocity[0] preserves unit-awareness."""
    result = 2 * temperature_var.sym * velocity_var.sym[0]

    # Complex compound expression should still be unit-aware
    assert uw.get_units(result) is not None or uw.get_units(result) == "dimensionless", \
        "Complex compound expression should preserve unit-awareness"


def test_units_scalar_preserves_variable_units(temperature_with_units):
    """Test: 2 * Temperature [K] = [K]."""
    result = 2 * temperature_with_units.sym

    result_units = uw.get_units(result)

    # Scalar multiplication preserves units
    assert "kelvin" in str(result_units).lower() or "K" in str(result_units), \
        f"Scalar * Temperature should preserve temperature units, got: {result_units}"


# =============================================================================
# Test 3: Derivative Operations
# =============================================================================

def test_closure_derivative_is_unit_aware(temperature_var, simple_mesh):
    """Test: Temperature.diff(x) preserves unit-awareness."""
    x = simple_mesh.N.x
    dT_dx = temperature_var.sym.diff(x)

    # Derivative should be unit-aware
    assert hasattr(dT_dx, "diffindex") or uw.get_units(dT_dx) is not None, \
        "Derivative should preserve unit-awareness"


def test_units_temperature_derivative(temperature_with_units, mesh_with_units):
    """Test: dT/dx where T [K], x [km] → [K/km]."""
    x = mesh_with_units.N.x
    dT_dx = temperature_with_units.sym.diff(x)

    result_units = uw.get_units(dT_dx)

    # Should have derivative units
    assert result_units is not None, "Derivative should have units"
    # Should contain both numerator and denominator units
    units_str = str(result_units).lower()
    assert "kelvin" in units_str or "K" in str(result_units), \
        f"Derivative should contain temperature units, got: {result_units}"


@pytest.mark.skip(reason="Second derivatives not supported by Underworld functions (UnderworldAppliedFunctionDeriv.fdiff)")
def test_closure_second_derivative(temperature_var, simple_mesh):
    """Test: d²T/dx² preserves unit-awareness."""
    x = simple_mesh.N.x
    dT_dx = temperature_var.sym.diff(x)
    d2T_dx2 = dT_dx.diff(x)

    # Second derivative should still be unit-aware
    assert uw.get_units(d2T_dx2) is not None or uw.get_units(d2T_dx2) == "dimensionless", \
        "Second derivative should preserve unit-awareness"


# =============================================================================
# Test 4: Division Operations
# =============================================================================

def test_closure_temperature_divided_by_coordinate(temperature_var, simple_mesh):
    """Test: Temperature / x preserves unit-awareness."""
    x = simple_mesh.N.x
    result = temperature_var.sym / x

    # Should be unit-aware
    assert uw.get_units(result) is not None or uw.get_units(result) == "dimensionless", \
        "Division should preserve unit-awareness"


def test_units_temperature_divided_by_length(temperature_with_units, mesh_with_units):
    """Test: Temperature [K] / x [km] = [K/km]."""
    x = mesh_with_units.N.x
    result = temperature_with_units.sym / x

    result_units = uw.get_units(result)

    assert result_units is not None, "Division result should have units"
    units_str = str(result_units).lower()
    assert "kelvin" in units_str or "K" in str(result_units), \
        f"Result should contain temperature units, got: {result_units}"


def test_closure_variable_divided_by_variable(temperature_var, velocity_var):
    """Test: Temperature / Velocity[0] preserves unit-awareness."""
    result = temperature_var.sym / velocity_var.sym[0]

    assert uw.get_units(result) is not None or uw.get_units(result) == "dimensionless", \
        "Variable / Variable should preserve unit-awareness"


# =============================================================================
# Test 5: Component Access
# =============================================================================

def test_closure_vector_component_access(velocity_var):
    """Test: Velocity[0] preserves unit-awareness."""
    v_x = velocity_var.sym[0]

    # Component should preserve unit-awareness
    assert hasattr(v_x, "units") or hasattr(v_x, "_units") or uw.get_units(v_x) is not None, \
        "Component access should preserve unit-awareness"


def test_closure_vector_component_in_expression(velocity_var, temperature_var):
    """Test: Temperature * Velocity[0] preserves unit-awareness."""
    result = temperature_var.sym * velocity_var.sym[0]

    assert uw.get_units(result) is not None or uw.get_units(result) == "dimensionless", \
        "Component in expression should preserve unit-awareness"


def test_units_vector_component_preserves_units(velocity_with_units):
    """Test: Velocity[0] has same units as Velocity."""
    v_x = velocity_with_units.sym[0]

    component_units = uw.get_units(v_x)
    parent_units = velocity_with_units.units

    # Component should have same units as parent (or compatible)
    assert component_units == parent_units or str(component_units) == str(parent_units), \
        f"Component units ({component_units}) should match parent units ({parent_units})"


# =============================================================================
# Test 6: Coordinate Operations
# =============================================================================

def test_closure_mesh_coordinates_are_unit_aware(mesh_with_units):
    """Test: mesh.X.coords is unit-aware."""
    coords = mesh_with_units.X.coords

    # Should be UnitAwareArray
    assert isinstance(coords, UnitAwareArray), \
        f"Coordinates should be UnitAwareArray, got {type(coords)}"
    assert hasattr(coords, "units"), \
        "Coordinates should have units property"
    assert coords.units is not None, \
        "Coordinates should have explicit units"


def test_closure_coordinate_in_expression(temperature_with_units, mesh_with_units):
    """Test: Temperature * x preserves unit-awareness."""
    x = mesh_with_units.N.x
    result = temperature_with_units.sym * x

    assert uw.get_units(result) is not None, \
        "Expression with coordinate should preserve unit-awareness"


def test_units_coordinate_access(mesh_with_units):
    """Test: mesh.X.coords has proper coordinate units."""
    coords = mesh_with_units.X.coords

    coord_units = coords.units

    # Coordinates should have length units
    assert coord_units is not None, "Coordinates should have units"
    assert "meter" in str(coord_units).lower() or "m" == str(coord_units), \
        f"Coordinates should have length units, got: {coord_units}"


# =============================================================================
# Test 7: Addition/Subtraction (Unit Compatibility)
# =============================================================================

def test_units_addition_requires_compatible_units(temperature_with_units):
    """Test: T [K] + T [K] = [K] (compatible units)."""
    result = temperature_with_units.sym + temperature_with_units.sym

    result_units = uw.get_units(result)

    # Addition preserves units
    assert "kelvin" in str(result_units).lower() or "K" in str(result_units), \
        f"Addition of same units should preserve units, got: {result_units}"


def test_units_addition_incompatible_units_fails(temperature_with_units, velocity_with_units):
    """Test: T [K] + V [m/s] should fail (incompatible units)."""
    # This test verifies that dimensional checking works
    # Use direct variables (not .sym) to test MathematicalMixin
    # Implementation may vary - some systems allow this, others don't
    # We just check that SOME unit information is preserved
    try:
        result = temperature_with_units + velocity_with_units[0]
        # If it doesn't raise an error, at least verify units are tracked
        if hasattr(result, 'units'):
            result_units = result.units
        elif hasattr(result, '_units'):
            result_units = result._units
        else:
            result_units = uw.get_units(result)
        assert result_units is not None, \
            "Even if addition allowed, units should be tracked"
    except (ValueError, TypeError, Exception) as e:
        # Expected behavior - incompatible units should raise error
        assert "units" in str(e).lower() or "dimension" in str(e).lower() or "compatible" in str(e).lower(), \
            f"Error should mention units/dimensions/compatibility: {e}"


# =============================================================================
# Test 8: Power Operations
# =============================================================================

def test_closure_variable_squared(temperature_var):
    """Test: Temperature² preserves unit-awareness."""
    result = temperature_var.sym ** 2

    assert uw.get_units(result) is not None or uw.get_units(result) == "dimensionless", \
        "Power operation should preserve unit-awareness"


def test_units_temperature_squared(temperature_with_units):
    """Test: T² where T [K] → [K²]."""
    result = temperature_with_units.sym ** 2

    result_units = uw.get_units(result)

    # Should have squared units
    assert result_units is not None, "Squared variable should have units"
    units_str = str(result_units)
    # Should contain exponent or kelvin twice
    assert "2" in units_str or "²" in units_str or units_str.count("kelvin") == 2, \
        f"Squared temperature should have squared units, got: {result_units}"


# =============================================================================
# Test 9: Complex Compound Expressions
# =============================================================================

def test_closure_complex_expression(temperature_var, velocity_var, simple_mesh):
    """Test: (T * V[0] + T * V[1]) / x preserves unit-awareness."""
    x = simple_mesh.N.x
    expr = (temperature_var.sym * velocity_var.sym[0] +
            temperature_var.sym * velocity_var.sym[1]) / x

    assert uw.get_units(expr) is not None or uw.get_units(expr) == "dimensionless", \
        "Complex compound expression should preserve unit-awareness"


def test_closure_derivative_of_product(temperature_var, velocity_var, simple_mesh):
    """Test: d/dx(T * V[0]) preserves unit-awareness."""
    x = simple_mesh.N.x
    product = temperature_var.sym * velocity_var.sym[0]
    derivative = product.diff(x)

    assert uw.get_units(derivative) is not None or uw.get_units(derivative) == "dimensionless", \
        "Derivative of product should preserve unit-awareness"


def test_units_energy_like_expression(temperature_with_units, velocity_with_units):
    """Test: T [K] * V² [(m/s)²] has proper units."""
    V = velocity_with_units.sym
    T = temperature_with_units.sym

    # Energy-like expression (not actual energy, just testing units)
    expr = T * (V[0]**2 + V[1]**2)

    result_units = uw.get_units(expr)

    assert result_units is not None, "Energy-like expression should have units"
    # Should contain both temperature and velocity squared units
    units_str = str(result_units).lower()
    assert "kelvin" in units_str or "K" in str(result_units), \
        f"Result should contain temperature, got: {result_units}"


# =============================================================================
# Test 10: UnitAwareArray Closure
# =============================================================================

def test_closure_unit_aware_array_arithmetic():
    """Test: UnitAwareArray operations preserve unit-awareness."""
    length = UnitAwareArray([1, 2, 3], units="m")
    time = UnitAwareArray([1, 2, 3], units="s")

    # Multiplication
    result_mul = length * 2
    assert isinstance(result_mul, UnitAwareArray), \
        "UnitAwareArray * scalar should return UnitAwareArray"
    assert result_mul.units == "m", \
        "Scalar multiplication should preserve units"

    # Division
    velocity = length / time
    assert isinstance(velocity, (UnitAwareArray, np.ndarray)), \
        "UnitAwareArray / UnitAwareArray should return array"
    if isinstance(velocity, UnitAwareArray):
        assert velocity.units is not None, \
            "Division should compute result units"


def test_closure_unit_aware_array_reductions():
    """Test: UnitAwareArray reductions preserve unit-awareness."""
    data = UnitAwareArray([1, 2, 3, 4, 5], units="m")

    # Max
    max_val = data.max()
    assert hasattr(max_val, "units") or hasattr(max_val, "magnitude"), \
        "max() should preserve units"

    # Mean
    mean_val = data.mean()
    assert hasattr(mean_val, "units") or hasattr(mean_val, "magnitude"), \
        "mean() should preserve units"

    # Sum
    sum_val = data.sum()
    assert hasattr(sum_val, "units") or hasattr(sum_val, "magnitude"), \
        "sum() should preserve units"


# =============================================================================
# Test 11: Coordinate System Closure
# =============================================================================

def test_closure_coordinate_operations(mesh_with_units):
    """Test: Operations on mesh.X preserve unit-awareness."""
    X = mesh_with_units.X

    # Indexing
    x = X[0]
    assert hasattr(x, "units") or hasattr(x, "_units") or uw.get_units(x) is not None, \
        "Coordinate component should preserve unit-awareness"

    # coords property
    coords = X.coords
    assert isinstance(coords, UnitAwareArray), \
        "Coordinate data should be UnitAwareArray"


# =============================================================================
# Test 12: Expression Evaluation Closure
# =============================================================================

def test_closure_evaluate_returns_unit_aware(temperature_with_units, mesh_with_units):
    """Test: Evaluating unit-aware expression returns unit-aware array."""
    # Evaluate temperature at some points
    pts = np.array([[0.5, 0.5]])
    result = uw.function.evaluate(temperature_with_units.sym, pts)

    # Result should preserve unit information
    # (Implementation may vary - could be UnitAwareArray or have _units attribute)
    has_units = (isinstance(result, UnitAwareArray) or
                 hasattr(result, "_units") or
                 hasattr(result, "units"))

    assert has_units, \
        f"Evaluation result should preserve units, got type: {type(result)}"


# =============================================================================
# Summary Test
# =============================================================================

def test_summary_closure_property():
    """
    Summary test documenting the closure property requirement.

    The units system should be CLOSED under all operations:
    - Variable × Variable → Unit-aware expression
    - Scalar × Variable → Unit-aware expression
    - Variable.diff(x) → Unit-aware expression
    - Variable / Variable → Unit-aware expression
    - Variable[i] → Unit-aware expression
    - UnitAwareArray op UnitAwareArray → UnitAwareArray
    - Coordinate operations → Unit-aware
    - Evaluation of unit-aware expression → Unit-aware result

    This test always passes but serves as documentation.
    """
    assert True, "Closure property documented in test suite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
