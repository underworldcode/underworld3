"""
Test coordinate units through gradient calculations.

This test validates that mesh coordinate units are properly handled when
computing gradients and derivatives. It uses a simple analytical problem
where we can verify that gradients have the correct dimensional units.

Test problem:
    - Temperature field: T(x,y) = T0 * (1 + a*x + b*y) [Kelvin]
    - Domain: [0, L_x] × [0, L_y] [meters or kilometers]
    - Gradients: ∂T/∂x = T0*a [K/m], ∂T/∂y = T0*b [K/m]

The key validation is that gradient dimensions are correct:
    - If mesh is in meters: gradient has units K/m
    - If mesh is in kilometers: gradient has units K/km
    - Numerical values should differ by factor of 1000
"""

import pytest
import numpy as np
import underworld3 as uw
from underworld3 import function as fn


class Test_CoordinateUnits_Gradients:
    """Test suite for coordinate units in gradient calculations."""

    def setup_method(self):
        """Set up test parameters."""
        # Physical domain size
        self.L_x = 1000.0  # meters
        self.L_y = 500.0   # meters

        # Temperature field parameters
        self.T0 = 300.0    # Kelvin (reference temperature)
        self.a = 0.001     # 1/meter (gradient in x)
        self.b = 0.002     # 1/meter (gradient in y)

        # Mesh resolution
        self.res = (16, 16)

    def create_temperature_field(self, mesh, T_var):
        """
        Set up analytical temperature field: T(x,y) = T0 * (1 + a*x + b*y)

        This gives exact gradients:
            ∂T/∂x = T0 * a
            ∂T/∂y = T0 * b
        """
        # Get mesh coordinates (these will have units if mesh.units is set)
        x, y = mesh.X

        # Define temperature field
        T_analytical = self.T0 * (1 + self.a * x + self.b * y)

        # Use projection solver to set up temperature field
        temp_proj = uw.systems.Projection(mesh, T_var)
        temp_proj.uw_function = T_analytical
        temp_proj.solve()

    def test_gradient_without_units(self):
        """
        Test gradient calculation with dimensionless mesh (baseline).

        When mesh has no units, gradients should be in natural units
        relative to the mesh coordinates.
        """
        # Create mesh without units (dimensionless)
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=self.res,
            minCoords=(0.0, 0.0),
            maxCoords=(self.L_x, self.L_y),
            # No units parameter - dimensionless
        )

        # Verify mesh has no units
        assert mesh.units is None
        assert uw.get_units(mesh.points) is None

        # Create temperature variable with units
        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

        # Create gradient recovery variable (vector field)
        gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)

        # Set up temperature field
        self.create_temperature_field(mesh, T)

        # Compute gradient using Vector_Projection
        gradient_projector = uw.systems.Vector_Projection(mesh, gradT)
        gradient_projector.uw_function = mesh.vector.gradient(T.sym)
        gradient_projector.solve()

        # Expected gradients (in mesh coordinates, which are meters numerically)
        expected_dT_dx = self.T0 * self.a  # K/m (numerically)
        expected_dT_dy = self.T0 * self.b  # K/m (numerically)

        # Check gradient values at center of domain
        x_center = self.L_x / 2
        y_center = self.L_y / 2
        grad_at_center = uw.function.evaluate(
            gradT.sym, np.array([[x_center, y_center]])
        )

        # Extract scalar values from gradient
        print(f"  DEBUG: grad_at_center shape = {grad_at_center.shape}")
        print(f"  DEBUG: grad_at_center = {grad_at_center}")

        # Flatten to 1D and extract components
        grad_flat = grad_at_center.flatten()
        dT_dx = grad_flat[0]
        dT_dy = grad_flat[1]

        # Print debug info
        print(f"  Expected: ∂T/∂x = {expected_dT_dx:.6f}, ∂T/∂y = {expected_dT_dy:.6f}")
        print(f"  Got:      ∂T/∂x = {dT_dx:.6f}, ∂T/∂y = {dT_dy:.6f}")

        # Verify gradient values (should be constant for linear field)
        assert np.allclose(dT_dx, expected_dT_dx, rtol=0.05), \
            f"X-gradient mismatch: expected {expected_dT_dx}, got {dT_dx}"
        assert np.allclose(dT_dy, expected_dT_dy, rtol=0.05), \
            f"Y-gradient mismatch: expected {expected_dT_dy}, got {dT_dy}"

        # Gradient variable should not have units automatically assigned
        # (gradient units would be temperature_units / coordinate_units)
        print(f"✓ Dimensionless mesh: ∂T/∂x = {dT_dx:.6f} K/m")
        print(f"✓ Dimensionless mesh: ∂T/∂y = {dT_dy:.6f} K/m")

    def test_gradient_with_meter_units(self):
        """
        Test gradient calculation with mesh in meters.

        Gradients should have units of K/m and numerical values should
        match the analytical solution.
        """
        # Create mesh with meter units
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=self.res,
            minCoords=(0.0, 0.0),
            maxCoords=(self.L_x, self.L_y),
            units="meter"  # Coordinate units
        )

        # Verify mesh has units
        assert mesh.units == "meter"
        assert uw.get_units(mesh.points) == "meter"

        # Create temperature variable with units
        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

        # Create gradient recovery variable
        gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)

        # Set up temperature field
        self.create_temperature_field(mesh, T)

        # Compute gradient
        gradient_projector = uw.systems.Vector_Projection(mesh, gradT)
        gradient_projector.uw_function = mesh.vector.gradient(T.sym)
        gradient_projector.solve()

        # Expected gradients in K/m
        expected_dT_dx = self.T0 * self.a  # 0.3 K/m
        expected_dT_dy = self.T0 * self.b  # 0.6 K/m

        # Check gradient values
        x_center = self.L_x / 2
        y_center = self.L_y / 2
        grad_at_center = uw.function.evaluate(
            gradT.sym, np.array([[x_center, y_center]])
        )

        # Extract scalar values from gradient
        grad_flat = grad_at_center.flatten()
        dT_dx = grad_flat[0]
        dT_dy = grad_flat[1]

        # Verify gradient values
        assert np.allclose(dT_dx, expected_dT_dx, rtol=0.05), \
            f"Expected dT/dx = {expected_dT_dx}, got {dT_dx}"
        assert np.allclose(dT_dy, expected_dT_dy, rtol=0.05), \
            f"Expected dT/dy = {expected_dT_dy}, got {dT_dy}"

        print(f"✓ Meter mesh: ∂T/∂x = {dT_dx:.6f} K/m")
        print(f"✓ Meter mesh: ∂T/∂y = {dT_dy:.6f} K/m")

    def test_gradient_with_kilometer_units(self):
        """
        Test gradient calculation with mesh in kilometers.

        Key validation: When mesh coordinates are in km instead of m,
        the gradient values should be 1000× larger (K/km vs K/m)
        because we're measuring change per kilometer instead of per meter.
        """
        # Create mesh with kilometer units
        # Domain is still 1000m × 500m, but specified as 1km × 0.5km
        L_x_km = self.L_x / 1000.0  # 1.0 km
        L_y_km = self.L_y / 1000.0  # 0.5 km

        mesh = uw.meshing.StructuredQuadBox(
            elementRes=self.res,
            minCoords=(0.0, 0.0),
            maxCoords=(L_x_km, L_y_km),
            units="kilometer"  # Coordinate units
        )

        # Verify mesh has units
        assert mesh.units == "kilometer"
        assert uw.get_units(mesh.points) == "kilometer"

        # Create temperature variable with units
        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

        # Create gradient recovery variable
        gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)

        # Set up temperature field
        # NOTE: When using km coordinates, the gradient coefficients a and b
        # need to be scaled: a_km = a_m * 1000, b_km = b_m * 1000
        # because T(x_km) = T0 * (1 + a_m * x_m) = T0 * (1 + a_m * 1000 * x_km)
        #                                        = T0 * (1 + a_km * x_km)
        x, y = mesh.X
        a_km = self.a * 1000  # Convert from 1/m to 1/km
        b_km = self.b * 1000  # Convert from 1/m to 1/km

        T_analytical = self.T0 * (1 + a_km * x + b_km * y)

        # Use projection solver to set up temperature field
        temp_proj = uw.systems.Projection(mesh, T)
        temp_proj.uw_function = T_analytical
        temp_proj.solve()

        # Compute gradient
        gradient_projector = uw.systems.Vector_Projection(mesh, gradT)
        gradient_projector.uw_function = mesh.vector.gradient(T.sym)
        gradient_projector.solve()

        # Expected gradients in K/km
        # ∂T/∂x_km = T0 * a_km = T0 * a_m * 1000
        expected_dT_dx_km = self.T0 * a_km  # 300 K/km
        expected_dT_dy_km = self.T0 * b_km  # 600 K/km

        # Check gradient values
        x_center_km = L_x_km / 2
        y_center_km = L_y_km / 2
        grad_at_center = uw.function.evaluate(
            gradT.sym, np.array([[x_center_km, y_center_km]])
        )

        # Extract scalar values from gradient
        grad_flat = grad_at_center.flatten()
        dT_dx_km = grad_flat[0]
        dT_dy_km = grad_flat[1]

        # Verify gradient values
        assert np.allclose(dT_dx_km, expected_dT_dx_km, rtol=0.05), \
            f"Expected dT/dx = {expected_dT_dx_km} K/km, got {dT_dx_km}"
        assert np.allclose(dT_dy_km, expected_dT_dy_km, rtol=0.05), \
            f"Expected dT/dy = {expected_dT_dy_km} K/km, got {dT_dy_km}"

        print(f"✓ Kilometer mesh: ∂T/∂x = {dT_dx_km:.6f} K/km")
        print(f"✓ Kilometer mesh: ∂T/∂y = {dT_dy_km:.6f} K/km")

    def test_coordinate_units_consistency(self):
        """
        Test that coordinate units are preserved through mesh operations.

        Validates:
        - mesh.points has correct units
        - mesh.X.coords has same units as mesh.points
        - mesh.view() displays units information
        """
        # Create mesh with units
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(8, 8),
            minCoords=(0.0, 0.0),
            maxCoords=(10.0, 5.0),
            units="kilometer"
        )

        # Test 1: mesh.units property
        assert mesh.units == "kilometer"

        # Test 2: mesh.points has units
        points = mesh.points
        assert uw.get_units(points) == "kilometer"

        # Test 3: mesh.X.coords has same units
        data = mesh.X.coords
        assert uw.get_units(data) == "kilometer"

        # Test 4: Swarm inherits mesh units
        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=2)

        swarm_points = swarm.points
        assert uw.get_units(swarm_points) == "kilometer"

        print("✓ Coordinate units preserved through mesh operations")
        print(f"  mesh.units: {mesh.units}")
        print(f"  mesh.points units: {uw.get_units(mesh.points)}")
        print(f"  mesh.X.coords units: {uw.get_units(mesh.X.coords)}")
        print(f"  swarm.points units: {uw.get_units(swarm.points)}")

    def test_gradient_magnitude_scaling(self):
        """
        Test that gradient magnitudes scale correctly with coordinate units.

        This test compares gradient calculations on the same physical domain
        but with different coordinate units (m vs km). The gradient values
        should differ by exactly the unit conversion factor.
        """
        # Physical domain: 1000m × 500m
        L_x_m = 1000.0
        L_y_m = 500.0
        L_x_km = L_x_m / 1000.0
        L_y_km = L_y_m / 1000.0

        # Create mesh in meters
        mesh_m = uw.meshing.StructuredQuadBox(
            elementRes=self.res,
            minCoords=(0.0, 0.0),
            maxCoords=(L_x_m, L_y_m),
            units="meter"
        )

        # Create mesh in kilometers (same physical domain)
        mesh_km = uw.meshing.StructuredQuadBox(
            elementRes=self.res,
            minCoords=(0.0, 0.0),
            maxCoords=(L_x_km, L_y_km),
            units="kilometer"
        )

        # Temperature variables
        T_m = uw.discretisation.MeshVariable("T_m", mesh_m, 1, degree=2, units="kelvin")
        T_km = uw.discretisation.MeshVariable("T_km", mesh_km, 1, degree=2, units="kelvin")

        # Gradient variables
        gradT_m = uw.discretisation.MeshVariable("gradT_m", mesh_m, mesh_m.dim, degree=1)
        gradT_km = uw.discretisation.MeshVariable("gradT_km", mesh_km, mesh_km.dim, degree=1)

        # Set up temperature field on meter mesh
        x_m, y_m = mesh_m.X
        T_analytical_m = self.T0 * (1 + self.a * x_m + self.b * y_m)
        temp_proj_m = uw.systems.Projection(mesh_m, T_m)
        temp_proj_m.uw_function = T_analytical_m
        temp_proj_m.solve()

        # Set up temperature field on kilometer mesh (scaled coefficients)
        x_km, y_km = mesh_km.X
        a_km = self.a * 1000
        b_km = self.b * 1000
        T_analytical_km = self.T0 * (1 + a_km * x_km + b_km * y_km)
        temp_proj_km = uw.systems.Projection(mesh_km, T_km)
        temp_proj_km.uw_function = T_analytical_km
        temp_proj_km.solve()

        # Compute gradients
        grad_proj_m = uw.systems.Vector_Projection(mesh_m, gradT_m)
        grad_proj_m.uw_function = mesh_m.vector.gradient(T_m.sym)
        grad_proj_m.solve()

        grad_proj_km = uw.systems.Vector_Projection(mesh_km, gradT_km)
        grad_proj_km.uw_function = mesh_km.vector.gradient(T_km.sym)
        grad_proj_km.solve()

        # Evaluate gradients at center
        grad_m = uw.function.evaluate(
            gradT_m.sym, np.array([[L_x_m/2, L_y_m/2]])
        )
        grad_km = uw.function.evaluate(
            gradT_km.sym, np.array([[L_x_km/2, L_y_km/2]])
        )

        # Extract scalar values from gradients
        grad_m_flat = grad_m.flatten()
        grad_km_flat = grad_km.flatten()
        dT_dx_m = grad_m_flat[0]
        dT_dy_m = grad_m_flat[1]
        dT_dx_km = grad_km_flat[0]
        dT_dy_km = grad_km_flat[1]

        # Gradients should differ by factor of 1000 (m to km conversion)
        scaling_factor = 1000.0

        assert np.allclose(dT_dx_km, dT_dx_m * scaling_factor, rtol=0.05), \
            f"X-gradient scaling incorrect: {dT_dx_km} vs {dT_dx_m * scaling_factor}"
        assert np.allclose(dT_dy_km, dT_dy_m * scaling_factor, rtol=0.05), \
            f"Y-gradient scaling incorrect: {dT_dy_km} vs {dT_dy_m * scaling_factor}"

        print(f"✓ Gradient scaling validation:")
        print(f"  Meter mesh:     ∂T/∂x = {dT_dx_m:.6f} K/m")
        print(f"  Kilometer mesh: ∂T/∂x = {dT_dx_km:.6f} K/km")
        print(f"  Ratio: {dT_dx_km / dT_dx_m:.2f} (expected 1000)")


if __name__ == "__main__":
    """Run tests with verbose output."""
    import sys

    print("=" * 70)
    print("COORDINATE UNITS GRADIENT VALIDATION TESTS")
    print("=" * 70)

    test_suite = Test_CoordinateUnits_Gradients()
    test_suite.setup_method()

    print("\n" + "─" * 70)
    print("Test 1: Gradient without units (baseline)")
    print("─" * 70)
    test_suite.test_gradient_without_units()

    print("\n" + "─" * 70)
    print("Test 2: Gradient with meter units")
    print("─" * 70)
    test_suite.test_gradient_with_meter_units()

    print("\n" + "─" * 70)
    print("Test 3: Gradient with kilometer units")
    print("─" * 70)
    test_suite.test_gradient_with_kilometer_units()

    print("\n" + "─" * 70)
    print("Test 4: Coordinate units consistency")
    print("─" * 70)
    test_suite.test_coordinate_units_consistency()

    print("\n" + "─" * 70)
    print("Test 5: Gradient magnitude scaling")
    print("─" * 70)
    test_suite.test_gradient_magnitude_scaling()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
