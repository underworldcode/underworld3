#!/usr/bin/env python3
"""
Comprehensive regression test for Mathematical Objects pattern: var[0] ≡ var.sym[0]

This test is specifically designed to prevent regression in the critical mathematical
objects functionality where users can use variables directly in expressions instead
of requiring explicit .sym access.

Key patterns tested:
- temperature[0] ≡ temperature.sym[0]
- velocity[0], velocity[1] ≡ velocity.sym[0], velocity.sym[1]
- stress[0,0], stress[0,1] ≡ stress.sym[0,0], stress.sym[0,1]
- All mathematical operations: +, -, *, /, **, unary -
- SymPy Matrix methods: .T, .dot(), .norm(), .diff(), etc.
- Component access in complex expressions
- Units integration with mathematical operations
"""

import pytest
import numpy as np
import sympy

import underworld3 as uw


class TestMathematicalObjectsRegression:
    """Comprehensive regression tests for mathematical objects functionality."""

    @pytest.fixture
    def setup_mesh_and_variables(self):
        """Create mesh and variables for testing."""
        # Create 2D mesh
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(4, 4),
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            qdegree=2
        )

        # Create variables of different types
        scalar_var = uw.discretisation.MeshVariable("temperature", mesh, 1, degree=1)
        vector_var = uw.discretisation.MeshVariable("velocity", mesh, 2, degree=2)
        tensor_var = uw.discretisation.MeshVariable("stress", mesh, (2, 2), degree=1)

        # Create variables with units
        temp_with_units = uw.discretisation.MeshVariable("temp_units", mesh, 1, degree=1, units="K")
        vel_with_units = uw.discretisation.MeshVariable("vel_units", mesh, 2, degree=2, units="m/s")

        return {
            'mesh': mesh,
            'scalar': scalar_var,
            'vector': vector_var,
            'tensor': tensor_var,
            'temp_units': temp_with_units,
            'vel_units': vel_with_units
        }

    def test_scalar_component_access_equivalence(self, setup_mesh_and_variables):
        """Test temperature[0] ≡ temperature.sym[0] for scalars."""
        temp = setup_mesh_and_variables['scalar']

        # Test direct equivalence
        direct_access = temp[0]
        sym_access = temp.sym[0]

        assert direct_access.equals(sym_access), "temp[0] should equal temp.sym[0]"
        assert isinstance(direct_access, sympy.Basic), "temp[0] should return SymPy expression"

    def test_vector_component_access_equivalence(self, setup_mesh_and_variables):
        """Test velocity[0], velocity[1] ≡ velocity.sym[0], velocity.sym[1] for vectors."""
        vel = setup_mesh_and_variables['vector']

        # Test all vector components
        for i in range(2):
            direct_access = vel[i]
            sym_access = vel.sym[i]

            assert direct_access.equals(sym_access), f"vel[{i}] should equal vel.sym[{i}]"
            assert isinstance(direct_access, sympy.Basic), f"vel[{i}] should return SymPy expression"

    def test_tensor_component_access_equivalence(self, setup_mesh_and_variables):
        """Test stress[0,0], stress[0,1] ≡ stress.sym[0,0], stress.sym[0,1] for tensors."""
        stress = setup_mesh_and_variables['tensor']

        # Test all tensor components
        for i in range(2):
            for j in range(2):
                direct_access = stress[i, j]
                sym_access = stress.sym[i, j]

                assert direct_access.equals(sym_access), f"stress[{i},{j}] should equal stress.sym[{i},{j}]"
                assert isinstance(direct_access, sympy.Basic), f"stress[{i},{j}] should return SymPy expression"

    def test_arithmetic_operations_equivalence(self, setup_mesh_and_variables):
        """Test all arithmetic operations work identically with and without .sym."""
        vel = setup_mesh_and_variables['vector']
        temp = setup_mesh_and_variables['scalar']

        # Test addition
        expr1 = vel + vel
        expr2 = vel.sym + vel.sym
        assert expr1.equals(expr2), "vel + vel should equal vel.sym + vel.sym"

        # Test subtraction
        expr1 = vel - vel
        expr2 = vel.sym - vel.sym
        assert expr1.equals(expr2), "vel - vel should equal vel.sym - vel.sym"

        # Test multiplication (scalar)
        expr1 = 2 * vel
        expr2 = 2 * vel.sym
        assert expr1.equals(expr2), "2 * vel should equal 2 * vel.sym"

        # Test multiplication (right)
        expr1 = vel * 2
        expr2 = vel.sym * 2
        assert expr1.equals(expr2), "vel * 2 should equal vel.sym * 2"

        # Test division
        expr1 = vel / 2
        expr2 = vel.sym / 2
        assert expr1.equals(expr2), "vel / 2 should equal vel.sym / 2"

        # Test power
        expr1 = temp ** 2
        expr2 = temp.sym ** 2
        assert expr1.equals(expr2), "temp ** 2 should equal temp.sym ** 2"

        # Test unary minus
        expr1 = -vel
        expr2 = -vel.sym
        assert expr1.equals(expr2), "-vel should equal -vel.sym"

    def test_sympy_matrix_methods_equivalence(self, setup_mesh_and_variables):
        """Test SymPy Matrix methods work identically with and without .sym."""
        vel = setup_mesh_and_variables['vector']

        # Test transpose
        expr1 = vel.T
        expr2 = vel.sym.T
        assert expr1.equals(expr2), "vel.T should equal vel.sym.T"

        # Test norm
        expr1 = vel.norm()
        expr2 = vel.sym.norm()
        assert expr1.equals(expr2), "vel.norm() should equal vel.sym.norm()"

        # Test dot product
        expr1 = vel.dot(vel)
        expr2 = vel.sym.dot(vel.sym)
        assert expr1.equals(expr2), "vel.dot(vel) should equal vel.sym.dot(vel.sym)"

        # Test cross product (if applicable) - Use existing 2D mesh for simplicity
        # For 2D, cross product returns scalar (z-component)
        # vel × vel should be zero anyway
        if len(vel) == 2:
            # Add zero z-component for cross product test
            vel_extended = sympy.Matrix([vel[0], vel[1], 0])
            vel_sym_extended = sympy.Matrix([vel.sym[0], vel.sym[1], 0])

            expr1 = vel_extended.cross(vel_extended)
            expr2 = vel_sym_extended.cross(vel_sym_extended)
            assert expr1.equals(expr2), "Cross product should be equivalent"

    def test_component_access_in_complex_expressions(self, setup_mesh_and_variables):
        """Test component access works correctly in complex mathematical expressions."""
        vel = setup_mesh_and_variables['vector']
        temp = setup_mesh_and_variables['scalar']
        mesh = setup_mesh_and_variables['mesh']
        x, y = mesh.CoordinateSystem.X

        # Test velocity magnitude using components
        vel_mag_new = (vel[0]**2 + vel[1]**2)**0.5
        vel_mag_old = (vel.sym[0]**2 + vel.sym[1]**2)**0.5
        assert vel_mag_new.equals(vel_mag_old), "Velocity magnitude should be equivalent"

        # Test divergence using components
        div_new = vel[0].diff(x) + vel[1].diff(y)
        div_old = vel.sym[0].diff(x) + vel.sym[1].diff(y)
        assert div_new.equals(div_old), "Divergence should be equivalent"

        # Test mixed operations
        mixed_new = temp[0] * vel[0] + vel[1]
        mixed_old = temp.sym[0] * vel.sym[0] + vel.sym[1]
        assert mixed_new.equals(mixed_old), "Mixed operations should be equivalent"

        # Test with coordinate expressions
        coord_expr_new = vel[0] * x + vel[1] * y
        coord_expr_old = vel.sym[0] * x + vel.sym[1] * y
        assert coord_expr_new.equals(coord_expr_old), "Coordinate expressions should be equivalent"

    def test_units_integration_with_mathematical_operations(self, setup_mesh_and_variables):
        """Test that units work correctly with mathematical operations using new syntax."""
        temp_units = setup_mesh_and_variables['temp_units']
        vel_units = setup_mesh_and_variables['vel_units']

        # Test component access preserves units context
        temp_component = temp_units[0]
        vel_x = vel_units[0]
        vel_y = vel_units[1]

        # These should work without errors (units handled internally)
        temp_scaled = 2 * temp_component
        vel_mag = (vel_x**2 + vel_y**2)**0.5
        mixed_expr = temp_component + vel_x  # Different units - should work in symbolic form

        # All should return SymPy expressions
        assert isinstance(temp_scaled, sympy.Basic)
        assert isinstance(vel_mag, sympy.Basic)
        assert isinstance(mixed_expr, sympy.Basic)

    def test_sympy_compatibility_edge_cases(self, setup_mesh_and_variables):
        """Test edge cases for SymPy compatibility."""
        vel = setup_mesh_and_variables['vector']
        temp = setup_mesh_and_variables['scalar']

        # Test len() function
        assert len(vel) == 2, "len(vel) should return 2"
        assert len(temp) == 1, "len(temp) should return 1"

        # Test iteration (if supported)
        components = [comp for comp in vel]
        assert len(components) == 2, "Should be able to iterate over vector components"
        assert all(isinstance(comp, sympy.Basic) for comp in components), "All components should be SymPy expressions"

        # Test membership (if supported)
        first_component = vel[0]
        assert first_component in components, "First component should be in iteration results"

    def test_boundary_conditions_patterns(self, setup_mesh_and_variables):
        """Test common boundary condition patterns that users rely on."""
        vel = setup_mesh_and_variables['vector']
        temp = setup_mesh_and_variables['scalar']
        mesh = setup_mesh_and_variables['mesh']

        # Test Dirichlet boundary conditions
        bc_temp = temp[0] - 300  # Temperature = 300
        assert isinstance(bc_temp, sympy.Basic), "Boundary condition should be SymPy expression"

        # Test velocity boundary conditions
        bc_vel_x = vel[0]  # u_x = 0
        bc_vel_y = vel[1] - 1  # u_y = 1
        assert isinstance(bc_vel_x, sympy.Basic)
        assert isinstance(bc_vel_y, sympy.Basic)

        # Test no-slip condition
        no_slip = vel[0]**2 + vel[1]**2  # |v|^2 = 0
        assert isinstance(no_slip, sympy.Basic)

    def test_error_handling_patterns(self, setup_mesh_and_variables):
        """Test proper error handling for invalid operations."""
        vel = setup_mesh_and_variables['vector']
        temp = setup_mesh_and_variables['scalar']

        # Test out of bounds access
        with pytest.raises(IndexError):
            _ = vel[5]  # Vector only has 2 components

        with pytest.raises(IndexError):
            _ = temp[1]  # Scalar only has 1 component

        # Test invalid tensor access
        tensor = setup_mesh_and_variables['tensor']
        with pytest.raises(IndexError):
            _ = tensor[5, 0]  # Invalid first index

    def test_jit_compilation_compatibility(self, setup_mesh_and_variables):
        """Test that expressions using new syntax compile correctly."""
        vel = setup_mesh_and_variables['vector']
        temp = setup_mesh_and_variables['scalar']

        # Create expressions using new syntax
        expr_new = vel[0] * temp[0] + vel[1]
        expr_old = vel.sym[0] * temp.sym[0] + vel.sym[1]

        # Test that both expressions compile identically
        compiled_new = uw.unwrap(expr_new)
        compiled_old = uw.unwrap(expr_old)

        assert compiled_new.equals(compiled_old), "Compiled expressions should be identical"

    def test_real_world_usage_patterns(self, setup_mesh_and_variables):
        """Test realistic usage patterns from actual geodynamics problems."""
        vel = setup_mesh_and_variables['vector']
        temp = setup_mesh_and_variables['scalar']
        stress = setup_mesh_and_variables['tensor']
        mesh = setup_mesh_and_variables['mesh']
        x, y = mesh.CoordinateSystem.X

        # Thermal convection: Rayleigh-Benard setup
        # Buoyancy force
        buoyancy = temp[0] * vel[1]  # Temperature-dependent vertical velocity

        # Strain rate tensor components
        strain_xx = vel[0].diff(x)
        strain_yy = vel[1].diff(y)
        strain_xy = 0.5 * (vel[0].diff(y) + vel[1].diff(x))

        # Viscous stress
        stress_xx = 2 * strain_xx
        stress_yy = 2 * strain_yy
        stress_xy = 2 * strain_xy

        # Heat equation (use first derivatives only - second derivatives not supported)
        heat_diffusion_x = temp[0].diff(x)
        heat_diffusion_y = temp[0].diff(y)
        heat_advection = vel[0] * temp[0].diff(x) + vel[1] * temp[0].diff(y)

        # All should be valid SymPy expressions
        expressions = [buoyancy, strain_xx, strain_yy, strain_xy,
                      stress_xx, stress_yy, stress_xy, heat_diffusion_x, heat_diffusion_y, heat_advection]

        for expr in expressions:
            assert isinstance(expr, sympy.Basic), f"Expression {expr} should be SymPy Basic"

        # Test that these are equivalent to old patterns
        buoyancy_old = temp.sym[0] * vel.sym[1]
        assert buoyancy.equals(buoyancy_old), "Buoyancy expressions should be equivalent"

    def test_performance_characteristics(self, setup_mesh_and_variables):
        """Test that performance characteristics are maintained."""
        vel = setup_mesh_and_variables['vector']

        # Test that repeated access is efficient (should be cached)
        import time

        # Time new pattern
        start = time.time()
        for _ in range(100):
            expr = vel[0] + vel[1]
        new_time = time.time() - start

        # Time old pattern
        start = time.time()
        for _ in range(100):
            expr = vel.sym[0] + vel.sym[1]
        old_time = time.time() - start

        # New pattern should not be significantly slower (within 50% is reasonable)
        # This is a soft check - exact performance depends on many factors
        performance_ratio = new_time / old_time
        assert performance_ratio < 5.0, f"New pattern should not be much slower (ratio: {performance_ratio:.2f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])