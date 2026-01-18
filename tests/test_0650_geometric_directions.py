#!/usr/bin/env python3

"""
Unit tests for geometric direction properties in coordinate systems.

Tests all coordinate system types for:
1. Basic geometric properties (unit_horizontal, unit_vertical, etc.)
2. Coordinate-system-specific properties (unit_radial, unit_tangential, etc.)
3. Metadata properties (geometric_dimension_names, primary_directions)
4. Consistency with existing unit_e_0, unit_e_1, unit_e_2 vectors
5. Error handling for inappropriate coordinate systems

These tests prevent regressions and ensure geometric properties work correctly
across all mesh types and coordinate systems.
"""

import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
import numpy as np
import sympy
import underworld3 as uw
from underworld3.coordinates import CoordinateSystemType


class TestGeometricDirections:
    """Test geometric direction properties for all coordinate system types"""

    def test_cartesian_2d_geometric_properties(self):
        """Test geometric properties for 2D Cartesian coordinate system"""
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
        )
        cs = mesh.CoordinateSystem

        # Test coordinate system identification
        assert cs.coordinate_type == CoordinateSystemType.CARTESIAN
        assert mesh.dim == 2

        # Test geometric dimension names
        assert cs.geometric_dimension_names == ["horizontal", "vertical"]

        # Test basic geometric properties
        assert cs.unit_horizontal.equals(sympy.Matrix([[1, 0]]))
        assert cs.unit_vertical.equals(sympy.Matrix([[0, 1]]))
        assert cs.unit_horizontal_0.equals(cs.unit_horizontal)
        assert cs.unit_horizontal_1.equals(sympy.Matrix([[0, 1]]))

        # Test consistency with existing unit vectors
        assert cs.unit_horizontal.equals(cs.unit_e_0)
        assert cs.unit_vertical.equals(cs.unit_e_1)

        # Test that coordinate-specific properties raise appropriate errors
        with pytest.raises(NotImplementedError, match="unit_radial not defined"):
            _ = cs.unit_radial
        with pytest.raises(NotImplementedError, match="unit_tangential not defined"):
            _ = cs.unit_tangential

        # Test primary_directions dictionary
        directions = cs.primary_directions
        expected_keys = {
            "unit_e_0",
            "unit_e_1",
            "unit_horizontal",
            "unit_horizontal_0",
            "unit_horizontal_1",
            "unit_vertical",
        }
        assert set(directions.keys()) == expected_keys

    def test_cartesian_3d_geometric_properties(self):
        """Test geometric properties for 3D Cartesian coordinate system"""
        mesh = uw.meshing.UnstructuredSimplexBox(
            regular=True, minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.5
        )
        cs = mesh.CoordinateSystem

        # Test coordinate system identification
        assert cs.coordinate_type == CoordinateSystemType.CARTESIAN
        assert mesh.dim == 3

        # Test geometric dimension names
        assert cs.geometric_dimension_names == ["horizontal_x", "horizontal_y", "vertical"]

        # Test basic geometric properties
        assert cs.unit_horizontal.equals(sympy.Matrix([[1, 0, 0]]))
        assert cs.unit_vertical.equals(sympy.Matrix([[0, 0, 1]]))
        assert cs.unit_horizontal_0.equals(cs.unit_horizontal)
        assert cs.unit_horizontal_1.equals(sympy.Matrix([[0, 1, 0]]))

        # Test consistency with existing unit vectors
        assert cs.unit_horizontal.equals(cs.unit_e_0)
        assert cs.unit_horizontal_1.equals(cs.unit_e_1)
        assert cs.unit_vertical.equals(cs.unit_e_2)

    def test_cylindrical_2d_geometric_properties(self):
        """Test geometric properties for 2D cylindrical coordinate system"""
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
        cs = mesh.CoordinateSystem

        # Test coordinate system identification
        assert cs.coordinate_type == CoordinateSystemType.CYLINDRICAL2D
        assert mesh.dim == 2

        # Test geometric dimension names
        assert cs.geometric_dimension_names == ["radial", "tangential"]

        # Test that radial and tangential properties work
        radial = cs.unit_radial
        tangential = cs.unit_tangential

        # Test consistency with existing unit vectors
        assert radial.equals(cs.unit_e_0)
        assert tangential.equals(cs.unit_e_1)

        # Test that radial and tangential are orthogonal (symbolically)
        # This is a key geometric property that must hold
        x, y = cs.mesh.X
        radial_at_point = radial.subs([(x, 1), (y, 0)])  # At (1,0)
        tangential_at_point = tangential.subs([(x, 1), (y, 0)])
        dot_product = float(radial_at_point.dot(tangential_at_point))
        assert abs(dot_product) < 1e-10, "Radial and tangential should be orthogonal"

        # Test vertical direction (should be Cartesian y)
        assert cs.unit_vertical.equals(sympy.Matrix([0, 1]))

        # Test horizontal direction (should be radial)
        assert cs.unit_horizontal.equals(cs.unit_radial)
        assert cs.unit_horizontal_1.equals(cs.unit_tangential)

        # Test that spherical-specific properties raise errors
        with pytest.raises(NotImplementedError, match="unit_meridional not defined"):
            _ = cs.unit_meridional
        with pytest.raises(NotImplementedError, match="unit_azimuthal not defined"):
            _ = cs.unit_azimuthal

        # Test primary_directions dictionary
        directions = cs.primary_directions
        expected_keys = {
            "unit_e_0",
            "unit_e_1",
            "unit_horizontal",
            "unit_horizontal_0",
            "unit_horizontal_1",
            "unit_vertical",
            "unit_radial",
            "unit_tangential",
        }
        assert set(directions.keys()) == expected_keys

    def test_spherical_geometric_properties(self):
        """Test geometric properties for spherical coordinate system"""
        mesh = uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.5, numElements=4)
        cs = mesh.CoordinateSystem

        # Test coordinate system identification
        assert cs.coordinate_type == CoordinateSystemType.SPHERICAL
        assert mesh.dim == 3

        # Test geometric dimension names
        assert cs.geometric_dimension_names == ["radial", "meridional", "azimuthal"]

        # Test spherical-specific properties
        radial = cs.unit_radial
        meridional = cs.unit_meridional
        azimuthal = cs.unit_azimuthal

        # Test consistency with existing unit vectors
        assert radial.equals(cs.unit_e_0)
        assert meridional.equals(cs.unit_e_1)
        assert azimuthal.equals(cs.unit_e_2)

        # Test that vertical is radial in spherical coordinates
        assert cs.unit_vertical.equals(cs.unit_radial)

        # Test that horizontal is meridional in spherical coordinates
        assert cs.unit_horizontal.equals(cs.unit_meridional)
        assert cs.unit_horizontal_1.equals(cs.unit_azimuthal)

        # Test that cylindrical-specific properties raise errors
        with pytest.raises(NotImplementedError, match="unit_tangential not defined"):
            _ = cs.unit_tangential

        # Test primary_directions dictionary
        directions = cs.primary_directions
        expected_keys = {
            "unit_e_0",
            "unit_e_1",
            "unit_e_2",
            "unit_horizontal",
            "unit_horizontal_0",
            "unit_horizontal_1",
            "unit_vertical",
            "unit_radial",
            "unit_meridional",
            "unit_azimuthal",
        }
        assert set(directions.keys()) == expected_keys

    def test_coordinate_system_consistency(self):
        """Test that geometric properties are consistent across coordinate systems"""

        # Test that all coordinate systems provide basic properties
        meshes = [
            uw.meshing.StructuredQuadBox(elementRes=(2, 2)),
            uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5),
            uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.5, numElements=2),
        ]

        for mesh in meshes:
            cs = mesh.CoordinateSystem

            # Every coordinate system should have these basic properties
            assert hasattr(cs, "unit_horizontal")
            assert hasattr(cs, "unit_vertical")
            assert hasattr(cs, "unit_horizontal_0")
            assert hasattr(cs, "geometric_dimension_names")
            assert hasattr(cs, "primary_directions")

            # unit_horizontal_0 should always equal unit_horizontal
            assert cs.unit_horizontal_0.equals(cs.unit_horizontal)

            # Dimension names should match mesh dimension
            assert len(cs.geometric_dimension_names) == mesh.dim

            # Primary directions should always include basic unit vectors
            directions = cs.primary_directions
            assert "unit_e_0" in directions
            assert "unit_e_1" in directions
            if mesh.dim >= 3:
                assert "unit_e_2" in directions

    def test_unit_vector_normalization(self):
        """Test that unit vectors are properly normalized"""

        # Test with annulus (has non-trivial unit vectors)
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.3)
        cs = mesh.CoordinateSystem

        # Test at specific points where we can evaluate symbolically
        x, y = cs.mesh.X
        test_points = [(1.0, 0.0), (0.0, 1.0), (0.7, 0.7)]

        for px, py in test_points:
            # Evaluate unit vectors at this point
            radial_at_point = cs.unit_radial.subs([(x, px), (y, py)])
            tangential_at_point = cs.unit_tangential.subs([(x, px), (y, py)])

            # Convert to float for numerical testing
            radial_vals = [float(val) for val in radial_at_point]
            tangential_vals = [float(val) for val in tangential_at_point]

            # Test normalization (should have unit length)
            radial_norm = np.sqrt(sum(val**2 for val in radial_vals))
            tangential_norm = np.sqrt(sum(val**2 for val in tangential_vals))

            assert abs(radial_norm - 1.0) < 1e-10, f"Radial vector not normalized at ({px}, {py})"
            assert (
                abs(tangential_norm - 1.0) < 1e-10
            ), f"Tangential vector not normalized at ({px}, {py})"

            # Test orthogonality
            dot_product = sum(r * t for r, t in zip(radial_vals, tangential_vals))
            assert abs(dot_product) < 1e-10, f"Vectors not orthogonal at ({px}, {py})"

    def test_geometric_property_errors(self):
        """Test appropriate error handling for unsupported coordinate systems"""

        # Create a mesh with Cartesian coordinate system
        mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
        cs = mesh.CoordinateSystem

        # Test that coordinate-specific properties raise appropriate errors
        with pytest.raises(NotImplementedError, match="unit_radial not defined.*CARTESIAN"):
            _ = cs.unit_radial

        with pytest.raises(NotImplementedError, match="unit_tangential not defined.*CARTESIAN"):
            _ = cs.unit_tangential

        with pytest.raises(NotImplementedError, match="unit_meridional not defined.*CARTESIAN"):
            _ = cs.unit_meridional

        with pytest.raises(NotImplementedError, match="unit_azimuthal not defined.*CARTESIAN"):
            _ = cs.unit_azimuthal

    def test_1d_coordinate_system_limitations(self):
        """Test that 1D coordinate systems handle horizontal_1 appropriately"""

        # Note: UW3 doesn't typically create 1D meshes, but test the logic
        # We'll simulate this by checking the logic in unit_horizontal_1

        # Create 2D mesh and verify it works
        mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
        cs = mesh.CoordinateSystem

        # This should work fine in 2D
        horizontal_1 = cs.unit_horizontal_1
        assert horizontal_1.equals(sympy.Matrix([[0, 1]]))

    def test_primary_directions_completeness(self):
        """Test that primary_directions includes all appropriate directions"""

        # Test Cartesian 2D
        mesh_2d = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
        directions_2d = mesh_2d.CoordinateSystem.primary_directions

        # Should have basic vectors plus geometric properties
        expected_2d = {
            "unit_e_0",
            "unit_e_1",
            "unit_horizontal",
            "unit_horizontal_0",
            "unit_horizontal_1",
            "unit_vertical",
        }
        assert set(directions_2d.keys()) == expected_2d

        # Test Cylindrical
        mesh_cyl = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5)
        directions_cyl = mesh_cyl.CoordinateSystem.primary_directions

        # Should include cylindrical-specific directions
        expected_cyl = expected_2d | {"unit_radial", "unit_tangential"}
        assert set(directions_cyl.keys()) == expected_cyl

        # Test Spherical
        mesh_sph = uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.5, numElements=2)
        directions_sph = mesh_sph.CoordinateSystem.primary_directions

        # Should include spherical-specific directions and unit_e_2
        expected_sph = {
            "unit_e_0",
            "unit_e_1",
            "unit_e_2",
            "unit_horizontal",
            "unit_horizontal_0",
            "unit_horizontal_1",
            "unit_vertical",
            "unit_radial",
            "unit_meridional",
            "unit_azimuthal",
        }
        assert set(directions_sph.keys()) == expected_sph

    def test_regression_against_existing_examples(self):
        """Test that our properties match usage in existing examples"""

        # Test case from Ex_Stokes_Disk_CylCoords.py
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
        cs = mesh.CoordinateSystem

        # In the example: unit_rvec = meshball.CoordinateSystem.unit_e_0
        # Our unit_radial should be identical
        assert cs.unit_radial.equals(cs.unit_e_0)

        # Test that this matches the expected radial form for cylindrical coordinates
        x, y = cs.mesh.X
        # At point (1, 0), radial should be [1, 0]
        radial_at_1_0 = cs.unit_radial.subs([(x, 1), (y, 0)])
        expected_at_1_0 = sympy.Matrix([[1, 0]])  # Row vector format
        assert radial_at_1_0.equals(expected_at_1_0)

        # At point (0, 1), radial should be [0, 1]
        radial_at_0_1 = cs.unit_radial.subs([(x, 0), (y, 1)])
        expected_at_0_1 = sympy.Matrix([[0, 1]])  # Row vector format
        assert radial_at_0_1.equals(expected_at_0_1)

    def test_cartesian_profile_sampling(self):
        """Test profile sampling for Cartesian coordinate systems"""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
        cs = mesh.CoordinateSystem

        # Test horizontal profile
        horizontal_sample = cs.create_profile_sample(
            "horizontal", y_position=0.3, x_range=(0.1, 0.9), num_points=5
        )

        # Check structure
        assert "cartesian_coords" in horizontal_sample
        assert "natural_coords" in horizontal_sample
        assert "parameters" in horizontal_sample

        # Check shapes
        assert horizontal_sample["cartesian_coords"].shape == (5, 2)
        assert horizontal_sample["natural_coords"].shape == (5, 2)
        assert horizontal_sample["parameters"].shape == (5,)

        # Check that y-coordinate is constant at 0.3
        y_coords = horizontal_sample["cartesian_coords"][:, 1]
        assert np.allclose(y_coords, 0.3)

        # Check x-range
        x_coords = horizontal_sample["cartesian_coords"][:, 0]
        assert np.allclose(x_coords[0], 0.1)
        assert np.allclose(x_coords[-1], 0.9)

        # For Cartesian, natural coords should equal Cartesian coords
        assert np.allclose(
            horizontal_sample["cartesian_coords"], horizontal_sample["natural_coords"]
        )

    def test_cylindrical_profile_sampling(self):
        """Test profile sampling for cylindrical coordinate systems"""
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.3)
        cs = mesh.CoordinateSystem

        # Test radial profile
        radial_sample = cs.create_profile_sample(
            "radial", theta=np.pi / 4, r_range=(0.6, 0.9), num_points=4
        )

        # Check structure
        assert radial_sample["cartesian_coords"].shape == (4, 2)
        assert radial_sample["natural_coords"].shape == (4, 2)

        # Check that theta is constant in natural coordinates
        theta_coords = radial_sample["natural_coords"][:, 1]
        assert np.allclose(theta_coords, np.pi / 4)

        # Check radial range in natural coordinates
        r_coords = radial_sample["natural_coords"][:, 0]
        assert np.allclose(r_coords[0], 0.6, atol=1e-10)
        assert np.allclose(r_coords[-1], 0.9, atol=1e-10)

        # Test tangential profile
        tangential_sample = cs.create_profile_sample(
            "tangential", radius=0.75, theta_range=(0, np.pi / 2), num_points=3
        )

        # Check that radius is constant in natural coordinates
        r_coords = tangential_sample["natural_coords"][:, 0]
        assert np.allclose(r_coords, 0.75)

        # Check theta range
        theta_coords = tangential_sample["natural_coords"][:, 1]
        assert np.allclose(theta_coords[0], 0.0, atol=1e-10)
        assert np.allclose(theta_coords[-1], np.pi / 2, atol=1e-10)

    def test_spherical_profile_sampling(self):
        """Test profile sampling for spherical coordinate systems"""
        mesh = uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.5, numElements=3)
        cs = mesh.CoordinateSystem

        # Test radial profile
        radial_sample = cs.create_profile_sample(
            "radial", theta=np.pi / 2, phi=0.0, r_range=(0.6, 0.9), num_points=4
        )

        # Check structure
        assert radial_sample["cartesian_coords"].shape == (4, 3)
        assert radial_sample["natural_coords"].shape == (4, 3)

        # Check that theta and phi are constant in natural coordinates
        theta_coords = radial_sample["natural_coords"][:, 1]
        phi_coords = radial_sample["natural_coords"][:, 2]
        assert np.allclose(theta_coords, np.pi / 2)
        assert np.allclose(phi_coords, 0.0)

        # Check radial range
        r_coords = radial_sample["natural_coords"][:, 0]
        assert np.allclose(r_coords[0], 0.6, atol=1e-10)
        assert np.allclose(r_coords[-1], 0.9, atol=1e-10)

    def test_coordinate_conversion_accuracy(self):
        """Test accuracy of coordinate conversions"""
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.3)
        cs = mesh.CoordinateSystem

        # Test specific known points
        test_points = np.array(
            [
                [1.0, 0.0],  # Should convert to (1, 0) in natural coords
                [0.0, 1.0],  # Should convert to (1, π/2) in natural coords
                [-1.0, 0.0],  # Should convert to (1, π) in natural coords
            ]
        )

        natural_coords = cs._cartesian_to_natural_coords(test_points)

        # Check radius values
        expected_r = [1.0, 1.0, 1.0]
        actual_r = natural_coords[:, 0]
        assert np.allclose(actual_r, expected_r)

        # Check theta values
        expected_theta = [0.0, np.pi / 2, np.pi]
        actual_theta = natural_coords[:, 1]
        assert np.allclose(actual_theta, expected_theta)

    def test_line_sampling_consistency(self):
        """Test generic line sampling consistency"""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        cs = mesh.CoordinateSystem

        # Sample along horizontal direction
        line_sample = cs.create_line_sample(
            start_point=[0.2, 0.3], direction_vector=cs.unit_horizontal, length=0.6, num_points=4
        )

        # Check that the line goes in the expected direction
        start_point = line_sample["cartesian_coords"][0]
        end_point = line_sample["cartesian_coords"][-1]

        # Should move only in x-direction
        assert np.allclose(start_point, [0.2, 0.3])
        assert np.allclose(end_point, [0.8, 0.3])  # 0.2 + 0.6 = 0.8

        # Check that intermediate points are evenly spaced
        x_coords = line_sample["cartesian_coords"][:, 0]
        expected_x = np.linspace(0.2, 0.8, 4)
        assert np.allclose(x_coords, expected_x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
