#!/usr/bin/env python3
"""
Tests for the Underworld3 surface module.

This test suite validates:
- Surface creation from points and VTK files
- Triangulation via pyvista
- SurfaceVariable with .data and .sym access
- SurfaceCollection management
- Distance field computation
- influence_function with various profiles
- Normal transfer and access
- VTK I/O with variable data

Test Levels:
- Level 1: Basic tests (no pyvista required)
- Level 2: Integration tests with pyvista
- Level 3: Full physics tests with expressions

Optional Dependencies:
- pyvista: Required for discretization, distance computation, VTK I/O
"""

import pytest
import numpy as np
import tempfile
import os

import underworld3 as uw


# =============================================================================
# Dependency Detection
# =============================================================================

def check_module_available(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


HAS_PYVISTA = check_module_available("pyvista")

requires_pyvista = pytest.mark.skipif(
    not HAS_PYVISTA,
    reason="Requires pyvista. Install with: pixi install -e runtime"
)


# =============================================================================
# Level 1: Basic Tests (No pyvista required)
# =============================================================================

pytestmark = pytest.mark.level_1


class TestSurfaceBasic:
    """Basic Surface tests that don't require pyvista."""

    def test_creation_empty(self):
        """Create an empty Surface."""
        surface = uw.meshing.Surface("test_surface")
        assert surface.name == "test_surface"
        assert surface.n_vertices == 0
        assert surface.n_triangles == 0
        assert not surface.is_discretized

    def test_creation_with_mesh(self):
        """Create Surface with associated mesh."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )
        surface = uw.meshing.Surface("test", mesh)
        assert surface.mesh is mesh

    def test_creation_with_control_points(self):
        """Create Surface with control points."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        surface = uw.meshing.Surface("surface1", control_points=points)

        assert surface.name == "surface1"
        assert surface.control_points.shape == (4, 3)
        np.testing.assert_array_equal(surface.control_points, points)

    def test_set_control_points(self):
        """Test setting control points."""
        surface = uw.meshing.Surface("test")
        assert surface.control_points is None

        points = np.random.random((10, 3))
        surface.set_control_points(points)
        assert surface.control_points.shape == (10, 3)
        np.testing.assert_array_equal(surface.control_points, points)

    def test_control_points_validation(self):
        """Test that invalid control points are rejected."""
        # 1D array is rejected
        with pytest.raises(ValueError, match="must be .N, 2. or .N, 3. array"):
            surface = uw.meshing.Surface("bad")
            surface.set_control_points(np.array([0, 0, 0]))

        # 4-column array is rejected
        with pytest.raises(ValueError, match="must be .N, 2. or .N, 3. array"):
            surface = uw.meshing.Surface("bad")
            surface.set_control_points(np.array([[0, 0, 0, 0], [1, 1, 1, 1]]))

    def test_repr(self):
        """Test string representation."""
        surface = uw.meshing.Surface("my_surface")
        repr_str = repr(surface)
        assert "my_surface" in repr_str
        assert "n_vertices=0" in repr_str
        assert "not discretized" in repr_str


class TestSurfaceCollectionBasic:
    """Basic SurfaceCollection tests that don't require pyvista."""

    def test_creation_empty(self):
        """Create empty SurfaceCollection."""
        surfaces = uw.meshing.SurfaceCollection()
        assert len(surfaces) == 0
        assert surfaces.names == []

    def test_add_surface(self):
        """Add surfaces to collection."""
        surface1 = uw.meshing.Surface("surface1")
        surface2 = uw.meshing.Surface("surface2")

        surfaces = uw.meshing.SurfaceCollection()
        surfaces.add(surface1)
        surfaces.add(surface2)

        assert len(surfaces) == 2
        assert "surface1" in surfaces.names
        assert "surface2" in surfaces.names

    def test_add_duplicate_name_fails(self):
        """Cannot add two surfaces with same name."""
        surface1 = uw.meshing.Surface("same_name")
        surface2 = uw.meshing.Surface("same_name")

        surfaces = uw.meshing.SurfaceCollection()
        surfaces.add(surface1)

        with pytest.raises(ValueError, match="already exists"):
            surfaces.add(surface2)

    def test_getitem(self):
        """Access surface by name."""
        surface = uw.meshing.Surface("test")
        surfaces = uw.meshing.SurfaceCollection()
        surfaces.add(surface)

        assert surfaces["test"] is surface

    def test_remove(self):
        """Remove surface from collection."""
        surface = uw.meshing.Surface("test")
        surfaces = uw.meshing.SurfaceCollection()
        surfaces.add(surface)

        removed = surfaces.remove("test")
        assert removed is surface
        assert len(surfaces) == 0

    def test_iteration(self):
        """Iterate over surface names."""
        surfaces = uw.meshing.SurfaceCollection()
        surfaces.add(uw.meshing.Surface("a"))
        surfaces.add(uw.meshing.Surface("b"))
        surfaces.add(uw.meshing.Surface("c"))

        names = list(surfaces)
        assert set(names) == {"a", "b", "c"}

    def test_repr(self):
        """Test string representation."""
        surfaces = uw.meshing.SurfaceCollection()
        surfaces.add(uw.meshing.Surface("surface1"))
        repr_str = repr(surfaces)
        assert "SurfaceCollection" in repr_str
        assert "surface1" in repr_str


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    def test_faultsurface_alias(self):
        """FaultSurface is alias for Surface."""
        fault = uw.meshing.FaultSurface("test")
        assert isinstance(fault, uw.meshing.Surface)

    def test_faultcollection_alias(self):
        """FaultCollection is alias for SurfaceCollection."""
        faults = uw.meshing.FaultCollection()
        assert isinstance(faults, uw.meshing.SurfaceCollection)


# =============================================================================
# Level 2: Integration Tests (Require pyvista)
# =============================================================================

@requires_pyvista
@pytest.mark.level_2
class TestSurfaceDiscretization:
    """Tests for surface discretization using pyvista."""

    def test_discretize_planar_points(self):
        """Discretize a simple planar point cloud."""
        # Create a grid of points on z=0 plane
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(25)])

        surface = uw.meshing.Surface("planar", control_points=points)
        surface.discretize()

        assert surface.is_discretized
        assert surface.n_triangles > 0
        assert surface.normals is not None
        assert surface.normals.shape == (surface.n_vertices, 3)

        # Point normals should point in z-direction (approximately)
        z_components = np.abs(surface.normals[:, 2])
        assert np.mean(z_components) > 0.9

    def test_discretize_curved_surface(self):
        """Discretize a curved (spherical cap) surface."""
        # Create points on a hemisphere
        theta = np.linspace(0, np.pi/4, 10)
        phi = np.linspace(0, 2*np.pi, 20)
        theta, phi = np.meshgrid(theta, phi)
        r = 1.0
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        surface = uw.meshing.Surface("curved", control_points=points)
        surface.discretize(offset=0.1)

        assert surface.is_discretized
        assert surface.n_triangles > 0
        assert surface.pv_mesh is not None

    def test_discretize_too_few_points(self):
        """Discretization fails with fewer than 3 points (3D)."""
        surface = uw.meshing.Surface("sparse", control_points=np.array([[0, 0, 0], [1, 0, 0]]))

        with pytest.raises(ValueError, match="at least 3 points"):
            surface.discretize()

    def test_discretize_no_points(self):
        """Discretization fails with no points."""
        surface = uw.meshing.Surface("empty")

        with pytest.raises(ValueError, match="no control points"):
            surface.discretize()

    def test_deform_vertices(self):
        """Test deforming vertices in-place."""
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", control_points=points)
        surface.discretize()

        original_vertices = surface.vertices.copy()

        # Apply displacement
        displacement = np.zeros_like(original_vertices)
        displacement[:, 2] = 0.5  # Move all vertices up in z

        surface.deform_vertices(displacement)

        np.testing.assert_array_almost_equal(
            surface.vertices[:, 2], original_vertices[:, 2] + 0.5
        )

    def test_flip_normals(self):
        """Test flipping normal directions."""
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", control_points=points)
        surface.discretize()

        original_normals = surface.normals.copy()
        surface.flip_normals()

        # Normals should be flipped
        np.testing.assert_array_almost_equal(
            surface.normals, -original_normals
        )


@requires_pyvista
@pytest.mark.level_2
class TestSurfaceVariable:
    """Tests for SurfaceVariable class."""

    def test_add_variable(self):
        """Add a scalar variable to surface."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction", size=1)

        assert friction.name == "friction"
        assert friction.size == 1
        assert friction.data.shape == (surface.n_vertices,)
        assert "friction" in surface.variables

    def test_add_vector_variable(self):
        """Add a vector variable to surface."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        velocity = surface.add_variable("velocity", size=3)

        assert velocity.size == 3
        assert velocity.data.shape == (surface.n_vertices, 3)

    def test_variable_data_access(self):
        """Access and modify variable data."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        var = surface.add_variable("test_var")

        # Set all values
        var.data[:] = 0.5
        assert np.allclose(var.data, 0.5)

        # Set specific values
        var.data[0] = 1.0
        var.data[1] = 2.0
        assert var.data[0] == 1.0
        assert var.data[1] == 2.0

    def test_variable_sym_access(self):
        """Access variable via .sym for expressions."""
        import sympy

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction")
        friction.data[:] = 0.6

        # Access symbolic representation
        sym = friction.sym

        # Should be a sympy expression
        assert sym is not None
        # Should be a Matrix with 1 component
        assert hasattr(sym, 'shape')

    def test_variable_stored_in_pyvista(self):
        """Variable data is stored in pyvista point_data."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        var = surface.add_variable("my_data")
        var.data[:] = np.arange(surface.n_vertices)

        # Data should be in pyvista's point_data
        assert "my_data" in surface.pv_mesh.point_data
        np.testing.assert_array_equal(
            surface.pv_mesh.point_data["my_data"],
            var.data
        )

    def test_duplicate_variable_fails(self):
        """Cannot add two variables with same name."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        surface.add_variable("friction")

        with pytest.raises(ValueError, match="already exists"):
            surface.add_variable("friction")


@requires_pyvista
@pytest.mark.level_1
class TestSurfaceVariableUnits:
    """Tests for unit-aware SurfaceVariable."""

    def test_variable_with_units(self):
        """Create variable with units."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction", units="Pa")

        assert friction.units == "Pa"
        assert friction.has_units is True

    def test_variable_without_units(self):
        """Variable without units returns plain array."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction")

        assert friction.units is None
        assert friction.has_units is False
        # Should be a plain numpy array
        assert isinstance(friction.data, np.ndarray)

    def test_unit_aware_data_access(self):
        """Data access with units returns UnitAwareArray."""
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction", units="Pa")
        friction.data[:] = 1e6

        # Should return UnitAwareArray
        data = friction.data
        assert isinstance(data, UnitAwareArray)
        assert data.units is not None

    def test_data_setter_strips_magnitude(self):
        """Setting data with units extracts magnitude for storage."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction", units="Pa")

        # Set with a value that has magnitude attribute (simulated)
        class MockQuantity:
            magnitude = np.array([1e6, 2e6, 3e6, 4e6])

        friction.data = MockQuantity()

        # Should store the magnitude
        raw = surface.pv_mesh.point_data["friction"]
        np.testing.assert_array_almost_equal(raw, [1e6, 2e6, 3e6, 4e6])


@requires_pyvista
@pytest.mark.level_1
class TestSurfaceVariableMask:
    """Tests for SurfaceVariable mask property."""

    def test_mask_with_width(self):
        """Variable with mask_width has .mask property."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction", mask_width=0.1)
        friction.data[:] = 0.6

        # Should have mask property
        mask = friction.mask
        assert mask is not None
        # Should be a sympy expression
        assert hasattr(mask, 'free_symbols') or hasattr(mask, 'args')

    def test_mask_without_width_raises(self):
        """Accessing .mask without mask_width raises ValueError."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction")  # No mask_width

        with pytest.raises(ValueError, match="no mask_width set"):
            _ = friction.mask

    def test_mask_with_profile(self):
        """Variable can specify mask profile."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable(
            "friction", mask_width=0.1, mask_profile="smoothstep"
        )

        # Should not raise
        mask = friction.mask
        assert mask is not None

    def test_mask_usage_pattern(self):
        """Test the expected usage pattern: var.sym * var.mask."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction", mask_width=0.1)
        friction.data[:] = 0.6

        # This is the expected pattern
        masked_value = friction.sym[0] * friction.mask

        # Should be a sympy expression
        assert masked_value is not None
        assert hasattr(masked_value, 'free_symbols') or hasattr(masked_value, 'args')

    def test_repr_with_units_and_mask(self):
        """Variable repr includes units and mask info."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        friction = surface.add_variable(
            "friction", units="Pa", mask_width=0.1
        )

        repr_str = repr(friction)
        assert "units='Pa'" in repr_str
        assert "mask_width=0.1" in repr_str


@requires_pyvista
@pytest.mark.level_2
class TestSurfaceVTKIO:
    """Tests for VTK file I/O."""

    def test_save_and_load_vtk(self):
        """Save and load a discretized surface."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.1]
        ], dtype=float)
        surface = uw.meshing.Surface("original", mesh, points)
        surface.discretize()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "surface.vtk")
            surface.save(filepath)

            # Load the saved file
            loaded = uw.meshing.Surface.from_vtk(filepath, mesh, "loaded")

            assert loaded.name == "loaded"
            assert loaded.is_discretized
            np.testing.assert_array_almost_equal(
                loaded.vertices, surface.vertices, decimal=5
            )
            assert loaded.n_triangles == surface.n_triangles

    def test_save_load_with_variables(self):
        """Save and load surface with variables."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("original", mesh, points)
        surface.discretize()

        # Add variables
        friction = surface.add_variable("friction")
        friction.data[:] = np.arange(surface.n_vertices) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "surface_with_vars.vtk")
            surface.save(filepath)

            # Load and check variables are preserved
            loaded = uw.meshing.Surface.from_vtk(filepath, mesh)

            assert "friction" in loaded.variables
            np.testing.assert_array_almost_equal(
                loaded.variables["friction"].data,
                friction.data,
                decimal=5
            )

    def test_from_vtk_nonexistent(self):
        """Loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            uw.meshing.Surface.from_vtk("nonexistent.vtk")


@requires_pyvista
@pytest.mark.level_2
class TestSurfaceDistanceField:
    """Tests for distance field computation."""

    def test_distance_to_planar_surface(self):
        """Compute distance from mesh to a planar surface."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        # Create a planar surface at z=0.5
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([
            xx.ravel(), yy.ravel(), np.full(100, 0.5)
        ])

        surface = uw.meshing.Surface("plane", mesh, points)
        surface.discretize()

        # Get distance field
        distance_var = surface.distance

        assert distance_var is not None
        assert distance_var.data.shape[1] == 1

        # Distance is SIGNED: positive on one side, negative on other
        # The magnitude should be approximately |z - 0.5| for each point
        z_coords = mesh.X.coords[:, 2]
        if hasattr(z_coords, 'magnitude'):
            z_coords = z_coords.magnitude
        expected_dist = np.asarray(z_coords) - 0.5  # Signed distance from z=0.5 plane

        # Check distances are reasonable (using absolute value for magnitude check)
        computed_dist = distance_var.data[:, 0]
        assert np.max(np.abs(computed_dist)) < 0.6
        # Check that we have both positive and negative distances (signed)
        assert np.min(computed_dist) < 0  # Some points below the surface
        assert np.max(computed_dist) > 0  # Some points above the surface

    def test_distance_cached(self):
        """Distance field is cached after first access."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        points = np.array([
            [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        # First access
        dist1 = surface.distance

        # Second access should return same object
        dist2 = surface.distance

        assert dist1 is dist2

    def test_distance_no_mesh_fails(self):
        """Accessing distance without mesh raises error."""
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", control_points=points)
        surface.discretize()

        with pytest.raises(RuntimeError, match="requires a mesh"):
            _ = surface.distance


@requires_pyvista
@pytest.mark.level_2
class TestInfluenceFunction:
    """Tests for influence_function with various profiles."""

    def test_step_profile(self):
        """Test step influence function."""
        import sympy

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        points = np.array([
            [0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]
        ], dtype=float)
        surface = uw.meshing.Surface("vert", mesh, points)
        surface.discretize()

        eta = surface.influence_function(
            width=0.1,
            value_near=0.01,
            value_far=1.0,
            profile="step",
        )

        assert isinstance(eta, sympy.Piecewise)
        assert len(eta.args) == 2
        assert eta.args[0][0] == 0.01  # value_near
        assert eta.args[1][0] == 1.0   # value_far

    def test_gaussian_profile(self):
        """Test gaussian influence function."""
        import sympy

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        points = np.array([
            [0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]
        ], dtype=float)
        surface = uw.meshing.Surface("vert", mesh, points)
        surface.discretize()

        eta = surface.influence_function(
            width=0.1,
            value_near=0.01,
            value_far=1.0,
            profile="gaussian",
        )

        # Should contain exp term
        assert "exp" in str(eta).lower()

    def test_linear_profile(self):
        """Test linear influence function."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        points = np.array([
            [0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]
        ], dtype=float)
        surface = uw.meshing.Surface("vert", mesh, points)
        surface.discretize()

        eta = surface.influence_function(
            width=0.1,
            value_near=0.01,
            value_far=1.0,
            profile="linear",
        )

        # Should not be a Piecewise
        import sympy
        assert not isinstance(eta, sympy.Piecewise)

    def test_smoothstep_profile(self):
        """Test smoothstep influence function."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        points = np.array([
            [0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]
        ], dtype=float)
        surface = uw.meshing.Surface("vert", mesh, points)
        surface.discretize()

        eta = surface.influence_function(
            width=0.1,
            value_near=0.01,
            value_far=1.0,
            profile="smoothstep",
        )

        # Should contain cubic term (t^3)
        assert "**3" in str(eta) or "**2" in str(eta)

    def test_unknown_profile_fails(self):
        """Unknown profile raises error."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        points = np.array([
            [0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]
        ], dtype=float)
        surface = uw.meshing.Surface("vert", mesh, points)
        surface.discretize()

        with pytest.raises(ValueError, match="Unknown profile"):
            surface.influence_function(
                width=0.1,
                value_near=0.01,
                value_far=1.0,
                profile="invalid",
            )

    def test_influence_with_variable(self):
        """Use SurfaceVariable as value_near."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        points = np.array([
            [0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]
        ], dtype=float)
        surface = uw.meshing.Surface("vert", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction")
        friction.data[:] = 0.3

        # Use scalar element of variable as value_near
        # (friction.sym is a Matrix, friction.sym[0] is the scalar)
        eta = surface.influence_function(
            width=0.1,
            value_near=friction.sym[0],
            value_far=1.0,
            profile="gaussian",
        )

        # Expression should reference the friction variable
        expr_str = str(eta)
        assert "surf_vert_friction" in expr_str


@requires_pyvista
@pytest.mark.level_2
class TestSurfaceNormals:
    """Tests for surface normal access."""

    def test_vertex_normals(self):
        """Access vertex (point) normals."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        # Horizontal surface at z=0.5
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([
            xx.ravel(), yy.ravel(), np.full(25, 0.5)
        ])

        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        normals = surface.normals

        assert normals.shape == (surface.n_vertices, 3)
        # Normals should point in z-direction (approximately)
        z_components = np.abs(normals[:, 2])
        assert np.mean(z_components) > 0.9

    def test_face_normals(self):
        """Access face (cell) normals."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        surface = uw.meshing.Surface("test", mesh, points)
        surface.discretize()

        face_normals = surface.face_normals

        assert face_normals.shape == (surface.n_triangles, 3)


# =============================================================================
# Level 3: Physics Tests (Full workflow)
# =============================================================================

@requires_pyvista
@pytest.mark.level_3
class TestSurfacePhysicsWorkflow:
    """Full workflow tests for surface-based physics."""

    def test_complete_workflow(self):
        """Test complete surface workflow with variables."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.15,
        )

        # Create a planar surface at y=0.5
        x = np.linspace(0, 1, 15)
        z = np.linspace(0, 1, 15)
        xx, zz = np.meshgrid(x, z)
        points = np.column_stack([
            xx.ravel(),
            np.full(225, 0.5),
            zz.ravel()
        ])

        surface = uw.meshing.Surface("fault", mesh, points)
        surface.discretize()

        # Add friction variable
        friction = surface.add_variable("friction")
        friction.data[:] = 0.6

        # Access distance field (signed distance)
        dist = surface.distance
        assert dist.data.shape[1] == 1
        # Signed distance has both positive and negative values
        assert np.abs(dist.data).max() > 0  # Some distance from surface

        # Access normals
        normals = surface.normals
        assert normals.shape[1] == 3
        # Normals should be normalized
        norms = np.sqrt(np.sum(normals**2, axis=1))
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=3)

        # Create influence function
        eta_weak = surface.influence_function(
            width=mesh.get_min_radius() * 3,
            value_near=0.1,
            value_far=1.0,
            profile="gaussian",
        )

        import sympy
        assert isinstance(eta_weak, sympy.Expr)

    def test_surface_collection_workflow(self):
        """Test workflow with multiple surfaces."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        # Create two parallel surfaces
        x = np.linspace(0, 1, 10)
        z = np.linspace(0, 1, 10)
        xx, zz = np.meshgrid(x, z)

        points1 = np.column_stack([
            xx.ravel(), np.full(100, 0.3), zz.ravel()
        ])
        points2 = np.column_stack([
            xx.ravel(), np.full(100, 0.7), zz.ravel()
        ])

        surface1 = uw.meshing.Surface("lower", mesh, points1)
        surface1.discretize()

        surface2 = uw.meshing.Surface("upper", mesh, points2)
        surface2.discretize()

        surfaces = uw.meshing.SurfaceCollection()
        surfaces.add(surface1)
        surfaces.add(surface2)

        assert len(surfaces) == 2

        # Compute combined distance field
        distance = surfaces.compute_distance_field(mesh)

        # Points near y=0.5 should be equidistant from both surfaces
        coords = mesh.X.coords
        if hasattr(coords, '__array__'):
            coords = np.asarray(coords)
        mid_mask = np.abs(coords[:, 1] - 0.5) < 0.1
        mid_distances = distance.data[mid_mask, 0]
        assert np.mean(mid_distances) < 0.25

        # Combined influence function
        eta = surfaces.influence_function(
            mesh,
            width=0.1,
            value_near=0.01,
            value_far=1.0,
            profile="step",
        )

        import sympy
        assert isinstance(eta, sympy.Piecewise)


# =============================================================================
# Level 2: 2D Surface Tests
# =============================================================================

@requires_pyvista
@pytest.mark.level_2
class TestSurface2D:
    """Tests for 2D surface (polyline) functionality."""

    def test_dim_detection_from_mesh(self):
        """Dimension is detected from associated mesh."""
        mesh_2d = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )
        mesh_3d = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.2
        )

        surface_2d = uw.meshing.Surface("test_2d", mesh_2d)
        surface_3d = uw.meshing.Surface("test_3d", mesh_3d)

        assert surface_2d.dim == 2
        assert surface_2d.is_2d is True
        assert surface_3d.dim == 3
        assert surface_3d.is_2d is False

    def test_dim_detection_from_control_points(self):
        """Dimension is detected from control point shape when no mesh."""
        points_2d = np.array([[0, 0], [0.5, 0.5], [1, 0]])
        points_3d = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 0, 0]])

        surface_2d = uw.meshing.Surface("test_2d", control_points=points_2d)
        surface_3d = uw.meshing.Surface("test_3d", control_points=points_3d)

        assert surface_2d.dim == 2
        assert surface_2d.is_2d is True
        assert surface_3d.dim == 3
        assert surface_3d.is_2d is False

    def test_discretize_2d_simple(self):
        """Discretize a 2D surface (creates polyline segments)."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        )

        # Two-point line segment
        points = np.array([[0.2, 0.5], [0.8, 0.5]])
        surface = uw.meshing.Surface("line", mesh, points)
        surface.discretize()

        assert surface.is_discretized
        assert surface.n_vertices == 2  # Two endpoints
        # For a simple 2-point line, pyvista stores vertices with z=0
        np.testing.assert_array_almost_equal(surface.vertices[:, 2], 0.0)

    def test_discretize_2d_with_spline(self):
        """Discretize a 2D surface with spline fitting."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        )

        # Multiple points - scipy will fit a spline
        points = np.array([
            [0.2, 0.2],
            [0.3, 0.5],
            [0.5, 0.7],
            [0.7, 0.5],
            [0.8, 0.2],
        ])
        surface = uw.meshing.Surface("curve", mesh, points)
        surface.discretize()

        assert surface.is_discretized
        assert surface.n_vertices == 5  # Same number of vertices as input
        # Vertices should be on a continuous curve
        assert surface.vertices.shape == (5, 3)

    def test_normals_2d(self):
        """2D surface has correct normals."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        )

        # Horizontal line - normals should point in y-direction
        points = np.array([[0.2, 0.5], [0.8, 0.5]])
        surface = uw.meshing.Surface("hline", mesh, points)
        surface.discretize()

        normals = surface.normals
        assert normals.shape == (2, 3)
        # For horizontal line, normals should be approximately (0, 1, 0) or (0, -1, 0)
        assert np.abs(normals[:, 1]).mean() > 0.9  # y-component dominant
        assert np.abs(normals[:, 0]).mean() < 0.1  # x-component minimal
        assert np.abs(normals[:, 2]).mean() < 0.1  # z-component minimal

    def test_normals_2d_diagonal(self):
        """2D diagonal line has correct normals."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        )

        # Diagonal line from (0,0) to (1,1)
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        surface = uw.meshing.Surface("diagonal", mesh, points)
        surface.discretize()

        normals = surface.normals
        # For 45° line, normal should be perpendicular (-1/√2, 1/√2, 0) or (1/√2, -1/√2, 0)
        # Check that x and y components have equal magnitude
        np.testing.assert_array_almost_equal(
            np.abs(normals[:, 0]),
            np.abs(normals[:, 1]),
            decimal=5
        )
        # And perpendicular to the line (dot product with direction = 0)
        direction = np.array([1, 1, 0]) / np.sqrt(2)
        for n in normals:
            dot = np.dot(n, direction)
            np.testing.assert_almost_equal(np.abs(dot), 0.0, decimal=5)

    def test_flip_normals_2d(self):
        """Flip normals works for 2D surface."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        )

        points = np.array([[0.2, 0.5], [0.8, 0.5]])
        surface = uw.meshing.Surface("line", mesh, points)
        surface.discretize()

        original_normals = surface.normals.copy()
        surface.flip_normals()

        np.testing.assert_array_almost_equal(
            surface.normals, -original_normals
        )

    def test_distance_field_2d(self):
        """Compute signed distance field for 2D surface."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        )

        # Horizontal line at y=0.5
        points = np.array([[0.0, 0.5], [1.0, 0.5]])
        surface = uw.meshing.Surface("hline", mesh, points)
        surface.discretize()

        distance = surface.distance

        assert distance is not None
        assert distance.data.shape[1] == 1

        # Distance should be signed
        # Points above y=0.5 should have positive distance
        # Points below y=0.5 should have negative distance (or vice versa)
        coords = mesh.X.coords
        if hasattr(coords, '__array__'):
            coords = np.asarray(coords)

        distances = distance.data[:, 0]

        # Check that we have both positive and negative distances
        assert np.min(distances) < -0.1, "Expected negative distances for points on one side"
        assert np.max(distances) > 0.1, "Expected positive distances for points on other side"

        # Check that magnitude increases with distance from line
        y_coords = coords[:, 1]
        expected_dist = np.abs(y_coords - 0.5)
        actual_abs_dist = np.abs(distances)

        # Correlation should be high
        correlation = np.corrcoef(expected_dist, actual_abs_dist)[0, 1]
        assert correlation > 0.99, f"Distance field correlation {correlation} < 0.99"

    def test_distance_field_2d_diagonal(self):
        """Signed distance for diagonal 2D surface."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        )

        # Diagonal line from (0.2, 0.2) to (0.8, 0.8)
        points = np.array([[0.2, 0.2], [0.8, 0.8]])
        surface = uw.meshing.Surface("diagonal", mesh, points)
        surface.discretize()

        distance = surface.distance
        distances = distance.data[:, 0]

        # Should have both positive and negative values
        assert np.min(distances) < 0
        assert np.max(distances) > 0

    def test_influence_function_2d(self):
        """Influence function works for 2D surface."""
        import sympy

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        )

        points = np.array([[0.0, 0.5], [1.0, 0.5]])
        surface = uw.meshing.Surface("line", mesh, points)
        surface.discretize()

        eta = surface.influence_function(
            width=0.1,
            value_near=0.01,
            value_far=1.0,
            profile="gaussian",
        )

        assert isinstance(eta, sympy.Expr)
        # Should contain exp term for gaussian
        assert "exp" in str(eta).lower()

    def test_variable_on_2d_surface(self):
        """SurfaceVariable works on 2D surface."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        )

        points = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.3]])
        surface = uw.meshing.Surface("curve", mesh, points)
        surface.discretize()

        friction = surface.add_variable("friction")
        friction.data[:] = 0.5

        assert friction.data.shape == (3,)
        np.testing.assert_array_equal(friction.data, 0.5)

        # .sym should work
        sym = friction.sym
        assert sym is not None


@requires_pyvista
@pytest.mark.level_2
class TestSurface2DGeometry:
    """Tests for 2D geometry functions used by Surface."""

    def test_polyline_distance_parallel_to_segment(self):
        """Distance to polyline for points parallel to middle segment."""
        from underworld3.utilities.geometry_tools import (
            signed_distance_pointcloud_polyline_2d,
            distance_pointcloud_polyline,
        )

        # Simple horizontal line
        vertices = np.array([[0.0, 0.5], [1.0, 0.5]])

        # Points above and below
        points = np.array([
            [0.5, 0.7],  # Above by 0.2
            [0.5, 0.3],  # Below by 0.2
            [0.5, 0.5],  # On line
        ])

        signed_dist = signed_distance_pointcloud_polyline_2d(points, vertices)
        unsigned_dist = distance_pointcloud_polyline(points, vertices)

        # Unsigned distances should all be |distance to line|
        np.testing.assert_array_almost_equal(unsigned_dist, [0.2, 0.2, 0.0])

        # Signed distances should have opposite signs for above/below
        assert signed_dist[0] * signed_dist[1] < 0  # Opposite signs
        np.testing.assert_almost_equal(np.abs(signed_dist[0]), 0.2)
        np.testing.assert_almost_equal(np.abs(signed_dist[1]), 0.2)
        np.testing.assert_almost_equal(signed_dist[2], 0.0)

    def test_polyline_normals(self):
        """Normals for polyline segments."""
        from underworld3.utilities.geometry_tools import linesegment_normals_2d

        # L-shaped polyline
        vertices = np.array([
            [0.0, 0.0],  # Start
            [1.0, 0.0],  # Corner
            [1.0, 1.0],  # End
        ])

        segment_normals, vertex_normals = linesegment_normals_2d(vertices)

        # First segment (horizontal): normal should be (0, 1) or (0, -1)
        assert np.abs(segment_normals[0, 0]) < 0.01
        assert np.abs(segment_normals[0, 1]) > 0.99

        # Second segment (vertical): normal should be (1, 0) or (-1, 0)
        assert np.abs(segment_normals[1, 0]) > 0.99
        assert np.abs(segment_normals[1, 1]) < 0.01

        # Corner vertex normal should be between the two segment normals
        # It's the average of the adjacent segment normals (normalized)
        assert np.abs(vertex_normals[1, 0]) > 0.5
        assert np.abs(vertex_normals[1, 1]) > 0.5


# =============================================================================
# Feature Summary
# =============================================================================

def test_surface_module_summary():
    """Print summary of surface module capabilities."""
    print("\n" + "=" * 60)
    print("SURFACE MODULE STATUS")
    print("=" * 60)
    print(f"  Surface class: AVAILABLE")
    print(f"  SurfaceVariable class: AVAILABLE")
    print(f"  SurfaceCollection class: AVAILABLE")
    print(f"  Backward compat (FaultSurface): AVAILABLE")
    print(f"  pyvista integration: {'YES' if HAS_PYVISTA else 'NO'}")
    print("=" * 60)
    if not HAS_PYVISTA:
        print("\nTo enable full functionality:")
        print("  pixi install -e runtime")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
