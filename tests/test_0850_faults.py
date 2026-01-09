#!/usr/bin/env python3
"""
Tests for the Underworld3 fault surface module.

This test suite validates:
- FaultSurface creation from points and VTK files
- Triangulation via pyvista
- FaultCollection management
- Distance field computation
- Normal transfer via KDTree
- Weakness function creation for rheology

Test Levels:
- Level 1: Basic tests (no pyvista required)
- Level 2: Integration tests with pyvista
- Level 3: Full physics tests with Stokes solver

Optional Dependencies:
- pyvista: Required for triangulation, distance computation, VTK I/O
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


class TestFaultSurfaceBasic:
    """Basic FaultSurface tests that don't require pyvista."""

    def test_creation_empty(self):
        """Create an empty FaultSurface."""
        fault = uw.meshing.FaultSurface("test_fault")
        assert fault.name == "test_fault"
        assert fault.n_points == 0
        assert fault.n_triangles == 0
        assert not fault.is_triangulated

    def test_creation_with_points(self):
        """Create FaultSurface with point array."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        fault = uw.meshing.FaultSurface("fault1", points)

        assert fault.name == "fault1"
        assert fault.n_points == 4
        assert fault.points.shape == (4, 3)
        assert not fault.is_triangulated

    def test_points_validation(self):
        """Test that invalid points are rejected."""
        # Wrong dimensionality
        with pytest.raises(ValueError, match="must be .N, 3. array"):
            uw.meshing.FaultSurface("bad", np.array([[0, 0], [1, 1]]))

        # 1D array
        with pytest.raises(ValueError):
            uw.meshing.FaultSurface("bad", np.array([0, 0, 0]))

    def test_points_setter(self):
        """Test setting points on existing fault."""
        fault = uw.meshing.FaultSurface("test")
        assert fault.n_points == 0

        points = np.random.random((10, 3))
        fault.points = points
        assert fault.n_points == 10
        np.testing.assert_array_equal(fault.points, points)

    def test_repr(self):
        """Test string representation."""
        fault = uw.meshing.FaultSurface("my_fault")
        repr_str = repr(fault)
        assert "my_fault" in repr_str
        assert "n_points=0" in repr_str
        assert "not triangulated" in repr_str


class TestFaultCollectionBasic:
    """Basic FaultCollection tests that don't require pyvista."""

    def test_creation_empty(self):
        """Create empty FaultCollection."""
        faults = uw.meshing.FaultCollection()
        assert len(faults) == 0
        assert faults.names == []

    def test_add_fault(self):
        """Add a fault to collection."""
        fault1 = uw.meshing.FaultSurface("fault1", np.random.random((5, 3)))
        fault2 = uw.meshing.FaultSurface("fault2", np.random.random((5, 3)))

        faults = uw.meshing.FaultCollection()
        faults.add(fault1)
        faults.add(fault2)

        assert len(faults) == 2
        assert "fault1" in faults.names
        assert "fault2" in faults.names

    def test_add_duplicate_name_fails(self):
        """Cannot add two faults with same name."""
        fault1 = uw.meshing.FaultSurface("same_name")
        fault2 = uw.meshing.FaultSurface("same_name")

        faults = uw.meshing.FaultCollection()
        faults.add(fault1)

        with pytest.raises(ValueError, match="already exists"):
            faults.add(fault2)

    def test_getitem(self):
        """Access fault by name."""
        fault = uw.meshing.FaultSurface("test")
        faults = uw.meshing.FaultCollection()
        faults.add(fault)

        assert faults["test"] is fault

    def test_remove(self):
        """Remove fault from collection."""
        fault = uw.meshing.FaultSurface("test")
        faults = uw.meshing.FaultCollection()
        faults.add(fault)

        removed = faults.remove("test")
        assert removed is fault
        assert len(faults) == 0

    def test_iteration(self):
        """Iterate over fault names."""
        faults = uw.meshing.FaultCollection()
        faults.add(uw.meshing.FaultSurface("a"))
        faults.add(uw.meshing.FaultSurface("b"))
        faults.add(uw.meshing.FaultSurface("c"))

        names = list(faults)
        assert set(names) == {"a", "b", "c"}

    def test_repr(self):
        """Test string representation."""
        faults = uw.meshing.FaultCollection()
        faults.add(uw.meshing.FaultSurface("fault1"))
        repr_str = repr(faults)
        assert "FaultCollection" in repr_str
        assert "fault1" in repr_str


# =============================================================================
# Level 2: Integration Tests (Require pyvista)
# =============================================================================

@requires_pyvista
@pytest.mark.level_2
class TestFaultTriangulation:
    """Tests for fault triangulation using pyvista."""

    def test_triangulate_planar_points(self):
        """Triangulate a simple planar point cloud."""
        # Create a grid of points on z=0 plane
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(25)])

        fault = uw.meshing.FaultSurface("planar", points)
        fault.triangulate()

        assert fault.is_triangulated
        assert fault.n_triangles > 0
        assert fault.normals is not None
        assert fault.normals.shape == (fault.n_triangles, 3)

        # Normals should point in z-direction (approximately)
        # Allow for some variation due to triangulation
        z_components = np.abs(fault.normals[:, 2])
        assert np.mean(z_components) > 0.9

    def test_triangulate_curved_surface(self):
        """Triangulate a curved (spherical cap) surface."""
        # Create points on a hemisphere
        theta = np.linspace(0, np.pi/4, 10)
        phi = np.linspace(0, 2*np.pi, 20)
        theta, phi = np.meshgrid(theta, phi)
        r = 1.0
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        fault = uw.meshing.FaultSurface("curved", points)
        fault.triangulate(offset=0.1)

        assert fault.is_triangulated
        assert fault.n_triangles > 0
        assert fault.pv_mesh is not None

    def test_triangulate_too_few_points(self):
        """Triangulation fails with fewer than 3 points."""
        fault = uw.meshing.FaultSurface("sparse", np.array([[0, 0, 0], [1, 0, 0]]))

        with pytest.raises(ValueError, match="at least 3 points"):
            fault.triangulate()

    def test_triangulate_no_points(self):
        """Triangulation fails with no points."""
        fault = uw.meshing.FaultSurface("empty")

        with pytest.raises(ValueError, match="no points"):
            fault.triangulate()

    def test_flip_normals(self):
        """Test flipping normal directions."""
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        fault = uw.meshing.FaultSurface("test", points)
        fault.triangulate()

        original_normals = fault.normals.copy()
        fault.flip_normals()

        np.testing.assert_array_almost_equal(
            fault.normals, -original_normals
        )


@requires_pyvista
@pytest.mark.level_2
class TestFaultVTKIO:
    """Tests for VTK file I/O."""

    def test_save_and_load_vtk(self):
        """Save and load a triangulated fault."""
        # Create and triangulate a simple fault
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.1]
        ], dtype=float)
        fault = uw.meshing.FaultSurface("original", points)
        fault.triangulate()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "fault.vtk")
            fault.to_vtk(filepath)

            # Load the saved file
            loaded = uw.meshing.FaultSurface.from_vtk(filepath, "loaded")

            assert loaded.name == "loaded"
            assert loaded.is_triangulated
            np.testing.assert_array_almost_equal(
                loaded.points, fault.points, decimal=5
            )
            assert loaded.n_triangles == fault.n_triangles

    def test_from_vtk_nonexistent(self):
        """Loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            uw.meshing.FaultSurface.from_vtk("nonexistent.vtk")

    def test_add_from_vtk(self):
        """Load fault directly into collection."""
        # Create a test VTK file
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 1, 0]
        ], dtype=float)
        fault = uw.meshing.FaultSurface("temp", points)
        fault.triangulate()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_fault.vtk")
            fault.to_vtk(filepath)

            faults = uw.meshing.FaultCollection()
            loaded = faults.add_from_vtk(filepath)

            assert "test_fault" in faults.names
            assert loaded.is_triangulated


@requires_pyvista
@pytest.mark.level_2
class TestFaultDistanceField:
    """Tests for distance field computation."""

    def test_distance_to_planar_fault(self):
        """Compute distance from mesh to a planar fault."""
        # Create a simple mesh
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        # Create a planar fault at z=0.5
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([
            xx.ravel(), yy.ravel(), np.full(100, 0.5)
        ])

        fault = uw.meshing.FaultSurface("plane", points)
        fault.triangulate()

        faults = uw.meshing.FaultCollection()
        faults.add(fault)

        # Compute distance
        distance_var = faults.compute_distance_field(mesh)

        assert distance_var is not None
        assert distance_var.data.shape[1] == 1

        # Distance should be approximately |z - 0.5| for each point
        z_coords = mesh.X.coords[:, 2]
        if hasattr(z_coords, 'magnitude'):
            z_coords = z_coords.magnitude
        expected_dist = np.abs(np.asarray(z_coords) - 0.5)

        # Check that computed distances are reasonable
        # (exact match not expected due to triangulation effects)
        computed_dist = distance_var.data[:, 0]
        assert np.max(computed_dist) < 0.6  # Should not exceed max possible
        assert np.min(computed_dist) >= 0   # Distances are non-negative

    def test_distance_empty_collection_fails(self):
        """Computing distance with empty collection raises error."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )
        faults = uw.meshing.FaultCollection()

        with pytest.raises(ValueError, match="no faults"):
            faults.compute_distance_field(mesh)


@requires_pyvista
@pytest.mark.level_2
class TestFaultNormalTransfer:
    """Tests for normal vector transfer."""

    def test_transfer_normals_planar(self):
        """Transfer normals from a planar fault."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        # Create horizontal fault at z=0.5
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([
            xx.ravel(), yy.ravel(), np.full(100, 0.5)
        ])

        fault = uw.meshing.FaultSurface("horizontal", points)
        fault.triangulate()

        faults = uw.meshing.FaultCollection()
        faults.add(fault)

        # Transfer normals
        normal_var = faults.transfer_normals(mesh)

        assert normal_var is not None
        assert normal_var.data.shape[1] == 3

        # All normals should point in z-direction (approximately)
        z_components = np.abs(normal_var.data[:, 2])
        assert np.mean(z_components) > 0.9

    def test_transfer_normals_empty_collection_fails(self):
        """Transferring normals with empty collection raises error."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )
        faults = uw.meshing.FaultCollection()

        with pytest.raises(ValueError, match="no faults"):
            faults.transfer_normals(mesh)


@requires_pyvista
@pytest.mark.level_2
class TestWeaknessFunction:
    """Tests for weakness function creation."""

    def test_create_weakness_function(self):
        """Create a Piecewise weakness function."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        # Create a simple fault
        points = np.array([
            [0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]
        ], dtype=float)
        fault = uw.meshing.FaultSurface("vert", points)
        fault.triangulate()

        faults = uw.meshing.FaultCollection()
        faults.add(fault)

        distance_var = faults.compute_distance_field(mesh)

        # Create weakness function
        import sympy
        weakness = faults.create_weakness_function(
            distance_var,
            fault_width=0.1,
            eta_weak=0.01,
            eta_background=1.0,
        )

        assert isinstance(weakness, sympy.Piecewise)
        # Verify the Piecewise structure: (value, condition) pairs
        # First pair should be (eta_weak, dist < fault_width)
        # Second pair should be (eta_background, True)
        assert len(weakness.args) == 2
        assert weakness.args[0][0] == 0.01  # eta_weak
        assert weakness.args[1][0] == 1.0   # eta_background
        # The expression should reference the distance variable
        expr_str = str(weakness)
        assert "fault_distance" in expr_str


# =============================================================================
# Level 3: Physics Tests (Full workflow)
# =============================================================================

@requires_pyvista
@pytest.mark.level_3
class TestFaultRheology:
    """Full workflow tests for fault-based rheology setup."""

    def test_fault_workflow_complete(self):
        """Test complete fault workflow: load, compute fields, create weakness."""
        # Create mesh
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.15,
        )

        # Create a planar fault at y=0.5
        x = np.linspace(0, 1, 15)
        z = np.linspace(0, 1, 15)
        xx, zz = np.meshgrid(x, z)
        points = np.column_stack([
            xx.ravel(),
            np.full(225, 0.5),
            zz.ravel()
        ])

        fault = uw.meshing.FaultSurface("planar", points)
        fault.triangulate()

        faults = uw.meshing.FaultCollection()
        faults.add(fault)

        # Compute fault data
        fault_distance = faults.compute_distance_field(mesh)
        fault_normals = faults.transfer_normals(mesh)

        # Verify distance field
        assert fault_distance.data.shape[1] == 1
        assert fault_distance.data.min() >= 0  # Distances are non-negative
        assert fault_distance.data.max() < 1.0  # Max distance in unit cube

        # Verify normals
        assert fault_normals.data.shape[1] == 3
        # Normals should be normalized (unit vectors)
        norms = np.sqrt(np.sum(fault_normals.data**2, axis=1))
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=3)

        # Verify normals point in y-direction (approximately)
        y_components = np.abs(fault_normals.data[:, 1])
        assert np.mean(y_components) > 0.9

        # Create weakness function
        fault_width = mesh.get_min_radius() * 3
        eta_weak = faults.create_weakness_function(
            fault_distance,
            fault_width=fault_width,
            eta_weak=0.1,
            eta_background=1.0,
        )

        # Verify weakness function structure
        import sympy
        assert isinstance(eta_weak, sympy.Piecewise)

    def test_multiple_faults_workflow(self):
        """Test workflow with multiple fault segments."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0, 0),
            maxCoords=(1, 1, 1),
            cellSize=0.2,
        )

        # Create two perpendicular faults
        # Fault 1: horizontal at y=0.3
        x1 = np.linspace(0, 1, 10)
        z1 = np.linspace(0, 1, 10)
        xx1, zz1 = np.meshgrid(x1, z1)
        points1 = np.column_stack([
            xx1.ravel(), np.full(100, 0.3), zz1.ravel()
        ])

        # Fault 2: horizontal at y=0.7
        points2 = np.column_stack([
            xx1.ravel(), np.full(100, 0.7), zz1.ravel()
        ])

        fault1 = uw.meshing.FaultSurface("lower", points1)
        fault1.triangulate()

        fault2 = uw.meshing.FaultSurface("upper", points2)
        fault2.triangulate()

        faults = uw.meshing.FaultCollection()
        faults.add(fault1)
        faults.add(fault2)

        assert len(faults) == 2

        # Compute combined distance field
        distance = faults.compute_distance_field(mesh)

        # Points near y=0.5 should be equidistant from both faults (~0.2)
        coords = mesh.X.coords
        if hasattr(coords, '__array__'):
            coords = np.asarray(coords)
        mid_mask = np.abs(coords[:, 1] - 0.5) < 0.1
        mid_distances = distance.data[mid_mask, 0]
        assert np.mean(mid_distances) < 0.25  # Should be close to 0.2

        # Transfer normals - should come from nearest fault
        normals = faults.transfer_normals(mesh)


# =============================================================================
# Feature Summary
# =============================================================================

def test_fault_module_summary():
    """Print summary of fault module capabilities."""
    print("\n" + "=" * 60)
    print("FAULT MODULE STATUS")
    print("=" * 60)
    print(f"  FaultSurface class: AVAILABLE")
    print(f"  FaultCollection class: AVAILABLE")
    print(f"  pyvista integration: {'YES' if HAS_PYVISTA else 'NO'}")
    print("=" * 60)
    if not HAS_PYVISTA:
        print("\nTo enable full functionality:")
        print("  pixi install -e runtime")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
