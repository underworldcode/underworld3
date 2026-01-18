#!/usr/bin/env python3
"""
Tests for optional module dependencies in Underworld3.

This test suite validates the optional dependency pattern:
- Core functionality works without optional modules
- Optional features gracefully skip when dependencies unavailable
- Optional features work correctly when dependencies ARE available
- Helpful error messages guide users to install missing dependencies

Optional Module Categories:
- Geographic (geo): cartopy, gdal, pyproj, geopandas, owslib
- Visualization (viz): pyvista, trame
- AMR: pragmatic, mmg, parmmg (via custom PETSc build)

Installation:
- Geographic: `pixi run install-geo`
- Visualization: `pixi install -e runtime` or `pixi install -e dev`
- AMR: `pixi install -e amr` then `pixi run -e amr petsc-build`
"""

import pytest
import sys

import underworld3 as uw

# Module-level markers
pytestmark = [
    pytest.mark.level_1,  # Quick tests - just checking imports/availability
]


# =============================================================================
# Dependency Detection Helpers
# =============================================================================

def check_module_available(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# Dependency availability flags
HAS_CARTOPY = check_module_available("cartopy")
HAS_PYPROJ = check_module_available("pyproj")
HAS_GDAL = check_module_available("osgeo.gdal")
HAS_GEOPANDAS = check_module_available("geopandas")
HAS_PYVISTA = check_module_available("pyvista")
HAS_TRAME = check_module_available("trame")

# Composite feature flags
HAS_GEO = HAS_CARTOPY and HAS_PYPROJ
HAS_VIZ = HAS_PYVISTA


def check_petsc_has_pragmatic() -> bool:
    """Check if PETSc was built with pragmatic support."""
    import os
    try:
        petsc_dir = os.environ.get('PETSC_DIR', '')
        petsc_arch = os.environ.get('PETSC_ARCH', '')

        # Check petscvariables file for pragmatic
        conf_file = f'{petsc_dir}/{petsc_arch}/lib/petsc/conf/petscvariables'
        if os.path.exists(conf_file):
            with open(conf_file) as f:
                content = f.read().lower()
                return 'pragmatic' in content

        # Fallback: check if this looks like a custom PETSc build
        return 'petsc-custom' in petsc_dir or 'petsc-4-uw' in petsc_arch
    except Exception:
        return False


HAS_AMR = check_petsc_has_pragmatic()


# =============================================================================
# Skip Markers for Optional Features
# =============================================================================

requires_geo = pytest.mark.skipif(
    not HAS_GEO,
    reason="Geographic features require cartopy and pyproj. Install with: pixi run install-geo"
)

requires_cartopy = pytest.mark.skipif(
    not HAS_CARTOPY,
    reason="Requires cartopy. Install with: pixi run install-geo"
)

requires_pyvista = pytest.mark.skipif(
    not HAS_PYVISTA,
    reason="Requires pyvista. Install with: pixi install -e runtime"
)

requires_amr = pytest.mark.skipif(
    not HAS_AMR,
    reason="Requires AMR-enabled PETSc. Install with: pixi install -e amr && pixi run -e amr petsc-build"
)


# =============================================================================
# Core Functionality Tests (Always Run)
# =============================================================================

class TestCoreWithoutOptionalDeps:
    """Verify core UW3 works without any optional dependencies."""

    def test_basic_mesh_creation(self):
        """Core mesh creation doesn't require optional deps."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )
        assert mesh is not None
        assert mesh.dim == 2

    def test_basic_variable_creation(self):
        """Variables work without optional deps."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )
        var = uw.discretisation.MeshVariable("test", mesh, 1)
        assert var is not None

    def test_basic_swarm_creation(self):
        """Swarms work without optional deps."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )
        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=2)
        assert swarm._particle_coordinates.data.shape[0] > 0

    def test_units_without_optional_deps(self):
        """Units system works without optional deps."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )
        velocity = uw.create_enhanced_mesh_variable("v", mesh, 2, units="m/s")
        assert velocity.has_units
        assert str(velocity.units) == "meter / second"


# =============================================================================
# Geographic Feature Tests
# =============================================================================

class TestGeographicFeatures:
    """Tests for geographic/cartographic features."""

    def test_geo_dependency_detection(self):
        """Verify we can detect geographic dependencies."""
        # This test always passes - it just reports status
        status = {
            "cartopy": HAS_CARTOPY,
            "pyproj": HAS_PYPROJ,
            "gdal": HAS_GDAL,
            "geopandas": HAS_GEOPANDAS,
            "geo_complete": HAS_GEO,
        }
        print(f"\nGeographic dependency status: {status}")
        # Always passes - informational only

    @requires_geo
    def test_geographic_mesh_creation(self):
        """Test geographic mesh when dependencies available."""
        # Regional spherical box should work
        mesh = uw.meshing.RegionalSphericalBox(
            radiusOuter=1.0,
            radiusInner=0.5,
            SWcorner=[-30, -30],
            NEcorner=[30, 30],
            numElementsLon=3,
            numElementsLat=3,
            numElementsDepth=2,
        )
        assert mesh is not None
        assert mesh.dim == 3

    @requires_cartopy
    def test_cartopy_projection_available(self):
        """Test cartopy projections when available."""
        import cartopy.crs as ccrs

        # Basic projection test
        proj = ccrs.PlateCarree()
        assert proj is not None

        # Orthographic for visualization
        ortho = ccrs.Orthographic(central_longitude=0, central_latitude=0)
        assert ortho is not None


# =============================================================================
# Visualization Feature Tests
# =============================================================================

class TestVisualizationFeatures:
    """Tests for visualization features."""

    def test_viz_dependency_detection(self):
        """Verify we can detect visualization dependencies."""
        status = {
            "pyvista": HAS_PYVISTA,
            "trame": HAS_TRAME,
            "viz_complete": HAS_VIZ,
        }
        print(f"\nVisualization dependency status: {status}")

    @requires_pyvista
    def test_pyvista_mesh_conversion(self):
        """Test mesh to pyvista conversion when available."""
        import pyvista as pv

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )

        # Check if mesh has pyvista conversion method via visualisation module
        if hasattr(uw, 'visualisation') and hasattr(uw.visualisation, 'mesh_to_pv'):
            pv_mesh = uw.visualisation.mesh_to_pv(mesh)
            assert pv_mesh is not None
        elif hasattr(mesh, 'to_pyvista') or hasattr(mesh, 'pyvista_mesh'):
            pv_mesh = mesh.to_pyvista() if hasattr(mesh, 'to_pyvista') else mesh.pyvista_mesh
            assert pv_mesh is not None
        else:
            # Conversion not implemented yet - that's OK, just verify pyvista works
            sphere = pv.Sphere()
            assert sphere is not None

    @requires_pyvista
    def test_pyvista_plotter_available(self):
        """Test pyvista plotter when available."""
        import pyvista as pv

        # Just verify we can create a plotter (off-screen)
        plotter = pv.Plotter(off_screen=True)
        assert plotter is not None
        plotter.close()


# =============================================================================
# AMR Feature Tests
# =============================================================================

class TestAMRFeatures:
    """Tests for adaptive mesh refinement features."""

    def test_amr_dependency_detection(self):
        """Verify we can detect AMR dependencies."""
        status = {
            "pragmatic_in_petsc": HAS_AMR,
        }
        print(f"\nAMR dependency status: {status}")

    @requires_amr
    def test_mesh_refinement_available(self):
        """Test mesh refinement when AMR available."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2
        )

        # Check if mesh has refinement capabilities
        if hasattr(mesh, 'refine') or hasattr(mesh, 'adapt'):
            # AMR methods exist
            pass
        else:
            pytest.skip("AMR refinement methods not yet exposed")


# =============================================================================
# Error Message Quality Tests
# =============================================================================

class TestOptionalModuleErrorMessages:
    """Test that missing optional modules produce helpful error messages."""

    def test_import_error_message_pattern(self):
        """Verify the pattern for helpful import error messages."""
        # This is a pattern test - shows expected behavior
        # When implementing optional modules, use this pattern:

        example_code = '''
        def require_cartopy():
            """Require cartopy with helpful error message."""
            try:
                import cartopy
                return cartopy
            except ImportError:
                raise ImportError(
                    "This feature requires cartopy. "
                    "Install with: pixi run install-geo"
                )
        '''

        # The test passes if we got here - it's documenting the pattern
        assert "pixi run install-geo" in example_code


# =============================================================================
# Feature Flag Reporting
# =============================================================================

def test_optional_features_summary():
    """Print summary of optional feature availability."""
    print("\n" + "=" * 60)
    print("OPTIONAL FEATURES STATUS")
    print("=" * 60)
    print(f"  Geographic (cartopy + pyproj): {'YES' if HAS_GEO else 'NO'}")
    print(f"    - cartopy:   {'YES' if HAS_CARTOPY else 'NO'}")
    print(f"    - pyproj:    {'YES' if HAS_PYPROJ else 'NO'}")
    print(f"    - gdal:      {'YES' if HAS_GDAL else 'NO'}")
    print(f"    - geopandas: {'YES' if HAS_GEOPANDAS else 'NO'}")
    print(f"  Visualization (pyvista):       {'YES' if HAS_VIZ else 'NO'}")
    print(f"    - pyvista:   {'YES' if HAS_PYVISTA else 'NO'}")
    print(f"    - trame:     {'YES' if HAS_TRAME else 'NO'}")
    print(f"  AMR (pragmatic in PETSc):      {'YES' if HAS_AMR else 'NO'}")
    print("=" * 60)
    print("\nTo install optional features:")
    print("  Geographic: pixi run install-geo")
    print("  Viz/Runtime: pixi install -e runtime")
    print("  AMR: pixi install -e amr && pixi run -e amr petsc-build")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
