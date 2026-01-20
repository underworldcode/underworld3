"""
Underworld3 Meshing Module

This module provides mesh generation functions organized by geometry type.
All functions maintain backward compatibility with the original meshing interface.
"""

# Import from submodules for backward compatibility
from .cartesian import (
    UnstructuredSimplexBox,
    StructuredQuadBox,
    BoxInternalBoundary,
)

from .spherical import (
    SphericalShell,
    SphericalShellInternalBoundary,
    SegmentofSphere,
    CubedSphere,
)

from .annulus import (
    Annulus,
    QuarterAnnulus,
    SegmentofAnnulus,
    AnnulusWithSpokes,
    AnnulusInternalBoundary,
    DiscInternalBoundaries,
)

from .geographic import (
    RegionalSphericalBox,
    RegionalGeographicBox,
)

from .segmented import (
    SegmentedSphericalSurface2D,
    SegmentedSphericalShell,
    SegmentedSphericalBall,
)

from .surfaces import (
    Surface,
    SurfaceVariable,
    SurfaceCollection,
)

from .faults import (
    FaultSurface,
    FaultCollection,
)

# Make all functions available at module level for backward compatibility
__all__ = [
    # Cartesian meshes
    "UnstructuredSimplexBox",
    "StructuredQuadBox",
    "BoxInternalBoundary",
    # Spherical meshes
    "SphericalShell",
    "SphericalShellInternalBoundary",
    "SegmentofSphere",
    "CubedSphere",
    # Annulus/cylindrical meshes
    "Annulus",
    "QuarterAnnulus",
    "SegmentofAnnulus",
    "AnnulusWithSpokes",
    "AnnulusInternalBoundary",
    "DiscInternalBoundaries",
    # Geographic meshes
    "RegionalSphericalBox",
    "RegionalGeographicBox",
    # Segmented meshes
    "SegmentedSphericalSurface2D",
    "SegmentedSphericalShell",
    "SegmentedSphericalBall",
    # Surfaces (general embedded surfaces)
    "Surface",
    "SurfaceVariable",
    "SurfaceCollection",
    # Backward compatibility aliases
    "FaultSurface",
    "FaultCollection",
]
