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
)

from .segmented import (
    SegmentedSphericalSurface2D,
    SegmentedSphericalShell,
    SegmentedSphericalBall,
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
    # Segmented meshes
    "SegmentedSphericalSurface2D",
    "SegmentedSphericalShell",
    "SegmentedSphericalBall",
]
