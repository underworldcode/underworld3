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
    SphericalManifold,
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
    fault_metric_tensor,
    fault_comb_metric,
    fault_metric,
    compose_metrics,
)

from .faults import (
    FaultSurface,
    FaultCollection,
)

from .smoothing import (
    smooth_mesh_interior,
    smooth_surface_field,
    node_redistribution,
    metric_density_from_gradient,
    mesh_metric_mismatch,
    follow_metric,
    ADAPT_STRATEGIES,
)

from .bounding_surface import (
    BoundingSurface,
    register_radial_surfaces,
    register_plane_surfaces,
    register_box_face_surfaces,
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
    "SphericalManifold",
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
    "fault_metric_tensor",
    "fault_comb_metric",
    "fault_metric",
    "compose_metrics",
    # Backward compatibility aliases
    "FaultSurface",
    "FaultCollection",
    # Mesh smoothing / node redistribution
    "smooth_mesh_interior",
    "smooth_surface_field",
    "node_redistribution",
    "metric_density_from_gradient",
    "mesh_metric_mismatch",
    "follow_metric",
    "ADAPT_STRATEGIES",
    # Bounding surfaces (tangent-slip providers for deforming boundaries)
    "BoundingSurface",
    "register_radial_surfaces",
    "register_plane_surfaces",
    "register_box_face_surfaces",
]
