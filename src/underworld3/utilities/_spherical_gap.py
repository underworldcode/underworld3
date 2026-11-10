r"""Quadrature between affine spherical-shell facets and the physical sphere.

A planar triangular facet is projected radially onto a concentric sphere.
For a point y on the facet, dOmega = |n.y| / |y|**3 dA. Surface integration
uses R**2 dOmega; radial gap integration uses t**2 dt dOmega between |y|
and R. The gap weights are positive: a shell assembly must add outer gaps
and subtract inner gaps in BOTH momentum and continuity equations.

This is geometry infrastructure, not a spherical Stokes solver. Multiplier
spaces, gauges and distributed assembly require separate validation.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


def _spherical_facet_quadrature(
    facets: np.ndarray, radius: float, surface_order: int, radial_order: int
) -> dict[str, np.ndarray]:
    """Integrate spherical patches and their gaps above affine triangular facets.

    Parameters
    ----------
    facets : numpy.ndarray
        Shape (number of facets, 3, 3); all vertices lie on the sphere centered
        at the origin. Vertex order does not affect the positive weights.
    radius : float
        Positive physical sphere radius.
    surface_order, radial_order : int
        Positive Gauss orders for each Duffy coordinate and the radial segment.

    Returns
    -------
    dict
        Planar/surface points and weights, outward radial surface normals,
        and gap points/weights. Empty facet batches retain the same axes.

    Notes
    -----
    The spherical area and volume Jacobians are exact geometric expressions;
    Gauss integration of their nonpolynomial dependence is approximate. Raise
    surface order to assess quadrature convergence independently of mesh error.
    No complete sphere or distributed mesh is gathered by this local operation.
    """
    facets = np.asarray(facets, dtype=float)
    if facets.ndim != 3 or facets.shape[1:] != (3, 3):
        raise ValueError("Spherical facets must have shape (N, 3, 3).")
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(facets).all():
        raise ValueError("Sphere radius must be positive and facet coordinates finite.")
    if any(not isinstance(order, (int, np.integer)) or isinstance(order, bool)
           or order < 1 for order in (surface_order, radial_order)):
        raise ValueError("Surface and radial quadrature orders must be positive integers.")
    if not np.allclose(np.linalg.norm(facets, axis=-1), radius, rtol=1e-12, atol=0):
        raise ValueError("All facet vertices must lie on the prescribed sphere.")

    edge_a = facets[:, 1] - facets[:, 0]
    edge_b = facets[:, 2] - facets[:, 0]
    cross = np.cross(edge_a, edge_b)
    jacobian = np.linalg.norm(cross, axis=-1)
    height_numerator = np.abs(np.einsum("fi,fi->f", cross, facets[:, 0]))
    if np.any(jacobian == 0) or np.any(height_numerator <= 1e-14 * radius * jacobian):
        raise ValueError("Spherical facets must be nondegenerate and radially projectable.")

    nodes, weights = leggauss(surface_order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    wu, wv = np.meshgrid(weights, weights, indexing="ij")
    u, v = u.ravel(), v.ravel()
    reference_weights = (wu * wv).ravel() * (1 - u)
    points = (facets[:, None, 0] + u[None, :, None] * edge_a[:, None]
              + ((1 - u) * v)[None, :, None] * edge_b[:, None])
    distance = np.linalg.norm(points, axis=-1)
    normals = points / distance[..., None]
    solid_angle_weights = (height_numerator[:, None] * reference_weights[None, :]
                           / distance**3)
    surface_points = radius * normals

    radial_nodes, radial_weights = leggauss(radial_order)
    radial_nodes, radial_weights = (radial_nodes + 1) / 2, radial_weights / 2
    gap_width = radius - distance
    radial_distance = distance[..., None] + gap_width[..., None] * radial_nodes
    gap_points = radial_distance[..., None] * normals[:, :, None, :]
    gap_weights = (solid_angle_weights[..., None] * gap_width[..., None]
                   * radial_weights * radial_distance**2)
    return {
        "planar_points": points,
        "planar_weights": jacobian[:, None] * reference_weights,
        "surface_points": surface_points,
        "surface_normals": normals,
        "surface_weights": radius**2 * solid_angle_weights,
        "gap_points": gap_points,
        "gap_weights": gap_weights,
    }
