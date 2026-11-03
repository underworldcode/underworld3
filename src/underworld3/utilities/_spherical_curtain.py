r"""Local radial-curtain integration for discontinuous spherical-gap extensions.

Independent tetrahedral polynomials need not agree above a shared boundary
edge. These consistency blocks use jump u_A-u_B with the curtain normal
pointing from patch A into B. They add no penalty or stabilization, and do
not establish stability of a distributed spherical Stokes discretisation.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from underworld3.utilities._tetrahedral_gap import _tetrahedral_p2_basis


def _spherical_curtain_quadrature(
    edge: np.ndarray, patch_a_center: np.ndarray, radius: float,
    edge_order: int, radial_order: int,
) -> dict[str, np.ndarray]:
    """Positive quadrature on the planar region between a chord and sphere arc.

    The normal points away from patch A. Endpoints lie on the sphere and
    the patch center must be strictly on one side of their radial plane.
    """
    edge, center = np.asarray(edge, dtype=float), np.asarray(patch_a_center, dtype=float)
    if edge.shape != (2, 3) or center.shape != (3,):
        raise ValueError("Require edge shape (2,3) and patch center shape (3,).")
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(edge).all() or not np.isfinite(center).all():
        raise ValueError("Require positive finite radius and finite curtain coordinates.")
    if not np.allclose(np.linalg.norm(edge, axis=1), radius, rtol=1e-12, atol=0):
        raise ValueError("Curtain edge endpoints must lie on the sphere.")
    if any(not isinstance(order, (int, np.integer)) or isinstance(order, bool) or order < 1
           for order in (edge_order, radial_order)):
        raise ValueError("Curtain quadrature orders must be positive integers.")
    cross = np.cross(edge[0], edge[1])
    magnitude = np.linalg.norm(cross)
    if magnitude <= 1e-14*radius**2:
        raise ValueError("Curtain endpoints must define a nondegenerate radial plane.")
    normal = cross/magnitude
    side = normal@center
    if abs(side) <= 1e-14*radius:
        raise ValueError("Patch A center must lie strictly on one side of the curtain.")
    normal *= -np.sign(side)
    s, sw = leggauss(edge_order)
    r, rw = leggauss(radial_order)
    s, sw, r, rw = (s+1)/2, sw/2, (r+1)/2, rw/2
    y = (1-s[:, None])*edge[0]+s[:, None]*edge[1]
    rho = np.linalg.norm(y, axis=1)
    direction = y/rho[:, None]
    width = radius-rho
    t = rho[:, None]+width[:, None]*r
    points = t[..., None]*direction[:, None, :]
    weight = (t*magnitude/rho[:, None]**2)*width[:, None]*sw[:, None]*rw
    return dict(points=points.reshape(-1, 3), weights=weight.ravel(), normal=normal)


def _curtain_stokes_blocks(
    vertices_a: np.ndarray, vertices_b: np.ndarray, points: np.ndarray,
    weights: np.ndarray, normal: np.ndarray, sign: int,
) -> dict[str, np.ndarray]:
    r"""Viscosity-one symmetric consistency and pressure-jump coupling blocks.

    A acts on concatenated 30-component velocities (A then B); B acts from
    those 60 velocities into concatenated four-node P1 pressures. The jump
    is u_A-u_B, traction is average(2 epsilon(u)) n, and pressure is averaged.
    The shell's outer/inner gap sign is respectively +1/-1.
    """
    points, weights, normal = np.asarray(points), np.asarray(weights), np.asarray(normal)
    if sign not in (-1, 1):
        raise ValueError("Curtain gap sign must be -1 or +1.")
    if points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points),):
        raise ValueError("Require curtain points (Q,3) and weights (Q,).")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Curtain geometric weights must be finite and nonnegative.")
    if normal.shape != (3,) or not np.allclose(np.linalg.norm(normal), 1, rtol=1e-12):
        raise ValueError("Curtain normal must be a finite unit three-vector.")
    shape_a, grad_a, pressure_a = _tetrahedral_p2_basis(np.asarray(vertices_a)[None], points[None])
    shape_b, grad_b, pressure_b = _tetrahedral_p2_basis(np.asarray(vertices_b)[None], points[None])
    jump = np.zeros((len(points), 3, 60))
    traction = np.zeros_like(jump)
    for side, shape, grad, jump_sign in (
        (0, shape_a[0], grad_a[0], 1), (30, shape_b[0], grad_b[0], -1),
    ):
        normal_derivative = grad@normal
        for component in range(3):
            jump[:, component, side+component:side+30:3] = jump_sign*shape
            for output in range(3):
                traction[:, output, side+component:side+30:3] = (
                    ((output == component)*normal_derivative
                     + grad[..., output]*normal[component])/2
                )
    average_pressure = np.concatenate((pressure_a[0], pressure_b[0]), axis=1)/2
    jt = np.einsum("q,qdi,qdj->ij", weights, jump, traction)
    return {
        "A": -sign*(jt+jt.T),
        "B": sign*np.einsum("q,ql,qdi,d->li", weights, average_pressure, jump, normal),
        "jump": jump,
        "average_traction": traction,
    }
