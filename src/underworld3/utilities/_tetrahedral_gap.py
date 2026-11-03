r"""Affine tetrahedral P2/P1 integration for a future spherical-gap backend.

Velocity components are interleaved x/y/z at four vertices, followed by edges
01, 02, 03, 12, 13, 23. B represents minus the divergence. Positive volume
rules also evaluate basis polynomials outside the element for physical gaps;
callers own inner/outer gap signs and distributed assembly.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _affine_geometry(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return affine map and signed determinant for nondegenerate tetrahedra."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
        raise ValueError("Tetrahedral vertices must have shape (N, 4, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("Tetrahedral coordinates must be finite.")
    edges = np.swapaxes(vertices[:, 1:] - vertices[:, None, 0], 1, 2)
    determinant = np.linalg.det(edges)
    if np.any(determinant == 0):
        raise ValueError("Tetrahedra must have nonzero volume.")
    return edges, determinant


def _tetrahedral_p2_basis(
    vertices: np.ndarray, points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return P2 values, Cartesian gradients and P1 barycentric values.

    Points have shape (N, Q, 3). Returned axes are (N,Q,10), (N,Q,10,3)
    and (N,Q,4). Points outside the element are intentionally supported.
    """
    vertices, points = np.asarray(vertices, dtype=float), np.asarray(points, dtype=float)
    edges, _ = _affine_geometry(vertices)
    if points.ndim != 3 or points.shape[0] != len(vertices) or points.shape[2] != 3:
        raise ValueError("Evaluation points must have shape (N, Q, 3).")
    if not np.isfinite(points).all():
        raise ValueError("Evaluation coordinates must be finite.")
    inverse = np.linalg.inv(edges)
    xi = np.einsum("eij,eqj->eqi", inverse, points - vertices[:, None, 0])
    bary = np.concatenate((1 - xi.sum(axis=-1, keepdims=True), xi), axis=-1)
    gradients = np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
    shape = np.stack(
        [bary[..., i] * (2*bary[..., i] - 1) for i in range(4)]
        + [4*bary[..., i]*bary[..., j] for i, j in _EDGES], axis=-1,
    )
    derivative = np.stack(
        [(4*bary[..., i, None] - 1) * gradients[:, None, i] for i in range(4)]
        + [4*(bary[..., i, None]*gradients[:, None, j]
              + bary[..., j, None]*gradients[:, None, i]) for i, j in _EDGES],
        axis=2,
    )
    return shape, derivative, bary


def _tetrahedron_quadrature(
    vertices: np.ndarray, order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive tensor-product Duffy quadrature, including reversed orientations."""
    vertices = np.asarray(vertices, dtype=float)
    _, determinant = _affine_geometry(vertices)
    if not isinstance(order, (int, np.integer)) or isinstance(order, bool) or order < 1:
        raise ValueError("Tetrahedral quadrature order must be a positive integer.")
    nodes, weights = leggauss(order)
    nodes, weights = (nodes + 1)/2, weights/2
    u, v, w = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    wu, wv, ww = np.meshgrid(weights, weights, weights, indexing="ij")
    bary = np.stack((1-u, u*(1-v), u*v*(1-w), u*v*w), axis=-1).reshape(-1, 4)
    points = np.einsum("qi,eid->eqd", bary, vertices)
    weight = np.abs(determinant)[:, None] * (u*u*v*wu*wv*ww).ravel()
    return points, weight


def _tetrahedral_stokes_blocks(
    vertices: np.ndarray, points: np.ndarray, weights: np.ndarray,
) -> dict[str, np.ndarray]:
    r"""Integrate viscosity-one 2 epsilon(u):epsilon(v) and -q div(u).

    A has shape (N,30,30), B (N,4,30), pressure_mean (N,4),
    rotation (N,30,3) and unit_force (N,10). Rotation columns integrate
    each velocity basis against e_x cross x, e_y cross x and e_z cross x.
    Weights may be signed for gap subtraction; no solver is constructed.
    """
    shape, grad, bary = _tetrahedral_p2_basis(vertices, points)
    points, weights = np.asarray(points, dtype=float), np.asarray(weights, dtype=float)
    if weights.shape != shape.shape[:2] or not np.isfinite(weights).all():
        raise ValueError("Quadrature weights must be finite with shape (N, Q).")
    stiffness = np.zeros((len(shape), 30, 30))
    divergence = np.zeros((len(shape), 4, 30))
    dot = np.einsum("eq,eqid,eqjd->eij", weights, grad, grad)
    for a in range(3):
        divergence[:, :, a::3] = -np.einsum("eq,eql,eqi->eli", weights, bary, grad[..., a])
        for b in range(3):
            stiffness[:, a::3, b::3] = (a == b)*dot + np.einsum(
                "eq,eqi,eqj->eij", weights, grad[..., b], grad[..., a],
            )
    rotations = np.stack(
        [np.cross(axis, points) for axis in np.eye(3)], axis=-1,
    )
    return {
        "A": stiffness,
        "B": divergence,
        "pressure_mean": np.einsum("eq,eql->el", weights, bary),
        "rotation": np.einsum("eq,eqi,eqdm->eidm", weights, shape, rotations).reshape(-1, 30, 3),
        "unit_force": np.einsum("eq,eqi->ei", weights, shape),
    }
