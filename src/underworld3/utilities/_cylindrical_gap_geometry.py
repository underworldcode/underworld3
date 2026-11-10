"""Affine annulus basis and signed-cap integration rules."""
import numpy as np
from petsc4py import PETSc

def _p2_basis(
    vertices: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(
        np.stack(
            (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=-1
        )
    )
    xi = np.einsum("eij,eqj->eqi", inverse, points - vertices[:, None, 0])
    bary = np.concatenate((1 - xi.sum(axis=-1, keepdims=True), xi), axis=-1)
    grad = np.stack((-inverse.sum(axis=1), inverse[:, 0], inverse[:, 1]), axis=1)
    shape = np.stack(
        [bary[..., i] * (2 * bary[..., i] - 1) for i in range(3)]
        + [4 * bary[..., i] * bary[..., j] for i, j in ((0, 1), (1, 2), (2, 0))],
        axis=-1,
    )
    derivative = np.stack(
        [(4 * bary[..., i, None] - 1) * grad[:, None, i] for i in range(3)]
        + [
            4
            * (
                bary[..., i, None] * grad[:, None, j]
                + bary[..., j, None] * grad[:, None, i]
            )
            for i, j in ((0, 1), (1, 2), (2, 0))
        ],
        axis=2,
    )
    return (shape, derivative, bary)

def _cap_quadrature(
    edges: np.ndarray, radius: float, angular_order: int, radial_order: int
) -> dict:
    """Exact circular cap parametrization, positive weights, independent of edge order."""
    delta = np.arctan2(
        edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0],
        np.sum(edges[:, 0] * edges[:, 1], axis=1),
    )
    if np.any(abs(delta) >= np.pi) or np.any(abs(delta) < 1e-14):
        raise ValueError("Cap rule requires nonzero minor arcs")
    a, wa = np.polynomial.legendre.leggauss(angular_order)
    b, wb = np.polynomial.legendre.leggauss(radial_order)
    angle0 = np.arctan2(edges[:, 0, 1], edges[:, 0, 0])
    angle = angle0[:, None] + delta[:, None] * (a + 1) / 2
    direction = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    chord_radius = (
        radius
        * np.cos(delta[:, None] / 2)
        / np.cos(angle - angle0[:, None] - delta[:, None] / 2)
    )
    gap = radius - chord_radius
    if np.any(gap < -1e-12):
        raise ValueError("Invalid circle/chord cap geometry")
    radial = chord_radius[..., None] + gap[..., None] * (b + 1) / 2
    points = radial[..., None] * direction[:, :, None, :]
    weight = (
        abs(delta[:, None, None])
        * wa[None, :, None]
        / 2
        * gap[..., None]
        * wb[None, None, :]
        / 2
        * radial
    )
    return dict(
        circle_points=radius * direction,
        circle_weight=radius * abs(delta[:, None]) * wa[None, :] / 2,
        cap_points=points.reshape(len(edges), angular_order * radial_order, 2),
        cap_weight=weight.reshape(len(edges), angular_order * radial_order),
    )

def _triangle_rule(vertices: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive tensor-product Duffy rule on each affine triangle."""
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = ((x + 1) / 2, w / 2)
    a, b = np.meshgrid(x, x, indexing="ij")
    bary = np.stack((1 - a, a * (1 - b), a * b), axis=-1).reshape(-1, 3)
    determinant = abs(
        np.linalg.det(
            np.stack(
                (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
                axis=-1,
            )
        )
    )
    return (
        np.einsum("qj,ejd->eqd", bary, vertices),
        determinant[:, None] * (a * w[:, None] * w[None, :]).ravel(),
    )

def _volume_blocks(
    vertices: np.ndarray, points: np.ndarray, weight: np.ndarray
) -> dict:
    """Viscosity-one element matrices; interleaved x/y velocity components."""
    shape, grad, bary = _p2_basis(vertices, points)
    stiffness = np.zeros((len(vertices), 12, 12))
    divergence = np.zeros((len(vertices), 3, 12))
    dot = np.einsum("eq,eqid,eqjd->eij", weight, grad, grad)
    for a in range(2):
        divergence[:, :, a::2] = -np.einsum(
            "eq,eql,eqi->eli", weight, bary, grad[..., a]
        )
        for b in range(2):
            stiffness[:, a::2, b::2] = (a == b) * dot + np.einsum(
                "eq,eqi,eqj->eij", weight, grad[..., b], grad[..., a]
            )
    rotation = np.stack((-points[..., 1], points[..., 0]), axis=-1)
    return dict(
        A=stiffness,
        B=divergence,
        pressure_mean=np.einsum("eq,eql->el", weight, bary),
        rotation=np.einsum("eq,eqi,eqd->eid", weight, shape, rotation),
        unit_force=np.einsum("eq,eqi->ei", weight, shape),
    )
