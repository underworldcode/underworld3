"""Cell-local quadrature of a tetrahedron outside a concentric ball.

Integrate signed radial cones over the four *actual cell faces*. Entry and
exit cones use the same cell polynomial, so their difference is its physical
restriction. Unlike a boundary-facet continuation, no neighbouring polynomial
is subtracted from this cell. Triangular faces are split at every circle/edge
intersection before polar quadrature; there is no pointwise clipping indicator.
Wholly void cells return empty rules exactly.
"""

from itertools import combinations

import numpy as np
from numpy.polynomial.legendre import leggauss


def _inner_cell_relation(vertices, radius):
    """Classify cells as wholly void (-1), cut (0), or wholly physical (1)."""
    vertices = np.asarray(vertices, dtype=float)
    tolerance = 8*np.finfo(float).eps*radius
    lengths = np.linalg.norm(vertices, axis=-1)
    result = np.ones(len(vertices), dtype=np.int8)
    void = lengths.max(axis=1) <= radius+tolerance
    result[void] = -1
    lower = np.linalg.norm(np.maximum(np.maximum(vertices.min(axis=1),
                                                -vertices.max(axis=1)), 0), axis=1)
    selected = np.flatnonzero(~void & (lower < radius))
    if not len(selected):
        return result
    v = vertices[selected]
    distance = np.min(np.sum(v*v, axis=-1), axis=1)
    for i, j in combinations(range(4), 2):
        edge = v[:, j]-v[:, i]
        t = np.clip(-np.einsum("ij,ij->i", v[:, i], edge)/np.sum(edge*edge, axis=1), 0, 1)
        distance = np.minimum(distance, np.sum((v[:, i]+t[:, None]*edge)**2, axis=1))
    for corners in combinations(range(4), 3):
        a, b, c = (v[:, i] for i in corners)
        normal = np.cross(b-a, c-a)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        projection = np.einsum("ij,ij->i", normal, a)[:, None]*normal
        e, f, p = b-a, c-a, projection-a
        ee, ef, ff = np.sum(e*e, axis=1), np.sum(e*f, axis=1), np.sum(f*f, axis=1)
        pe, pf = np.sum(p*e, axis=1), np.sum(p*f, axis=1)
        determinant = ee*ff-ef*ef
        s, t = (pe*ff-pf*ef)/determinant, (pf*ee-pe*ef)/determinant
        inside = (s >= 0) & (t >= 0) & (s+t <= 1)
        distance[inside] = np.minimum(distance[inside], np.sum(projection[inside]**2, axis=1))
    xi = np.linalg.solve(np.swapaxes(v[:, 1:]-v[:, None, 0], 1, 2), -v[:, 0, :, None])[..., 0]
    origin_inside = (xi.min(axis=1) >= 0) & (xi.sum(axis=1) <= 1)
    distance[origin_inside] = 0
    result[selected[distance < (radius-tolerance)**2]] = 0
    return result


def _triangle_outside_disk(vertices, radius, order):
    """Positive area rule for a planar triangle minus a centred disk."""
    edge = np.roll(vertices, -1, axis=0)-vertices
    cross = lambda a, b: a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    if cross(edge[0], vertices[2]-vertices[0]) < 0:
        vertices = vertices[::-1]
        edge = np.roll(vertices, -1, axis=0)-vertices
    angles = [0., 2*np.pi, *np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2*np.pi)]
    if radius > 0:
        for a, e in zip(vertices, edge, strict=True):
            coefficients = (e@e, 2*(a@e), a@a-radius**2)
            discriminant = coefficients[1]**2-4*coefficients[0]*coefficients[2]
            if discriminant >= 0:
                for t in ((-coefficients[1]-np.sqrt(discriminant))/(2*coefficients[0]),
                          (-coefficients[1]+np.sqrt(discriminant))/(2*coefficients[0])):
                    if 0 < t < 1:
                        point = a+t*e
                        angles.append(np.mod(np.arctan2(point[1], point[0]), 2*np.pi))
    angles = np.unique(angles)
    nodes, weights = leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    points, areas = [], []
    for start, end in zip(angles[:-1], angles[1:], strict=True):
        theta = start+(end-start)*nodes
        direction = np.column_stack((np.cos(theta), np.sin(theta)))
        lower, upper = np.full(order, radius), np.full(order, np.inf)
        for a, e in zip(vertices, edge, strict=True):
            coefficient = cross(e, direction)
            value = cross(e, a)
            positive, negative = coefficient > 0, coefficient < 0
            lower[positive] = np.maximum(lower[positive], value/coefficient[positive])
            upper[negative] = np.minimum(upper[negative], value/coefficient[negative])
            upper[(coefficient == 0) & (value > 0)] = -np.inf
        selected = np.isfinite(upper) & (upper > lower)
        lo, hi = lower[selected], upper[selected]
        rho = lo[:, None]+(hi-lo)[:, None]*nodes
        points.append((rho[..., None]*direction[selected, None]).reshape(-1, 2))
        areas.append(((end-start)*weights[selected, None]*weights[None]
                      *(hi-lo)[:, None]*rho).ravel())
    return np.concatenate(points), np.concatenate(areas)


def _cell_faces(vertices):
    """Yield outward face geometry and an orthonormal planar frame."""
    for corners in combinations(range(4), 3):
        triangle = vertices[list(corners)]
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        normal /= np.linalg.norm(normal)
        other = next(i for i in range(4) if i not in corners)
        if normal@(vertices[other]-triangle[0]) > 0:
            normal *= -1
        height = normal@triangle[0]
        first = triangle[1]-triangle[0]
        first /= np.linalg.norm(first)
        yield triangle, normal, height, np.stack((first, np.cross(normal, first)))


def _face_outside_ball(triangle, radius, order, normal, height, frame):
    planar = (triangle-height*normal)@frame.T
    disk = np.sqrt(max(0., radius**2-height**2))
    xy, weight = _triangle_outside_disk(planar, disk, order)
    return xy@frame+height*normal, weight


def _clip_cone(polygon, planes):
    """Intersect a planar convex polygon with three radial half spaces."""
    for plane in planes:
        if not len(polygon):
            break
        distance = polygon@plane
        clipped = []
        for i in range(len(polygon)):
            j = (i+1) % len(polygon)
            a, b = distance[i], distance[j]
            if a >= 0:
                clipped.append(polygon[i])
            if (a >= 0) != (b >= 0):
                clipped.append(polygon[i]+a/(a-b)*(polygon[j]-polygon[i]))
        polygon = np.asarray(clipped).reshape(-1, 3)
    return polygon


def _spherical_cut_cell_surface(vertices, radius, order, facets, lids):
    """Inner-wall rule partitioned by the ORIGINAL multiplier facet cones.

    Every velocity sample evaluates this cell's polynomial. Every multiplier
    sample belongs to one known original facet. Angular boundaries of the
    piecewise multiplier are clipped geometrically before quadrature, rather
    than located after an unsplit rule has crossed them.
    """
    points, weights, ids, values = [], [], [], []
    components = lids.shape[1]//6
    for triangle, normal, height, frame in _cell_faces(vertices):
        if height == 0:
            continue
        area = np.linalg.norm(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]))
        for facet, lid in zip(facets, lids, strict=True):
            inverse = np.linalg.inv(facet.T)
            planes = inverse/np.linalg.norm(inverse, axis=1)[:, None]
            polygon = _clip_cone(triangle.copy(), planes)
            for i in range(1, len(polygon)-1):
                piece = polygon[[0, i, i+1]]
                piece_area = np.linalg.norm(np.cross(piece[1]-piece[0], piece[2]-piece[0]))
                if piece_area <= 64*np.finfo(float).eps*area:
                    continue
                y, weight = _face_outside_ball(piece, radius, order, normal, height, frame)
                distance = np.linalg.norm(y, axis=1)
                x = radius*y/distance[:, None]
                coefficients = x@inverse.T
                total = coefficients.sum(axis=1)
                bary = coefficients/total[:, None]
                multiplier = np.column_stack((bary*(2*bary-1),
                    4*bary[:, 0]*bary[:, 1], 4*bary[:, 0]*bary[:, 2], 4*bary[:, 1]*bary[:, 2]))
                if components == 3:
                    multiplier *= total[:, None]
                points.append(x)
                weights.append(radius**2*height*weight/distance**3)
                ids.append(np.broadcast_to(lid, (len(x), len(lid))))
                values.append(multiplier)
    return (np.concatenate(points) if points else np.empty((0, 3)),
            np.concatenate(weights) if weights else np.empty(0),
            np.concatenate(ids) if ids else np.empty((0, lids.shape[1]), dtype=lids.dtype),
            np.concatenate(values) if values else np.empty((0, 6)))


def _spherical_cut_cell_quadrature(vertices, radius, surface_order, radial_order):
    """Return volume, inner-sphere and planar-face rules for one physical cell.

    Surface weights are signed radial-cone weights; normals on the physical
    inner wall are ``-x/radius``. Face weights are positive and face normals
    point out of the tetrahedron. The volume rule integrates the portion with
    ``|x| >= radius``. Signed cone samples can lie outside the tetrahedron;
    they must ALL evaluate this cell's polynomial, including its prescribed
    FE force. These algebraic cones are not assigned to neighbouring cells.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (4, 3) or not np.isfinite(vertices).all():
        raise ValueError("Require four finite tetrahedral vertices.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Require a positive finite inner radius.")
    for order in (surface_order, radial_order):
        if not isinstance(order, (int, np.integer)) or isinstance(order, bool) or order < 1:
            raise ValueError("Quadrature orders must be positive integers.")
    if np.linalg.det(vertices[1:]-vertices[0]) == 0:
        raise ValueError("Require a nondegenerate tetrahedron.")
    result = {name: [] for name in ("volume_points", "volume_weights", "surface_points",
                                    "surface_weights", "face_points", "face_weights", "face_normals")}
    if np.max(np.linalg.norm(vertices, axis=1)) <= radius*(1+8*np.finfo(float).eps):
        return {name: np.empty((0, 3) if name.endswith(("points", "normals")) else (0,))
                for name in result}
    rn, rw = leggauss(radial_order)
    rn, rw = (rn+1)/2, rw/2
    for triangle, normal, height, frame in _cell_faces(vertices):
        y, weight = _face_outside_ball(triangle, radius, surface_order, normal, height, frame)
        result["face_points"].append(y)
        result["face_weights"].append(weight)
        result["face_normals"].append(np.broadcast_to(normal, y.shape))
        if height == 0:
            continue
        distance = np.linalg.norm(y, axis=1)
        direction = y/distance[:, None]
        solid_angle = height*weight/distance**3
        t = radius+(distance-radius)[:, None]*rn
        result["volume_points"].append((t[..., None]*direction[:, None]).reshape(-1, 3))
        result["volume_weights"].append((solid_angle[:, None]*(distance-radius)[:, None]*rw*t**2).ravel())
        result["surface_points"].append(radius*direction)
        result["surface_weights"].append(radius**2*solid_angle)
    return {name: np.concatenate(values) if values else np.empty(
                (0, 3) if name.endswith(("points", "normals")) else (0,))
            for name, values in result.items()}
