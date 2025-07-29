from itertools import count
import numpy as np

# TODO: Add the 2D version of this: distance to a line segment.
# TODO: Signed distance to plane (NaN if outside prism)

## Note: Distance from triangle to cloud of points
## Follows the open source embree (ray-shading) code (assumed fast). This is the numpy /
## python version of their code. See the diagram here for how the spatial division works:
## https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d


def points_in_simplex2D(p, a, b, c):
    """
    p - numpy array of points in 3D
    a, b, c - triangle points (numpy 1x2 arrays)

    returns:
        numpy array of truth values for each of the points in p (is this point in the triangle)
    """

    v0 = (c - a).reshape(1, 2)
    v1 = (b - a).reshape(1, 2)
    v2 = (p - a).reshape(-1, 2)

    # Compute dot products
    def dot(v1, v2):
        d = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
        return d

    dot00 = dot(v0, v0)
    dot01 = dot(v0, v1)
    dot02 = dot(v0, v2)
    dot11 = dot(v1, v1)
    dot12 = dot(v1, v2)

    # Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    w = 1 - u - v

    return np.logical_and((u >= 0), np.logical_and((v >= 0), (w >= 0)))


def points_in_simplex3D(p, a, b, c, d):

    vap = (p - a).reshape(-1, 3)
    vbp = (p - b).reshape(-1, 3)

    vab = (b - a).reshape(-1, 3)
    vac = (c - a).reshape(-1, 3)
    vad = (d - a).reshape(-1, 3)

    vbc = (c - b).reshape(-1, 3)
    vbd = (d - b).reshape(-1, 3)

    # ScTP computes the scalar triple product a.(bxc):

    def ScTP(a, b, c):
        c0 = b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
        c1 = b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2]
        c2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
        return a[:, 0] * c0 + a[:, 1] * c1 + a[:, 2] * c2

    va6 = ScTP(vbp, vbd, vbc)
    vb6 = ScTP(vap, vac, vad)
    vc6 = ScTP(vap, vad, vab)
    vd6 = ScTP(vap, vab, vac)
    v6 = 1 / ScTP(vab, vac, vad)

    u = va6 * v6
    v = vb6 * v6
    w = vc6 * v6
    t = vd6 * v6

    return np.logical_and(
        np.logical_and((u >= 0), (t >= 0)), np.logical_and((v >= 0), (w >= 0))
    )


def distance_pointcloud_triangle(p, a, b, c):
    """
    p - numpy array of points in 3D
    a, b, c - triangle points (numpy 1x3 arrays)

    returns:
        numpy array of distances from each of the points to the nearest point within the triangle (0 if in the plane, within the triangle)
    """

    ab = (b - a).reshape(1, 3)
    ac = (c - a).reshape(1, 3)
    bc = (c - b).reshape(1, 3)

    ap = p - a
    bp = p - b
    cp = p - c

    def dot(v1, v2):
        d = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1] + v1[:, 2] * v2[:, 2]
        return d

    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = 1 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    # There are multiple branches so we will have to mask to find those.
    # The first / general case is the perpendicular distance to the plane
    # of the triangle.

    pt = a + v.reshape(-1, 1) * ab + w.reshape(-1, 1) * ac

    # If outside the prism defined by the triangle extruded along its normal,
    # over-write the near point appropriately

    ## Region 1
    mask = np.logical_and(d1 < 0, d2 < 0)
    pt[mask] = a

    ## Region 2
    mask = np.logical_and(d3 > 0, d4 <= d3)
    pt[mask] = b

    ## Region 3
    mask = np.logical_and(d6 >= 0, d5 <= d6)
    pt[mask] = c

    ## Region 4
    mask = np.logical_and(np.logical_and(vc <= 0, d1 >= 0), d3 <= 0)
    v = d1 / (d1 - d3)
    if np.count_nonzero(mask):
        pt[mask] = a + v[mask] * ab

    ## Region 5
    mask = np.logical_and(np.logical_and(vb <= 0, d2 >= 0), d6 <= 0)
    v = d2 / (d2 - d6)
    if np.count_nonzero(mask):
        pt[mask] = a + v[mask] * ac

    ## Region 6
    mask = np.logical_and(np.logical_and(va <= 0, (d4 - d3) >= 0), (d5 - d6) >= 0)
    v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if np.count_nonzero(mask):
        pt[mask] = b + v[mask] * bc

    d = np.sqrt(dot(p - pt, p - pt))

    return d


def distance_pointcloud_linesegment(p, a, b):
    """
    p - numpy array of points
    a, b - line-segment points (numpy 1xdim arrays)

    returns:
        numpy array of distances from each of the points to the nearest point within the triangle (0 if in the plane, within the triangle)
    """

    dim = p.shape[1]

    ab = (b - a).reshape(1, dim)
    ap = p - a
    bp = p - b

    if dim == 3:

        def dot(v1, v2):
            d = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1] + v1[:, 2] * v2[:, 2]
            return d

    else:

        def dot(v1, v2):
            d = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
            return d

    ab_norm = np.sqrt(dot(ab, ab))
    adotp = dot(ab, ap) / ab_norm

    P = adotp / ab_norm

    # Three different cases:
    #     1: P < 0 return distance p to a
    #     2: P > 1 return distance p to b
    #     3: 0 <=  P <= 1 return perpendicular distance

    # First, perpendicular distance for every point (case 3)
    d = np.sqrt(dot(ap, ap) - adotp**2)

    # Over-write these cases
    mask = P < 0
    d[mask] = np.sqrt(dot(ap[mask], ap[mask]))

    mask = P > 1
    d[mask] = np.sqrt(dot(bp[mask], bp[mask]))

    return d
