"""
Generate data for the element-location demo figure (two-panel version).

Emits two JSON files representing the same algorithm in two cell shapes:

  * element-location-normal-data.json -- a "normal" asymmetric cell.
    The control-point KDTree returns the highlight, as does a naive
    centroid-only lookup (essentially -- the failure region is a tiny
    sliver of the cell).  This is the "works well" panel: the
    algorithm is doing its job, but in this geometry it would have
    worked even without the nudges.

  * element-location-sliver-data.json -- two opposing slivers sharing
    an edge incident to the highlight's acute tip.  Naive centroid-only
    lookup picks the WRONG cell across ~21% of the highlight.  The
    nudge is what makes the algorithm robust here.

Shared parameters (NUDGE, mesh density) so the two panels are visually
comparable side by side.  The figure-NUDGE is exaggerated to ~12% so
the control-point dots are visually distinct from the vertex dots;
real UW3 uses 1%, and the caption notes this.

Output schema (per file)::

    {
      "vertices":  [[x, y], ...],
      "triangles": [[i, j, k], ...],
      "highlight": int,
      "neighbour": int,
      "x_p":       [x, y],
      "nudge":     float,
    }

Run::

    pixi run python generate-element-location-data.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

SEED = 5
BOX = 2.0
SPACING = 1.20
JITTER = 0.15
NUDGE = 0.12
SAFETY = 0.05


def circumcircle(a, b, c):
    ax, ay = a; bx, by = b; cx_, cy_ = c
    d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy_)
          + (bx**2 + by**2) * (cy_ - ay)
          + (cx_**2 + cy_**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx_ - bx)
          + (bx**2 + by**2) * (ax - cx_)
          + (cx_**2 + cy_**2) * (bx - ax)) / d
    r = float(np.hypot(ux - ax, uy - ay))
    return float(ux), float(uy), r


def lattice_points(rng):
    dy = SPACING * np.sqrt(3) / 2
    pts = []
    ys = np.arange(-BOX, BOX + dy, dy)
    for row, y in enumerate(ys):
        offset = SPACING / 2 if row % 2 else 0.0
        for x in np.arange(-BOX + offset, BOX + 1e-9, SPACING):
            pts.append([x, y])
    pts = np.array(pts)
    pts += rng.uniform(-JITTER * SPACING, JITTER * SPACING, size=pts.shape)
    return pts


def make_normal():
    """Normal (non-sliver) cell -- previous figure's geometry."""
    TRI = np.array([
        [-0.85, -0.10],   # v1 -- acute tip on the left
        [ 0.60, -0.55],   # v2 -- lower-right shoulder
        [ 0.50,  0.55],   # v3 -- upper-right shoulder
    ])
    XP_FRAC = 0.28

    rng = np.random.default_rng(SEED)
    cx, cy, r = circumcircle(*TRI)
    lattice = lattice_points(rng)
    d = np.hypot(lattice[:, 0] - cx, lattice[:, 1] - cy)
    lattice = lattice[d > r + SAFETY]

    points = np.vstack([TRI, lattice])
    tri = Delaunay(points)

    target = {0, 1, 2}
    highlight = next(i for i, s in enumerate(tri.simplices) if set(s) == target)
    simplices = tri.simplices.copy()
    simplices[highlight] = [0, 1, 2]

    centroid = TRI.mean(axis=0)
    xp = TRI[0] + XP_FRAC * (centroid - TRI[0])

    def cell_centroid(idx):
        return points[simplices[idx]].mean(axis=0)

    neighbours_of_v1 = [i for i, s in enumerate(simplices)
                       if i != highlight and 0 in s]
    neighbour = min(neighbours_of_v1,
                   key=lambda i: np.linalg.norm(cell_centroid(i) - xp))

    return points, simplices.tolist(), highlight, neighbour, xp


def make_sliver():
    """Two opposing slivers -- naive centroid lookup structurally fails."""
    V1 = np.array([-0.85,  0.00])
    V2 = np.array([ 0.25,  0.15])
    V3 = np.array([ 0.25, -0.15])
    V4 = np.array([-0.80,  0.30])
    XP_FRAC = 0.30
    central = np.vstack([V1, V2, V3, V4])

    rng = np.random.default_rng(SEED)
    hi_cx, hi_cy, hi_r = circumcircle(V1, V2, V3)
    nb_cx, nb_cy, nb_r = circumcircle(V1, V2, V4)
    lattice = lattice_points(rng)
    d_hi = np.hypot(lattice[:, 0] - hi_cx, lattice[:, 1] - hi_cy)
    d_nb = np.hypot(lattice[:, 0] - nb_cx, lattice[:, 1] - nb_cy)
    lattice = lattice[(d_hi > hi_r + SAFETY) & (d_nb > nb_r + SAFETY)]

    points = np.vstack([central, lattice])
    tri = Delaunay(points)
    simplices = tri.simplices.tolist()

    def find_simplex(target):
        for i, s in enumerate(simplices):
            if set(s) == target:
                return i
        return None

    hi_idx = find_simplex({0, 1, 2})
    nb_idx = find_simplex({0, 1, 3})

    if hi_idx is None or nb_idx is None:
        # Delaunay flipped the central diagonal -- force the two slivers.
        simplices = [s for s in simplices
                    if not set(s).issubset({0, 1, 2, 3})]
        simplices.append([0, 1, 2])
        simplices.append([0, 1, 3])
        hi_idx = len(simplices) - 2
        nb_idx = len(simplices) - 1
    else:
        simplices[hi_idx] = [0, 1, 2]
        simplices[nb_idx] = [0, 1, 3]

    centroid = central[:3].mean(axis=0)
    xp = V1 + XP_FRAC * (centroid - V1)

    return points, simplices, hi_idx, nb_idx, xp


def save(name, points, simplices, hi_idx, nb_idx, xp):
    H = simplices[hi_idx]
    N = simplices[nb_idx]
    centroid = points[H].mean(axis=0)
    nc = points[N].mean(axis=0)
    c1 = points[H[0]] + NUDGE * (centroid - points[H[0]])

    d_c  = float(np.linalg.norm(centroid - xp))
    d_nc = float(np.linalg.norm(nc - xp))
    d_c1 = float(np.linalg.norm(c1 - xp))
    # The nudge MUST win against both alternatives.
    assert d_c1 < d_nc, f"{name}: nudge must beat neighbour centroid"
    assert d_c1 < d_c,  f"{name}: nudge must beat highlight centroid"

    data = {
        "vertices":  [[round(float(x), 4), round(float(y), 4)] for x, y in points],
        "triangles": [[int(i), int(j), int(k)] for i, j, k in simplices],
        "highlight": int(hi_idx),
        "neighbour": int(nb_idx),
        "x_p":       [round(float(xp[0]), 4), round(float(xp[1]), 4)],
        "nudge":     NUDGE,
    }
    out = Path(__file__).with_name(f"element-location-{name}-data.json")
    out.write_text(json.dumps(data, indent=2))
    print(f"{name:7}: {len(points)} verts, {len(simplices)} tris, "
          f"d(xp,c)={d_c:.3f}, d(xp,c')={d_nc:.3f}, d(xp,c1)={d_c1:.3f}, "
          f"naive {'fails' if d_nc < d_c else 'OK'} by "
          f"margin={abs(d_c - d_nc):.3f}")


save("normal", *make_normal())
save("sliver", *make_sliver())
