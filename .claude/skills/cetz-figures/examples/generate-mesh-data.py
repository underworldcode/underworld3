"""
Generate a deterministic 2D triangular mesh for the cetz demo figure.

The central "highlight" triangle is forced into the Delaunay output by
seeding its vertices first and rejecting any other interior point that
would fall inside its circumcircle (Delaunay-empty-circle property).

Output schema (identical to what a future underworld3 export will emit):

    {
      "vertices":  [[x, y], ...],      # N x 2
      "triangles": [[i, j, k], ...],   # M x 3 indices
      "highlight": int                  # index into triangles
    }

Run: python3 generate-mesh-data.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

SEED = 3
BOX = 2.0                 # point cloud extent — further out to ensure
                          # enough lattice points survive for cut edges on
                          # every side at the coarse spacing below.
SPACING = 1.25            # target edge length of the FE-like lattice —
                          # matched to the central triangle's average edge
                          # (~1.29) so it reads as a typical cell.
JITTER = 0.15             # fraction of SPACING — small randomness so the
                          # mesh reads as a real FE discretisation rather
                          # than a perfect lattice

# Central triangle — deliberately asymmetric.
TRI = np.array([
    [-0.60, -0.38],
    [ 0.58, -0.45],        # v2 pulled in slightly from (0.62, -0.48)
                           # so it isn't flush with the frame.
    [ 0.32,  0.70],        # v3 shifted up-right so CA is the longest edge.
])


def circumcircle(a, b, c):
    """Return (cx, cy, r) of the circumscribed circle of triangle abc."""
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


cx, cy, r = circumcircle(*TRI)
SAFETY = 0.05             # keep candidates a little further out than the circle

rng = np.random.default_rng(SEED)

# Jittered equilateral-triangular lattice — visually uniform triangle
# sizes, like an FE mesh from gmsh with a single size field.  Every other
# row is offset by half a spacing to build equilateral triangles.
dy = SPACING * np.sqrt(3) / 2
lattice = []
ys = np.arange(-BOX, BOX + dy, dy)
for row, y in enumerate(ys):
    offset = SPACING / 2 if row % 2 else 0.0
    for x in np.arange(-BOX + offset, BOX + 1e-9, SPACING):
        lattice.append([x, y])
lattice = np.array(lattice)

# Add a small jitter (after placing) so the mesh doesn't read as a
# perfect lattice — closer to a real FE mesh in appearance.
lattice += rng.uniform(-JITTER * SPACING, JITTER * SPACING,
                      size=lattice.shape)

# Keep only lattice points outside the central triangle's circumcircle.
dist = np.hypot(lattice[:, 0] - cx, lattice[:, 1] - cy)
lattice = lattice[dist > r + SAFETY]

points = np.vstack([TRI, lattice])
tri = Delaunay(points)

# Locate the central triangle among the simplices (any vertex order),
# then canonicalise its row to (0, 1, 2) so the Typst figure can rely
# on triangle[highlight].at(k) referring to the kth original TRI vertex.
target = {0, 1, 2}
highlight = next(i for i, s in enumerate(tri.simplices) if set(s) == target)
simplices = tri.simplices.copy()
simplices[highlight] = [0, 1, 2]

data = {
    "vertices":  [[round(float(x), 4), round(float(y), 4)] for x, y in points],
    "triangles": [[int(i), int(j), int(k)] for i, j, k in simplices],
    "highlight": int(highlight),
}

out = Path(__file__).with_name("mesh-data.json")
out.write_text(json.dumps(data, indent=2))

print(f"wrote {out.name}: "
      f"{len(data['vertices'])} vertices, "
      f"{len(data['triangles'])} triangles, "
      f"highlight={highlight}")
