"""Geometry for the rotated-boundary-conditions figure.

The cetz skill's rule: geometry computation happens in Python and Typst just
draws. Doing the triangulation here rather than in the figure is what stops
nodes being left out of the mesh -- an earlier version connected nodes by a
distance threshold and silently missed several.

Emits the schema in the skill's `underworld-bridge.md`:

    {"vertices": [[x, y], ...],
     "triangles": [[i, j, k], ...],
     "surface": [i, ...],              indices of the constrained nodes
     "frames": [{"p": [x, y], "n": [nx, ny], "t": [tx, ty]}, ...],
     "curve": [[x, y], ...]}           the surface, finely sampled

Run:  python3 generate-rotated-basis.py
"""
import json
import pathlib

import numpy as np
from scipy.spatial import Delaunay

OUT = pathlib.Path(__file__).with_name("rotated-basis-data.json")

X0, X1 = -3.5, 3.5
BASE = -2.9
NX = 9          # columns of nodes
NY = 4          # rows, surface included


def surface_y(x):
    """A free surface: rises on the left, falls on the right, with an
    inflection between, so the normal swings through a wide range and the
    curvature changes sign. No global rotation straightens this out, which is
    the reason the figure exists."""
    x = np.asarray(x, dtype=float)
    return (0.95 * np.exp(-(((x + 1.75) / 1.30) ** 2))
            - 0.80 * np.exp(-(((x - 1.70) / 1.15) ** 2))
            + 0.75)


def surface_slope(x, eps=1.0e-4):
    return (surface_y(x + eps) - surface_y(x - eps)) / (2 * eps)


# Nodes on a grid warped to sit under the surface. Columns are staggered on
# alternate rows so the Delaunay triangulation comes out as triangles rather
# than as near-degenerate right angles on a perfect lattice.
xs = np.linspace(X0, X1, NX)
pts = []
surface_idx = []
for row in range(NY):
    frac = row / (NY - 1)                      # 0 at the base, 1 at the surface
    offset = 0.0 if row % 2 == 0 else 0.5 * (xs[1] - xs[0])
    cols = xs + offset
    if row == NY - 1:
        cols = xs                              # surface row unstaggered
    for x in cols:
        if x < X0 - 1e-9 or x > X1 + 1e-9:
            continue
        top = float(surface_y(x))
        y = BASE + (top - BASE) * frac
        if row == NY - 1:
            surface_idx.append(len(pts))
        pts.append([float(x), float(y)])

pts = np.array(pts)
tri = Delaunay(pts)

# Drop the slivers Delaunay leaves along a non-convex top edge: any triangle
# whose centroid sits above the surface is outside the domain.
keep = []
for simplex in tri.simplices:
    c = pts[simplex].mean(axis=0)
    if c[1] <= float(surface_y(c[0])) + 1.0e-9:
        keep.append([int(i) for i in simplex])

frames = []
for i in surface_idx:
    x = pts[i][0]
    m = float(surface_slope(x))
    n = np.array([-m, 1.0])
    n /= np.linalg.norm(n)
    t = np.array([1.0, m])
    t /= np.linalg.norm(t)
    frames.append({"p": [float(pts[i][0]), float(pts[i][1])],
                   "n": [float(n[0]), float(n[1])],
                   "t": [float(t[0]), float(t[1])]})

curve_x = np.linspace(X0, X1, 121)
data = {
    "vertices": [[float(a), float(b)] for a, b in pts],
    "triangles": keep,
    "surface": [int(i) for i in surface_idx],
    "frames": frames,
    "curve": [[float(a), float(b)] for a, b in zip(curve_x, surface_y(curve_x))],
}
OUT.write_text(json.dumps(data, indent=1))
print("wrote %s: %d vertices, %d triangles, %d surface nodes"
      % (OUT.name, len(data["vertices"]), len(data["triangles"]),
         len(data["surface"])))
