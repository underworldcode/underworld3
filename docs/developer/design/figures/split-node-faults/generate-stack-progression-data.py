"""Geometry for the stack-on progression figure (cetz draws, Python
computes).

Five panels from deterministic point sets over [0,2] x [0,1.2] with a
slanted fault segment: (a) the static base mesh with the fault MANIFOLD
overlaid (geometry only — not in the mesh); (b,c) two adapt-on-top
layers, points graded toward the manifold; (d) the cut — vertices placed
ON the segment so the fault becomes a conforming facet chain; (e) the
split, exploded for display exactly as in split-anatomy.
"""
import json
import os

import numpy as np
from scipy.spatial import Delaunay

A = np.array([0.50, 0.32])
B = np.array([1.50, 0.66])
T = (B - A) / np.linalg.norm(B - A)
N = np.array([-T[1], T[0]])
LEN = float(np.linalg.norm(B - A))


def seg_dist(p):
    s = np.clip((p - A) @ T, 0.0, LEN)
    return float(np.linalg.norm(p - (A + s * T)))


def grid(h):
    xs = np.arange(0.0, 2.0 + 1e-9, h)
    ys = np.arange(0.0, 1.0 + 1e-9, h)
    return np.array([[x, y] for x in xs for y in ys])


def band(h, width):
    """Refinement points near the fault, deliberately NOT conforming to it:
    an axis-aligned lattice with deterministic jitter, restricted to the
    distance band. A real adapted mesh refines TOWARD the line without
    containing it — the conforming chain must be visibly the CUT's
    contribution, not an accident of the refinement layout."""
    out = []
    xs = np.arange(0.06, 1.95, h)
    ys = np.arange(0.05, 0.96, h)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            q = np.array([
                x + 0.33 * h * np.sin(12.9898 * i + 78.233 * j),
                y + 0.33 * h * np.sin(39.3460 * i + 11.135 * j)])
            if (seg_dist(q) < width
                    and 0.03 < q[0] < 1.97 and 0.03 < q[1] < 0.97):
                out.append(q)
    return np.array(out)


def dedup(points, spacing):
    kept = []
    for p in points:
        if all(np.linalg.norm(p - q) > spacing for q in kept):
            kept.append(p)
    return np.array(kept)


def triangulate(points):
    tri = Delaunay(points)
    return points, tri.simplices


def pack(points, tris):
    return dict(coords=points.tolist(), tris=tris.tolist())


panels = {}

# (a) base: coarse uniform
base_pts = grid(0.25)
panels["base"] = pack(*triangulate(base_pts))

# (b) layer 1: base + a coarse band toward the manifold
l1 = dedup(np.vstack([grid(0.25), band(0.125, 0.22)]), 0.07)
panels["l1"] = pack(*triangulate(l1))

# (c) layer 2: tighter band on top
l2 = dedup(np.vstack([grid(0.25), band(0.125, 0.24), band(0.062, 0.10)]), 0.036)
panels["l2"] = pack(*triangulate(l2))

# (d) cut: clear a corridor around the segment, then place vertices ON it
keep = np.array([p for p in l2 if seg_dist(p) > 0.048])
on_seg = np.array([A + s * T for s in np.arange(0.0, LEN + 1e-9, 0.062)])
cut_pts = np.vstack([keep, on_seg])
cut_pts, cut_tris = triangulate(cut_pts)
panels["cut"] = pack(cut_pts, cut_tris)
chain = list(range(len(keep), len(cut_pts)))          # in s-order already

# (e) split: explode the Minus flank (side of -N), tips pinned
interior = chain[1:-1]
tips = [chain[0], chain[-1]]
side = []
for t in cut_tris:
    cen = cut_pts[t].mean(axis=0)
    s = np.clip((cen - A) @ T, 0.0, LEN)
    touches = any(v in chain for v in t)
    inside = 0.0 < (cen - A) @ T < LEN
    side.append(-1 if ((cen - (A + s * T)) @ N < 0 and touches and inside)
                else +1)
DELTA = 0.06
exploded = [list(p) for p in cut_pts]
replicas = {}
for v in interior:
    replicas[v] = len(exploded)
    exploded.append(list(cut_pts[v] - DELTA * N))
moved_tris, flank = [], set()
for t, s in zip(cut_tris.tolist(), side):
    if s < 0:
        moved_tris.append([replicas.get(v, v) for v in t])
        flank.update(v for v in t if v not in chain)
    else:
        moved_tris.append(list(t))
for v in flank:
    exploded[v] = list(np.array(exploded[v]) - DELTA * N)
panels["split"] = dict(coords=exploded,
                       tris=[list(t) for t in moved_tris], side=side)

out = dict(panels=panels, chain=chain, tips=tips, interior=interior,
           replicas={str(k): v for k, v in replicas.items()},
           fault=[A.tolist(), B.tolist()], cut_side=side)
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "stack-progression-data.json"), "w") as f:
    json.dump(out, f)
for name, p in panels.items():
    print(f"{name}: {len(p['coords'])} pts, {len(p['tris'])} tris")
