"""Geometry for the split-anatomy figure (cetz draws, Python computes).

A small structured triangulation of [0,3] x [0,1] with a fault chain along
y = 0.5 from (0.5, 0.5) to (2.5, 0.5): 4 fault edges, 3 interior vertices
(duplicated by the split), 2 tips (shared). The "after" panel is EXPLODED
for display — the Minus side translated down — because the real copies are
geometrically coincident.
"""
import json
import os

H = 0.5
NX, NY = 6, 2
DELTA = 0.22          # display-only explosion offset

verts = {}
coords = []


def vid(i, j):
    if (i, j) not in verts:
        verts[(i, j)] = len(coords)
        coords.append([i * H, j * H])
    return verts[(i, j)]


tris = []
for i in range(NX):
    for j in range(NY):
        a, b = vid(i, j), vid(i + 1, j)
        c, d = vid(i + 1, j + 1), vid(i, j + 1)
        tris.append([a, b, c])
        tris.append([a, c, d])

chain = [vid(i, 1) for i in range(1, 6)]      # (0.5,0.5) .. (2.5,0.5)
tips = [chain[0], chain[-1]]
interior = chain[1:-1]

# Minus side = triangles whose centroid lies below the chain SEGMENT span;
# triangles below y=0.5 but outside the span stay welded (no fault there).
side = []
for t in tris:
    cx = sum(coords[v][0] for v in t) / 3.0
    cy = sum(coords[v][1] for v in t) / 3.0
    touches = any(v in chain for v in t)
    side.append(-1 if (cy < 0.5 and touches and 0.5 < cx < 2.5) else +1)

# Exploded positions: replicas of the interior chain vertices move down;
# to keep the Minus flank rigid, the row below them (y = 0) within the span
# moves too. Tips stay put — the lens pins there.
exploded = [list(c) for c in coords]
replicas = {}
for v in interior:
    replicas[v] = len(exploded)
    exploded.append([coords[v][0], coords[v][1] - DELTA])
moved_tris = []
flank = set()
for t, s in zip(tris, side):
    if s < 0:
        moved_tris.append([replicas.get(v, v) for v in t])
        flank.update(v for v in t if v not in chain)
    else:
        moved_tris.append(list(t))
for v in flank:
    exploded[v][1] -= DELTA

out = dict(coords=coords, exploded=exploded, tris=tris,
           moved_tris=moved_tris, side=side, chain=chain, tips=tips,
           interior=interior, replicas=replicas, delta=DELTA)
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "split-anatomy-data.json"), "w") as f:
    json.dump(out, f)
print("wrote split-anatomy-data.json:",
      f"{len(coords)} verts, {len(tris)} tris, chain {len(chain)},",
      f"replicas {len(replicas)}")
