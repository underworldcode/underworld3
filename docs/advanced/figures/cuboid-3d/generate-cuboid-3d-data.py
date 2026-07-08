"""
Data for the 3D cuboid "boundary labels" sketch.

Isometric projection of a rectangular box with the standard UW3 face
labels (Upper / Lower / Left / Right / Front / Back).  Each face gets
an arrow from outside the box toward its centroid, and a text label
at the arrow tail.  The cetz figure just renders what this script
emits — no 3D math in Typst.

Output schema:

  {
    "vertices_2d": [[sx, sy], ... 8 entries],
    "faces": {
      "<name>": {
        "vertex_indices": [i, j, k, l],      # CCW when looking at face
        "visible":       bool,                # w.r.t. isometric viewer
        "arrow_from":    [sx, sy],            # tail, outside the box
        "arrow_to":      [sx, sy],            # tip, at face centroid
        "label_pos":     [sx, sy],            # anchor for label text
        "label_anchor":  "west"|"east"|"north"|"south"|...,
      }, ...
    },
    "faces_back_to_front": [<name>, ...],    # painter's-algorithm order
    "edges": [
      {"vertices": [i, j], "visible": bool}, ...
    ]
  }
"""
import json
import math
from pathlib import Path

# Cuboid half-extents — deliberately non-cubic so the axes read differently.
X_HALF = 1.0
Y_HALF = 0.75
Z_HALF = 0.5

# Isometric projection.
# Axes: x → right-down, y → left-down, z → up
# Point (x, y, z) → 2D (sx, sy):
#   sx = (x - y) * cos(30°)
#   sy = z - (x + y) * sin(30°)
COS30 = math.cos(math.radians(30.0))
SIN30 = 0.5


def project(p):
    x, y, z = p
    return ((x - y) * COS30, z - (x + y) * SIN30)


def project_direction(nx, ny, nz):
    """Project a 3D direction vector onto the screen plane."""
    return ((nx - ny) * COS30, nz - (nx + ny) * SIN30)


# ── 8 vertices, indexed by (xs, ys, zs) with xs, ys, zs ∈ {−1, +1} ───
def vertex(xs, ys, zs):
    return (xs * X_HALF, ys * Y_HALF, zs * Z_HALF)


V = [
    vertex(-1, -1, -1),  # 0
    vertex(+1, -1, -1),  # 1
    vertex(+1, +1, -1),  # 2
    vertex(-1, +1, -1),  # 3
    vertex(-1, -1, +1),  # 4
    vertex(+1, -1, +1),  # 5
    vertex(+1, +1, +1),  # 6
    vertex(-1, +1, +1),  # 7
]

# ── Faces: vertex indices, outward normals, and UW3 label names ──────
# In this convention +x → Right, -x → Left, +y → Back, -y → Front,
# +z → Upper, -z → Lower.
FACES = {
    "Upper": {"vertex_indices": [4, 5, 6, 7], "normal": (0, 0, +1)},
    "Lower": {"vertex_indices": [0, 1, 2, 3], "normal": (0, 0, -1)},
    "Front": {"vertex_indices": [0, 1, 5, 4], "normal": (0, -1, 0)},
    "Back":  {"vertex_indices": [2, 3, 7, 6], "normal": (0, +1, 0)},
    "Left":  {"vertex_indices": [0, 3, 7, 4], "normal": (-1, 0, 0)},
    "Right": {"vertex_indices": [1, 2, 6, 5], "normal": (+1, 0, 0)},
}

# ── Edges and which two faces each one borders ────────────────────────
EDGES = [
    # bottom square
    ([0, 1], ("Lower", "Front")),
    ([1, 2], ("Lower", "Right")),
    ([2, 3], ("Lower", "Back")),
    ([3, 0], ("Lower", "Left")),
    # top square
    ([4, 5], ("Upper", "Front")),
    ([5, 6], ("Upper", "Right")),
    ([6, 7], ("Upper", "Back")),
    ([7, 4], ("Upper", "Left")),
    # verticals
    ([0, 4], ("Front", "Left")),
    ([1, 5], ("Front", "Right")),
    ([2, 6], ("Back",  "Right")),
    ([3, 7], ("Back",  "Left")),
]

# ── Visibility ────────────────────────────────────────────────────────
# With the projection above (+x → right-down, +y → left-down, +z → up),
# the viewer sits at the (+1, +1, +1) corner direction — upper-back-right.
# A face is visible iff its outward normal has a positive component along
# that viewer direction, i.e. iff nx + ny + nz > 0.
# ─────────────────────────────────────────────────────────────────────
def face_centroid_3d(name):
    idxs = FACES[name]["vertex_indices"]
    return tuple(sum(V[i][k] for i in idxs) / 4 for k in range(3))


def depth_from_viewer(p3d):
    """Larger = closer to the viewer (at +x+y+z direction)."""
    return p3d[0] + p3d[1] + p3d[2]


for name in FACES:
    c3d = face_centroid_3d(name)
    FACES[name]["centroid_3d"] = c3d
    nx, ny, nz = FACES[name]["normal"]
    FACES[name]["visible"] = (nx + ny + nz) > 0

# ── 2D projections ────────────────────────────────────────────────────
vertices_2d = [project(v) for v in V]

for name, face in FACES.items():
    c2d = project(face["centroid_3d"])
    face["centroid_2d"] = c2d
    # Outward-normal direction in 2D, for placing arrow + label outside box.
    nx, ny, nz = face["normal"]
    dir_2d = project_direction(nx, ny, nz)
    dlen = math.hypot(dir_2d[0], dir_2d[1])
    if dlen > 1e-9:
        dir_2d = (dir_2d[0] / dlen, dir_2d[1] / dlen)
    face["dir_2d"] = dir_2d

# ── Arrow tail, tip, and label position for each face ────────────────
# Arrow goes FROM outside (at distance TAIL_DIST along face's 2D outward
# normal) TO the face centroid.  Label sits just beyond the arrow tail.
TAIL_DIST  = 0.80
LABEL_DIST = 1.15

for name, face in FACES.items():
    dx, dy = face["dir_2d"]
    cx, cy = face["centroid_2d"]
    face["arrow_from"] = [cx + TAIL_DIST * dx,  cy + TAIL_DIST * dy]
    face["arrow_to"]   = [cx,                   cy]
    face["label_pos"]  = [cx + LABEL_DIST * dx, cy + LABEL_DIST * dy]
    # Pick a text anchor so the label sits past the tail, not on top of it.
    face["label_anchor"] = (
        "south" if dy >  0.5 else
        "north" if dy < -0.5 else
        "west"  if dx >  0.0 else
        "east"
    )

# ── Edge visibility: an edge is hidden if *both* its faces are hidden ─
edges_out = []
for verts, (face_a, face_b) in EDGES:
    visible = FACES[face_a]["visible"] or FACES[face_b]["visible"]
    edges_out.append({"vertices": verts, "visible": visible})

# ── Painter's-algorithm order for fills: farthest first ──────────────
# Farthest from viewer ↔ smallest depth_from_viewer ↔ smallest x+y+z.
faces_back_to_front = sorted(
    FACES.keys(), key=lambda n: depth_from_viewer(FACES[n]["centroid_3d"])
)

# ── Assemble JSON ─────────────────────────────────────────────────────
data = {
    "vertices_2d": [[round(x, 4), round(y, 4)] for x, y in vertices_2d],
    "faces": {
        name: {
            "vertex_indices": face["vertex_indices"],
            "visible":        face["visible"],
            "arrow_from":     [round(c, 4) for c in face["arrow_from"]],
            "arrow_to":       [round(c, 4) for c in face["arrow_to"]],
            "label_pos":      [round(c, 4) for c in face["label_pos"]],
            "label_anchor":   face["label_anchor"],
        }
        for name, face in FACES.items()
    },
    "faces_back_to_front": faces_back_to_front,
    "edges": edges_out,
}

out = Path(__file__).with_name("cuboid-3d-data.json")
out.write_text(json.dumps(data, indent=2))
print(f"wrote {out.name}: "
      f"{len(data['vertices_2d'])} vertices, "
      f"{len(data['faces'])} faces "
      f"({sum(1 for f in data['faces'].values() if f['visible'])} visible), "
      f"{len(data['edges'])} edges")
