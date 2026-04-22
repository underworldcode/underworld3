"""
Generate a 2D triangular mesh partitioned into two domains for the
parallel particle migration blog post figure.

The mesh is larger than the element-level demo, with more equant
triangles. Two domains (A and B) are defined by a partition line.
A test particle is placed so it is closer to domain A's centroid
but actually inside domain B — illustrating why centroid-nearest
is insufficient for parallel migration.

Output schema:
    {
      "vertices":  [[x, y], ...],
      "triangles": [[i, j, k], ...],
      "domain_a":  [tri_index, ...],    # triangle indices in domain A
      "domain_b":  [tri_index, ...],    # triangle indices in domain B
      "centroid_a": [x, y],
      "centroid_b": [x, y],
      "test_point": [x, y]
    }

Run: python3 generate-domain-demo-data.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

SEED = 42
BOX = 2.5
SPACING = 0.28
JITTER = 0.10

rng = np.random.default_rng(SEED)

# Jittered equilateral-triangular lattice
dy = SPACING * np.sqrt(3) / 2
lattice = []
ys = np.arange(-BOX, BOX + dy, dy)
for row, y in enumerate(ys):
    offset = SPACING / 2 if row % 2 else 0.0
    for x in np.arange(-BOX + offset, BOX + 1e-9, SPACING):
        lattice.append([x, y])
lattice = np.array(lattice)

# Jitter
lattice += rng.uniform(-JITTER * SPACING, JITTER * SPACING, size=lattice.shape)

# Triangulate
tri = Delaunay(lattice)

# Compute triangle centroids
centroids = lattice[tri.simplices].mean(axis=1)

# Partition into four real domains, all with centroids:
# - Domain A (blue): L-shaped, left and lower
# - Domain B (red): upper-right plus top strip (contiguous)
# - Domain C (green): left strip
# - Domain D (yellow): bottom strip plus right edge of A's boundary

def assign_domain(cx, cy):
    """Assign a triangle centroid to a domain."""
    # Domain C (green): far left
    if cx <= -1.2:
        return "C"
    # Domain D (yellow): bottom region, plus right boundary strip
    if cy <= -0.75 or (cx > 1.4 and cy <= 0.1):
        return "D"
    # Domain B (red): upper-right quadrant plus entire top strip
    if (cx > 0.1 and cy > 0.1) or cy > 1.4:
        return "B"
    # Domain A (blue): everything else (L-shaped, central-left and lower)
    return "A"

domains = {}
for label in ["A", "B", "C", "D"]:
    domains[label] = []

for i, (cx, cy) in enumerate(centroids):
    label = assign_domain(cx, cy)
    domains[label].append(i)

# Compute domain centroids
VIEW = 1.8

# Compute domain centroids using only triangles visible in the view,
# so the dots match what the reader sees in the clipped figure.
domain_centroids = {}
for label, indices in domains.items():
    visible = [i for i in indices
               if abs(centroids[i, 0]) <= VIEW and abs(centroids[i, 1]) <= VIEW]
    if visible:
        domain_centroids[label] = centroids[visible].mean(axis=0)
    elif indices:
        domain_centroids[label] = centroids[indices].mean(axis=0)

centroid_a = domain_centroids["A"]
centroid_b = domain_centroids["B"]

# Place test particle: inside domain B but closer to centroid A.
test_point = [0.30, 0.30]

dist_a = np.hypot(test_point[0] - centroid_a[0], test_point[1] - centroid_a[1])
dist_b = np.hypot(test_point[0] - centroid_b[0], test_point[1] - centroid_b[1])

print(f"Domain A centroid: ({centroid_a[0]:.3f}, {centroid_a[1]:.3f})")
print(f"Domain B centroid: ({centroid_b[0]:.3f}, {centroid_b[1]:.3f})")
print(f"Test point: ({test_point[0]:.3f}, {test_point[1]:.3f})")
print(f"Distance to A: {dist_a:.3f}, Distance to B: {dist_b:.3f}")
print(f"Closer to A: {dist_a < dist_b}")

if dist_a >= dist_b:
    for ty in np.arange(0.12, 0.8, 0.02):
        for tx in np.arange(0.12, 0.8, 0.02):
            if assign_domain(tx, ty) != "B":
                continue
            da = np.hypot(tx - centroid_a[0], ty - centroid_a[1])
            db = np.hypot(tx - centroid_b[0], ty - centroid_b[1])
            if da < db:
                test_point = [tx, ty]
                print(f"Adjusted to ({tx:.3f}, {ty:.3f}), dist_a={da:.3f}, dist_b={db:.3f}")
                break
        else:
            continue
        break

# All domain centroids
domain_centroids_out = {}
for label, indices in domains.items():
    if indices:
        c = domain_centroids[label]
        domain_centroids_out[label] = [round(float(c[0]), 4), round(float(c[1]), 4)]

data = {
    "vertices": [[round(float(x), 4), round(float(y), 4)] for x, y in lattice],
    "triangles": [[int(i), int(j), int(k)] for i, j, k in tri.simplices],
    "domains": {label: [int(i) for i in indices] for label, indices in domains.items()},
    "domain_centroids": domain_centroids_out,
    "centroid_a": [round(float(centroid_a[0]), 4), round(float(centroid_a[1]), 4)],
    "centroid_b": [round(float(centroid_b[0]), 4), round(float(centroid_b[1]), 4)],
    "test_point": [round(float(test_point[0]), 4), round(float(test_point[1]), 4)],
    "view": VIEW,
}

out = Path(__file__).with_name("domain-demo-data.json")
out.write_text(json.dumps(data, indent=2))

print(f"\nWrote {out.name}: "
      f"{len(data['vertices'])} vertices, "
      f"{len(data['triangles'])} triangles")
for label, indices in domains.items():
    c = domain_centroids[label]
    print(f"  Domain {label}: {len(indices)} triangles, centroid ({c[0]:.3f}, {c[1]:.3f})")
