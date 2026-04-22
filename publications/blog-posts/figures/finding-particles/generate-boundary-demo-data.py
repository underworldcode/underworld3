"""
Generate boundary-zone classification data for the claim hierarchy figure.

For each domain, classify triangles into three zones based on distance
from the domain boundary:
  - "inside":   clearly inside (distance > threshold from boundary)
  - "boundary": near the boundary (ambiguous ownership zone)
  - "outside":  clearly outside

Uses the same mesh and domain partition as domain-demo-data.json.

Output: boundary-demo-data.json with zone classifications per domain.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

# Load the existing mesh and domain data
src = json.load(Path(__file__).with_name("domain-demo-data.json").open())
vertices = np.array(src["vertices"])
triangles = np.array(src["triangles"])
domains = src["domains"]
VIEW = src["view"]

# Compute triangle centroids
centroids = vertices[triangles].mean(axis=1)

# For each domain, find boundary triangles (triangles in this domain
# that share an edge with a triangle in a different domain).
# Approximate: a triangle is "near boundary" if its centroid is within
# a threshold distance of any triangle centroid in a different domain.

BOUNDARY_THRESHOLD = 0.35  # distance from nearest foreign triangle
                           # to count as "near boundary" (~2 cells)

def classify_zones(domain_label):
    """Classify all triangles relative to one domain."""
    own_indices = set(domains[domain_label])
    other_indices = []
    for label, indices in domains.items():
        if label != domain_label:
            other_indices.extend(indices)

    # KDTree of centroids of triangles NOT in this domain
    other_centroids = centroids[other_indices]
    tree = KDTree(other_centroids)

    zones = {}  # tri_index -> "inside" | "boundary" | "outside"

    for i in range(len(triangles)):
        c = centroids[i]
        dist_to_boundary, _ = tree.query(c)

        if i in own_indices:
            if dist_to_boundary < BOUNDARY_THRESHOLD:
                zones[i] = "boundary"
            else:
                zones[i] = "inside"
        else:
            if dist_to_boundary < BOUNDARY_THRESHOLD:
                # This triangle is outside but near the boundary of
                # the domain — it's in the "maybe" zone from the
                # perspective of domain_label
                zones[i] = "boundary"
            else:
                zones[i] = "outside"

    return zones

# Wait — for outside triangles, we want distance to the nearest triangle
# IN the domain, not distance to the nearest foreign triangle.
# Let me redo this properly.

def classify_zones_v2(domain_label):
    """Classify all triangles relative to one domain.

    For triangles inside the domain: distance to nearest foreign triangle.
    For triangles outside: distance to nearest triangle in the domain.
    """
    own_indices = list(domains[domain_label])
    own_set = set(own_indices)

    own_centroids = centroids[own_indices]
    own_tree = KDTree(own_centroids)

    other_indices = []
    for label, indices in domains.items():
        if label != domain_label:
            other_indices.extend(indices)
    other_centroids = centroids[other_indices]
    other_tree = KDTree(other_centroids)

    zones = {}

    for i in range(len(triangles)):
        if i in own_set:
            # Inside domain: how far from the border?
            dist_to_foreign, _ = other_tree.query(centroids[i])
            if dist_to_foreign < BOUNDARY_THRESHOLD:
                zones[i] = "boundary"
            else:
                zones[i] = "inside"
        else:
            # Outside domain: how far from the domain?
            dist_to_own, _ = own_tree.query(centroids[i])
            if dist_to_own < BOUNDARY_THRESHOLD:
                zones[i] = "boundary"
            else:
                zones[i] = "outside"

    return zones

zones_a = classify_zones_v2("A")
zones_b = classify_zones_v2("B")

# Count zones for reporting
for label, zones in [("A", zones_a), ("B", zones_b)]:
    counts = {"inside": 0, "boundary": 0, "outside": 0}
    for z in zones.values():
        counts[z] += 1
    print(f"Domain {label}: {counts}")

# Compute domain boundary edges.
# An edge is a domain boundary if the two triangles sharing it belong
# to different domains. We find these by building an edge-to-triangle map.

tri_domain = {}
for label, indices in domains.items():
    for i in indices:
        tri_domain[i] = label

def edge_key(a, b):
    return (min(a, b), max(a, b))

edge_tris = {}  # edge -> list of triangle indices
for i, tri in enumerate(triangles):
    for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]:
        ek = edge_key(e[0], e[1])
        edge_tris.setdefault(ek, []).append(i)

boundary_edges = []
boundary_control_points = []  # list of {"inside": [x,y], "outside": [x,y]}

CONTROL_OFFSET = 0.04  # visible offset for the figure (much larger than
                        # the 1e-8 used in the actual code, for visibility)

for ek, tris_list in edge_tris.items():
    if len(tris_list) == 2:
        d0 = tri_domain.get(tris_list[0])
        d1 = tri_domain.get(tris_list[1])
        if d0 != d1:
            boundary_edges.append([int(ek[0]), int(ek[1]), d0, d1])

            # Compute control point pair for this boundary edge.
            # The "inside" point is offset toward the cell centroid of
            # the triangle in the domain we're viewing from. Since this
            # figure is domain-agnostic, we store both sides.
            p0 = vertices[ek[0]]
            p1 = vertices[ek[1]]
            face_mid = 0.5 * (p0 + p1)

            # Face normal (2D: rotate edge vector 90 degrees)
            edge_vec = p1 - p0
            normal = np.array([-edge_vec[1], edge_vec[0]])
            normal = normal / np.linalg.norm(normal)

            # Determine which side is which domain by checking which
            # triangle centroid is on which side of the edge
            c0 = centroids[tris_list[0]]
            sign = np.sign(np.dot(normal, c0 - face_mid))
            # sign > 0 means tri_0 is on the +normal side

            # Store control points: one on each side of the face
            cp_plus = face_mid + CONTROL_OFFSET * normal
            cp_minus = face_mid - CONTROL_OFFSET * normal

            # Label by which domain each side belongs to
            boundary_control_points.append({
                "pos_plus": [round(float(cp_plus[0]), 4), round(float(cp_plus[1]), 4)],
                "pos_minus": [round(float(cp_minus[0]), 4), round(float(cp_minus[1]), 4)],
                "domain_plus": d0 if sign > 0 else d1,
                "domain_minus": d1 if sign > 0 else d0,
            })
    elif len(tris_list) == 1:
        # Mesh boundary edge — skip (not a domain boundary)
        pass

print(f"Domain boundary edges: {len(boundary_edges)}")
print(f"Boundary control point pairs: {len(boundary_control_points)}")

data = {
    "vertices": src["vertices"],
    "triangles": src["triangles"],
    "domains": src["domains"],
    "domain_centroids": src["domain_centroids"],
    "test_point": src["test_point"],
    "view": VIEW,
    "zones_a": {str(k): v for k, v in zones_a.items()},
    "zones_b": {str(k): v for k, v in zones_b.items()},
    "boundary_edges": boundary_edges,
    "boundary_control_points": boundary_control_points,
    "boundary_threshold": BOUNDARY_THRESHOLD,
}

# Test points:
# x_p (purple): contested, in the boundary zone of both A and B
# x_a (blue): clearly inside A, far from any boundary
# x_b (red): clearly inside B, far from any boundary
# x_a is also clearly outside B, and x_b is clearly outside A.

test_point_a = [-0.5, -0.2]   # well inside A's dark zone
test_point_b = [1.1, 1.0]     # well inside B's dark zone

data["test_point_a"] = [round(float(test_point_a[0]), 4), round(float(test_point_a[1]), 4)]
data["test_point_b"] = [round(float(test_point_b[0]), 4), round(float(test_point_b[1]), 4)]

# Verify zones using the zone classifications we already computed
for label, zones, pt_name, pt_val in [("A", zones_a, "x_a", test_point_a),
                                       ("B", zones_b, "x_b", test_point_b)]:
    # Find nearest triangle centroid to this point
    dists = np.sqrt(((centroids - pt_val) ** 2).sum(axis=1))
    nearest_tri = int(np.argmin(dists))
    zone = zones.get(nearest_tri, "outside")
    print(f"  {pt_name} at ({pt_val[0]:.2f}, {pt_val[1]:.2f}) -> zone '{zone}' in {label}'s view (want 'inside')")

out = Path(__file__).with_name("boundary-demo-data.json")
out.write_text(json.dumps(data, indent=2))
print(f"\nWrote {out.name}")
