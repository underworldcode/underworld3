"""
Data for the "facet normals vs. true normals" figure in
`docs/advanced/curved-boundary-conditions.md`.

The figure illustrates why `mesh.Gamma` (PETSc facet normals) diverges
from the true smooth-surface normal on a curved boundary:

  - Three straight facets approximate a circular arc.
  - At each Gauss quadrature point (2 per facet), the facet normal
    is constant across the facet while the true (radial) normal
    rotates with position.

Output schema:

  {
    "centre": [cx, cy],
    "radius": R,
    "arc_points": [[x, y], ...],        # densely sampled true arc
    "facet_vertices": [[x, y], ...],    # points on the circle
    "quadrature": [
      {"pos": [x, y],
       "facet_normal": [nx, ny],
       "true_normal":  [nx, ny],
       "facet_idx": int},
      ...
    ],
    "radius_end": [x, y]                 # arc point at mid-angle
                                          # (end of the radius indicator)
  }
"""
import json
import math
from pathlib import Path

CENTRE       = (0.0, 0.0)
RADIUS       = 1.5
ANGLE_START  = 30.0       # degrees — chosen so the error angle at
ANGLE_END    = 150.0      # quadrature points is visually obvious (~12°)
N_FACETS     = 3
ARC_SAMPLES  = 120


def point_at_angle(deg, r=RADIUS):
    rad = math.radians(deg)
    return (CENTRE[0] + r * math.cos(rad),
            CENTRE[1] + r * math.sin(rad))


# 1. Dense sampling of the true arc (for a dashed polyline in Typst)
arc_points = [
    list(point_at_angle(
        ANGLE_START + (i / ARC_SAMPLES) * (ANGLE_END - ANGLE_START)
    ))
    for i in range(ARC_SAMPLES + 1)
]

# 2. Facet vertices — N_FACETS + 1 points evenly spaced on the arc
facet_vertices = [
    list(point_at_angle(
        ANGLE_START + (i / N_FACETS) * (ANGLE_END - ANGLE_START)
    ))
    for i in range(N_FACETS + 1)
]

# 3. Three-point Gauss–Legendre quadrature on [-1, 1]:  0, ±sqrt(3/5).
#    Mapped to t ∈ [0, 1]:  0.5 ± sqrt(3/5)/2  and  0.5.
#    The middle node lies exactly at the chord midpoint, where the
#    facet normal and the radial true normal coincide — that's the
#    "error vanishes at facet midpoint" case the figure needs to show.
_HALF = 0.5 * math.sqrt(0.6)
GAUSS_T = (
    0.5 - _HALF,
    0.5,
    0.5 + _HALF,
)

quadrature = []
for facet_idx in range(N_FACETS):
    p0 = facet_vertices[facet_idx]
    p1 = facet_vertices[facet_idx + 1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)

    # Outward perpendicular to the chord: pick the rotation of (dx, dy)
    # whose dot-product with (midpoint - centre) is positive.
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    cand = (-dy / length, dx / length)
    if (cand[0] * (mid[0] - CENTRE[0])
        + cand[1] * (mid[1] - CENTRE[1])) < 0:
        cand = (dy / length, -dx / length)
    facet_normal = cand

    for t in GAUSS_T:
        pos = (p0[0] + t * dx, p0[1] + t * dy)
        # True normal: unit radial vector from circle centre through pos.
        rx, ry = pos[0] - CENTRE[0], pos[1] - CENTRE[1]
        rlen = math.hypot(rx, ry)
        true_normal = (rx / rlen, ry / rlen)
        quadrature.append({
            "pos":          [pos[0], pos[1]],
            "facet_normal": [facet_normal[0], facet_normal[1]],
            "true_normal":  [true_normal[0], true_normal[1]],
            "facet_idx":    facet_idx,
        })

# 4. Radius indicator: from centre to arc midpoint
radius_end = list(point_at_angle((ANGLE_START + ANGLE_END) / 2))

data = {
    "centre":         list(CENTRE),
    "radius":         RADIUS,
    "arc_points":     [[round(x, 4), round(y, 4)] for x, y in arc_points],
    "facet_vertices": [[round(x, 4), round(y, 4)] for x, y in facet_vertices],
    "quadrature": [
        {k: ([round(v[0], 4), round(v[1], 4)] if isinstance(v, list) else v)
         for k, v in q.items()}
        for q in quadrature
    ],
    "radius_end":     [round(radius_end[0], 4), round(radius_end[1], 4)],
}

out = Path(__file__).with_name("curved-bc-data.json")
out.write_text(json.dumps(data, indent=2))
print(f"wrote {out.name}: "
      f"{len(arc_points)} arc samples, "
      f"{len(facet_vertices)} facet vertices, "
      f"{len(quadrature)} quadrature points")
