"""Domain membership + point location must use the DEFORMED geometry.

Regression for the deformed-domain bug: points_in_domain() and
uw.function.evaluate() locate points using a boundary-skeleton kd-tree built
from mesh._nav_coords. That was captured as a reference to the ORIGINAL
coordinates in __init__ and never refreshed in nuke_coords_and_rebuild (only
the per-cell face-control-point arrays were invalidated, on adapt). So on a
volume mesh _nav_coords stayed at the undeformed boundary after a deform, and a
region that bulged OUT past the original boundary (r > r_o on a free surface)
read as EXTERIOR — stranding semi-Lagrangian trace-back feet there and
mis-locating evaluations to the old boundary, injecting the cold boundary value
at topographic highs.

The fix invalidates the boundary kd-tree AND refreshes _nav_coords from the
current DM coordinates on every deform.

Run: pixi run python -m pytest tests/test_0057_deformed_domain_membership.py -v
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw

pytestmark = [
    pytest.mark.level_1,
    pytest.mark.tier_a,
    # Probes a rank-local bulge; the membership/eval logic is verified serial.
    pytest.mark.skipif(uw.mpi.size > 1, reason="serial point-location check"),
]


def test_membership_and_eval_track_bulged_surface():
    r_i, r_o, cs = 0.5, 1.0, 0.15
    mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=cs, qdegree=3)

    # T = radius (a field whose value identifies where a point actually is)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    r_sym = sympy.sqrt(mesh.X[0] ** 2 + mesh.X[1] ** 2)
    T.data[:, 0] = np.asarray(uw.function.evaluate(r_sym, T.coords)).flatten()

    # Bulge the outer surface OUTWARD in mode-3 (only the outward half), so the
    # surface near theta=0 sits well beyond the original radius r_o.
    X = np.asarray(mesh.X.coords).copy()
    R = np.sqrt((X ** 2).sum(1)); TH = np.arctan2(X[:, 1], X[:, 0])
    surf = np.abs(R - r_o) < 0.5 * cs
    Xd = X.copy()
    Xd[surf] *= (1.0 + 0.12 * np.maximum(np.cos(3 * TH[surf]), 0.0))[:, None]
    mesh.deform(Xd, dt=1.0)

    Rn = np.sqrt((np.asarray(mesh.X.coords) ** 2).sum(1))
    crest_r = Rn[surf & (np.abs(TH) < 0.2)].max() if (surf & (np.abs(TH) < 0.2)).any() else Rn.max()
    assert crest_r > r_o + 0.05, "test setup: surface should bulge past r_o"

    # nav coords must now reflect the deformed geometry
    nav_max = np.sqrt((np.asarray(mesh._nav_coords) ** 2).sum(1)).max()
    assert nav_max > r_o + 0.05, (
        f"_nav_coords stale: max radius {nav_max:.3f} (deformed ~ {crest_r:.3f})")

    # probe points along the bulge crest (theta=0), from inside r_o out to the
    # deformed surface. ALL must be inside the (deformed) domain.
    radii = np.linspace(0.95, crest_r - 0.01, 10)
    pts = np.column_stack([radii, np.zeros_like(radii)])

    inside = mesh.points_in_domain(pts, strict_validation=True)
    assert inside.all(), (
        f"points in the bulge (r in [{radii[0]:.2f},{radii[-1]:.2f}], surface "
        f"at {crest_r:.2f}) wrongly flagged exterior: {inside}")

    # evaluate must LOCATE them (not clamp to the old boundary). T is carried
    # Lagrangian-ly, so the value need not equal r; but it must NOT all collapse
    # to ~r_o (the cold-clamp signature), and must vary smoothly with radius.
    Tev = np.asarray(uw.function.evaluate(T.sym[0], pts)).flatten()
    assert np.ptp(Tev) > 0.02, (
        f"evaluate collapsed to a single boundary value (cold-clamp): {Tev}")
    assert np.all(np.diff(Tev) >= -1e-6), (
        f"evaluate not monotone along the carried-T crest: {Tev}")
