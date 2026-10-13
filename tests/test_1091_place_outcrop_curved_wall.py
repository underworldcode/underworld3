"""A ribbon placed into an annulus: every cell size, and a DIPPING band that
outcrops through the curved wall (the Zhong & Gurnis trench, 2026-10-14).

Two placer defects, each with the probe that found it:

* the domain tool's boundary loops arrived with whichever winding the walk
  started on; a hole loop wound against the exterior was ADDED by OCC, the
  tool had the area of the full disc plus the hole, and the ribbon's
  intersection with it was empty ("meshed to no triangles") — for cell sizes
  0.04 and 0.08 but not 0.05 or 0.06, which happened to wind both loops the
  same way;
* the 2-D outcrop frame read every chord of the faceted wall as its own
  wall, so a dipping band's clearance cavity, spanning a few chords beside
  the one it crosses, was refused as contact "away from the outcrop"; and
  the splice then chorded across the wall's own vertices, losing the sagitta
  the area guard refuses. The frame now groups chords into smooth walls
  (the 3-D rule) and the splice keeps the surviving wall vertices.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.place_surface import place_thin_volume

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1, reason="serial suite")]


def _census(new, zone_value):
    fS, fE = new.getHeightStratum(1)
    lower, upper, trace = (new.getLabel(n) for n in ("Lower", "Upper", "Zone_trace"))
    n_trace = n_bare = 0
    for f in range(fS, fE):
        if len(new.getSupport(f)) != 1:
            continue
        if lower.getValue(f) < 0 and upper.getValue(f) < 0:
            n_bare += 1
        if trace.getValue(f) == zone_value:
            n_trace += 1
    return n_trace, n_bare


def _dipping(theta, top_r, length, dip=np.deg2rad(45.0)):
    t_hat = np.array([np.cos(theta), np.sin(theta)])
    s_hat = np.array([-np.sin(theta), np.cos(theta)])
    d_hat = -np.cos(dip) * t_hat + np.sin(dip) * s_hat
    return np.array([t_hat * top_r, t_hat * top_r + d_hat * length])


@pytest.mark.parametrize("cell_size", [0.04, 0.05, 0.06])
def test_an_interior_ribbon_places_at_every_cell_size(cell_size):
    from underworld3.utilities.line_cut import cell_areas

    base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=cell_size, qdegree=2)
    zv = base._boundaries_with("Zone")["Zone"].value
    ray = np.array([np.cos(0.4), np.sin(0.4)])
    before = float(cell_areas(base.dm).sum())
    new, info = place_thin_volume(base.dm, [np.array([0.62 * ray, 0.88 * ray])],
                                  width=0.03, label="Zone", label_value=zv)
    assert info["n_zone_cells"] > 0
    assert float(cell_areas(new).sum()) == pytest.approx(before, rel=1e-12)


def test_a_dipping_band_outcrops_through_the_curved_wall():
    from underworld3.utilities.line_cut import cell_areas

    base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.55, cellSize=0.04,
                              refinement=1, qdegree=2)
    zv = base._boundaries_with("Zone")["Zone"].value
    before = float(cell_areas(base.dm).sum())
    new, info = place_thin_volume(base.dm, [_dipping(0.0, 1.03, 0.25)], width=0.02,
                                  label="Zone", label_value=zv, size=0.005, clearance=0.7)
    n_trace, n_bare = _census(new, zv)
    assert info["n_trace_facets"] > 0 and n_trace == info["n_trace_facets"], \
        "the dipping band left no outcrop trace on the wall"
    assert n_bare == 0, "the relabel left boundary edges without a wall label"
    assert float(cell_areas(new).sum()) == pytest.approx(before, rel=1e-9), \
        "the splice chorded across the wall's vertices"
    assert float(cell_areas(new).min()) > 0.0


def test_the_structured_ladder_outcrops_through_a_cap():
    """The ladder (rungs on the spine, one element either side) used to be
    refused at a wall outright. It now runs to one rung short of the wall
    and a clipped unstructured cap closes it to the outcrop, so a TI band
    reaches the surface and a split spine stops on its own rung vertices
    below it. Both placement routes: the thin volume, and the ribbon placer
    with a trace shorter than its spine."""
    from underworld3.utilities.line_cut import cell_areas
    from underworld3.utilities.place_surface import place_fault_ribbon_2d

    S = 0.005
    base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.55, cellSize=0.16,
                              refinement=3, qdegree=2)
    zv = base._boundaries_with("Zone")["Zone"].value
    dip = np.deg2rad(45.0)
    d_hat = np.array([-np.cos(dip), np.sin(dip)])
    top = np.array([1.02, 0.0])
    n = int(round(0.2 / S))
    spine = np.array([top + d_hat * (0.2 * i / n) for i in range(n + 1)])
    before = float(cell_areas(base.dm).sum())
    new, info = place_thin_volume(base.dm, [spine], width=2 * S, label="Zone",
                                  label_value=zv, size=S, clearance=0.7, mesher="ladder")
    n_trace, n_bare = _census(new, zv)
    assert info["n_trace_facets"] > 0 and n_trace == info["n_trace_facets"]
    assert n_bare == 0
    assert float(cell_areas(new).sum()) == pytest.approx(before, rel=1e-9)
    assert float(cell_areas(new).min()) > 0.0

    r = np.hypot(spine[:, 0], spine[:, 1])
    trace = spine[r <= 1.0 - 3 * S]
    mesh, rinfo = place_fault_ribbon_2d(base, [("Trench", trace)], 2 * S,
                                        spines=[("Trench", spine)], margin_rings=1,
                                        clearance=0.7, split=True, mesher="ladder")
    pairs = mesh._fault_point_pairs["Trench"]
    # interior vertices plus the edges between them (P2 carries edge points)
    assert len(pairs) == 2 * len(trace) - 3, "the spine is split on its own rung vertices"
    X = np.asarray(mesh.dm.getCoordinatesLocal().array).reshape(-1, 2)
    vS, vE = mesh.dm.getDepthStratum(0)
    pp = np.array([int(q) for q in pairs] + [int(q) for q in pairs.values()])
    pv = pp[(pp >= vS) & (pp < vE)] - vS             # the vertex points of the pairing
    assert np.hypot(X[pv, 0], X[pv, 1]).max() < 1.0 - 2 * S, "no pair node near the wall"
    assert len(mesh._custom_mg_coarse_meshes) == 4, "the base tail is adopted"
