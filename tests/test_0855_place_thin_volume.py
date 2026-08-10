"""The embedded thin-volume fault mesh (:func:`place_surface.place_thin_volume`).

The finite-width representation: patches are thickened by ±width/2 in OCC,
``fragment`` resolves the network's junctions in CAD — the only kernel that
can — the assembly is meshed standalone at layer scale, and the meshed
assembly is embedded whole into the existing mesh: cavity carved, annular gap
filled by gmsh with the assembly's skin as an interior HOLE, both constraint
surfaces verbatim. Junctions need no geometric treatment: they are ordinary
cells of the union, and the rheology decides (the design ruling this exists
to serve).

What is asserted is what the construction promises: the zone's CELLS carry
the label (that is the point of a volume representation), the skin is a
separate face/edge label (a cell-bearing stratum is a volume, invisible to
the interface machinery — the skin label is what protects the zone from later
surgery), domain volume/area is conserved to round-off, sub-h widths work,
and every refusal names its cause. Absolute cell counts are never pinned —
they are gmsh-version-dependent (the test_0842 lesson).

Measured basis: ~/+Simulations/mesh_reconnection_study/thin_volume_spike.py
(widths h, h/2, h/4; junction angles to 10 degrees).
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.place_surface import place_thin_volume

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite; the parallel form is "
                                        "ptest_0855")]


def _box3(cell):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=cell, regular=False, qdegree=2)


def _box2(cell):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell,
        regular=False, qdegree=2)


CROSS_3D = [np.array([[0.3, 0.3, 0.5], [0.7, 0.3, 0.5],
                      [0.7, 0.7, 0.5], [0.3, 0.7, 0.5]]),
            np.array([[0.3, 0.5, 0.3], [0.7, 0.5, 0.3],
                      [0.7, 0.5, 0.7], [0.3, 0.5, 0.7]])]


def _volume(dm):
    from underworld3.utilities.place_surface import _owned_cell_volume
    return _owned_cell_volume(dm)


# ------------------------------------------------------------------------ 3-D

def test_a_crossing_pair_embeds_with_the_junction_in_the_volume():
    mesh = _box3(0.09)
    before = _volume(mesh.dm)
    new, info = place_thin_volume(mesh.dm, CROSS_3D, width=0.045,
                                  label="Zone", label_value=5)
    assert info["n_zone_cells"] > 0
    assert new.getLabel("Zone").getStratumSize(5) == info["n_zone_cells"]
    assert new.getLabel("Zone_skin").getStratumSize(5) == info["n_skin_faces"]
    assert _volume(new) == pytest.approx(before, rel=1e-12)

    # Every zone point is a CELL and every skin point a FACE — the label
    # split that keeps the skin visible to the interface machinery.
    cS, cE = new.getHeightStratum(0)
    fS, fE = new.getHeightStratum(1)
    zone = new.getLabel("Zone").getStratumIS(5).getIndices()
    assert all(cS <= p < cE for p in zone)
    skin = new.getLabel("Zone_skin").getStratumIS(5).getIndices()
    assert all(fS <= p < fE for p in skin)
    # Skin faces are interior: two cells each.
    assert all(len(new.getSupport(int(p))) == 2 for p in skin)

    # No isolated vertex: the carve's growth can swallow the whole star of a
    # non-victim vertex, which then rides through the rebuild referenced by
    # nothing (global Euler 2 — caught by CI, gmsh-version-dependent). The
    # carve promotes such orphans to victims; this is the direct probe.
    vS, vE = new.getDepthStratum(0)
    assert all(len(new.getSupport(int(v))) > 0 for v in range(vS, vE))


def test_sub_h_width_is_supported():
    """Width below the background h is the whole point (V = 2 edot w)."""
    mesh = _box3(0.12)
    new, info = place_thin_volume(
        mesh.dm, [CROSS_3D[0]], width=0.03, label="Thin", label_value=2)
    assert info["n_zone_cells"] > 0
    assert _volume(new) == pytest.approx(_volume(mesh.dm), rel=1e-12)


def test_the_zone_survives_a_later_sheet_and_vice_versa():
    """The skin label is the guard: later surgery must hold clear of it."""
    from underworld3.utilities.place_surface import place_sheet

    # The margins are arithmetic, not luck: at cellSize 0.07 the cell
    # diameter is ~0.11, the sheet's reach 0.6*0.11 = 0.066, so a sheet at
    # z = 0.25 keeps its cavity off the floor (0.25 - 0.066 - 0.11 > 0) and
    # its victim stars off the zone's held cells (0.48 - 0.316 > 0.11).
    mesh = _box3(0.07)
    new, info = place_thin_volume(mesh.dm, [CROSS_3D[0]], width=0.04,
                                  label="Zone", label_value=5)

    # A sheet placed well away embeds; the zone comes through intact.
    n = 4
    s = np.linspace(-0.12, 0.12, n)
    pts = np.array([[0.5 + a, 0.5 + b, 0.25] for a in s for b in s])
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, d), (a, d, c)]
    two, _ = place_sheet(new, pts, np.array(tris, dtype=np.int64),
                         label="Sheet", label_value=9)
    assert two.getLabel("Zone").getStratumSize(5) == info["n_zone_cells"]
    assert two.getLabel("Zone_skin").getStratumSize(5) == info["n_skin_faces"]

    # A sheet THROUGH the zone must refuse, naming the embedded surface.
    mid = np.array([[0.5 + a, 0.5 + b, 0.5] for a in s for b in s])
    with pytest.raises(RuntimeError,
                       match="already embedded|held|hole|clearance"):
        place_sheet(two, mid, np.array(tris, dtype=np.int64), label="Bad")


def test_the_wall_is_refused_with_the_reason():
    mesh = _box3(0.12)
    low = np.array([[0.3, 0.3, 0.08], [0.7, 0.3, 0.08],
                    [0.7, 0.7, 0.08], [0.3, 0.7, 0.08]])
    with pytest.raises(RuntimeError, match="domain wall"):
        place_thin_volume(mesh.dm, [low], width=0.04)


def test_a_non_planar_patch_is_refused():
    mesh = _box3(0.2)
    warped = np.array([[0.3, 0.3, 0.5], [0.7, 0.3, 0.5],
                       [0.7, 0.7, 0.6], [0.3, 0.7, 0.5]])
    with pytest.raises(RuntimeError, match="planar"):
        place_thin_volume(mesh.dm, [warped], width=0.04)


# ------------------------------------------------------------------------ 2-D

def test_a_ribbon_x_junction_embeds_in_two_dimensions():
    from underworld3.utilities.line_cut import cell_areas

    mesh = _box2(0.05)
    before = float(cell_areas(mesh.dm).sum())
    l1 = np.array([[0.3, 0.35], [0.7, 0.65]])
    l2 = np.array([[0.3, 0.65], [0.7, 0.35]])
    new, info = place_thin_volume(mesh.dm, [l1, l2], width=0.02,
                                  label="Zone", label_value=5)
    assert info["n_zone_cells"] > 0
    assert new.getLabel("Zone").getStratumSize(5) == info["n_zone_cells"]
    assert new.getLabel("Zone_skin").getStratumSize(5) == info["n_skin_faces"]
    assert float(cell_areas(new).sum()) == pytest.approx(before, rel=1e-12)
    assert info["min_angle"] > 10.0


def test_a_kinked_ribbon_does_not_sliver():
    """The mitre-join lesson: per-segment quads fragmented together sliver at
    every kink (the inner-side overlap lens meshes at ~2 degrees, measured);
    one mitred outline per polyline has no internal seam to lens."""
    mesh = _box2(0.05)
    kinked = np.array([[0.15, 0.15], [0.4, 0.18], [0.6, 0.12]])
    new, info = place_thin_volume(mesh.dm, [kinked], width=0.02,
                                  label="Kink", label_value=3)
    assert info["min_angle"] > 10.0


def test_a_second_zone_leaves_the_first_intact():
    mesh = _box2(0.05)
    l1 = np.array([[0.3, 0.35], [0.7, 0.65]])
    one, info1 = place_thin_volume(mesh.dm, [l1], width=0.02,
                                   label="Z1", label_value=1)
    l2 = np.array([[0.2, 0.8], [0.8, 0.8]])
    two, _info2 = place_thin_volume(one, [l2], width=0.02,
                                    label="Z2", label_value=2)
    assert two.getLabel("Z1").getStratumSize(1) == info1["n_zone_cells"]
    assert two.getLabel("Z1_skin").getStratumSize(1) == info1["n_skin_faces"]


def test_a_sharp_kink_is_refused_with_the_reason():
    mesh = _box2(0.1)
    hairpin = np.array([[0.3, 0.5], [0.6, 0.5], [0.32, 0.55]])
    with pytest.raises((RuntimeError, ValueError), match="mitre|sharply"):
        place_thin_volume(mesh.dm, [hairpin], width=0.02)


def test_zero_width_is_refused():
    mesh = _box2(0.2)
    with pytest.raises(ValueError, match="width"):
        place_thin_volume(mesh.dm, [np.array([[0.3, 0.5], [0.7, 0.5]])],
                          width=0.0)
