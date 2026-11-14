"""The add-on partition's region for a ribbon holds the placement's own region.

In parallel a ribbon is placed after the add-on has been partitioned for its
bands, so that the placement's gather moves no cell. That holds only if every
cell the placement gathers lies in the region the partition kept on one rank.
Both regions are functions of the mesh and the band alone, so one rank shows
whether the first contains the second (#1172).

The fixture is a band that outcrops at a low angle with its top one rung below
the wall. There the band's cap runs along the wall past the end of the band's
path, which is where a region measured from the path falls short of the band.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import addon_partition, place_surface

pytestmark = [pytest.mark.level_2,
              pytest.mark.skipif(uw.mpi.size > 1, reason="serial suite")]

RUNG = 0.03                 # the band's rung and the base cell size
WIDTH = 2.0 * RUNG
CLEARANCE = 0.7
GATHER_LAYERS = 1           # the placement's gather
PARTITION_LAYERS = 2        # the ribbon's call of partition_addon


def _low_angle_outcrop(dip_degrees):
    dip = np.radians(dip_degrees)
    top = np.array([0.3, 1.0 - RUNG])
    along = np.array([np.cos(dip), -np.sin(dip)])
    return top + (np.arange(12) * RUNG)[:, None] * along


def _place_and_keep_the_gather(monkeypatch, mesher):
    """Place the ribbon; return the first gather's mesh and vertex marks, and
    the ribbon's record."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=RUNG, qdegree=2)
    seen = {}
    gather = place_surface._gather_regions

    def keep(dm, chart, *args, **kwargs):
        seen.setdefault("first", (dm, np.array(chart, dtype=np.int32)))
        return gather(dm, chart, *args, **kwargs)

    monkeypatch.setattr(place_surface, "_gather_regions", keep)
    _mesh, info = place_surface.place_fault_ribbon_2d(
        base, [("f", _low_angle_outcrop(20.0))], WIDTH, margin_rings=1,
        clearance=CLEARANCE, split=False, mesher=mesher)
    dm, placed = seen["first"]
    return dm, placed, info


def _outside(dm, placed, rule):
    """Vertices the placement marks and the rule does not; cells of the
    placement's region outside the rule's region; the placement's region."""
    vS, vE = dm.getDepthStratum(0)
    pStart, _pEnd = dm.getChart()
    v = slice(vS - pStart, vE - pStart)
    n_vertices = int(np.count_nonzero((placed[v] != 0) & (rule[v] == 0)))
    gathered = np.asarray(
        place_surface._region_cells(dm, placed, layers=GATHER_LAYERS)[0])
    kept = np.asarray(
        place_surface._region_cells(dm, rule, layers=PARTITION_LAYERS)[0])
    n_cells = int(np.count_nonzero((gathered != 0) & (kept == 0)))
    return n_vertices, n_cells, int(np.count_nonzero(gathered))


@pytest.mark.tier_b
@pytest.mark.parametrize("mesher", ["ladder", "network"])
def test_partition_region_holds_the_gather_at_a_low_angle_outcrop(
        monkeypatch, mesher):
    dm, placed, info = _place_and_keep_the_gather(monkeypatch, mesher)
    marks = place_surface._ribbon_region_marks(
        dm, info["extended"], WIDTH, CLEARANCE, info["spacing"], mesher)
    rule = np.asarray(marks(dm), dtype=np.int32)

    n_vertices, n_cells, n_gathered = _outside(dm, placed, rule)
    assert n_gathered > 0                # the placement gathered a region
    assert (n_vertices, n_cells) == (0, 0)


@pytest.mark.tier_b
def test_the_fixture_reaches_past_a_region_measured_from_the_path(monkeypatch):
    """The ladder's cap carries the band past the reach of
    :func:`addon_partition.mark_band`, the rule for a band whose skin stays
    within a mitred half width of its path. If this stops holding the test
    above no longer exercises the outcrop."""
    dm, placed, info = _place_and_keep_the_gather(monkeypatch, "ladder")
    rule = np.asarray(
        addon_partition.mark_band(info["extended"], WIDTH, CLEARANCE)(dm),
        dtype=np.int32)

    n_vertices, n_cells, _n = _outside(dm, placed, rule)
    assert min(n_vertices, n_cells) > 0
