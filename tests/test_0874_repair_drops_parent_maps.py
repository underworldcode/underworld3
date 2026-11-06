"""A repaired edge_split pass leaves no cell-parent map on any level through it (#961).

``mesh.adapt(engine="edge_split", repair=True)`` flips cells after each split. A flipped
cell can straddle two coarse cells, so no level whose span includes a repaired pass may
carry a cell-parent map: the any-degree transfer has to use the geometric builder. The
subsampler used to derive the map for every retained span from the composed vertex
transfer, and kept it wherever the derivation found a unique parent for each cell. On one
rank that held exactly when no local flip straddled two coarse cells, so at np 2 one rank
kept a map and the other did not (ptest_0844_reconnect, #961).

The fixture makes that case deterministic and serial: a repair pass that reports flips
but changes nothing. Every cell then has a unique parent, so deriving the map succeeds,
and only the record of the repair can drop it.

- A repaired pass leaves every retained level without a map.
- The same adapt without repair keeps maps, so the first assertion is not vacuous.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import reconnect

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

CENTRE = np.array([0.4, 0.55])


def _metric(centroids):
    d = np.linalg.norm(np.asarray(centroids) - CENTRE, axis=1)
    return 1.0 / np.where(d < 0.2, 0.04, 0.15) ** 2


def _base():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.2, regular=False, refinement=1,
                                             qdegree=3)


def test_a_repaired_pass_leaves_no_parent_map(monkeypatch):
    # a repair that reports one flip per pass and moves nothing
    monkeypatch.setattr(reconnect, "flip_to_reduce_max_angle", lambda dm, **kw: (dm, 1))
    child = _base().adapt(_metric, max_levels=2, engine="edge_split", repair=True,
                          mg_coarsening_ratio=1.0)
    assert len(child._adapt_parent_cells) > 1
    assert all(pc is None for pc in child._adapt_parent_cells), [
        None if pc is None else len(pc) for pc in child._adapt_parent_cells]


def test_without_repair_the_maps_are_kept():
    child = _base().adapt(_metric, max_levels=2, engine="edge_split", repair=False,
                          mg_coarsening_ratio=1.0)
    assert all(pc is not None for pc in child._adapt_parent_cells), [
        None if pc is None else len(pc) for pc in child._adapt_parent_cells]
