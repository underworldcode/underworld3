#!/usr/bin/env python3
"""Regression test: Mesh.extract_region (codim-0 region submesh) — serial.

extract_region was only covered by the parallel suite (test_0770); the serial
path raised ``AttributeError: 'CoordinateSystem' object has no attribute
'_get_kdtree'`` from ``_build_vertex_map`` (UW3 issue #197). ``mesh.X`` is a
CoordinateSystem, which has no ``_get_kdtree`` — that method lives on
MeshVariable / swarm variables. The fix builds the vertex-coincidence KDTree
directly on the coordinate arrays, mirroring ``extract_surface``.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = pytest.mark.level_1


@pytest.fixture
def shell():
    return uw.meshing.AnnulusInternalBoundary(
        radiusInner=0.4, radiusInternal=0.7, radiusOuter=1.0, cellSize=0.3
    )


def test_extract_region_does_not_raise(shell):
    """The headline #197 symptom: extract_region must not raise."""
    sub = shell.extract_region("Inner")
    cStart, cEnd = sub.dm.getHeightStratum(0)
    assert cEnd - cStart > 0, "Inner region submesh has no cells"


def test_extract_region_vertex_map_is_consistent(shell):
    """The vertex map pairs each matched submesh vertex with a parent vertex."""
    sub = shell.extract_region("Inner")
    sub_rows, parent_rows = sub._build_vertex_map()
    assert len(sub_rows) == len(parent_rows)
    assert len(sub_rows) > 0
    # Matched pairs are bit-exact coincident coordinates (submesh ⊂ parent).
    sub_coords = np.asarray(sub._coords)
    parent_coords = np.asarray(shell._coords)
    assert np.allclose(
        sub_coords[sub_rows], parent_coords[parent_rows], atol=1.0e-10
    )


def test_extract_region_inner_radii_bounded(shell):
    """The Inner region lies within [radiusInner, radiusInternal]."""
    sub = shell.extract_region("Inner")
    r = np.linalg.norm(np.asarray(sub._coords), axis=1)
    # small tolerance for the faceted boundary
    assert r.min() >= 0.4 - 0.05
    assert r.max() <= 0.7 + 0.05
