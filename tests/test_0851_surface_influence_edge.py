#!/usr/bin/env python3
"""Regression test: Surface.influence_function must respect a FINITE edge.

A surface (here a short 2D line segment) has ends. The influence/weak zone
should decay beyond the segment tip — it must NOT bleed out along the segment's
infinite-line extension.

The bug (fixed by using the unsigned, edge-clamped distance field rather than
Abs() of the *signed* field): the signed distance changes sign across the line
and its zero-contour runs along the infinite line past the finite tip, so any
non-nodal evaluation interpolates the signed field through ~0 there and Abs()
leaves a spurious high-influence streak beyond the end of the segment.

We embed a short vertical segment and check the influence at NON-NODAL points
(which is where the interpolation artifact appears).
"""
import numpy as np
import pytest

import underworld3 as uw


def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


requires_pyvista = pytest.mark.skipif(
    not _has("pyvista"), reason="pyvista required for surface distance"
)

pytestmark = pytest.mark.level_2


@requires_pyvista
def test_influence_decays_beyond_finite_edge():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.05, qdegree=3
    )

    # Short vertical segment, interior, ends at y = 0.5 (tip).
    y = np.linspace(0.3, 0.5, 11)
    pts = np.column_stack([np.full_like(y, 0.5), y, np.zeros_like(y)])
    surf = uw.meshing.Surface("seg", mesh, pts, symbol="S")
    surf.discretize()

    width = 0.05
    influence = surf.influence_function(
        width=width, value_near=1.0, value_far=0.0, profile="gaussian"
    )

    # On the segment (non-nodal): full influence.
    v_on = float(np.asarray(uw.function.evaluate(influence, np.array([[0.5, 0.40]]))).reshape(-1)[0])
    assert v_on > 0.9, f"influence on segment should be ~1, got {v_on}"

    # Probe a BAND along the segment's line EXTENSION, well beyond the tip
    # (y >= 0.65 is >= 0.15 past the tip at y=0.5, i.e. >= 3*width, so the true
    # influence is < 1e-3 everywhere here). The bug spikes the influence to ~1
    # on the mesh edges that cross the extension; the fix keeps it ~0. Use a
    # band in x (not just x=0.5) so we actually hit those crossing edges.
    xs = np.linspace(0.45, 0.55, 11)
    ys = np.linspace(0.65, 0.92, 28)
    band = np.array([[x, y] for y in ys for x in xs])
    v_band = np.asarray(uw.function.evaluate(influence, band)).reshape(-1)
    assert v_band.max() < 0.05, (
        f"influence bled past the finite edge: max={v_band.max():.3f} in the "
        f"extension band >=0.15 beyond the tip (width={width}); expected ~0"
    )

    # Sanity: a perpendicular-far point also decays.
    v_perp = float(np.asarray(uw.function.evaluate(influence, np.array([[0.85, 0.40]]))).reshape(-1)[0])
    assert v_perp < 0.05, f"influence far perpendicular should be ~0, got {v_perp}"


@requires_pyvista
def test_abs_distance_matches_geometry_beyond_edge():
    """The unsigned distance field is edge-aware between nodes too."""
    from underworld3.utilities.geometry_tools import (
        signed_distance_pointcloud_polyline_2d,
    )

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.05, qdegree=3
    )
    y = np.linspace(0.3, 0.5, 11)
    xy = np.column_stack([np.full_like(y, 0.5), y])
    surf = uw.meshing.Surface("seg2", mesh, np.column_stack([xy, np.zeros(len(xy))]), symbol="S")
    surf.discretize()

    probe = np.array([[0.5, 0.80], [0.5, 0.40], [0.85, 0.40], [0.65, 0.65]])
    d_field = np.asarray(
        uw.function.evaluate(surf.abs_distance.sym[0], probe)
    ).reshape(-1)
    d_true = np.abs(signed_distance_pointcloud_polyline_2d(probe, xy))

    # Interpolated unsigned field tracks the true edge-clamped distance
    # (linear interp of a >=0 cone, so within a cell-size tolerance).
    assert np.allclose(d_field, d_true, atol=0.05), (
        f"abs_distance field {d_field} != geometric {d_true}"
    )
