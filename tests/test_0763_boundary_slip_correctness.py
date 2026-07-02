"""Correctness of the mesh-owned ``mesh.boundary_slip`` contract.

Step 2 of the boundary tangent-slip refactor swapped every metric mover from the
private ``_ot_adapt._build_slip_projector`` onto ``mesh.boundary_slip`` (see
``docs/developer/design/boundary-slip-strategy.md``) and removed the old
projector. This test locks the replacement's behaviour directly: slip vertices
land **exactly** on their analytic bounding surface (radius / plane), junctions
and unregistered-surface corners pin, the transient ``facet`` fallback keeps
vertices on the reference-facet polygon, and a FREE surface slides without snap.

Historical note: the swap was validated against ``_build_slip_projector`` before
that engine was deleted — agreement was machine-precision (~1e-16) on a centred
annulus (boundary COM == analytic centre to fp) and exact on box faces. These
absolute-landing checks are strictly tighter than that parity comparison.
"""
import numpy as np

import underworld3 as uw
from underworld3.meshing.smoothing import _pinned_mask, _auto_pinned_labels


def _annulus():
    return uw.meshing.Annulus(
        radiusInner=0.547, radiusOuter=1.0, cellSize=0.1, qdegree=2)


def _box():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=2)


def _perturb(X0, is_bnd, seed):
    """A tangential-ish perturbation on boundary vertices (interior fixed)."""
    rng = np.random.default_rng(seed)
    Y = X0.copy()
    Y[is_bnd] = X0[is_bnd] + 0.02 * rng.standard_normal(X0[is_bnd].shape)
    return Y


def test_annulus_radial_lands_exactly_on_radius():
    m = _annulus()
    X0 = np.asarray(m.X.coords, dtype=float).copy()
    labels = _auto_pinned_labels(m)
    is_bnd = _pinned_mask(m.dm, labels)
    is_pinned, project = m.boundary_slip(
        True, reference_coords=X0, boundary_labels=labels)
    # Full annulus: every boundary vertex slips, none pinned (no junctions).
    assert not is_pinned[is_bnd].any()
    Y2 = project(_perturb(X0, is_bnd, seed=0))
    r0 = np.linalg.norm(X0, axis=1)
    r2 = np.linalg.norm(Y2, axis=1)
    up = np.isclose(r0, 1.0, atol=1e-6)
    lo = np.isclose(r0, 0.547, atol=1e-6)
    # slipped nodes land EXACTLY on their analytic radius
    assert np.abs(r2[up] - 1.0).max() < 1e-12
    assert np.abs(r2[lo] - 0.547).max() < 1e-12
    # interior untouched
    assert np.allclose(Y2[~is_bnd], X0[~is_bnd])


def test_box_plane_corners_pin_edges_on_face():
    m = _box()
    X0 = np.asarray(m.X.coords, dtype=float).copy()
    labels = _auto_pinned_labels(m)
    is_bnd = _pinned_mask(m.dm, labels)
    is_pinned, project = m.boundary_slip(
        True, reference_coords=X0, boundary_labels=labels)
    corner = ((np.isclose(X0[:, 0], 0) | np.isclose(X0[:, 0], 1)) &
              (np.isclose(X0[:, 1], 0) | np.isclose(X0[:, 1], 1)))
    assert corner.sum() == 4
    assert is_pinned[corner].all()              # junctions pin
    Y2 = project(_perturb(X0, is_bnd, seed=1))
    # left-edge slip nodes keep x == 0 exactly (plane restore)
    left = is_bnd & ~is_pinned & np.isclose(X0[:, 0], 0)
    assert left.any() and np.abs(Y2[left, 0]).max() < 1e-12
    # bottom-edge slip nodes keep y == 0
    bot = is_bnd & ~is_pinned & np.isclose(X0[:, 1], 0)
    assert bot.any() and np.abs(Y2[bot, 1]).max() < 1e-12


def test_box_facet_fallback_stays_on_polygon():
    """Unregistered slip labels build transient ``facet`` surfaces; projected
    vertices lie on the reference-facet polygon and the transient surfaces do
    not leak into the persistent collection."""
    from underworld3.meshing._ot_adapt import (
        _boundary_facets, _nearest_on_facets_2d)
    m = _box()
    m.bounding_surfaces.clear()                 # force the facet fallback path
    X0 = np.asarray(m.X.coords, dtype=float).copy()
    labels = _auto_pinned_labels(m)
    is_bnd = _pinned_mask(m.dm, labels)
    is_pinned, project = m.boundary_slip(
        True, reference_coords=X0, boundary_labels=labels)
    assert len(m.bounding_surfaces) == 0        # no leak
    # corners still pin (junction of two labels)
    corner = ((np.isclose(X0[:, 0], 0) | np.isclose(X0[:, 0], 1)) &
              (np.isclose(X0[:, 1], 0) | np.isclose(X0[:, 1], 1)))
    assert is_pinned[corner].all()
    Y2 = project(_perturb(X0, is_bnd, seed=2))
    facets, _ = _boundary_facets(m, m.cdim)
    seg = X0[facets]
    slip_b = np.nonzero(is_bnd & ~is_pinned)[0]
    assert np.allclose(Y2[slip_b], _nearest_on_facets_2d(Y2[slip_b], seg),
                       atol=1e-9)


def test_single_label_slips_other_pins():
    m = _annulus()
    X0 = np.asarray(m.X.coords, dtype=float).copy()
    is_pinned, _ = m.boundary_slip("Upper", reference_coords=X0)
    lower = _pinned_mask(m.dm, ("Lower",))
    upper = _pinned_mask(m.dm, ("Upper",))
    assert is_pinned[lower].all()               # Lower pinned (not a slip label)
    assert not is_pinned[upper].any()           # Upper slips


def test_free_surface_slides_without_restore():
    """A FREE slip surface (dict ``{label: False}``) slides tangentially but is
    NOT snapped back onto |r| — distinct from a restored radial surface."""
    m = _annulus()
    X0 = np.asarray(m.X.coords, dtype=float).copy()
    labels = _auto_pinned_labels(m)
    is_bnd = _pinned_mask(m.dm, labels)
    is_pinned, project = m.boundary_slip(
        {"Upper": False, "Lower": True}, reference_coords=X0,
        boundary_labels=labels)
    Y2 = project(_perturb(X0, is_bnd, seed=3))
    r0 = np.linalg.norm(X0, axis=1)
    up = is_bnd & ~is_pinned & np.isclose(r0, 1.0, atol=1e-6)
    lo = is_bnd & ~is_pinned & np.isclose(r0, 0.547, atol=1e-6)
    # Lower (restored) lands exactly on |r|; Upper (free) does not snap back.
    assert np.abs(np.linalg.norm(Y2[lo], axis=1) - 0.547).max() < 1e-12
    assert np.isfinite(Y2[up]).all()
    # at least one free Upper node moved off the exact radius (no restore)
    assert np.abs(np.linalg.norm(Y2[up], axis=1) - 1.0).max() > 1e-9
