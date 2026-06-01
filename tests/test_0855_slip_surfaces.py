"""Named-surface tangent slip for the metric movers.

Locks the ``slip_surfaces`` API in ``_ot_adapt._build_slip_projector``:

* slip-vs-pin is **label-driven** — a boundary vertex slips iff it lies on
  exactly one slip surface; this fixes the old topology classifier that
  spuriously pinned the coarse-but-smooth annulus *inner* ring.
* junctions of two slip surfaces (box corners) **pin** (ambiguous normal).
* the tangential slide uses the projected P1 normal (``mesh.Gamma_P1``).
* non-free surfaces are returned to their reference facets (stay on the
  boundary); a ``dict`` value of ``False`` marks a FREE surface (no snap).

See project_mover_tangent_slip_surfaces.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import _ot_adapt as ota
from underworld3.meshing.smoothing import _pinned_mask


@pytest.mark.level_1
@pytest.mark.tier_a
def test_annulus_inner_ring_slips():
    """Both rings must slip fully — the inner ring was the bug (4/12)."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.12)
    coords = np.asarray(mesh.X.coords)
    n_verts = coords.shape[0]
    is_bnd = _pinned_mask(mesh.dm, ota._all_boundary_labels(mesh))

    is_pinned, project = ota._build_slip_projector(
        mesh, coords.copy(), is_bnd, n_verts, True)
    slip = is_bnd & ~is_pinned

    r = np.linalg.norm(coords, axis=1)
    outer = r > 0.9
    inner = (r > 0.4) & (r < 0.6)
    # every ring vertex slips (no spurious pinning of the coarse inner ring)
    assert (slip & outer).sum() == (is_bnd & outer).sum() > 0
    assert (slip & inner).sum() == (is_bnd & inner).sum() > 0

    # a tangential nudge + return-to-bounds keeps nodes on their rings
    rng = np.random.default_rng(0)
    Y = coords.copy()
    Y[slip] += 0.05 * rng.standard_normal((int(slip.sum()), mesh.cdim))
    Y = project(Y)
    rnew = np.linalg.norm(Y, axis=1)
    assert np.abs(rnew[slip & outer] - 1.0).max() < 0.02   # chord sag only
    assert np.abs(rnew[slip & inner] - 0.5).max() < 0.02


@pytest.mark.level_1
@pytest.mark.tier_a
def test_box_corners_pin_edges_slip():
    """Box corners (on two labels) pin; edge nodes slip along their line."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.15)
    coords = np.asarray(mesh.X.coords)
    n_verts = coords.shape[0]
    is_bnd = _pinned_mask(mesh.dm, ota._all_boundary_labels(mesh))

    is_pinned, project = ota._build_slip_projector(
        mesh, coords.copy(), is_bnd, n_verts, True)
    slip = is_bnd & ~is_pinned

    corner = ((np.isclose(coords[:, 0], 0) | np.isclose(coords[:, 0], 1)) &
              (np.isclose(coords[:, 1], 0) | np.isclose(coords[:, 1], 1)))
    assert corner.sum() == 4
    assert (corner & slip).sum() == 0          # junctions pinned

    # tangential nudge: a left-edge node keeps x == 0
    rng = np.random.default_rng(1)
    Y = coords.copy()
    Y[slip] += 0.05 * rng.standard_normal((int(slip.sum()), 2))
    Y = project(Y)
    left = slip & np.isclose(coords[:, 0], 0)
    assert np.abs(Y[left, 0]).max() < 1.0e-9


@pytest.mark.level_1
@pytest.mark.tier_a
def test_named_subset_and_free_surface_dict():
    """A label subset slips while others pin; a dict ``False`` value marks a
    free surface that slides without being snapped back."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.15)
    coords = np.asarray(mesh.X.coords)
    n_verts = coords.shape[0]
    is_bnd = _pinned_mask(mesh.dm, ota._all_boundary_labels(mesh))
    r = np.linalg.norm(coords, axis=1)
    outer = r > 0.9
    inner = (r > 0.4) & (r < 0.6)

    # only the Upper (outer) ring slips; Lower pins
    is_pinned, _ = ota._build_slip_projector(
        mesh, coords.copy(), is_bnd, n_verts, ["Upper"])
    slip = is_bnd & ~is_pinned
    assert (slip & outer).sum() > 0
    assert (slip & inner).sum() == 0           # Lower pinned

    # dict free-surface form must resolve both labels as slipping and run the
    # no-snap branch for Upper without error
    is_pinned2, project2 = ota._build_slip_projector(
        mesh, coords.copy(), is_bnd, n_verts, {"Upper": False, "Lower": True})
    slip2 = is_bnd & ~is_pinned2
    assert (slip2 & outer).sum() > 0 and (slip2 & inner).sum() > 0
    Y = coords.copy()
    Y[slip2] += 0.01 * np.ones((int(slip2.sum()), 2))
    Y = project2(Y)                            # must not raise
    assert np.isfinite(Y).all()


@pytest.mark.level_1
@pytest.mark.tier_a
def test_mmpde_slip_with_meshvariable_metric():
    """Regression: mmpde + slip + a MeshVariable-valued metric must not abort.

    Touching mesh.Gamma_P1 (to build the slip normals) creates the _n_proj
    MeshVariable; doing so mid-mover restructured the DM and invalidated the
    interpolation state a MeshVariable metric needs — a hard abort.
    smooth_mesh_interior now pre-creates Gamma_P1 before dispatching, so this
    runs cleanly. Pure-sympy metrics never hit it (no DM interpolation)."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1.0 / 10)
    f = uw.discretisation.MeshVariable("Fm", mesh, 1, degree=1)
    r = np.linalg.norm(np.asarray(f.coords), axis=1)
    f.data[:, 0] = 1.0 + 4.0 * np.exp(-((r - 1.0) / 0.1) ** 2)  # refine outer ring
    uw.meshing.smooth_mesh_interior(
        mesh, metric=f.sym[0], method="mmpde", slip_surfaces=True,
        method_kwargs=dict(n_outer=6, step_frac=0.2, tol=5.0e-3))
    assert np.isfinite(np.asarray(mesh.X.coords)).all()


@pytest.mark.level_1
@pytest.mark.tier_a
def test_resolve_slip_forms():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
    assert ota._resolve_slip(mesh, False) == ()
    assert ota._resolve_slip(mesh, None) == ()
    assert set(ota._resolve_slip(mesh, True)) == set(ota._all_boundary_labels(mesh))
    assert ota._resolve_slip(mesh, "Upper") == ("Upper",)
    assert set(ota._resolve_slip(mesh, ["Upper", "Lower"])) == {"Upper", "Lower"}
    assert set(ota._resolve_slip(mesh, {"Upper": False, "Lower": True})) == {"Upper", "Lower"}
