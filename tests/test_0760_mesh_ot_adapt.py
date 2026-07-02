"""Locks the mesh.OT_adapt() public API.

OT_adapt() runs the validated optimal-transport reset adapt as a method
on the mesh: reset to a cached reference, FE-remap the driving field onto
the clean canvas, build a gradient metric, run the OT mover, then FE-remap
fields and zero a cold-restart set. These tests pin:

* the mesh actually moves and the driving field's spatial pattern survives
  the FE-remap (Annulus and Box);
* boundary behaviour — radial (Annulus) boundary nodes slide tangentially
  but stay on the circle; Cartesian (Box) boundaries are pinned (their
  vertex-evaluated normal is degenerate, so we do not slip with it);
* the lazy reference-coordinate cache and OT_adapt_reset_reference();
* skip_threshold short-circuits an already-aligned mesh;
* constrained-manifold meshes (dim != cdim) raise NotImplementedError.

Behaviour is validated, not bit-for-bit: the unified Gamma_N-based slip
is a different algorithm from the historical hardcoded box/ring slip.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import smoothing as _sm


def _annulus_with_field(slope=15.0, r_bl=0.75):
    """Annulus + a tanh boundary-layer T field at radius r_bl (set
    directly on the DOFs — no projection solve needed)."""
    m = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0,
                           cellSize=0.08, qdegree=3)
    T = uw.discretisation.MeshVariable(
        "T", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    feat = lambda c: np.tanh((np.linalg.norm(c, axis=1) - r_bl) * slope)
    T.data[:, 0] = feat(np.asarray(T.coords))
    return m, T, feat


def _box_with_field(slope=15.0, y_bl=0.5):
    m = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=0.08, qdegree=3)
    T = uw.discretisation.MeshVariable(
        "Tb", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    feat = lambda c: np.tanh((c[:, 1] - y_bl) * slope)
    T.data[:, 0] = feat(np.asarray(T.coords))
    return m, T, feat


def _boundary_mask(mesh):
    from underworld3.meshing.smoothing import (
        _pinned_mask, _auto_pinned_labels)
    return _pinned_mask(mesh.dm, tuple(_auto_pinned_labels(mesh)))


# ---------------------------------------------------------------------------
# Annulus: moves, preserves field, boundary stays on the circle
# ---------------------------------------------------------------------------
@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_moves_mesh_annulus():
    m, T, _ = _annulus_with_field()
    X0 = np.asarray(m.X.coords).copy()
    moved = m.OT_adapt(T, refinement=3.0, fields_to_remap=[T])
    assert moved is True
    X1 = np.asarray(m.X.coords)
    # interior nodes moved appreciably
    assert float(np.linalg.norm(X1 - X0, axis=1).max()) > 1.0e-3


@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_annulus_boundary_stays_on_circle():
    m, T, _ = _annulus_with_field()
    is_bnd = _boundary_mask(m)
    X0 = np.asarray(m.X.coords).copy()
    r0 = np.linalg.norm(X0[is_bnd], axis=1)
    m.OT_adapt(T, refinement=3.0, fields_to_remap=[T])
    X1 = np.asarray(m.X.coords)
    r1 = np.linalg.norm(X1[is_bnd], axis=1)
    # boundary nodes slid tangentially ...
    bnd_disp = float(np.linalg.norm(X1[is_bnd] - X0[is_bnd], axis=1).max())
    assert bnd_disp > 1.0e-4
    # ... but stayed on their original radii (radial drift ~ rounding)
    assert float(np.abs(r1 - r0).max()) < 1.0e-3


@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_preserves_field_pattern_annulus():
    m, T, feat = _annulus_with_field()
    m.OT_adapt(T, refinement=3.0, fields_to_remap=[T])
    # T at the adapted DOFs should match the analytic feature there
    err = np.abs(np.asarray(T.data)[:, 0] - feat(np.asarray(T.coords))).max()
    assert float(err) < 5.0e-2


# ---------------------------------------------------------------------------
# Box: interior moves, Cartesian boundary pinned, field preserved
# ---------------------------------------------------------------------------
@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_box_moves_interior_slides_boundary_on_faces():
    # Generic topology-based slip is now active on Cartesian boxes too:
    # boundary nodes on the box faces SLIDE tangentially; corners are pinned
    # (incident face normals disagree); the box shape (axis-aligned bounding
    # planes) is preserved exactly.
    m, T, feat = _box_with_field()
    is_bnd = _boundary_mask(m)
    X0 = np.asarray(m.X.coords).copy()
    moved = m.OT_adapt(T, refinement=3.0, fields_to_remap=[T])
    assert moved is True
    X1 = np.asarray(m.X.coords)
    # interior refined
    assert float(np.linalg.norm(X1[~is_bnd] - X0[~is_bnd], axis=1).max()) > 1.0e-3
    # box CORNERS are pinned (each of the four unit-square corners is still
    # present at its original position)
    for cc in ([0., 0.], [0., 1.], [1., 0.], [1., 1.]):
        assert np.any(np.all(np.abs(X1 - np.array(cc)) < 1.0e-9, axis=1)), \
            f"corner {cc} lost"
    # face vertices stay on the bounding planes (x=0, x=1, y=0 or y=1):
    # for each originally-boundary node, at least one coord is 0 or 1
    on_face = (np.minimum(X1[is_bnd], 1 - X1[is_bnd]).min(axis=1) < 1.0e-9)
    assert on_face.all(), "boundary node left the box face"
    # field pattern preserved within FE-remap tolerance — looser than the
    # old pinned-box test (5e-2) because face slip reallocates some node
    # budget from the BL interior to the boundary, mildly increasing remap
    # error on a sharp tanh feature at this coarse base mesh. Still
    # well-controlled (≪ the BL amplitude of 1).
    err = np.abs(np.asarray(T.data)[:, 0] - feat(np.asarray(T.coords))).max()
    assert float(err) < 1.5e-1


# ---------------------------------------------------------------------------
# Reference-coordinate cache + reset
# ---------------------------------------------------------------------------
@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_reference_cache_and_reset():
    m, T, _ = _annulus_with_field()
    X0 = np.asarray(m.X.coords).copy()
    # first call snapshots the (pristine) coords as the reset reference
    m.OT_adapt(T, refinement=3.0, fields_to_remap=[T])
    assert np.allclose(m._ot_adapt_reference_coords, X0)
    # the mesh itself has moved away from the reference
    assert not np.allclose(np.asarray(m.X.coords), X0)
    # reset_reference(None) re-baselines to the current (moved) coords
    m.OT_adapt_reset_reference()
    assert np.allclose(m._ot_adapt_reference_coords,
                       np.asarray(m.X.coords))
    # explicit coords override
    m.OT_adapt_reset_reference(coords=X0)
    assert np.allclose(m._ot_adapt_reference_coords, X0)


# ---------------------------------------------------------------------------
# skip_threshold short-circuits an already-aligned mesh
# ---------------------------------------------------------------------------
@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_skip_threshold_returns_false():
    m, T, _ = _annulus_with_field()
    X0 = np.asarray(m.X.coords).copy()
    # misalignment is in [0, 1]; threshold > 1 always skips
    moved = m.OT_adapt(T, refinement=3.0, fields_to_remap=[T],
                       skip_threshold=2.0)
    assert moved is False
    # mesh untouched
    assert np.allclose(np.asarray(m.X.coords), X0)


# ---------------------------------------------------------------------------
# Constrained-manifold mesh (dim != cdim) is not supported
# ---------------------------------------------------------------------------
@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_manifold_raises():
    try:
        sm = uw.meshing.SegmentedSphericalSurface2D(
            radius=1.0, cellSize=0.2, numSegments=6)
    except Exception:
        pytest.skip("manifold surface mesh not constructible in this build")
    assert sm.dim != sm.cdim
    T = uw.discretisation.MeshVariable(
        "Tm", sm, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
    with pytest.raises(NotImplementedError):
        sm.OT_adapt(T)


# ---------------------------------------------------------------------------
# grad_smoothing_length default ("auto" ≈ grid size) de-noises the metric so
# production refinement stays sliver-free; "off" (None) does not.
# ---------------------------------------------------------------------------
def _qmin(mesh):
    tris = _sm._tri_cells(mesh.dm)
    X = np.asarray(mesh.X.coords); p = X[tris]
    e = (np.linalg.norm(p[:, 1]-p[:, 0], axis=1)**2
         + np.linalg.norm(p[:, 2]-p[:, 1], axis=1)**2
         + np.linalg.norm(p[:, 0]-p[:, 2], axis=1)**2)
    A = _sm._signed_areas(X, tris)
    return float((4*np.sqrt(3)*np.abs(A)/(e+1e-30)).min())


@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_auto_smoothing_default_avoids_slivers():
    # Sharp boundary layer: with smoothing OFF the metric chases sub-cell
    # noise and slivers form; the "auto" default de-noises it.
    m_off, T_off, _ = _annulus_with_field(slope=28.0)
    m_off.OT_adapt(T_off, refinement=3.0, fields_to_remap=[T_off],
                   grad_smoothing_length=None)
    q_off = _qmin(m_off)

    m_auto, T_auto, _ = _annulus_with_field(slope=28.0)
    m_auto.OT_adapt(T_auto, refinement=3.0, fields_to_remap=[T_auto])  # default "auto"
    q_auto = _qmin(m_auto)

    assert q_auto > q_off            # auto is strictly better
    assert q_auto > 0.2              # and sliver-free


@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_smoothing_accepts_explicit_and_unit_aware_length():
    # explicit non-dimensional length
    m, T, _ = _annulus_with_field()
    assert m.OT_adapt(T, refinement=3.0, fields_to_remap=[T],
                      grad_smoothing_length=0.0625) is True
    # unit-aware (Pint) length is accepted (routed through the projection's
    # non-dimensionalisation)
    m2, T2, _ = _annulus_with_field()
    L = 0.0625 * uw.scaling.units.meter
    assert m2.OT_adapt(T2, refinement=3.0, fields_to_remap=[T2],
                       grad_smoothing_length=L) is True


@pytest.mark.tier_a
@pytest.mark.level_1
def test_ot_adapt_invalid_smoothing_string_raises():
    m, T, _ = _annulus_with_field()
    with pytest.raises(ValueError):
        m.OT_adapt(T, refinement=3.0, grad_smoothing_length="bogus")
