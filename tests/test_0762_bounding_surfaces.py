"""Bounding-surface objects: per-surface tangent-slip + restore (step 1).

Locks the additive, self-contained API from
docs/developer/design/boundary-slip-strategy.md:

  * analytic constructors register radial BoundingSurface objects;
  * mesh.boundaries (the persisted gmsh/DMPlex labelling) is left untouched;
  * radial/plane restore land points on the surface;
  * release() flips a surface to free (restore becomes a no-op);
  * mesh.boundary_slip orchestrates slip-vs-pin and keeps slipped boundary
    vertices exactly on the boundary while leaving interior vertices alone.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing.bounding_surface import BoundingSurface


def _annulus():
    return uw.meshing.Annulus(
        radiusInner=0.5, radiusOuter=1.0, cellSize=0.2, qdegree=2)


def test_constructor_registers_radial_surfaces():
    m = _annulus()
    bs = m.bounding_surfaces
    assert {"Upper", "Lower"} <= set(bs)
    assert bs["Upper"].kind == "radial"
    assert bs["Lower"].kind == "radial"
    assert np.isclose(bs["Upper"].radius, 1.0)
    assert np.isclose(bs["Lower"].radius, 0.5)
    assert np.allclose(bs["Upper"].centre, [0.0, 0.0])


def test_boundaries_enum_untouched():
    m = _annulus()
    # mesh.boundaries is still the persisted gmsh/DMPlex Enum (name/value)
    names = {b.name for b in m.boundaries}
    assert {"Upper", "Lower"} <= names
    assert all(hasattr(b, "value") for b in m.boundaries)
    # bounding_surfaces is a SEPARATE collection
    assert m.bounding_surfaces is not m.boundaries
    assert isinstance(m.bounding_surfaces, dict)


def test_radial_restore_lands_on_radius():
    m = _annulus()
    bs = m.bounding_surfaces["Upper"]
    pts = np.array([[1.2, 0.0], [0.0, 0.8], [-0.9, 0.0], [0.5, 0.5]])
    out = bs.restore(pts)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    # direction preserved (snap is purely radial)
    assert np.allclose(out / np.linalg.norm(out, axis=1, keepdims=True),
                       pts / np.linalg.norm(pts, axis=1, keepdims=True))


def test_plane_restore_zeroes_offplane():
    m = _annulus()
    bs = BoundingSurface(m, "face", "plane", point=[0.0, 0.0], normal=[0.0, 1.0])
    pts = np.array([[0.3, 0.5], [1.0, -0.2]])
    out = bs.restore(pts)
    assert np.allclose(out[:, 1], 0.0)        # on the y=0 plane
    assert np.allclose(out[:, 0], pts[:, 0])  # tangential coord unchanged


def test_release_makes_restore_a_noop():
    m = _annulus()
    bs = m.bounding_surfaces["Upper"]
    assert not bs.is_free
    bs.release()
    assert bs.is_free and bs.kind == "free"
    pts = np.array([[1.2, 0.0], [0.0, 0.8]])
    assert np.allclose(bs.restore(pts), pts)  # follows live surface → no snap


def test_register_provider_and_type_check():
    m = _annulus()
    s = BoundingSurface(m, "Upper", "radial", centre=[0.0, 0.0], radius=2.0)
    m.register_tangent_slip_provider("Upper", s)
    assert m.bounding_surfaces["Upper"].radius == 2.0
    with pytest.raises(TypeError):
        m.register_tangent_slip_provider("X", object())


def test_invalid_kind_and_missing_geometry_raise():
    m = _annulus()
    with pytest.raises(ValueError):
        BoundingSurface(m, "x", "bogus")
    with pytest.raises(ValueError):
        BoundingSurface(m, "x", "radial")           # needs centre+radius
    with pytest.raises(ValueError):
        BoundingSurface(m, "x", "plane", point=[0, 0])  # needs normal


def test_degenerate_geometry_raises():
    # A zero / non-finite normal would make plane restore() a silent no-op;
    # a non-positive / non-finite radius produces invalid radial projections.
    # Both must be rejected at construction.
    m = _annulus()
    with pytest.raises(ValueError):
        BoundingSurface(m, "x", "plane", point=[0, 0], normal=[0, 0])
    with pytest.raises(ValueError):
        BoundingSurface(m, "x", "plane", point=[0, 0],
                        normal=[np.nan, 0])
    with pytest.raises(ValueError):
        BoundingSurface(m, "x", "radial", centre=[0, 0], radius=-1.0)
    with pytest.raises(ValueError):
        BoundingSurface(m, "x", "radial", centre=[0, 0], radius=np.inf)
    # radius == 0 is VALID (a solid sphere/annulus registers its inner boundary
    # at the centre, radius 0) — must NOT raise.
    BoundingSurface(m, "x", "radial", centre=[0, 0], radius=0.0)


def test_boundary_slip_keeps_nodes_on_boundary():
    m = _annulus()
    ref = np.asarray(m.X.coords, dtype=float).copy()
    is_pinned, project = m.boundary_slip(True, reference_coords=ref)

    r_ref = np.linalg.norm(ref, axis=1)
    upper = np.isclose(r_ref, 1.0, atol=1e-6)
    lower = np.isclose(r_ref, 0.5, atol=1e-6)
    interior = ~(upper | lower)

    # Deterministic perturbation of every vertex, then project.
    th = np.arctan2(ref[:, 1], ref[:, 0])
    Y = ref + 0.05 * np.column_stack([np.cos(th + 1.0), np.sin(th + 1.0)])
    Yin = Y.copy()
    Y2 = project(Y)

    # Slipped Upper/Lower vertices land EXACTLY on their radius.
    su = (~is_pinned) & upper
    sl = (~is_pinned) & lower
    assert su.any() and sl.any()
    assert np.allclose(np.linalg.norm(Y2[su], axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(Y2[sl], axis=1), 0.5, atol=1e-9)

    # Interior vertices are untouched by the projector.
    assert np.allclose(Y2[interior], Yin[interior])


def test_boundary_slip_facet_fallback_when_no_surface_registered():
    # Step-2: a slip label with NO registered analytic surface no longer pins;
    # mesh.boundary_slip builds a transient `facet` surface from the reference
    # facets, so the vertices slip along the boundary polygon (the same path a
    # mesh loaded from file takes). See boundary-slip-strategy.md.
    from underworld3.meshing.smoothing import (
        _boundary_facets, _nearest_on_facets_2d)
    m = _annulus()
    m.bounding_surfaces.clear()      # remove the analytic surfaces
    ref = np.asarray(m.X.coords, dtype=float).copy()
    is_pinned, project = m.boundary_slip(True, reference_coords=ref)
    r_ref = np.linalg.norm(ref, axis=1)
    bnd = np.isclose(r_ref, 1.0, atol=1e-6) | np.isclose(r_ref, 0.5, atol=1e-6)
    # Most boundary vertices now SLIP (only true junctions/degenerate pin).
    assert not is_pinned[bnd].all()
    assert (~is_pinned[bnd]).sum() > 0.5 * bnd.sum()
    # Transient facet surfaces are local to the call — they don't leak in.
    assert len(m.bounding_surfaces) == 0
    # A tangential perturbation slips ON the reference-facet polygon: projected
    # boundary nodes lie on the nearest reference boundary facet (chord), to fp.
    th = np.arctan2(ref[:, 1], ref[:, 0])
    Y = ref.copy()
    Y[bnd] = ref[bnd] + 0.03 * np.column_stack(
        [np.cos(th[bnd] + 1.0), np.sin(th[bnd] + 1.0)])
    Y2 = project(Y.copy())
    facets, _ = _boundary_facets(m, m.cdim)
    seg = ref[facets]                                    # all boundary chords
    slip_b = np.nonzero(bnd & ~is_pinned)[0]
    nearest = _nearest_on_facets_2d(Y2[slip_b], seg)
    assert np.allclose(Y2[slip_b], nearest, atol=1e-9)


# NOTE: SphericalShell (3D radial) registration is tested in
# tests/test_0002_bounding_surface_3d.py — it must run in the early test batch
# because SphericalShell construction is fragile under the accumulated PETSc
# state of the heavy test_05*/07* batch (a pre-existing mesh-lifecycle issue,
# unrelated to boundary slip). The Annulus tests above cover the radial logic.


def test_box_registers_plane_surfaces():
    m = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.34)
    bs = m.bounding_surfaces
    assert {"Left", "Right", "Bottom", "Top"} <= set(bs)
    assert all(bs[k].kind == "plane" for k in ("Left", "Right", "Bottom", "Top"))
    # Left face is x=0: restore zeroes x, keeps y.
    out = bs["Left"].restore(np.array([[0.3, 0.5]]))
    assert np.isclose(out[0, 0], 0.0) and np.isclose(out[0, 1], 0.5)
    # Top face is y=1.
    out = bs["Top"].restore(np.array([[0.4, 0.7]]))
    assert np.isclose(out[0, 1], 1.0) and np.isclose(out[0, 0], 0.4)
    # Box corner (0,0) is a junction of two faces → pinned.
    ref = np.asarray(m.X.coords, dtype=float)
    is_pinned, _project = m.boundary_slip(True, reference_coords=ref)
    corner = np.isclose(ref[:, 0], 0.0) & np.isclose(ref[:, 1], 0.0)
    assert corner.any() and is_pinned[corner].all()
