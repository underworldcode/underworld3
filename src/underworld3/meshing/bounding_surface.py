"""Boundary-surface objects: per-surface tangent-slip + restore.

See ``docs/developer/design/boundary-slip-strategy.md``.

A :class:`BoundingSurface` binds a boundary *label* (a name in
``mesh.boundaries`` — the persisted gmsh/DMPlex labelling, which is **not**
replaced here) to that surface's geometry, slip state, and the tangent-project /
restore operations for it. Surfaces live in the separate
``mesh.bounding_surfaces`` collection.

This is the step-1 (additive, self-contained) implementation: ``radial`` and
``plane`` restore are analytic; ``facet`` (nearest reference facet) and the
``free`` live-surface follow are follow-ups (the orchestrator pins labels with
no analytic surface). The module depends only on the primitive
``_ot_adapt`` helpers that exist on ``development`` (``_slip_normals``), not on
the mover feature branch's unified projector.
"""
import numpy as np

import underworld3 as uw

_VALID_KINDS = ("radial", "plane", "facet", "free")


def _unit(v):
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _as_float(x):
    """Coerce a radius/length that may be a UWQuantity / pint Quantity to a
    bare non-dimensional float in the mesh's coordinate units."""
    try:
        return float(x)
    except (TypeError, ValueError):
        nd = uw.non_dimensionalise(x)
        return float(getattr(nd, "magnitude", getattr(nd, "value", nd)))


class BoundingSurface:
    """One bounding surface of a mesh — its geometry, slip state, and methods.

    Parameters
    ----------
    mesh : underworld3 mesh
        The mesh this surface belongs to.
    label : str
        The boundary label (a name in ``mesh.boundaries``) this surface is for.
    kind : {"radial", "plane", "facet", "free"}
        ``radial`` — snap to ``|r| = radius`` about ``centre`` (annulus / sphere
        / cylinder); ``plane`` — project onto the plane through ``point`` with
        unit ``normal`` (box face); ``facet`` — nearest reference facet
        (follow-up); ``free`` — follow the live deformed surface, analytic
        restore is a no-op (follow-up).
    centre, radius : for ``radial``.
    point, normal : for ``plane``.
    is_free : bool
        If True (or ``kind == "free"``), :meth:`restore` is a no-op — the
        surface follows the live discrete boundary rather than a fixed target.
    """

    def __init__(self, mesh, label, kind, *, centre=None, radius=None,
                 point=None, normal=None, reference_facets=None, is_free=False):
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"BoundingSurface kind must be one of {_VALID_KINDS}; got {kind!r}")
        self._mesh = mesh
        self.label = str(label)
        self.kind = kind
        self.is_free = bool(is_free) or kind == "free"
        self.centre = None if centre is None else np.asarray(centre, dtype=float).ravel()
        self.radius = None if radius is None else _as_float(radius)
        self.point = None if point is None else np.asarray(point, dtype=float).ravel()
        self.normal = None if normal is None else _unit(normal)
        # reference_facets: (nf, cdim, cdim) — line segments (2D) / triangles
        # (3D) of the surface, captured from a FIXED reference, for the `facet`
        # nearest-point restore on non-analytic surfaces.
        self.reference_facets = (
            None if reference_facets is None
            else np.ascontiguousarray(reference_facets, dtype=float))
        if kind == "radial" and (self.centre is None or self.radius is None):
            raise ValueError("radial BoundingSurface requires centre and radius")
        if kind == "plane" and (self.point is None or self.normal is None):
            raise ValueError("plane BoundingSurface requires point and normal")
        if kind == "facet" and self.reference_facets is None:
            raise ValueError("facet BoundingSurface requires reference_facets")

    @property
    def mesh(self):
        return self._mesh

    # -- normals -------------------------------------------------------------
    def normals(self, coords):
        """Outward unit normals at ``coords`` from the projected P1
        boundary-normal field (``mesh.Gamma_P1``).

        Returns ``(normals, valid)`` where ``valid`` is False at nodes whose
        projected normal is degenerate (box corners, unlocatable points) — those
        should be pinned, not slipped.
        """
        from underworld3.meshing._ot_adapt import _slip_normals
        return _slip_normals(self._mesh, np.ascontiguousarray(coords, dtype=float))

    # -- tangent slide -------------------------------------------------------
    def tangent_project(self, coords, reference):
        """Tangent-slide: remove this surface's normal component of the
        displacement ``coords - reference`` (the projected P1 normal is taken at
        ``reference``). Nodes with a degenerate normal keep ``reference``.
        """
        coords = np.asarray(coords, dtype=float)
        reference = np.asarray(reference, dtype=float)
        n, valid = self.normals(reference)
        disp = coords - reference
        dn = (disp * n).sum(axis=1, keepdims=True)
        slid = reference + (disp - dn * n)
        return np.where(valid[:, None], slid, reference)

    # -- restore to surface --------------------------------------------------
    def restore(self, coords):
        """Snap ``coords`` back onto this surface (kind-specific).

        ``radial`` — re-impose ``|r| = radius`` about ``centre`` (exact,
        concave-safe). ``plane`` — orthogonal projection onto the plane.
        ``facet`` — nearest point on the surface's reference facets (segments
        in 2D, triangles in 3D); convex-safe, with a documented concave bias.
        A ``free``/``is_free`` surface returns ``coords`` unchanged (it follows
        the live discrete surface — a follow-up).
        """
        coords = np.asarray(coords, dtype=float)
        if self.is_free or self.kind == "free":
            return coords
        if self.kind == "radial":
            v = coords - self.centre
            nrm = np.linalg.norm(v, axis=1)
            nrm = np.where(nrm > 1.0e-30, nrm, 1.0)
            return self.centre + v * (self.radius / nrm)[:, None]
        if self.kind == "plane":
            d = ((coords - self.point) * self.normal).sum(axis=1, keepdims=True)
            return coords - d * self.normal
        if self.kind == "facet":
            if coords.shape[0] == 0:
                return coords
            from underworld3.meshing._ot_adapt import (
                _nearest_on_facets_2d, _nearest_on_facets_3d)
            if coords.shape[1] == 2:
                return _nearest_on_facets_2d(coords, self.reference_facets)
            return _nearest_on_facets_3d(coords, self.reference_facets)
        return coords

    # -- state transition ----------------------------------------------------
    def release(self):
        """Flip a rigid (``radial``/``plane``) surface to ``free``: subsequent
        :meth:`restore` follows the live discrete surface instead of the analytic
        target. Records the previous kind on ``self._prev_kind``."""
        self._prev_kind = self.kind
        self.kind = "free"
        self.is_free = True
        return self

    def __repr__(self):
        g = ""
        if self.kind == "radial":
            g = f", centre={self.centre}, radius={self.radius}"
        elif self.kind == "plane":
            g = f", point={self.point}, normal={self.normal}"
        return (f"BoundingSurface(label={self.label!r}, kind={self.kind!r}, "
                f"is_free={self.is_free}{g})")


# -- constructor-side registration helpers ---------------------------------
def register_radial_surfaces(mesh, centre, label_radius):
    """Register ``radial`` surfaces: ``label_radius = {label_name: radius}``.

    Called by analytic radial-boundary constructors (Annulus, SphericalShell,
    CubedSphere). ``centre`` is the common centre (e.g. the origin)."""
    centre = np.asarray(centre, dtype=float).ravel()
    for lab, r in label_radius.items():
        mesh.register_tangent_slip_provider(
            lab, BoundingSurface(mesh, lab, "radial", centre=centre, radius=r))


def register_plane_surfaces(mesh, label_plane):
    """Register ``plane`` surfaces: ``label_plane = {label_name: (point, normal)}``.

    Called by box constructors for axis-aligned (or general) faces."""
    for lab, (p, n) in label_plane.items():
        mesh.register_tangent_slip_provider(
            lab, BoundingSurface(mesh, lab, "plane", point=p, normal=n))


def register_box_face_surfaces(mesh, minCoords, maxCoords):
    """Register ``plane`` surfaces for an axis-aligned box's faces.

    Matches the box constructors' labels:
    2D — ``Left``/``Right`` (``x = x_min/max``), ``Bottom``/``Top``
    (``y = y_min/max``); 3D — ``Left``/``Right`` (``x``), ``Front``/``Back``
    (``y``), ``Bottom``/``Top`` (``z``). Box corners/edges are junctions of two
    faces and are pinned by the slip orchestrator (the normal is ambiguous).
    """
    lo = np.asarray(minCoords, dtype=float).ravel()
    hi = np.asarray(maxCoords, dtype=float).ravel()
    dim = lo.shape[0]
    planes = {}
    if dim == 2:
        ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        planes["Left"] = (lo, ex)
        planes["Right"] = ([hi[0], lo[1]], ex)
        planes["Bottom"] = (lo, ey)
        planes["Top"] = ([lo[0], hi[1]], ey)
    elif dim == 3:
        ex, ey, ez = (np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0]),
                      np.array([0.0, 0.0, 1.0]))
        planes["Left"] = (lo, ex)
        planes["Right"] = ([hi[0], lo[1], lo[2]], ex)
        planes["Front"] = (lo, ey)
        planes["Back"] = ([lo[0], hi[1], lo[2]], ey)
        planes["Bottom"] = (lo, ez)
        planes["Top"] = ([lo[0], lo[1], hi[2]], ez)
    else:
        return
    register_plane_surfaces(mesh, planes)
