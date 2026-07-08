"""Prototype: surface submesh extraction (third submesh flavour).

A *surface submesh* is a real ``uw.Mesh`` extracted from a parent mesh's
codimension-1 boundary, sharing exact vertex positions with the parent.
On a 3D ``SphericalShell``, ``extract_surface(shell, "Upper")`` returns
a 2-manifold embedded in 3-space: ``dim=2, cdim=3``.

This is the third flavour, alongside:

| Flavour          | Constructor                       | PETSc mechanism                          | dim, cdim         |
|------------------|-----------------------------------|------------------------------------------|-------------------|
| Subdomain        | ``mesh.extract_region(label)``    | ``DMPlexFilter``                         | parent.dim, cdim  |
| Resolution level | ``coarsened_companion(...)``      | ``dm.refine()`` hierarchy                | parent.dim, cdim  |
| **Surface**      | ``extract_surface(mesh, label)``  | ``DMPlexCreateSubmesh`` + ``DMPlexFilter`` | parent.dim-1, cdim |

All three share the same usage pattern:

    get a submesh -> build a solver on it -> map fields back and forth.

This is investigation code under docs/examples/, not a merged API.

Design notes
------------

* **Two PETSc primitives, composed.** ``DMPlexCreateSubmesh`` on a face
  label produces a cd-1 DM, but PETSc retains an upward-DAG phantom
  depth-3 stratum on it (one point per parent volume cell, celltype 12)
  so the result can still be navigated as "a slice of the parent". For
  a standalone surface mesh we don't use that parent-DAG linkage, and
  the phantom points break naive ``closure[-n:]`` slicing in the
  volume-mesh kd-tree builder and centroid pickers. The second stage,
  ``DMPlexFilter(depth, 2)``, drops the phantom: the result is a clean
  3-stratum chart (depth = 0/1/2, celltypes ``[0, 1, 3]``). The two
  subpoint IS's compose into a single surface->parent point map.
* **Design contract (loud-failure).** Surface extraction *requires* a
  non-empty boundary-face stratum on the parent. If the named label
  is absent or its face stratum is empty on this rank,
  ``extract_surface`` raises rather than returning a degenerate mesh.
* **Manifold navigation.** With the chart clean, the standard
  ``Mesh._build_kd_tree_index`` runs on the cd-1 surface: a centroid
  kd-tree on a convex manifold (sphere) ranks faces by chord distance,
  which agrees with the owning-face ranking to first order. Surface
  ``evaluate(expr, pts)`` works for query points on the surface (the
  only meaningful regime — querying R^3 points off the surface is a
  separate, ill-posed problem).
"""

from enum import Enum

import numpy as np
from petsc4py import PETSc

import underworld3 as uw
from underworld3.cython.petsc_discretisation import (
    petsc_dm_create_submesh_from_label,
    petsc_dm_filter_by_label,
)


def _attach_vertex_map(surf_mesh, parent_mesh):
    """Build sub_rows / parent_rows for coincident-vertex pairs.

    Workaround for UW3 issue #197 (``Mesh._build_vertex_map`` calls
    ``self.X._get_kdtree()``, but ``mesh.X`` is a ``CoordinateSystem``
    with no such method — ``extract_region`` is affected too). Once the
    core fix lands, this and the call site can collapse back to
    ``surf_mesh._build_vertex_map()``.

    Surface vertex coords ARE an exact subset of the parent's vertex
    coords (DMPlexCreateSubmesh preserves vertex positions), so a 1e-10
    coordinate match is bit-exact.
    """
    sub_coords = np.asarray(surf_mesh._coords)
    parent_coords = np.asarray(parent_mesh._coords)

    tree = uw.kdtree.KDTree(sub_coords)
    dists, indices = tree.query(parent_coords, sqr_dists=False)
    dists = np.asarray(dists).reshape(-1)
    indices = np.asarray(indices).reshape(-1)
    matched = dists < 1.0e-10

    parent_rows = np.where(matched)[0]
    sub_rows = indices[matched]
    surf_mesh._vertex_map = (sub_rows, parent_rows)


def _resolve_label_value(parent_mesh, label_name, label_value):
    """Resolve a boundary label name to its integer stratum value.

    Mirrors the resolve step in ``Mesh.extract_region`` but pulls from
    ``boundaries`` (face labels) rather than ``regions`` (cell labels).
    """
    if label_value is not None:
        return label_value

    if parent_mesh.boundaries is None:
        raise ValueError(
            "No boundaries defined on parent mesh. "
            "Provide label_value explicitly."
        )
    try:
        return parent_mesh.boundaries[label_name].value
    except KeyError:
        raise ValueError(
            f"Boundary '{label_name}' not found on parent mesh. "
            f"Available: {[b.name for b in parent_mesh.boundaries]}"
        )


def _require_non_empty_face_stratum(parent_dm, label_name, label_value):
    """Loud-fail if the parent has no faces marked with this label.

    Cd-1 submesh extraction is meaningless on an empty stratum and
    silently producing a zero-cell mesh is exactly the kind of soft
    failure the established submesh pattern rejects.

    Important: we must *not* call ``getStratumIS(value)`` for a value
    that isn't in the label's live value set — that hard-aborts PETSc
    on at least some labels (the same class of abort that protects the
    annulus/shell ``"Centre"`` pseudo-label). Check via ``getValueIS()``
    first.
    """
    label = parent_dm.getLabel(label_name)
    if label is None:
        raise ValueError(
            f"Parent DM has no label '{label_name}'. "
            f"Cannot extract a surface submesh from it."
        )
    vals_is = label.getValueIS()
    live_values = (
        set(int(v) for v in vals_is.getIndices())
        if vals_is is not None
        else set()
    )
    if int(label_value) not in live_values:
        raise ValueError(
            f"Label '{label_name}' has no stratum with value "
            f"{label_value} on the parent (live values: "
            f"{sorted(live_values)}). There is no surface to extract."
        )
    sis = label.getStratumIS(label_value)
    size = sis.getSize() if sis is not None else 0
    if size == 0:
        raise ValueError(
            f"Label '{label_name}' (value {label_value}) has an empty "
            f"face stratum on the parent. There is no surface to extract."
        )
    return size


def _surviving_boundaries(surface_dm, parent_boundaries):
    """Build a boundaries Enum from labels that landed on the surface.

    DMPlexCreateSubmesh propagates parent labels onto the submesh: the
    label that selected the surface (e.g. ``"Upper"``) carries the
    surface cells themselves; sibling boundary labels (e.g. ``"Lower"``
    on a SphericalShell) survive as empty strata. We keep only the
    labels whose stratum is non-empty on the submesh.

    Two contract details:

    1. We enumerate the *submesh's* labels (by index, ``getLabelName``)
       rather than probing parent labels by name. ``surface_dm.getLabel(
       name)`` for some parent labels (notably the annulus/shell
       ``"Centre"`` pseudo-label) hits PETSc machinery that hard-aborts
       rather than raises a Python error — the probe revealed this.
    2. Within a label, we obtain the live value set via ``getValueIS()``
       *before* asking for any stratum — calling ``getStratumIS(v)``
       with a value that has an empty stratum on this submesh likewise
       hard-aborts (the probe path avoided this by getting values
       first).
    """
    if parent_boundaries is None:
        return None

    parent_by_name = {b.name: b.value for b in parent_boundaries}

    # Enumerate submesh labels by index (the only safe path on a
    # submesh DM — see docstring).
    sm_label_names = [
        surface_dm.getLabelName(i)
        for i in range(surface_dm.getNumLabels())
    ]

    surviving = {}
    for name in sm_label_names:
        if name not in parent_by_name:
            continue  # internal PETSc label (celltype, depth) — ignore
        if name in ("Null_Boundary", "All_Boundaries", "Centre"):
            continue
        lab = surface_dm.getLabel(name)
        if lab is None:
            continue
        # Get the live value set on this submesh BEFORE asking for any
        # stratum — empty-stratum getStratumIS hard-aborts.
        try:
            vals_is = lab.getValueIS()
            vals = (
                set(int(v) for v in vals_is.getIndices())
                if vals_is is not None
                else set()
            )
        except Exception:
            continue
        parent_value = parent_by_name[name]
        if parent_value not in vals:
            continue
        sis = lab.getStratumIS(parent_value)
        if sis is not None and sis.getSize() > 0:
            surviving[name] = parent_value

    return Enum("Boundaries", surviving) if surviving else None


def extract_surface(parent_mesh, label_name, label_value=None, verbose=False):
    """Extract the codimension-1 surface marked by ``label_name`` as a uw.Mesh.

    Wraps ``DMPlexCreateSubmesh`` to produce a real ``uw.Mesh`` for the
    surface stratum. The submesh carries:

    - ``dim = parent.dim - 1`` (one less than the parent: 2 for a sphere
      surface extracted from a 3D shell);
    - ``cdim = parent.cdim`` (the embedding space is preserved; surface
      vertices keep their 3-component coordinates);
    - ``parent``, ``subpoint_is`` and registration with
      ``parent._registered_submeshes`` (standard submesh lineage).

    Parent ↔ submesh DOF transfer reuses ``Mesh.restrict`` /
    ``Mesh.prolongate`` (the same KDTree-coord-match-at-1e-10 path that
    ``extract_region`` uses); surface vertex coordinates are an *exact*
    subset of the parent's, so this is bit-exact.

    Parameters
    ----------
    parent_mesh : uw.discretisation.Mesh
        The parent (typically 3D) mesh with a boundary-face label.
    label_name : str
        Name of the parent boundary label whose marked faces become the
        cells of the surface submesh. E.g. ``"Upper"`` on a
        ``SphericalShell``.
    label_value : int, optional
        Stratum value within the label. If ``None``, resolved from
        ``parent_mesh.boundaries[label_name].value``.

    Returns
    -------
    uw.discretisation.Mesh
        A solver-ready surface mesh with ``parent`` set to
        ``parent_mesh`` and the standard submesh state attached.

    Raises
    ------
    ValueError
        If ``label_name`` is missing from the parent or its face stratum
        is empty (loud-fail contract — there is deliberately no
        geometric fallback).
    """
    label_value = _resolve_label_value(parent_mesh, label_name, label_value)
    _require_non_empty_face_stratum(parent_mesh.dm, label_name, label_value)

    # Stage 1: DMPlexCreateSubmesh produces a cd-1 DM with the right
    # cells, edges, and vertices, but PETSc retains an upward-DAG
    # phantom depth-3 stratum (one point per parent volume cell,
    # celltype 12). The phantom points show up inside surface cell
    # closures and break any `closure[-n:]` slicing in downstream
    # code — kd-tree builder, cell-centroid pickers, anything that
    # walks the closure naively. We don't use the parent-volume-cell
    # linkage for anything in the surface-only mesh use case, so we
    # strip it cleanly.
    sub_with_phantoms = petsc_dm_create_submesh_from_label(
        parent_mesh.dm, label_name, label_value, marked_faces=True,
    )

    # Stage 2: DMPlexFilter on (depth, 2) keeps only the surface cells
    # and their downward closure (edges, vertices). The result is a
    # genuine standalone 2-manifold mesh with a clean 3-stratum chart
    # (depth = 0, 1, 2) — celltypes [0, 1, 3] only, no phantom.
    surf_dm = petsc_dm_filter_by_label(sub_with_phantoms, "depth", 2)

    # Get subpoint IS's *before* the Mesh constructor wraps the DM,
    # since constructor side-effects (createDS, projectCoordinates) can
    # invalidate cached IS handles otherwise. Compose to get a single
    # surface -> parent point map.
    stage1_sp = sub_with_phantoms.getSubpointIS()  # sub1 point -> parent point
    stage2_sp = surf_dm.getSubpointIS()            # surf point -> sub1 point
    composed_indices = stage1_sp.getIndices()[stage2_sp.getIndices()]
    subpoint_is = PETSc.IS().createGeneral(
        composed_indices, comm=surf_dm.getComm(),
    )

    sub_boundaries = _surviving_boundaries(surf_dm, parent_mesh.boundaries)

    # Construct the Mesh. The constructor's kd-tree builder uses a
    # vertex-only placeholder when dim != cdim (the chart layout of a
    # DMPlexCreateSubmesh result breaks the volume-mesh closure
    # assumptions); point-location-based evaluate() on a surface
    # submesh is not supported by this prototype — that path needs the
    # manifold-aware solver work scheduled for Session 2.
    surf_mesh = uw.discretisation.Mesh(
        surf_dm,
        degree=parent_mesh.degree,
        qdegree=parent_mesh.qdegree,
        boundaries=sub_boundaries,
        coordinate_system_type=parent_mesh.CoordinateSystemType,
        verbose=verbose,
    )

    # Submesh lineage — same shape as extract_region
    surf_mesh.parent = parent_mesh
    surf_mesh.subpoint_is = subpoint_is
    surf_mesh._parent_mesh_version = parent_mesh._mesh_version
    surf_mesh._extract_label_name = label_name
    surf_mesh._extract_label_value = label_value
    surf_mesh._is_surface_submesh = True  # disambiguates from extract_region
    surf_mesh._dof_maps = {}

    # Build the vertex map (sub_rows -> parent_rows for shared verts).
    # Inlined here rather than calling Mesh._build_vertex_map() because
    # that method is broken on development (see UW3 issue #197 —
    # extract_region is also affected). Once #197 lands, this can
    # collapse back to surf_mesh._build_vertex_map().
    _attach_vertex_map(surf_mesh, parent_mesh)

    # Register with parent for coordinate-sync notifications
    parent_mesh._registered_submeshes.add(surf_mesh)

    return surf_mesh
