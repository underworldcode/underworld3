r"""Optimal-transport mesh adaptation — the validated reset-to-uniform step.

This module factors the production pattern that was inlined in
``scripts/stagnant_lid_adapt_loop.py`` (the ``ot-reset`` branch) into a
reusable library function. The public entry point is :meth:`Mesh.OT_adapt`
(see ``discretisation/discretisation_mesh.py``); this module holds the
algorithm and the boundary-slip helpers it shares with the OT mover
(``_winslow_equidistribute`` in ``smoothing.py``).

The algorithm, per adapt event:

1. Reset the mesh to its reference (IC uniform) coordinates.
2. FE-remap the driving ``field`` onto the reference-mesh DOFs.
3. Build the gradient density metric ``ρ`` on that clean canvas.
4. Run the OT mover from the uniform canvas (``smooth_mesh_interior``,
   ``method="ot"``).
5. FE-remap the requested fields onto the adapted positions and zero any
   fields flagged for a cold restart.

The "reset every event" discipline is load-bearing: carrying mesh state
*across* time steps is the broken incremental pattern (slivers lock in).
Composition *within* an adapt is fine. See
``docs/developer/design/ot-adapt-api-proposal.md`` and the
``project_ot_reset_validated`` memory note.

Boundary slip uses the mesh's **projected boundary normals**
(``mesh.Gamma_P1`` / ``mesh._update_projected_normals``) — the symbolic
``mesh.Gamma`` projected to a P1 vector field and normalised. This is the
general, free-surface-ready normal source: it is re-projected on demand here
because the projected field goes stale every time the mesh deforms. No
per-mesh-class normal code is used. Nodes whose projected normal is
degenerate (box corners, or an occasional unlocatable vertex) are pinned
rather than slipped.
"""

import numpy as np

import underworld3 as uw

# Validated OT-mover constants (2026-05-23/24 investigation). These are
# deliberately *not* exposed on the public OT_adapt signature — they are the
# settled production point, not user dials.
_OT_N_OUTER = 5
_OT_RELAX = 0.1
_OT_STEP_FRAC = 0.3


def _is_radial_coords(mesh) -> bool:
    """True for coordinate systems with a radial boundary (the snap-back
    target is a fixed ``|r|``). Cartesian boundaries are flat — zeroing the
    normal displacement keeps nodes on the face, so no snap-back is needed."""
    from underworld3.coordinates import CoordinateSystemType as CT

    return mesh.CoordinateSystem.coordinate_type in (
        CT.CYLINDRICAL2D,
        CT.CYLINDRICAL3D,
        CT.SPHERICAL,
        CT.GEOGRAPHIC,
    )


def _auto_grad_smoothing_length(mesh):
    """The mesh's characteristic (uniform) cell size — mean edge length,
    parallel-safe — returned as a unit-aware length when the mesh carries
    coordinate units, else a bare (non-dimensional) float. Used as the
    default ``grad_smoothing_length`` so gradient de-noising is on by
    default at a scale comparable to the grid (the validated production
    setting); ``None`` turns it off."""
    from underworld3.meshing.smoothing import _edge_pairs

    ep = _edge_pairs(mesh.dm)
    X = np.asarray(mesh.X.coords)
    if ep.shape[0]:
        h0 = float(np.linalg.norm(
            X[ep[:, 1]] - X[ep[:, 0]], axis=1).mean())
    else:
        h0 = 1.0
    if uw.mpi.size > 1:
        h0 = uw.mpi.comm.allreduce(h0) / uw.mpi.size
    units = getattr(mesh.X, "units", None)
    return h0 if units is None else h0 * units


def _boundary_centre(mesh, boundary_coords: np.ndarray) -> np.ndarray:
    """Parallel-safe centroid of the boundary node coordinates (the centre
    used for the radial snap-back)."""
    n_loc = int(boundary_coords.shape[0])
    s_loc = (boundary_coords.sum(axis=0)
             if n_loc else np.zeros(mesh.cdim))
    if uw.mpi.size > 1:
        from mpi4py import MPI as _MPI

        s = uw.mpi.comm.allreduce(s_loc, op=_MPI.SUM)
        n = uw.mpi.comm.allreduce(n_loc, op=_MPI.SUM)
    else:
        s, n = s_loc, n_loc
    return s / max(n, 1)


def _slip_normals(mesh, boundary_coords: np.ndarray):
    """Unit outward normals at ``boundary_coords`` from the projected
    boundary-normal field.

    Re-projects ``mesh._projected_normals`` (``mesh.Gamma_P1``) first so the
    normals reflect the mesh's *current* coordinates — the projected field is
    stale after any deform. Returns ``(normals, valid)`` where ``normals`` is
    ``(k, cdim)`` and ``valid`` is a boolean mask; ``valid`` is ``False`` for
    nodes with a degenerate (zero / non-finite) normal (e.g. box corners
    where opposing face normals cancel, or an occasional unlocatable vertex).
    Such nodes should be pinned, not slipped.
    """
    cdim = mesh.cdim
    n = np.zeros((boundary_coords.shape[0], cdim))
    try:
        mesh._update_projected_normals()
        n = np.asarray(
            uw.function.evaluate(mesh.Gamma_P1, boundary_coords)
        ).reshape(-1, cdim)
    except Exception:
        # Projection unavailable / degenerate on this mesh — fall back to
        # all-pinned boundaries (valid stays all-False below).
        n = np.zeros((boundary_coords.shape[0], cdim))
    mag = np.linalg.norm(n, axis=1)
    valid = np.isfinite(mag) & (mag > 0.5)
    out = np.zeros_like(n)
    out[valid] = n[valid] / mag[valid, None]
    return out, valid


def _ot_adapt_step(
    mesh,
    field,
    *,
    refinement=3.0,
    coarsening="auto",
    grad_smoothing_length="auto",
    metric_choice="front-following",
    fields_to_remap=None,
    fields_to_zero=None,
    skip_threshold=None,
    reference_coords=None,
    verbose=False,
) -> bool:
    r"""Run one OT-reset adapt event. Returns ``True`` if the mesh moved,
    ``False`` if the skip-on-aligned check short-circuited.

    See the module docstring for the algorithm. ``field`` is the scalar
    MeshVariable whose gradient drives refinement; it is always FE-remapped
    onto the adapted mesh. ``reference_coords`` overrides the reset target
    for this call only (defaults to ``mesh._ot_adapt_reference_coords``).

    ``grad_smoothing_length`` de-noises ``|∇field|`` before the metric is
    built: ``"auto"`` (default) ≈ the mesh's uniform cell size — the
    validated setting that keeps the metric clean at production refinement;
    ``None`` turns it off; a number or Pint length sets it explicitly
    (user-supplied lengths are unit-aware via the projection's
    non-dimensionalisation).
    """
    cdim = mesh.cdim
    ref_R = float(refinement)
    coar = coarsening
    if coar != "auto":
        coar = float(coar)
    # Resolve the gradient de-noising length: "auto" ≈ uniform grid size.
    if isinstance(grad_smoothing_length, str):
        if grad_smoothing_length.strip().lower() != "auto":
            raise ValueError(
                "grad_smoothing_length string must be 'auto'; got "
                f"{grad_smoothing_length!r}. Pass None (off) or a "
                "unit-aware length.")
        grad_smoothing_length = _auto_grad_smoothing_length(mesh)
    # R for the alignment clamp matches follow_metric: max(refine, coarsen).
    coar_val = (ref_R ** (1.0 / cdim)) if coar == "auto" else float(coar)
    R_clamp = max(ref_R, coar_val)

    if reference_coords is not None:
        ref_X = np.asarray(reference_coords)
    else:
        ref_X = np.asarray(mesh._ot_adapt_reference_coords)

    # For radial coordinate systems (where boundary slip is used), create the
    # projected-normal field up front — before the metric builder / OT mover
    # set up any solver DM. Creating that MeshVariable mid-mover would stale
    # those DM handles (see project_uw3_smoother_footguns). Cartesian meshes
    # pin their boundary (no slip), so no normal field is needed there.
    if _is_radial_coords(mesh):
        try:
            mesh._update_projected_normals()
        except Exception:
            pass

    # --- skip-on-aligned -------------------------------------------------
    if skip_threshold is not None:
        rho_now = uw.meshing.metric_density_from_gradient(
            mesh, field, refinement=ref_R, coarsening=coar,
            metric_choice=metric_choice,
            gradient_smoothing_length=grad_smoothing_length,
            degree=1, name="ot_adapt_skip")
        mm = uw.meshing.mesh_metric_mismatch(
            mesh, rho_now, resolution_ratio=R_clamp)
        if mm["misalignment"] < float(skip_threshold):
            if verbose:
                uw.pprint(
                    f"  OT_adapt: skip — misalignment "
                    f"{mm['misalignment']:.3f} < {float(skip_threshold):.3f}")
            return False

    # Phase-1 remesh redesign: the snapshot/move/transfer dance is now
    # owned by the adapt op via remesh_with_field_transfer. The closure
    # below performs the reset-to-reference + metric-canvas write +
    # OT-mover steps. The helper snapshots `field` (and every other
    # REMAP variable on the mesh — including hidden solver history) at
    # entry, runs the closure (which may clobber `field` for the metric
    # canvas — that write is INTENDED to be discarded by the helper's
    # post-move transfer), then performs ONE deform-back /
    # global_evaluate / deform-forward pair to bring every REMAP var
    # onto the adapted positions. Fields the user previously listed in
    # ``fields_to_remap`` are now transferred automatically; the kwarg
    # is preserved for API compatibility (vars must be REMAP-policy,
    # which is the default — so listing them is a no-op).
    from underworld3.discretisation.remesh import (
        remesh_with_field_transfer)

    def _do_move():
        # Phase-2 ALE opt-out: the OT reset-to-reference step is a
        # discrete jump in node positions, not a smooth displacement,
        # so the linear ``v_mesh = Δx / dt`` interpretation that
        # SemiLagrangian.on_remesh uses for ALE is meaningless here.
        # Publish a flag so DDt hooks fall back to Phase-1 REMAP for
        # this adapt; the mesh's _remesh_pending_scratch dict is the
        # pre-fire channel into ctx.scratch.
        if hasattr(mesh, "_remesh_pending_scratch"):
            scratch = getattr(mesh, "_remesh_pending_scratch", None)
            if scratch is not None:
                scratch["ale_opt_out"] = True

        old_X_local = np.asarray(mesh.X.coords).copy()
        # --- step 1: capture `field` at the reference-mesh DOF positions
        mesh._deform_mesh(ref_X)
        ref_field_coords = np.asarray(field.coords).copy()
        mesh._deform_mesh(old_X_local)
        field_at_ref = np.asarray(
            uw.function.global_evaluate(
                field.sym[0], ref_field_coords)).reshape(-1)
        # --- step 2: load the reference (clean) mesh with the remapped field
        mesh._deform_mesh(ref_X)
        field.data[:, 0] = field_at_ref
        # --- step 3: build the gradient metric + run the OT mover
        rho = uw.meshing.metric_density_from_gradient(
            mesh, field, refinement=ref_R, coarsening=coar,
            metric_choice=metric_choice,
            gradient_smoothing_length=grad_smoothing_length,
            degree=1, name="ot_adapt")
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="ot", boundary_slip=True,
            method_kwargs=dict(n_outer=_OT_N_OUTER, relax=_OT_RELAX,
                               step_frac=_OT_STEP_FRAC),
            verbose=verbose)

    return remesh_with_field_transfer(
        mesh, _do_move,
        extra_zero=fields_to_zero,
        verbose=verbose,
    )

# ===== grafted from feature/elliptic-ma: slip helpers for mmpde mover =====
def _boundary_facets(mesh, cdim):
    """Boundary facets + opposite cell-vertex, found from the cell topology.

    For each cell, every facet (edge in 2D, triangle in 3D) is a candidate
    boundary facet; one that occurs in **exactly one** cell is on the
    boundary. Returns ``(facets, opp)`` where ``facets`` is ``(n_bnd, k)``
    (``k=2`` for 2D edges, ``k=3`` for 3D triangles) and ``opp`` is the
    cell vertex opposite each facet — used to orient the facet normal
    outward. Returns ``(None, None)`` for non-simplicial meshes.
    """
    from underworld3.meshing.smoothing import _tri_cells, _tet_cells
    if cdim == 2:
        cells = _tri_cells(mesh.dm)
        if cells is None:
            return None, None
        rows = []
        for k in range(3):
            v0 = cells[:, k]; v1 = cells[:, (k + 1) % 3]
            vopp = cells[:, (k + 2) % 3]
            vmin = np.minimum(v0, v1); vmax = np.maximum(v0, v1)
            rows.append(np.column_stack([vmin, vmax, vopp]))
        e = np.vstack(rows)
        idx = np.lexsort((e[:, 1], e[:, 0]))
        e = e[idx]
        same_prev = np.zeros(len(e), dtype=bool)
        same_prev[1:] = ((e[1:, 0] == e[:-1, 0])
                         & (e[1:, 1] == e[:-1, 1]))
        same_next = np.zeros(len(e), dtype=bool)
        same_next[:-1] = same_prev[1:]
        bnd_mask = (~same_prev) & (~same_next)
        bnd = e[bnd_mask]
        return bnd[:, :2], bnd[:, 2]
    if cdim == 3:
        cells = _tet_cells(mesh.dm)
        if cells is None:
            return None, None
        rows = []
        for k in range(4):
            others = [(k + 1) % 4, (k + 2) % 4, (k + 3) % 4]
            tri = np.sort(np.column_stack(
                [cells[:, others[0]], cells[:, others[1]],
                 cells[:, others[2]]]), axis=1)
            rows.append(np.column_stack([tri, cells[:, k]]))
        f = np.vstack(rows)
        idx = np.lexsort((f[:, 2], f[:, 1], f[:, 0]))
        f = f[idx]
        same_prev = np.zeros(len(f), dtype=bool)
        same_prev[1:] = ((f[1:, 0] == f[:-1, 0])
                         & (f[1:, 1] == f[:-1, 1])
                         & (f[1:, 2] == f[:-1, 2]))
        same_next = np.zeros(len(f), dtype=bool)
        same_next[:-1] = same_prev[1:]
        bnd_mask = (~same_prev) & (~same_next)
        bnd = f[bnd_mask]
        return bnd[:, :3], bnd[:, 3]
    return None, None


def _all_boundary_labels(mesh):
    """Named codim-1 boundary labels of the mesh, skipping the synthetic /
    non-geometric ones (``All_Boundaries``, ``Null_Boundary``, and the
    Annulus single-point ``Centre`` pseudo-label that hard-aborts PETSc)."""
    skip = {"All_Boundaries", "Null_Boundary", "Centre"}
    out = []
    try:
        names = [b.name for b in mesh.boundaries]
    except Exception:
        names = []
    for nm in names:
        if nm in skip:
            continue
        out.append(nm)
    return tuple(out)


def _label_vertex_mask(dm, label_name):
    """Local-chart boolean vertex mask for one named label (closure of its
    tagged points/edges/faces). Thin single-label wrapper over the same
    logic as :func:`_pinned_mask`."""
    from underworld3.meshing.smoothing import _pinned_mask
    return _pinned_mask(dm, (label_name,))


def _resolve_slip(mesh, slip_spec):
    """Resolve the ``slip_spec`` (the value passed as ``boundary_slip`` /
    ``slip_surfaces``) into a tuple of named slip-surface labels, and
    pre-touch ``mesh.Gamma_P1`` so the projected-normal field ``_n_proj``
    exists BEFORE any mover builds its solver DM (creating that MeshVariable
    mid-mover would stale the DM handle — see project_uw3_smoother_footguns;
    the matrix-free ``mmpde`` mover has no such DM but the elliptic /
    anisotropic movers do).

    Accepted forms (back-compatible):
      * ``True`` / truthy / legacy ``'ring'``,``'box'`` strings → ALL named
        codim-1 boundary surfaces slip.
      * ``False`` / ``None`` / ``[]`` → no slip (pin all boundaries).
      * a label name, or a list of label names → only those surfaces slip.
      * a ``dict`` ``{label: snap_bool}`` → those labels slip; ``snap_bool``
        is the per-surface return-to-bounds flag (``False`` = FREE surface,
        slip but do not snap back). The dict keys are the slip labels.

    Returns the tuple of slip-surface label names (possibly empty).
    """
    if slip_spec is None or slip_spec is False:
        return ()
    if slip_spec is True:
        labels = _all_boundary_labels(mesh)
    elif isinstance(slip_spec, dict):
        labels = tuple(slip_spec.keys())
    elif isinstance(slip_spec, str):
        s = slip_spec.strip().lower()
        if s in ("ring", "box", "axes", "axis", "true", "on", "1", "all"):
            labels = _all_boundary_labels(mesh)
        elif s in ("false", "off", "0", "none", ""):
            return ()
        else:
            labels = (slip_spec,)            # a single explicit label name
    else:
        # an iterable of label names
        labels = tuple(slip_spec)
    if labels:
        # Pre-create the projected-normal field (footgun-safe; see docstring).
        try:
            _ = mesh.Gamma_P1
        except Exception:
            pass
    return labels


def _gamma_p1_at_vertices(mesh, n_verts, cdim):
    """Projected P1 outward unit normal at every local-chart vertex, as an
    ``(n_verts, cdim)`` array. Reads the cached ``_n_proj`` MeshVariable and
    maps its DOF order onto the local-chart vertex order via the vertices'
    coordinates (degree-1 ⇒ one DOF per vertex). Non-boundary rows are
    whatever the projection holds there (unused — only slip rows are read)."""
    _ = mesh.Gamma_P1                                  # ensure built
    nproj = mesh._projected_normals
    ndata = np.asarray(nproj.data).reshape(-1, cdim)
    ncoords = np.asarray(nproj.coords)
    vcoords = np.asarray(mesh.X.coords)
    out = np.zeros((n_verts, cdim))
    if ndata.shape[0] == vcoords.shape[0]:
        # Common case: same count — match by nearest coordinate (robust to
        # any DOF-vs-vertex reordering).
        from scipy.spatial import cKDTree
        tree = cKDTree(ncoords)
        _, idx = tree.query(vcoords)
        out[:] = ndata[idx]
    else:
        from scipy.spatial import cKDTree
        tree = cKDTree(ncoords)
        _, idx = tree.query(vcoords)
        out[:] = ndata[idx]
    # renormalise (projection may leave |n|≈1 but be safe)
    mag = np.linalg.norm(out, axis=1)
    ok = mag > 1.0e-30
    out[ok] /= mag[ok, None]
    return out


def _nearest_on_facets_2d(pts, seg):
    """Closest point on a set of 2D line segments. ``pts`` (m,2),
    ``seg`` (nf,2,2). Returns (m,2) closest points (over all segments)."""
    a = seg[:, 0]; b = seg[:, 1]            # (nf,2)
    ab = b - a
    ab2 = np.einsum('fi,fi->f', ab, ab)
    ab2 = np.where(ab2 > 1.0e-30, ab2, 1.0)
    out = np.empty_like(pts)
    for i, p in enumerate(pts):
        t = np.clip(((p - a) * ab).sum(axis=1) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab           # (nf,2)
        d2 = ((proj - p) ** 2).sum(axis=1)
        out[i] = proj[d2.argmin()]
    return out


def _nearest_on_facets_3d(pts, tri):
    """Closest point on a set of 3D triangles. ``pts`` (m,3),
    ``tri`` (nf,3,3). Returns (m,3). Per-point loop, vectorised over
    triangles via the standard region-based closest-point algorithm."""
    A = tri[:, 0]; B = tri[:, 1]; C = tri[:, 2]
    AB = B - A; AC = C - A
    out = np.empty_like(pts)
    for i, p in enumerate(pts):
        AP = p - A
        d1 = np.einsum('fi,fi->f', AB, AP)
        d2 = np.einsum('fi,fi->f', AC, AP)
        BP = p - B
        d3 = np.einsum('fi,fi->f', AB, BP)
        d4 = np.einsum('fi,fi->f', AC, BP)
        CP = p - C
        d5 = np.einsum('fi,fi->f', AB, CP)
        d6 = np.einsum('fi,fi->f', AC, CP)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = va + vb + vc
        denom = np.where(np.abs(denom) > 1.0e-30, denom, 1.0)
        v = vb / denom
        w = vc / denom
        # interior barycentric point; clamp handles edge/vertex regions well
        # enough for a small return-to-bounds correction on convex surfaces.
        v = np.clip(v, 0.0, 1.0); w = np.clip(w, 0.0, 1.0)
        s = v + w
        over = s > 1.0
        v = np.where(over, v / np.where(s > 0, s, 1.0), v)
        w = np.where(over, w / np.where(s > 0, s, 1.0), w)
        proj = A + v[:, None] * AB + w[:, None] * AC
        dd = ((proj - p) ** 2).sum(axis=1)
        out[i] = proj[dd.argmin()]
    return out


def _build_slip_projector(mesh, old_coords, is_bnd, n_verts, slip_spec):
    """Build ``(is_pinned, project_fn)`` for named-surface tangent slip,
    shared by all metric movers.

    ``slip_spec`` is whatever ``_resolve_slip`` accepts (``True`` = all
    boundaries, a label, a list of labels, or a ``dict`` ``{label: snap_bool}``
    whose ``False`` values mark FREE surfaces that slip without snapping back).
    For each named slip surface:

      * **slip-vs-pin is label-driven** (not normal-agreement): a boundary
        vertex slips iff it belongs to **exactly one** slip surface. Vertices
        on a non-slip boundary (count 0) or at a **junction** of two slip
        surfaces (count ≥2 — e.g. a box corner, where the normal is
        ambiguous) are pinned. This fixes the old topology classifier, which
        spuriously pinned a *coarse but smooth* curved ring (adjacent facet
        normals diverge >15° on a low-resolution polygon, yet it is no
        corner).
      * the tangential slide uses the **projected P1 normal**
        (:attr:`mesh.Gamma_P1`) — smooth and consistently oriented, reliable
        on curved boundaries where the raw face normal is noisy.
      * **return-to-bounds**: after the tangent step, each slip node is
        re-projected onto the nearest point of its surface's **reference
        facets** (captured once from ``old_coords``), so it stays on the
        (convex) surface instead of creeping inward chord-wise over many
        iterations. A surface whose dict value is ``False`` skips this (FREE
        surfaces, where the geometry is itself the unknown).
    """
    slip_labels = _resolve_slip(mesh, slip_spec)
    # FREE surfaces (snap_bool == False in a dict spec) slip but don't snap.
    no_snap = (
        {lab for lab, snap in slip_spec.items() if not snap}
        if isinstance(slip_spec, dict) else set()
    )
    if not (slip_labels and is_bnd.any()):
        def _project(Y):
            return Y
        return is_bnd.copy(), _project

    cdim = mesh.cdim
    dm = mesh.dm
    # per-label vertex masks → slip count per vertex
    label_masks = {lab: _label_vertex_mask(dm, lab) for lab in slip_labels}
    count = np.zeros(n_verts, dtype=int)
    for m in label_masks.values():
        count += m.astype(int)
    slip_mask = is_bnd & (count == 1)            # exactly one slip surface
    is_pinned = is_bnd & ~slip_mask              # non-slip + junctions pinned
    slip_b = np.nonzero(slip_mask)[0]
    if slip_b.size == 0:
        def _project(Y):
            return Y
        return is_pinned, _project

    n_all = _gamma_p1_at_vertices(mesh, n_verts, cdim)
    n_slip = n_all[slip_b]
    old_slip = old_coords[slip_b]

    # Return-to-bounds. Two snap modes, per the design's cure menu:
    #   (1) ANALYTIC snap for known radial geometries (annulus / sphere /
    #       cylinder) — re-impose each slip node's reference |r| about the
    #       boundary centre. EXACT (no chord sag) and, crucially, free of the
    #       concave-inward bias the facet snap suffers on the inner ring.
    #   (2) FACET snap (nearest reference boundary facet) as the
    #       geometry-general fallback for surfaces with no analytic form.
    # FREE surfaces (dict value False) skip snapping in either mode.
    radial = _is_radial_coords(mesh)
    centre = r_target = snap_radial = None
    if radial:
        bidx = np.nonzero(is_bnd)[0]
        centre = _boundary_centre(mesh, old_coords[bidx])
        # reference radius per slip vertex (each ring snaps to its own |r|)
        r_target = np.linalg.norm(old_slip - centre, axis=1)
        # snap unless the vertex's slip surface is FREE (no_snap)
        free_vert = np.zeros(n_verts, dtype=bool)
        for lab in no_snap:
            free_vert |= label_masks[lab]
        snap_radial = ~free_vert[slip_b]

    # Reference facets per slip label, for the FACET fallback. A boundary
    # facet belongs to label L iff all its vertices carry L; captured from
    # old_coords (the FIXED reference surface).
    facets, _opp = _boundary_facets(mesh, cdim)
    snap_facets_by_label = {}
    if (not radial) and facets is not None and facets.size:
        for lab, lm in label_masks.items():
            if lab in no_snap:
                continue
            fac_in = lm[facets].all(axis=1)      # facet fully in label L
            if fac_in.any():
                snap_facets_by_label[lab] = old_coords[facets[fac_in]]
    # vertex -> its (single) slip label, for facet-snap routing
    vert_label = np.empty(n_verts, dtype=object)
    for lab, lm in label_masks.items():
        vert_label[lm & slip_mask] = lab

    def _project(Y):
        # tangential slide: remove the projected-normal component
        disp = Y[slip_b] - old_slip
        dn = (disp * n_slip).sum(axis=1, keepdims=True)
        Y[slip_b] = old_slip + (disp - dn * n_slip)
        if radial:
            # (1) analytic |r| snap — exact, concave-safe; skip FREE surfaces
            v = Y[slip_b] - centre
            nrm = np.linalg.norm(v, axis=1)
            nrm = np.where(nrm > 1.0e-30, nrm, 1.0)
            snapped = centre + v * (r_target / nrm)[:, None]
            Y[slip_b] = np.where(snap_radial[:, None], snapped, Y[slip_b])
        else:
            # (2) facet fallback. TODO(watch): facet return-to-bounds is
            # exact-to-the-POLYGON — safe for CONVEX surfaces but biases a
            # CONCAVE one (chords sit inside the true arc, so nodes creep
            # inward over many iterations). Radial geometries take the
            # analytic branch above and are immune; a genuinely concave,
            # non-analytic surface would need a smoothness / mean-preserving
            # constraint (cure (2) in the design). Watching how fast it
            # degrades on such a case before adding that.
            for lab, fcoords in snap_facets_by_label.items():
                sel = np.array([vert_label[v] == lab for v in slip_b])
                if not sel.any():
                    continue
                pts = Y[slip_b[sel]]
                if cdim == 2:
                    Y[slip_b[sel]] = _nearest_on_facets_2d(pts, fcoords)
                else:
                    Y[slip_b[sel]] = _nearest_on_facets_3d(pts, fcoords)
        return Y

    return is_pinned, _project


