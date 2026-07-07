r"""Optimal-transport mesh adaptation — the validated reset-to-uniform step.

This module factors the production pattern that was inlined in
``scripts/stagnant_lid_adapt_loop.py`` (the ``ot-reset`` branch) into a
reusable library function. The public entry point is :meth:`Mesh.OT_adapt`
(see ``discretisation/discretisation_mesh.py``); this module holds the
algorithm and the boundary-slip helpers it shares with the OT mover
(``_ot_improvement_step`` in ``smoothing.py``).

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
