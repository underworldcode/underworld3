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


def _resolve_slip(mesh, boundary_slip):
    """Resolve ``boundary_slip`` (bool, or legacy ``'ring'/'box'/'axes'``
    string) to a radial-gated slip-on flag, and pre-create the projected
    boundary-normal field (footgun-safe) so the mover can read it.

    Projected-normal slip is reliable only for *radial* coordinate systems
    (cylindrical / spherical / geographic), where ``mesh.Gamma`` is the
    coordinate-derived radial field and evaluates cleanly at vertices; for
    Cartesian boundaries the vertex normal is degenerate, so we pin instead.
    Call this ONCE before the mover builds its solver DM — creating the
    ``_n_proj`` MeshVariable mid-mover would stale that DM handle
    (project_uw3_smoother_footguns). Returns the bool slip-on flag.
    """
    if isinstance(boundary_slip, str):
        req = boundary_slip.strip().lower() in (
            "ring", "box", "axes", "axis", "true", "on", "1")
    else:
        req = bool(boundary_slip)
    slip_on = req and _is_radial_coords(mesh)
    if slip_on:
        try:
            mesh._update_projected_normals()
        except Exception:
            slip_on = False
    return slip_on


def _build_slip_projector(mesh, old_coords, is_bnd, n_verts, slip_on):
    """Build ``(is_pinned, project_fn)`` for the unified Gamma_N boundary
    slip, shared by the OT and Monge–Ampère movers.

    Boundary nodes slide tangentially — ``project_fn`` zeros the
    projected-normal component of their displacement — and, for radial
    coordinate systems, snaps them back to their reference ``|r|`` so they
    stay exactly on the surface. Nodes with a degenerate projected normal
    (box corners where opposing face normals cancel, or an occasional
    unlocatable vertex) are pinned. When ``slip_on`` is False (or there is no
    boundary) the boundary is fully pinned.
    """
    if not (slip_on and is_bnd.any()):
        def _project(Y):
            return Y
        return is_bnd.copy(), _project

    bidx = np.nonzero(is_bnd)[0]
    bcoords = old_coords[bidx]
    n_hat, valid = _slip_normals(mesh, bcoords)
    slip_b = bidx[valid]
    is_pinned = np.zeros(n_verts, dtype=bool)
    is_pinned[bidx[~valid]] = True            # degenerate-normal nodes pinned
    n_slip = n_hat[valid]
    old_slip = old_coords[slip_b]
    radial = _is_radial_coords(mesh)
    if radial:
        centre = _boundary_centre(mesh, bcoords)
        r_target = np.linalg.norm(old_slip - centre, axis=1)

    def _project(Y):
        # tangential slide: remove the normal component of the displacement
        disp = Y[slip_b] - old_slip
        dn = (disp * n_slip).sum(axis=1, keepdims=True)
        Y[slip_b] = old_slip + (disp - dn * n_slip)
        # snap curved boundaries back onto the surface (fixed |r|)
        if radial:
            v = Y[slip_b] - centre
            nrm = np.linalg.norm(v, axis=1)
            nrm = np.where(nrm > 1.0e-30, nrm, 1.0)
            Y[slip_b] = centre + v * (r_target / nrm)[:, None]
        return Y

    return is_pinned, _project


def _build_box_slip_projector(mesh, ref_coords, is_bnd, n_verts, cdim,
                              tol=None):
    """Axis-aligned **box-face** boundary slip (the Cartesian counterpart to
    the radial ring/normal slip).

    The projected boundary normal (``Gamma_P1``) is degenerate at the
    vertices of a Cartesian box (opposing face normals cancel; raw
    ``Gamma_N`` is even NaN there), so the projected-normal slip of
    :func:`_build_slip_projector` cannot be used. Instead detect the
    axis-aligned bounding-box faces geometrically from ``ref_coords`` (the
    *undeformed* reference coordinates): a boundary node on exactly one face
    slides **along** that face (its perpendicular coordinate is snapped back
    to the face plane each step), while a node on two or more faces (a box
    edge / corner) is pinned. This lets a fault that reaches the domain
    boundary refine across it on both ends, instead of being blocked by a
    fully-pinned boundary.

    Unlike the projected-normal path this creates **no** MeshVariable, so it
    is free of the mid-mover DM-stale footgun. If the domain is not an
    axis-aligned box (some boundary node lies off every extent plane) the
    boundary is fully pinned (safe fallback).

    Returns ``(is_pinned, project_fn)``.
    """
    bidx = np.nonzero(is_bnd)[0]
    if bidx.size == 0:
        return is_bnd.copy(), (lambda Y: Y)
    bcoords = np.asarray(ref_coords)[bidx]
    lo = bcoords.min(axis=0)
    hi = bcoords.max(axis=0)
    if tol is None:
        ext = float(np.max(hi - lo)) if (hi - lo).size else 0.0
        tol = 1.0e-6 * ext if ext > 0.0 else 1.0e-9
    # on[i, j, side] : boundary node i sits on the lo/hi extent plane of dim j
    on = np.zeros((bidx.size, cdim, 2), dtype=bool)
    for j in range(cdim):
        on[:, j, 0] = np.abs(bcoords[:, j] - lo[j]) < tol
        on[:, j, 1] = np.abs(bcoords[:, j] - hi[j]) < tol
    nfaces = on.reshape(bidx.size, -1).sum(axis=1)
    if not bool((nfaces >= 1).all()):
        # not an axis-aligned box — pin everything (safe)
        return is_bnd.copy(), (lambda Y: Y)

    is_pinned = np.zeros(n_verts, dtype=bool)
    pin_local = nfaces >= 2                     # edges / corners
    is_pinned[bidx[pin_local]] = True
    slip_local = ~pin_local
    slip_b = bidx[slip_local]
    on_slip = on[slip_local]                    # (n_slip, cdim, 2)
    # the single fixed dimension and its plane value for each slip node
    fixed_dim = np.argmax(on_slip.any(axis=2), axis=1)      # (n_slip,)
    plane_val = np.where(on_slip[np.arange(slip_b.size), fixed_dim, 0],
                         lo[fixed_dim], hi[fixed_dim])

    def _project(Y):
        # snap each face node's perpendicular coordinate back to its plane;
        # the tangential coordinate(s) move freely.
        Y[slip_b, fixed_dim] = plane_val
        return Y

    return is_pinned, _project


def _ot_adapt_step(
    mesh,
    field,
    *,
    refinement=3.0,
    coarsening="auto",
    grad_smoothing_length="auto",
    metric_choice="front-following",
    mover="ot",
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

    old_X = np.asarray(mesh.X.coords).copy()

    # Fields to FE-remap: `field` is always remapped; append extras (deduped).
    remap = [field]
    for f in (fields_to_remap or []):
        if f is not field and f not in remap:
            remap.append(f)
    old_data = {f: np.asarray(f.data).copy() for f in remap}

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

    # --- step 1: capture `field` at the reference-mesh DOF positions -----
    mesh._deform_mesh(ref_X)
    ref_field_coords = np.asarray(field.coords).copy()
    mesh._deform_mesh(old_X)
    field.data[...] = old_data[field]
    field_at_ref = np.asarray(
        uw.function.evaluate(field.sym[0], ref_field_coords)).reshape(-1)

    # --- step 2: load the reference (clean) mesh with the remapped field -
    mesh._deform_mesh(ref_X)
    field.data[:, 0] = field_at_ref

    # --- step 3: build the gradient metric + run the OT mover ------------
    rho = uw.meshing.metric_density_from_gradient(
        mesh, field, refinement=ref_R, coarsening=coar,
        metric_choice=metric_choice,
        gradient_smoothing_length=grad_smoothing_length,
        degree=1, name="ot_adapt")
    if mover in ("ma", "monge-ampere", "monge_ampere"):
        # Elliptic Monge–Ampère: one Caffarelli-clean convex-potential map
        # from the reset canvas (untangled by construction; no polish).
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="ma", boundary_slip=True,
            method_kwargs=dict(n_outer=1, n_picard=25), verbose=verbose)
    elif mover in ("ot", "equidistribute"):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="ot", boundary_slip=True,
            method_kwargs=dict(n_outer=_OT_N_OUTER, relax=_OT_RELAX,
                               step_frac=_OT_STEP_FRAC),
            verbose=verbose)
    else:
        raise ValueError(
            f"OT_adapt mover must be 'ot' or 'ma', got {mover!r}")
    new_X = np.asarray(mesh.X.coords).copy()

    # --- step 4: FE-remap all fields from old_X onto the adapted mesh ----
    # The metric-canvas write to `field` (step 2) is discarded here by
    # design: every remapped field is re-derived from its *original*
    # (old_X) data, so the final field is the true physical field carried
    # onto the new positions.
    new_coords = {f: np.asarray(f.coords).copy() for f in remap}
    mesh._deform_mesh(old_X)
    for f in remap:
        f.data[...] = old_data[f]
    remapped = {}
    for f in remap:
        val = np.asarray(uw.function.evaluate(f.sym, new_coords[f]))
        remapped[f] = val.reshape(np.asarray(f.data).shape)
    mesh._deform_mesh(new_X)
    for f in remap:
        f.data[...] = remapped[f]
    for f in (fields_to_zero or []):
        f.data[...] = 0.0

    return True
