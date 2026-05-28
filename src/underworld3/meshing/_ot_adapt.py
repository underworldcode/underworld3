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

Boundary slip uses **topology-based outward vertex normals**
(:func:`_boundary_vertex_normals`) — the geometric face normals of the
boundary facets incident to each vertex, area-weighted averaged. This is
truly generic: works on any 2D/3D simplicial mesh (Cartesian box, annulus,
sphere, polyhedron, curved surface) because it depends only on the cell
coordinates and connectivity, not on a symbolic normal field. (The old
``mesh.Gamma_P1`` path evaluated PETSc's quadrature-point ``petsc_n``
symbol at *vertices* — undefined off boundary quadrature, which is why it
gave garbage normals on Cartesian boxes.) Face vertices slide tangentially;
corners / edges (where incident facet normals disagree by more than
~15°) are pinned. For radial coordinate systems a snap-back to fixed
``|r|`` is layered on top so curved boundaries stay on the surface.
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


def _boundary_vertex_normals(mesh, parallel_tol_deg=15.0):
    """Topology-based outward unit normal at each boundary vertex.

    The generic alternative to ``mesh.Gamma_P1`` — works on any 2D/3D
    simplicial mesh (Cartesian box, annulus, sphere, polyhedron, curved
    surface), because the boundary facet normals are computed
    **geometrically** from the cell coordinates (not from the symbolic
    PETSc face-normal ``petsc_n``, which is only defined at boundary
    integration points and gives garbage when evaluated at vertices —
    why ``Gamma_P1`` is unreliable on a Cartesian box).

    For each boundary vertex, the per-facet outward normals are
    **area-weighted averaged**, then we classify by how strongly the
    incident normals agree:

    * **face slip** (``is_face_slip=True``): all incident-facet normals lie
      within ``parallel_tol_deg`` of the average → the vertex sits on one
      smooth face (or a single tangent plane). Tangential slide is well-
      defined; the projector removes the displacement's component along
      ``normal``.
    * **pin** (``is_face_slip=False``): the incident normals disagree
      (corner, edge between two faces in 3D, …). The simple-and-safe
      treatment is to pin these.

    Returns ``(normals, is_face_slip)`` of shape ``(n_verts, cdim)`` and
    ``(n_verts,)``; non-boundary vertices have zero normal and False.
    """
    cdim = mesh.cdim
    facets, opp = _boundary_facets(mesh, cdim)
    coords = np.asarray(mesh.X.coords)
    n_verts = coords.shape[0]
    if facets is None or len(facets) == 0:
        return (np.zeros((n_verts, cdim)),
                np.zeros(n_verts, dtype=bool))

    if cdim == 2:
        p0 = coords[facets[:, 0]]; p1 = coords[facets[:, 1]]
        t = p1 - p0; tlen = np.linalg.norm(t, axis=1)
        t = t / np.where(tlen > 1.0e-30, tlen, 1.0)[:, None]
        ncand = np.stack([-t[:, 1], t[:, 0]], axis=1)
        mid = 0.5 * (p0 + p1)
        out = mid - coords[opp]
        sgn = np.sign(np.einsum("ij,ij->i", out, ncand))
        sgn = np.where(sgn == 0, 1.0, sgn)
        fnorm = ncand * sgn[:, None]
        farea = tlen                                       # edge length
    else:
        p0 = coords[facets[:, 0]]; p1 = coords[facets[:, 1]]
        p2 = coords[facets[:, 2]]
        cross = np.cross(p1 - p0, p2 - p0)
        clen = np.linalg.norm(cross, axis=1)
        ncand = cross / np.where(clen > 1.0e-30, clen, 1.0)[:, None]
        centr = (p0 + p1 + p2) / 3.0
        out = centr - coords[opp]
        sgn = np.sign(np.einsum("ij,ij->i", out, ncand))
        sgn = np.where(sgn == 0, 1.0, sgn)
        fnorm = ncand * sgn[:, None]
        farea = 0.5 * clen                                 # triangle area

    sum_n = np.zeros((n_verts, cdim))
    for col in range(facets.shape[1]):
        np.add.at(sum_n, facets[:, col], fnorm * farea[:, None])
    nmag = np.linalg.norm(sum_n, axis=1)
    on = nmag > 1.0e-30
    avg = np.zeros_like(sum_n)
    avg[on] = sum_n[on] / nmag[on, None]

    # classify: a boundary vertex is "face-slip" iff every incident facet
    # normal is within `parallel_tol_deg` of the average — i.e. it sits on
    # one smooth face.
    cos_tol = float(np.cos(np.radians(parallel_tol_deg)))
    bad_count = np.zeros(n_verts, dtype=int)
    for col in range(facets.shape[1]):
        vi = facets[:, col]
        cos_a = np.einsum("ij,ij->i", fnorm, avg[vi])
        bad = cos_a < cos_tol
        np.add.at(bad_count, vi[bad], 1)
    is_face_slip = on & (bad_count == 0)
    return avg, is_face_slip


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
    # Generic topology-based slip works on any 2D/3D simplicial mesh —
    # Cartesian boxes, annulus, sphere, polyhedra. No radial gate.
    return req


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

    # Topology-based outward vertex normals — generic across geometries
    # (Cartesian boxes, annulus, sphere, polyhedra, curved surfaces).
    # Face-slip vertices get a tangential slide; corners/edges (where
    # incident facet normals disagree) are pinned.
    avg_n, is_face_slip = _boundary_vertex_normals(mesh)
    slip_mask = is_bnd & is_face_slip
    is_pinned = is_bnd & ~slip_mask              # everything on the boundary
                                                  # that isn't face-slip
    slip_b = np.nonzero(slip_mask)[0]
    if slip_b.size == 0:
        def _project(Y):
            return Y
        return is_pinned, _project
    n_slip = avg_n[slip_b]
    old_slip = old_coords[slip_b]
    radial = _is_radial_coords(mesh)
    if radial:
        bidx = np.nonzero(is_bnd)[0]
        centre = _boundary_centre(mesh, old_coords[bidx])
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
