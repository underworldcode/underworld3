"""Topology, boundary masks, parallel scalar reductions, and the
graph-based smoothing primitives shared by every mover.

Part of the ``underworld3.meshing.smoothing`` package — see the package
docstring in ``__init__.py`` for the module map.
"""

from typing import Optional, Sequence

import numpy as np
from mpi4py import MPI as _MPI

import underworld3 as uw


def _global_sum(value):
    """Scalar MPI SUM of a rank-local value (as float; serial no-op)."""
    if uw.mpi.size > 1:
        return uw.mpi.comm.allreduce(float(value), op=_MPI.SUM)
    return float(value)


def _global_min(value):
    """Scalar MPI MIN of a rank-local value (as float; serial no-op)."""
    if uw.mpi.size > 1:
        return uw.mpi.comm.allreduce(float(value), op=_MPI.MIN)
    return float(value)


def _global_max(value):
    """Scalar MPI MAX of a rank-local value (as float; serial no-op)."""
    if uw.mpi.size > 1:
        return uw.mpi.comm.allreduce(float(value), op=_MPI.MAX)
    return float(value)


def _global_mean(value):
    """Mean over RANKS of a rank-local scalar (allreduce / size; serial
    no-op). NOTE: this is the movers' historical rank-mean of rank-local
    means — cheap and adequate for the scale factors it feeds (h0, patch
    normalisers), not an ownership-weighted global mean. Kept bit-for-bit."""
    if uw.mpi.size > 1:
        return uw.mpi.comm.allreduce(float(value)) / uw.mpi.size
    return float(value)


def _auto_pinned_labels(mesh) -> tuple:
    """All non-sentinel geometric boundary labels on the mesh.

    Skips ``All_Boundaries`` / ``Null_Boundary`` (sentinels) and
    known non-geometric pressure-pin markers such as ``Centre`` on
    the Annulus (a single-point marker whose underlying ``DMLabel``
    has an invalid communicator and hard-crashes any
    ``getNumValues`` / ``getValueIS`` / ``view`` call).
    """
    skip = {"All_Boundaries", "Null_Boundary", "Centre"}
    names = []
    for member in mesh.boundaries:
        name = getattr(member, "name", None)
        if name and name not in skip:
            names.append(name)
    return tuple(names)


def _owned_vertex_mask(dm):
    """Local-chart boolean mask: True for owned vertices, False for
    ghosts (leaves of the point StarForest). Used by the parallel
    tests; the smoother itself derives ownership from the global
    section attached to its scalar DM clone.
    """
    pStart, pEnd = dm.getDepthStratum(0)
    n_verts = pEnd - pStart
    is_owned = np.ones(n_verts, dtype=bool)
    sf = dm.getPointSF()
    if sf is None:
        return is_owned
    try:
        _n_roots, leaves, _remote = sf.getGraph()
    except Exception:
        return is_owned
    if leaves is None or len(leaves) == 0:
        return is_owned
    for leaf in leaves:
        if pStart <= leaf < pEnd:
            is_owned[leaf - pStart] = False
    return is_owned


def _pinned_mask(dm, pinned_labels):
    """Local-chart boolean mask: True where the vertex belongs to (or
    is the endpoint of an edge in) any of ``pinned_labels``.

    UW3 mesh generators tag boundaries by EDGE rather than by
    vertex; the vertex stratum sometimes misses 1-2 endpoint
    vertices at the gmsh seam (e.g. θ=0°/180° on the Annulus outer
    rim). Pinning by vertex-stratum-only would leave those
    "seam" vertices free, and the smoother would pull them
    inward. Taking the closure of the tagged edges recovers them.

    Tolerates labels that are present but empty (e.g. the
    ``Centre`` pressure-pin marker on an Annulus, whose underlying
    ``DMLabel`` has no strata and hard-crashes any query)."""
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    n_verts = pEnd - pStart
    is_pinned = np.zeros(n_verts, dtype=bool)
    for lname in pinned_labels:
        label = dm.getLabel(lname)
        if label is None:
            continue
        try:
            if label.getNumValues() == 0:
                continue
            vIS = label.getValueIS()
        except Exception:
            continue
        if vIS is None:
            continue
        for val in vIS.getIndices():
            try:
                iset = label.getStratumIS(int(val))
            except Exception:
                continue
            if iset is None:
                continue
            for idx in iset.getIndices():
                if pStart <= idx < pEnd:
                    # Tagged vertex — pin directly.
                    is_pinned[idx - pStart] = True
                elif eStart <= idx < eEnd:
                    # Tagged edge — pin both endpoint vertices.
                    cone = dm.getCone(idx)
                    for c in cone:
                        if pStart <= c < pEnd:
                            is_pinned[c - pStart] = True
    return is_pinned


def _build_scalar_dm(dm):
    """A clone of the topological DM with a 1-dof-per-vertex local
    section. Used to size the adjacency Mat and to produce the global
    vertex numbering."""
    from petsc4py import PETSc
    chart_start, chart_end = dm.getChart()
    pStart, pEnd = dm.getDepthStratum(0)
    section = PETSc.Section().create(comm=dm.getComm())
    section.setChart(chart_start, chart_end)
    for p in range(chart_start, chart_end):
        section.setDof(p, 1 if pStart <= p < pEnd else 0)
    section.setUp()
    dm_scalar = dm.clone()
    dm_scalar.setLocalSection(section)
    return dm_scalar


def _build_adjacency_matrix(mesh):
    """Build the parallel vertex-vertex adjacency as a PETSc AIJ Mat.

    Each rank inserts entries for every locally-visible edge using
    GLOBAL vertex indices; ``mat.assemble()`` combines cross-rank
    contributions, so that after assembly an owned-vertex row has
    every neighbour it would in a serial run — even when the
    incident edge lives in a cell owned by another rank that is not
    in this rank's overlap.

    Returns
    -------
    A : PETSc.Mat
        Unweighted vertex-vertex adjacency, entries are 1.0 where an
        edge exists. Divide the result of ``A @ x`` by the degree
        vector to get the neighbour average.
    dm_scalar : PETSc.DMPlex
        Clone of ``mesh.dm`` with a 1-dof-per-vertex section. Owns
        the parallel layout for the Mat and any vectors of the same
        shape.
    gsection : PETSc.Section
        Global section of ``dm_scalar`` — the owned-vertex numbering.
    """
    from petsc4py import PETSc
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)

    dm_scalar = _build_scalar_dm(dm)
    gsection = dm_scalar.getGlobalSection()

    def gidx(p):
        off = gsection.getOffset(p)
        return off if off >= 0 else -(off + 1)

    A = dm_scalar.createMatrix()
    A.setOption(A.Option.NEW_NONZERO_LOCATION_ERR, False)
    A.setOption(A.Option.IGNORE_OFF_PROC_ENTRIES, False)

    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0], cone[1]
        if not (pStart <= v0 < pEnd and pStart <= v1 < pEnd):
            continue
        g0, g1 = gidx(v0), gidx(v1)
        A.setValues([g0], [g1], [1.0], PETSc.InsertMode.INSERT)
        A.setValues([g1], [g0], [1.0], PETSc.InsertMode.INSERT)
    A.assemble()
    return A, dm_scalar, gsection


def _min_incident_edge(dm, coords):
    """Per-vertex minimum incident edge length (local-chart
    v-pStart order). Used as an optional secondary per-node cap on
    the spring step (the primary tangle guard is the coherent global
    signed-area backtrack in ``_spring_equilibrium_mover``)."""
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    h = np.full(pEnd - pStart, np.inf)
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0], cone[1]
        if not (pStart <= v0 < pEnd and pStart <= v1 < pEnd):
            continue
        i0, i1 = v0 - pStart, v1 - pStart
        L = float(np.linalg.norm(coords[i0] - coords[i1]))
        if L < h[i0]:
            h[i0] = L
        if L < h[i1]:
            h[i1] = L
    return h


def _tri_cells(dm):
    """Triangle vertex-index triples (local-chart, v-pStart order).

    Returns an ``(n_tri, 3)`` int array, or ``None`` if the mesh is
    not all-triangle (then the global signed-area backtrack is
    skipped and only the optional per-node edge cap guards against
    tangling).
    """
    cStart, cEnd = dm.getHeightStratum(0)
    pStart, pEnd = dm.getDepthStratum(0)
    tris = []
    for c in range(cStart, cEnd):
        closure = dm.getTransitiveClosure(c)[0]
        vs = [p - pStart for p in closure if pStart <= p < pEnd]
        if len(vs) != 3:
            return None
        tris.append(vs)
    if not tris:
        return None
    return np.asarray(tris, dtype=np.int64)


def _signed_areas(coords, tris):
    """Signed area of each triangle (sign = orientation)."""
    a = coords[tris[:, 0]]
    b = coords[tris[:, 1]]
    c = coords[tris[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _cap_step_to_edge_fraction(step, dm, coords, step_frac):
    """Per-vertex displacement cap: ``|step_i| <= step_frac * h_i`` with
    ``h_i`` the shortest edge incident on vertex ``i``.

    Prevents a mover step from creating LOCAL cell folds near sharp
    features (where the source is strongest) without killing the global
    motion the way the coherent global signed-area backtrack does.
    No-op when ``step_frac`` is ``None`` or non-finite. (The MMPDE mover
    has its own, structurally different per-node cap — do not fold it in
    here.)"""
    if step_frac is None or not np.isfinite(step_frac):
        return step
    h = _min_incident_edge(dm, coords)
    mag = np.linalg.norm(step, axis=1)
    cap = float(step_frac) * h
    clip = np.isfinite(cap) & (mag > cap) & (mag > 0.0)
    sc = np.ones_like(mag)
    sc[clip] = cap[clip] / mag[clip]
    return step * sc[:, None]


def _backtracked_move(old_coords, step, free, tris, project,
                      area_floor=0.0):
    """Coherent global signed-area backtrack shared by the MA / OT /
    anisotropic movers.

    Apply ``step`` to the ``free`` vertices at a global scale, halving
    the scale (up to 10 times) until no triangle inverts and none drops
    to (or below) ``area_floor``. The min-area test is reduced globally
    (MPI MIN) so every rank takes the same accept/backtrack branch — the
    loop is collective. ``project`` re-imposes the boundary-slip
    constraint on each trial (slip vertices snap back to their bounding
    surface).

    ``area_floor=0.0`` reproduces the historical flip-only acceptance
    (``a1min > 0``) bit-for-bit; the anisotropic mover passes a positive
    floor (a fraction of the undeformed median cell area) so
    near-degenerate slivers are rejected as well as inverted cells.
    ``tris is None`` (non-triangle mesh) applies the step unguarded.

    Returns
    -------
    (new_coords, scale) : (ndarray, float)
        ``scale == 0.0`` means no acceptable move was found and
        ``new_coords`` equals ``old_coords``.
    """
    scale = 1.0
    new_coords = old_coords.copy()
    if tris is not None:
        a0 = _signed_areas(old_coords, tris)
        orient = np.sign(np.median(a0)) or 1.0
        for _bt in range(10):
            trial = old_coords.copy()
            trial[free] += scale * step[free]
            trial = project(trial)
            a1min = _global_min(
                (_signed_areas(trial, tris) * orient).min())
            if a1min > area_floor:
                new_coords = trial
                break
            scale *= 0.5
        else:
            scale = 0.0
            new_coords = old_coords.copy()
    else:
        new_coords[free] += step[free]
        new_coords = project(new_coords)
    return new_coords, scale


def _reweight_displacement_radial_tangential(disp, coords,
                                             move_anisotropy):
    """Directional move-weighting (approach (2), opt-in; 2D only).

    The annulus node budget is anisotropic — radial is scarce and
    pinned, tangential is abundant and free ("spare" angular nodes). A
    scalar equidistribution is isotropic and cannot express "prefer
    tangential"; rescale the realised displacement in the local
    radial / tangential frame (``move_anisotropy = (w_r, w_θ)``) so
    the same metric is met mostly by sliding nodes around rather than
    crushing radially. Lightweight and solver-consistent — the mover's
    operator algebra is untouched, only the realised move is
    reweighted. Centre = the coordinate centroid (the origin for a
    centred annulus). Degenerate radii (< 1e-30) keep a zero frame and
    therefore a zero reweighted move."""
    w_r, w_t = (float(move_anisotropy[0]),
                float(move_anisotropy[1]))
    ctr = coords.mean(axis=0)
    rv = coords - ctr
    rn = np.linalg.norm(rv, axis=1)
    ok = rn > 1.0e-30
    rhat = np.zeros_like(rv)
    rhat[ok] = rv[ok] / rn[ok, None]
    that = np.stack([-rhat[:, 1], rhat[:, 0]], axis=1)
    d_r = (disp * rhat).sum(axis=1)
    d_t = (disp * that).sum(axis=1)
    return (w_r * d_r[:, None] * rhat
            + w_t * d_t[:, None] * that)


def _edge_pairs(dm):
    """``(n_edge, 2)`` int array of edge endpoint vertex indices in
    local-chart (v - pStart) order — the spring network's bars.

    Skips edges whose endpoints are not both in the local vertex
    stratum (rank-ghost incomplete edges); the spring path is
    serial-exact (see module docstring)."""
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    pairs = []
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0], cone[1]
        if not (pStart <= v0 < pEnd and pStart <= v1 < pEnd):
            continue
        pairs.append((v0 - pStart, v1 - pStart))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _mean_edge_length(dm, coords):
    """Global mean edge length of the mesh at ``coords`` (the movers'
    rank-mean of rank-local means — :func:`_global_mean`); 1.0 on a
    rank with no complete local edges.

    Callers that re-adapt repeatedly must measure this ONCE on the
    undeformed mesh and cache it — re-measuring from an
    already-refined mesh shrinks the value every adapt and compounds
    refinement (see the ``_FOLLOW_METRIC_H0_CACHE`` note at the top of
    the module)."""
    ep = _edge_pairs(dm)
    if ep.shape[0]:
        h = float(np.linalg.norm(
            coords[ep[:, 1]] - coords[ep[:, 0]], axis=1).mean())
    else:
        h = 1.0
    return _global_mean(h)


def _build_local_to_owned_map(dm, gsection, vec):
    """Compute, for each local owned vertex, its position in the
    rank's slice of the global Vec.

    Returns (owned_local_indices, owned_vec_positions, is_owned_local)
    where:
      * owned_local_indices : local-chart indices of owned vertices
        (shape n_owned, dtype int64)
      * owned_vec_positions : positions in vec.array (same shape)
      * is_owned_local : bool mask over the local chart
    """
    pStart, pEnd = dm.getDepthStratum(0)
    n_local = pEnd - pStart
    rstart, rend = vec.getOwnershipRange()
    is_owned = np.zeros(n_local, dtype=bool)
    owned_local = []
    owned_vec_pos = []
    for v in range(pStart, pEnd):
        off = gsection.getOffset(v)
        if off < 0:
            continue  # ghost
        is_owned[v - pStart] = True
        owned_local.append(v - pStart)
        owned_vec_pos.append(off - rstart)
    return (np.asarray(owned_local, dtype=np.int64),
            np.asarray(owned_vec_pos, dtype=np.int64),
            is_owned)


def smooth_surface_field(
    field,
    n_iters: int = 10,
    alpha: float = 0.5,
    taubin: bool = True,
    mu: Optional[float] = None,
    passband: float = 0.1,
    pinned_labels: Optional[Sequence[str]] = None,
):
    r"""Low-pass a scalar field over a (surface) mesh's vertex-edge graph.

    The *field* analogue of the coordinate graph-Laplacian Jacobi path in
    :func:`smooth_mesh_interior`. Each sweep blends every vertex value toward
    the mean of its edge-neighbours,

    .. math::

        h_i \leftarrow h_i + f\,\Big( \tfrac{1}{|N(i)|}\sum_{j\in N(i)} h_j
                                       - h_i \Big),

    attenuating high-wavenumber (facet-scale / sawtooth) content while leaving
    the smooth, long-wavelength field. Plain Laplacian smoothing
    (``taubin=False``) also damps the smooth part and shrinks amplitudes; the
    **Taubin** :math:`\lambda\,|\,\mu` scheme — a positive blend ``alpha``
    followed by a negative back-step ``mu`` each iteration — is a near-flat
    passband low-pass that leaves the mean and the long-wavelength amplitude
    essentially unchanged (the discrete analogue of a volume-preserving
    filter).

    Designed for the free-surface height field carried on a codim-1 surface
    submesh (:meth:`Mesh.extract_surface`): the graph operator needs only the
    edge connectivity, so it works on a 1-manifold loop where an FE solve is
    not yet available. No global gather, no FFT — unlike a spectral
    (Fourier ``h(θ)``) low-pass it is purely local on the surface graph.

    Parameters
    ----------
    field : degree-1 scalar ``MeshVariable``
        Smoothed **in place** (its nodal values are overwritten).
    n_iters : int, default 10
        Number of Taubin (or plain-Laplacian) iterations.
    alpha : float, default 0.5
        Positive blend factor :math:`\lambda \in (0, 1]`.
    taubin : bool, default True
        Apply the volume-preserving :math:`\lambda\,|\,\mu` pair each
        iteration. ``False`` ⇒ plain Laplacian smoothing (shrinks).
    mu : float, optional
        Negative back-step factor. Default derived from ``passband`` as
        :math:`\mu = 1/(k_{pb} - 1/\lambda)` (the standard Taubin choice;
        gives :math:`|\mu| \gtrsim \lambda`). Ignored when ``taubin`` is False.
    passband : float, default 0.1
        Taubin passband wavenumber :math:`k_{pb}` used to derive ``mu``.
    pinned_labels : sequence of str, optional
        Boundary labels whose vertices are held fixed (e.g. the endpoints of
        an open surface arc). A closed loop (annulus / shell ``Upper``) has
        none, so the default ``None`` is correct there.

    Notes
    -----
    Parallel-correct and **bit-identical serial vs parallel**. The
    neighbour-average is a single ``A.mult`` against the parallel
    vertex-vertex adjacency from :func:`_build_adjacency_matrix` (assembled
    with GLOBAL vertex indices, so an owned row sees every neighbour — even
    those owned by another rank not in this rank's overlap); the field is
    loaded into / read out of the global Vec via
    :func:`_build_local_to_owned_map`, and the smoothed result is scattered
    back to the field's local array (ghosts filled) with ``globalToLocal``.
    The constant mode (eigenvalue 0) is preserved exactly on any rank count.
    """
    from petsc4py import PETSc

    mesh = field.mesh

    # Parallel vertex-vertex adjacency (entries 1.0; global indices) + the
    # 1-dof-per-vertex scalar DM that owns the Vec/section layout.
    A, dm_scalar, gsection = _build_adjacency_matrix(mesh)

    g = dm_scalar.createGlobalVector()
    tmp = g.duplicate()
    ones = g.duplicate()
    ones.set(1.0)
    deg = g.duplicate()
    A.mult(ones, deg)                       # row sums = vertex degrees
    deg_arr = deg.array_r
    deg_safe = np.where(deg_arr > 0.0, deg_arr, 1.0)

    owned_local, owned_vec_pos, is_owned = _build_local_to_owned_map(
        dm_scalar, gsection, g)

    if taubin and mu is None:
        # Standard Taubin: 1/λ + 1/μ = k_pb  ⇒  μ = 1/(k_pb − 1/λ) < 0.
        mu = 1.0 / (passband - 1.0 / alpha)

    # Load the field (local: owned+ghost) into the global Vec (owned rows).
    # field.data[i] ↔ vertex pStart+i (degree-1), matching the owned map.
    fvals = np.asarray(field.data[:, 0], dtype=float)
    g.array[owned_vec_pos] = fvals[owned_local]

    # Positions of pinned owned vertices within the global Vec (held fixed).
    pin_vec_pos = None
    if pinned_labels:
        pmask = _pinned_mask(mesh.dm, tuple(pinned_labels))   # local-chart mask
        full_to_vec = np.full(is_owned.shape[0], -1, dtype=np.int64)
        full_to_vec[owned_local] = owned_vec_pos
        pin_owned = owned_local[pmask[owned_local]]
        pin_vec_pos = full_to_vec[pin_owned]
        pin_vals = g.array_r[pin_vec_pos].copy()

    def _blend(f):
        A.mult(g, tmp)                       # tmp = Σ_{neighbours} (global-correct)
        a = g.array                          # writable view of owned rows
        a += f * (tmp.array_r / deg_safe - a)
        if pin_vec_pos is not None:
            a[pin_vec_pos] = pin_vals

    for _ in range(n_iters):
        _blend(alpha)
        if taubin:
            _blend(mu)

    # Scatter owned -> local (fills ghosts) and write back to the field.
    lvec = dm_scalar.createLocalVector()
    dm_scalar.globalToLocal(g, lvec)
    field.data[:, 0] = lvec.array_r
    return field


def _tet_cells(dm):
    """Tetrahedron vertex-index quadruples (local-chart), or ``None`` if the
    mesh is not all-tet. The 3D analogue of :func:`_tri_cells` — used by the
    3D boundary-face extraction in ``_ot_adapt`` (the MMPDE mover itself is
    currently 2D-only)."""
    cStart, cEnd = dm.getHeightStratum(0)
    pStart, pEnd = dm.getDepthStratum(0)
    tets = []
    for c in range(cStart, cEnd):
        closure = dm.getTransitiveClosure(c)[0]
        vs = [p - pStart for p in closure if pStart <= p < pEnd]
        if len(vs) != 4:
            return None
        tets.append(vs)
    if not tets:
        return None
    return np.asarray(tets, dtype=np.int64)


def _owned_cell_mask(dm):
    """Local-chart boolean mask over cells (height stratum 0): True for
    owned cells, False for ghost/overlap cells (leaves of the point SF).
    Indexed like ``_tri_cells`` / ``_signed_areas`` (cell i ↔ point
    cStart+i). Assembly must sum over OWNED cells only so that a
    ``localToGlobal(ADD_VALUES)`` ghost reduction does not double-count
    overlap cells.
    """
    cStart, cEnd = dm.getHeightStratum(0)
    is_owned = np.ones(cEnd - cStart, dtype=bool)
    sf = dm.getPointSF()
    if sf is None:
        return is_owned
    try:
        _n_roots, leaves, _remote = sf.getGraph()
    except Exception:
        return is_owned
    if leaves is None or len(leaves) == 0:
        return is_owned
    for leaf in leaves:
        if cStart <= leaf < cEnd:
            is_owned[leaf - cStart] = False
    return is_owned


def _min_incident_edge_nd(cells, coords):
    """Dimension-general shortest-incident-edge per vertex. ``cells`` is
    (n_cells, d+1); returns (n_verts,). Used by the MMPDE per-node step
    cap. (The 2D-only ``_min_incident_edge`` reads the DM directly; this
    works for tets too and takes an explicit cell array so the caller can
    restrict the stencil.)"""
    n_verts = coords.shape[0]
    ncorner = cells.shape[1]
    v = np.full(n_verts, np.inf)
    for a in range(ncorner):
        for b in range(a + 1, ncorner):
            e = np.linalg.norm(coords[cells[:, a]] - coords[cells[:, b]],
                               axis=1)
            np.minimum.at(v, cells[:, a], e)
            np.minimum.at(v, cells[:, b], e)
    return v
