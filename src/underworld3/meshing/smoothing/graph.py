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
                else:
                    # Tagged edge (2D labels) or face (3D labels tag
                    # faces ONLY, so stopping at edges left every 3D
                    # boundary vertex unmarked — no pin, no slip, the
                    # boundary drifted freely): close the point down to
                    # its vertices. For an edge the closure is exactly
                    # its endpoints, so the 2D mask is bit-identical.
                    closure = dm.getTransitiveClosure(idx)[0]
                    for c in closure:
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



def _tri_cells(dm):
    """Triangle vertex-index triples (local-chart, v-pStart order).

    Returns an ``(n_tri, 3)`` int array, or ``None`` if the mesh is
    not all-triangle (callers that need the cell array — fold checks,
    boundary-facet extraction — then skip those steps).
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


def _signed_volumes(coords, tets):
    """Signed volume of each tetrahedron (sign = orientation) — the 3D
    analogue of :func:`_signed_areas`, indexed the same way. Note the
    DMPlex reference tet is det-NEGATIVE, so a healthy all-DMPlex-ordered
    tet mesh yields uniformly negative values; callers (the MMPDE mover's
    fold guard) already orientation-normalise with the global median
    sign, exactly as they do for 2D."""
    a = coords[tets[:, 0]]
    e = np.stack([coords[tets[:, 1]] - a,
                  coords[tets[:, 2]] - a,
                  coords[tets[:, 3]] - a], axis=1)
    return np.linalg.det(e) / 6.0





def _edge_pairs(dm):
    """``(n_edge, 2)`` int array of edge endpoint vertex indices in
    local-chart (v - pStart) order.

    Skips edges whose endpoints are not both in the local vertex
    stratum (rank-ghost incomplete edges), so the result is the
    rank-local edge graph."""
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
    3D boundary-face extraction in :func:`_boundary_facets` and by the
    MMPDE mover's 3D discretization."""
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
    cap. Works for tets too and takes an explicit cell array so the
    caller can restrict the stencil."""
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


# ---------------------------------------------------------------------------
# Boundary-facet and boundary-slip primitives.
#
# Relocated from ``meshing/_ot_adapt.py`` when the OT-reset adapt was
# retired (2026-07; the spring / Monge-Ampere / OT / anisotropic-Winslow
# movers are superseded by the MMPDE mover). These helpers are mover-
# agnostic and remain in use by the MMPDE mover, the mesh's
# ``boundary_slip`` orchestration, and ``BoundingSurface`` facet restore.
# ---------------------------------------------------------------------------

def _slip_normals(mesh, boundary_coords):
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


def _boundary_facets(mesh, cdim):
    """Boundary facets + opposite cell-vertex, found from the cell topology.

    For each cell, every facet (edge in 2D, triangle in 3D) is a candidate
    boundary facet; one that occurs in **exactly one** cell is on the
    boundary. Returns ``(facets, opp)`` where ``facets`` is ``(n_bnd, k)``
    (``k=2`` for 2D edges, ``k=3`` for 3D triangles) and ``opp`` is the
    cell vertex opposite each facet — used to orient the facet normal
    outward. Returns ``(None, None)`` for non-simplicial meshes.
    """
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
        # Meshes without a boundaries enum (e.g. hand-built DMs) simply
        # expose no named labels — treat as label-free.
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
    mid-mover would stale the DM handle — see project_uw3_smoother_footguns).

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
            # No projected-normal field on this mesh (e.g. a manifold or
            # label-free DM) — slip resolution still returns the labels;
            # the caller's per-node normal lookup pins degenerate nodes.
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
