"""Mesh smoothing utilities.

Currently provides a Winslow-style Jacobi smoother for interior
vertex positions: each interior vertex is moved toward the average
position of its edge neighbours, with boundary vertices held fixed.

Use after a mesh deformation has left some cells highly distorted
(e.g. free-surface evolution that has crushed cells near the
surface). Topology is unchanged — vertex indices, DOFs, and the
parallel partition are all preserved; only coordinates move.

Parallel: the per-sweep update is a local scipy CSR Mat-Vec on the
local DMPlex chart (which already includes ghost vertices). A
halo exchange via PETSc localToGlobal/globalToLocal on the
coordinate DM runs between sweeps so each rank's ghost-vertex
copies see the new owned values from neighbours.

Future extensions (separate PRs):
  - PR B: nicer pinning API (per-boundary explicit lists, callable
    masks)
  - PR C: non-uniform metric (swarm-anchored target spacing,
    mirroring ``mesh.adapt`` semantics)
"""

from typing import Optional, Sequence

import numpy as np

import underworld3 as uw


# Cached adjacency keyed by (mesh-id, pinned-label-tuple, topology).
# Rebuilt automatically when the mesh topology changes.
_ADJ_CACHE: dict = {}


def _auto_pinned_labels(mesh) -> tuple:
    """All non-sentinel boundary labels on the mesh."""
    skip = {"All_Boundaries", "Null_Boundary"}
    names = []
    for member in mesh.boundaries:
        name = getattr(member, "name", None)
        if name and name not in skip:
            names.append(name)
    return tuple(names)


def _owned_vertex_mask(dm):
    """Return a local-chart boolean mask: True for owned vertices,
    False for ghosts (leaves of the point StarForest).

    In serial (size == 1), every local vertex is owned.
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


def _build_adjacency(mesh, pinned_labels):
    """Build row-normalised vertex-vertex adjacency + pinned mask.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Row-normalised so ``A @ x`` gives the average of each vertex's
        edge-neighbour ``x`` values.
    is_pinned : numpy.ndarray of bool, shape (n_local_verts,)
        True where the vertex belongs to any of ``pinned_labels``.
    """
    from scipy.sparse import csr_matrix

    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)      # vertex stratum
    eStart, eEnd = dm.getDepthStratum(1)      # edge stratum (2D / 3D)
    n_verts = pEnd - pStart

    rows, cols = [], []
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        # An edge's cone is its two endpoint vertices (any mesh).
        if len(cone) != 2:
            continue
        v0, v1 = cone[0] - pStart, cone[1] - pStart
        if 0 <= v0 < n_verts and 0 <= v1 < n_verts:
            rows.append(v0); cols.append(v1)
            rows.append(v1); cols.append(v0)

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    data = np.ones_like(rows, dtype=np.float64)
    A_pat = csr_matrix((data, (rows, cols)),
                       shape=(n_verts, n_verts))
    n_nbr = np.asarray(A_pat.sum(axis=1)).ravel()
    n_nbr_safe = np.where(n_nbr > 0, n_nbr, 1.0)
    inv = 1.0 / n_nbr_safe
    A = csr_matrix((data * inv[rows], (rows, cols)),
                   shape=(n_verts, n_verts))

    is_pinned = np.zeros(n_verts, dtype=bool)
    for lname in pinned_labels:
        label = dm.getLabel(lname)
        if label is None:
            continue
        vIS = label.getValueIS()
        if vIS is None:
            continue
        for val in vIS.getIndices():
            iset = label.getStratumIS(int(val))
            if iset is None:
                continue
            for idx in iset.getIndices():
                if pStart <= idx < pEnd:
                    is_pinned[idx - pStart] = True
    return A, is_pinned


def smooth_mesh_interior(
    mesh,
    pinned_labels: Optional[Sequence[str]] = None,
    n_iters: int = 5,
    alpha: float = 0.5,
    verbose: bool = False,
):
    r"""Apply Winslow Jacobi smoothing to a mesh's interior vertices.

    Each interior vertex is replaced by a blend of its current
    position and the unweighted mean of its edge-neighbour positions:

    .. math::

        x_i^{n+1} = (1 - \alpha)\, x_i^n
                    + \alpha \cdot \frac{1}{|N(i)|}
                    \sum_{j \in N(i)} x_j^n

    Vertices in any of ``pinned_labels`` are held fixed (preserves
    boundary geometry). The mesh's coordinate vector is updated in
    place via ``mesh._deform_mesh`` once after all sweeps — so the
    DM rebuild / cache invalidation cost is paid once rather than
    per sweep.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The mesh to smooth. Modified in place.
    pinned_labels : sequence of str, optional
        Names of boundary labels whose vertices stay fixed. If
        ``None`` (default), all non-sentinel labels on
        ``mesh.boundaries`` are pinned — i.e. every named boundary
        stays put. Pass an explicit list to release some boundaries.
    n_iters : int, default 5
        Number of Jacobi sweeps. 5-10 is typical for surface-
        deformation cleanup.
    alpha : float, default 0.5
        Under-relaxation in ``(0, 1]``. 1.0 is pure Jacobi; smaller
        is more damped (slower but safer on irregular meshes).
    verbose : bool, default False
        Print per-sweep RMS interior displacement.

    Notes
    -----
    **Parallel implementation**: per-sweep update is a local scipy
    CSR Mat-Vec on the local DMPlex chart, which includes ghost
    (off-rank) vertices as neighbours of owned vertices. Only owned
    interior vertices are written each sweep; a halo exchange via
    ``coordDM.localToGlobal`` (INSERT) + ``globalToLocal`` pushes
    those new owned values out to the ghost copies on receiving
    ranks before the next sweep's Mat-Vec.

    **Topology preservation**: vertex IDs, DOF mappings, and the
    rank partition are unchanged. Only coordinates move. Anything
    cached against the topology version stays valid; anything
    cached against coords is invalidated by the final
    ``mesh._deform_mesh`` call.

    Examples
    --------
    Pin all named boundaries (the usual case)::

        import underworld3 as uw
        from underworld3.meshing import smooth_mesh_interior

        mesh = uw.meshing.Annulus(...)
        # ... some deformation that leaves bad cells ...
        smooth_mesh_interior(mesh, n_iters=5, alpha=0.5)

    Pin only the outer boundary, allowing the inner to drift::

        smooth_mesh_interior(mesh, pinned_labels=["Upper"])

    Pin nothing (free-floating; rare — boundary will collapse)::

        smooth_mesh_interior(mesh, pinned_labels=[])
    """
    if pinned_labels is None:
        pinned_labels = _auto_pinned_labels(mesh)
    pinned_labels = tuple(pinned_labels)

    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    cone_size = dm.getConeSize(cStart) if cEnd > cStart else 0
    cache_key = (id(mesh), pinned_labels,
                 pEnd - pStart, cEnd - cStart, cone_size)

    cache = _ADJ_CACHE.get(cache_key)
    if cache is None:
        A, is_pinned = _build_adjacency(mesh, pinned_labels)
        is_owned = _owned_vertex_mask(dm)
        _ADJ_CACHE[cache_key] = (A, is_pinned, is_owned)
    else:
        A, is_pinned, is_owned = cache

    # Owned interior vertices are the writes per sweep.
    # Ghost-vertex values come from the halo exchange.
    is_int_owned = is_owned & ~is_pinned

    coord_dm = dm.getCoordinateDM()
    local_vec = dm.getCoordinatesLocal()
    global_vec = dm.getCoordinates()
    cdim = mesh.cdim
    parallel = uw.mpi.size > 1

    # Working buffer (initially copies the local Vec contents, which
    # already includes ghosts in parallel).
    coords = np.asarray(
        local_vec.array, dtype=np.double).reshape(-1, cdim).copy()

    for sweep in range(n_iters):
        avg = A @ coords
        new_int = ((1.0 - alpha) * coords[is_int_owned]
                   + alpha * avg[is_int_owned])
        if verbose:
            disp = float(np.linalg.norm(
                new_int - coords[is_int_owned]))
            if parallel:
                disp = uw.mpi.comm.allreduce(
                    disp ** 2) ** 0.5
            uw.pprint(
                f"  smooth_mesh_interior sweep "
                f"{sweep+1}/{n_iters}: "
                f"||Δx||_interior = {disp:.3e}")
        coords[is_int_owned] = new_int

        if parallel:
            # Halo exchange so the next sweep's Mat-Vec sees the
            # updated owned-vertex values on every rank's ghost copies.
            # 1. write our owned updates into the local Vec
            local_vec.array[:] = coords.ravel()
            # 2. localToGlobal with INSERT — push owned to global
            coord_dm.localToGlobal(
                local_vec, global_vec, addv=False)
            # 3. globalToLocal — refresh ghost copies from new owned
            coord_dm.globalToLocal(global_vec, local_vec)
            # 4. read back into our numpy buffer
            coords[:] = np.asarray(
                local_vec.array).reshape(-1, cdim)

    # Single DM-coords update at the end: one rebuild, not N.
    mesh._deform_mesh(coords)
