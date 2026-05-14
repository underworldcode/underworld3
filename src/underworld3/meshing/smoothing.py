"""Mesh smoothing utilities.

Currently provides a Winslow-style Jacobi smoother for interior
vertex positions: each interior vertex is moved toward the average
position of its edge neighbours, with boundary vertices held fixed.

Use after a mesh deformation has left some cells highly distorted
(e.g. free-surface evolution that has crushed cells near the
surface). Topology is unchanged — vertex indices, DOFs, and the
parallel partition are all preserved; only coordinates move.

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
    **Parallel safety**: currently a serial scipy CSR matrix-vector
    operation on the local DMPlex chart. Running under
    ``mpi.size > 1`` raises ``NotImplementedError``. A future change
    will replace the scipy Mat-Vec with a PETSc Mat-Vec with halo
    exchange between sweeps.

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
    if uw.mpi.size > 1:
        raise NotImplementedError(
            "smooth_mesh_interior is currently serial-only. "
            "Parallel (PETSc Mat-Vec) implementation pending — "
            "see docs/developer/subsystems/mesh-smoothing.md.")

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
        _ADJ_CACHE[cache_key] = (A, is_pinned)
    else:
        A, is_pinned = cache

    is_interior = ~is_pinned
    coords = np.asarray(mesh.X.coords, dtype=np.double).copy()

    for sweep in range(n_iters):
        avg = A @ coords
        new_int = ((1.0 - alpha) * coords[is_interior]
                   + alpha * avg[is_interior])
        if verbose:
            disp = np.linalg.norm(new_int - coords[is_interior])
            print(f"  smooth_mesh_interior sweep "
                  f"{sweep+1}/{n_iters}: "
                  f"||Δx||_interior = {disp:.3e}", flush=True)
        coords[is_interior] = new_int

    # Single DM-coords update at the end: one rebuild, not N.
    mesh._deform_mesh(coords)
