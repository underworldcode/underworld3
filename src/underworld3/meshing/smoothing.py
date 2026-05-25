"""Mesh smoothing utilities.

Currently provides a Winslow-style Jacobi smoother for interior
vertex positions: each interior vertex is moved toward the average
position of its edge neighbours, with boundary vertices held fixed.

Use after a mesh deformation has left some cells highly distorted
(e.g. free-surface evolution that has crushed cells near the
surface). Topology is unchanged — vertex indices, DOFs, and the
parallel partition are all preserved; only coordinates move.

Parallel: a PETSc parallel AIJ matrix represents the vertex-vertex
adjacency. Each rank inserts entries for every edge it sees locally
using GLOBAL vertex indices; ``mat.assemble()`` combines cross-rank
contributions so that owned-vertex rows are complete after assembly.
Without this, UW3's default cell-overlap-0 distribution under-counts
neighbours for vertices on the rank partition boundary, producing
visibly wrong updates along the rank cut.

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
    local_to_global_owned : numpy.ndarray, shape (n_owned,)
        ``local_to_global_owned[i]`` is the offset (in the *local*
        owned portion of the global Vec) at which the ``i``-th
        OWNED local vertex appears. Use this to pack/unpack between
        ``coords[is_owned, d]`` and ``vec.array``.
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
    **Parallel implementation**: the vertex-vertex adjacency is
    assembled as a parallel PETSc AIJ matrix; each rank inserts
    entries for every locally-visible edge using GLOBAL vertex
    indices and ``mat.assemble()`` routes cross-rank contributions
    so that owned-vertex rows are complete after assembly. The
    per-sweep update is then a per-component ``A.mult`` followed by
    a pointwise divide by the precomputed degree vector. Results
    are bit-identical (to a single ULP) between serial and parallel
    runs at any rank count.

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
        A, dm_scalar, gsection = _build_adjacency_matrix(mesh)
        # A scratch global Vec of the right shape — also used to read
        # the ownership range when packing/unpacking coord components.
        x_vec = A.createVecRight()
        y_vec = A.createVecLeft()
        ones = A.createVecLeft()
        ones.set(1.0)
        degrees = A.createVecLeft()
        A.mult(ones, degrees)
        owned_local, owned_vec_pos, is_owned = (
            _build_local_to_owned_map(dm, gsection, x_vec))
        is_pinned = _pinned_mask(dm, pinned_labels)
        _ADJ_CACHE[cache_key] = (
            A, dm_scalar, gsection, x_vec, y_vec, degrees,
            owned_local, owned_vec_pos, is_owned, is_pinned)
    else:
        (A, dm_scalar, gsection, x_vec, y_vec, degrees,
         owned_local, owned_vec_pos, is_owned, is_pinned) = cache

    # is_int_owned over the LOCAL chart — selects interior owned
    # vertices for displacement reporting.
    is_int_owned = is_owned & ~is_pinned
    # Subset of owned_local that's also interior (i.e. not pinned)
    # — used to write the per-sweep updates into the numpy buffer.
    int_mask_on_owned = ~is_pinned[owned_local]
    int_owned_local = owned_local[int_mask_on_owned]
    int_owned_vec_pos = owned_vec_pos[int_mask_on_owned]

    coord_dm = dm.getCoordinateDM()
    local_vec = dm.getCoordinatesLocal()
    global_vec = dm.getCoordinates()
    cdim = mesh.cdim
    parallel = uw.mpi.size > 1

    coords = np.asarray(
        local_vec.array, dtype=np.double).reshape(-1, cdim).copy()

    for sweep in range(n_iters):
        new_int = np.empty((int_owned_local.shape[0], cdim),
                           dtype=np.double)
        # For each coordinate component, do A @ coord_comp (PETSc
        # handles cross-rank communication), then divide by degree
        # to get the per-vertex neighbour average.
        for d in range(cdim):
            x_vec.array[owned_vec_pos] = coords[owned_local, d]
            A.mult(x_vec, y_vec)
            y_vec.pointwiseDivide(y_vec, degrees)
            avg_owned = np.asarray(y_vec.array)
            new_int[:, d] = (
                (1.0 - alpha) * coords[int_owned_local, d]
                + alpha * avg_owned[int_owned_vec_pos])

        if verbose:
            disp = float(np.linalg.norm(
                new_int - coords[int_owned_local]))
            if parallel:
                disp = uw.mpi.comm.allreduce(
                    disp ** 2) ** 0.5
            uw.pprint(
                f"  smooth_mesh_interior sweep "
                f"{sweep+1}/{n_iters}: "
                f"||Δx||_interior = {disp:.3e}")

        coords[int_owned_local] = new_int

        if parallel:
            # Halo exchange so the next sweep sees updated owned
            # values on every rank's ghost copies. (PETSc's mat.mult
            # handles cross-rank READS internally via the matrix's
            # column communication, so this halo exchange is only
            # needed to keep the LOCAL coord array consistent for
            # the final ``mesh._deform_mesh`` call.)
            local_vec.array[:] = coords.ravel()
            coord_dm.localToGlobal(
                local_vec, global_vec, addv=False)
            coord_dm.globalToLocal(global_vec, local_vec)
            coords[:] = np.asarray(
                local_vec.array).reshape(-1, cdim)

    mesh._deform_mesh(coords)
