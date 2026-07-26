"""
Custom geometric-multigrid prolongation injection.

Build the multigrid prolongation ``P`` **ourselves** (barycentric or RBF) from a
sequence of independent — possibly *non-nested* — coarse meshes, and install it
into a solver's PETSc ``PCMG`` via ``PC.setMGInterpolation``. The coarse
operators are then formed by Galerkin RAP (``Pᵀ A P``), exactly as UW3's FMG
already does.

Why
---
UW3's geometric FMG requires a nested ``refine()`` hierarchy with PETSc's exact
nested interpolation. That couples multigrid to uniform refinement and rules out
local adaptation / a dynamic node budget. PETSc's ``PCMG`` does **not** require
its transfer operator to come from the DM hierarchy, however: any prolongation
supplied with ``setMGInterpolation`` is used, and ``pc_mg_galerkin=both`` builds
consistent coarse operators from it. The prolongation "evaluate the coarse FE
function at the fine DOF coordinates" is exactly what the nested path computes —
so we can build it ourselves by barycentric (FE-exact) or RBF interpolation for
*any* coarse/fine pair. This decouples *how the mesh refines* from *what transfer
multigrid uses*.

Validated (scalar Poisson, refinement=2 box): custom barycentric ``P`` from
independent coarse meshes reaches FMG iteration counts (1–3 vs FMG 2). On a mesh
with **no** refine hierarchy (FMG unavailable → GAMG, 13 iters) custom-P
geometric MG converges in ~9 iters.

Notes
-----
* The finest level is the solver's real (BC-eliminated) operator space; its
  ``P`` rows are reduced to the global/MG ordering. Coarse levels use the full
  DOF coordinates of each coarse mesh (no BC reduction needed — Galerkin RAP and
  the finest reduction keep the coarse correction consistent).
* Injection happens at solver build time, *before* the first ``PCSetUp``: the
  first Galerkin assembly is built from our ``P`` directly, avoiding PETSc's
  ``MatProductReplaceMats`` shape error that a live matrix swap triggers.
* ``KSPSetDMActive(OPERATOR, False)`` stops PETSc re-deriving interpolation /
  operators from the DM hierarchy, so our explicit ``P`` is used.
"""

from typing import NamedTuple

import numpy as np
from petsc4py import PETSc

__all__ = ["barycentric_prolongation", "rbf_prolongation", "inject_custom_mg",
           "CustomMGHierarchy", "set_custom_fmg", "sbr_refine", "sbr_refine_where",
           "nvb_refine"]


# --------------------------------------------------------------------------- #
#  Prolongation builders (return scipy CSR matrices)
# --------------------------------------------------------------------------- #
def barycentric_prolongation(coarse_coords, fine_coords):
    """Prolongation by linear interpolation over a Delaunay triangulation of the
    coarse DOF cloud. Partition of unity (row sums = 1) and linear fields are
    reproduced exactly; fine points outside the coarse hull fall back to the
    nearest coarse DOF, and orphaned coarse DOFs are repaired below.

    .. note::

       This is **not** the coarse FE space's embedding, despite what this
       docstring claimed until 2026-07. It triangulates the coarse DOF *point
       cloud* from scratch and never consults the coarse mesh's cells, so
       "the coarse element containing the fine DOF" is only what gets used when
       the Delaunay triangulation happens to coincide with the mesh. Measured
       agreement between the two:

       ===========================  ==========  ==========  ==========
       case                         mesh cells  Delaunay    agreement
       ===========================  ==========  ==========  ==========
       2D uniform base, P1                 168         168       100 %
       2D adapt child, P1                  238         238      90.8 %
       3D uniform base, P1                 800         812      58.8 %
       3D adapt child, P1                 4685        5201      17.1 %
       3D adapt child, P2                 4685       39597         --
       ===========================  ==========  ==========  ==========

       Reproducing linears is enough for multigrid to converge well, which is
       why this works. But two limits follow. For P2 and above the coarse
       space is not represented exactly. And because Delaunay simplices do not
       respect mesh topology, on a non-convex domain or one with an internal
       interface (a fault) a simplex can bridge across the discontinuity and
       smear the coarse correction over it — predicted from the algorithm,
       not yet measured.

       Where a parent/child relation exists, prefer the exact nested transfer
       (#425), which is the true FE embedding, cannot bridge across features,
       and cannot orphan a coarse DOF.
    """
    import scipy.sparse as sp
    from scipy.spatial import Delaunay, cKDTree

    tri = Delaunay(coarse_coords)
    simp = tri.find_simplex(fine_coords)
    tree = cKDTree(coarse_coords)
    dim = coarse_coords.shape[1]
    rows, cols, vals = [], [], []
    for i in range(fine_coords.shape[0]):
        s = simp[i]
        if s < 0:  # outside coarse hull → nearest coarse DOF
            rows.append(i)
            cols.append(int(tree.query(fine_coords[i])[1]))
            vals.append(1.0)
            continue
        verts = tri.simplices[s]
        T = tri.transform[s]
        bary = T[:dim].dot(fine_coords[i] - T[dim])
        w = np.append(bary, 1.0 - bary.sum())
        for k in range(dim + 1):
            rows.append(i)
            cols.append(int(verts[k]))
            vals.append(float(w[k]))
    P = sp.csr_matrix((vals, (rows, cols)),
                      shape=(fine_coords.shape[0], coarse_coords.shape[0]))

    # Repair orphaned coarse DOFs (columns with no fine image). Point location
    # has LOCAL support, so a coarse DOF is reached only if some fine DOF lands
    # in a simplex touching it; move the fine coordinates (mesh.relax, boundary
    # snapping, free-surface deformation) and a coarse DOF can lose every fine
    # image. Its column goes to zero and the Galerkin coarse operator PᵀAP is
    # singular (#424).
    #
    # Give each orphan its nearest fine DOF as a pure injection (weight 1) —
    # exactly the fallback already used above for fine points outside the
    # coarse hull. Partition of unity is preserved (the row still sums to 1),
    # sparsity is preserved (~d+1 nnz/row), and the column is no longer empty.
    # Distinct rows are used so two orphans cannot claim the same fine DOF.
    # The alternative — switching to the global-support RBF builder — removes
    # the singularity but returns a DENSE transfer (nnz/row == n_coarse), which
    # makes PᵀAP dense and is unusable at production sizes.
    orphans = np.nonzero(np.asarray((P != 0).sum(axis=0)).ravel() == 0)[0]
    if orphans.size:
        k = int(min(orphans.size + 8, fine_coords.shape[0]))
        _, cand = cKDTree(fine_coords).query(coarse_coords[orphans], k=k)
        cand = np.atleast_2d(cand)
        claimed, fixes = set(), {}
        for i, c in enumerate(orphans):
            for f in cand[i]:
                f = int(f)
                if f not in claimed:
                    claimed.add(f)
                    fixes[f] = int(c)
                    break
        if fixes:
            keep = ~np.isin(np.asarray(rows), list(fixes))
            rows = list(np.asarray(rows)[keep]) + list(fixes.keys())
            cols = list(np.asarray(cols)[keep]) + list(fixes.values())
            vals = list(np.asarray(vals)[keep]) + [1.0] * len(fixes)
            P = sp.csr_matrix((vals, (rows, cols)),
                              shape=(fine_coords.shape[0],
                                     coarse_coords.shape[0]))
    return P


def rbf_prolongation(coarse_coords, fine_coords, smooth=0.0):
    """RBF prolongation: polyharmonic (r² log r) kernel + affine polynomial tail
    (reproduces linear fields), Shepard row-normalised to a partition of unity.
    Works for arbitrary (non-nested) point sets; software-equivalent to the
    barycentric builder as an MG transfer operator."""
    import scipy.sparse as sp
    from scipy.spatial.distance import cdist

    def phi(r):
        # r² log r → 0 as r → 0; the clamp only keeps log(0) finite at
        # coincident points — it does not perturb the kernel value.
        r = np.where(r == 0.0, 1e-30, r)
        return r ** 2 * np.log(r)

    nc, dim = coarse_coords.shape
    Pc = np.hstack([np.ones((nc, 1)), coarse_coords])          # affine tail
    Acc = phi(cdist(coarse_coords, coarse_coords)) + smooth * np.eye(nc)
    M = np.block([[Acc, Pc], [Pc.T, np.zeros((dim + 1, dim + 1))]])
    B = np.hstack([phi(cdist(fine_coords, coarse_coords)),
                   np.ones((fine_coords.shape[0], 1)), fine_coords])
    # Solve M Xᵀ = Bᵀ rather than forming M⁻¹ explicitly (faster, more stable). M is
    # symmetric, so B M⁻¹ = solve(M, Bᵀ)ᵀ.
    Praw = np.linalg.solve(M, B.T).T[:, :nc]
    rs = Praw.sum(axis=1, keepdims=True)
    rs[np.abs(rs) < 1e-12] = 1.0
    return sp.csr_matrix(Praw / rs)


_BUILDERS = {"barycentric": barycentric_prolongation, "rbf": rbf_prolongation}


def _require_serial(where):
    """Custom-P transfers are serial-only (experimental). The reduced maps use
    rank-local DOF indices and the prolongations assemble as serial AIJ; at np>1
    they would silently build wrong P / mis-numbered transfers. Parallel support
    (nested co-partitioned, rank-local P + MPIAIJ, global-section reduction) is a
    designed fast-follow — until then, fail loudly rather than wrong."""
    from underworld3 import mpi
    if mpi.size > 1:
        raise NotImplementedError(
            f"custom_mg ({where}): custom-P geometric MG is serial-only "
            f"(running on {mpi.size} ranks). Parallel (np>1) support is not yet "
            f"implemented; use preconditioner='fmg' (nested) or 'gamg' in parallel.")


# --------------------------------------------------------------------------- #
#  Local Skeleton-Based Refinement (no MMG; on-rank; conforming)
# --------------------------------------------------------------------------- #
_DM_ADAPT_REFINE = 1   # PETSc DM_ADAPT_REFINE


def _sbr_apply(dm, mark):
    """Clone ``dm``, let ``mark(clone, adapt_label)`` flag cells, and SBR-refine.
    Conforming (edge bisection + propagation, no hanging nodes), on-rank (no
    redistribution), no external mesh libraries. Cell geometry / marking is done
    on the CLONE (the un-cloned mesh DM raises err73 from computeCellGeometryFVM).

    IMPORTANT: ``dm_plex_transform_type`` is a *global* PETSc option; leaving it as
    ``refine_sbr`` breaks UW3's uniform ``dm.refine()`` (FMG hierarchy build) with
    PETSc error 73. It is set only for the duration of the call and restored."""
    opts = PETSc.Options()
    had = opts.hasName("dm_plex_transform_type")
    prev = opts.getString("dm_plex_transform_type") if had else None
    opts.setValue("dm_plex_transform_type", "refine_sbr")
    try:
        d = dm.clone()
        d.createLabel("adapt")
        lab = d.getLabel("adapt")
        lab.setDefaultValue(0)
        mark(d, lab)
        return d.adaptLabel("adapt")
    finally:
        if had:
            opts.setValue("dm_plex_transform_type", prev)
        else:
            opts.delValue("dm_plex_transform_type")


def sbr_refine(dm, cells):
    """SBR-refine an explicit list of ``cells``."""
    def mark(d, lab):
        for c in cells:
            lab.setValue(int(c), _DM_ADAPT_REFINE)
    return _sbr_apply(dm, mark)


def sbr_refine_where(dm, predicate):
    """SBR-refine cells whose centroid satisfies ``predicate(centroid)``."""
    def mark(d, lab):
        cs, ce = d.getHeightStratum(0)
        for c in range(cs, ce):
            if predicate(d.computeCellGeometryFVM(c)[1]):
                lab.setValue(c, _DM_ADAPT_REFINE)
    return _sbr_apply(dm, mark)


# --------------------------------------------------------------------------- #
#  Newest-vertex bisection (NVB) — GRADED refinement (serial Route A)
# --------------------------------------------------------------------------- #
def nvb_refine(dm, cells, boundaries=(), regions=()):
    """NVB-refine an explicit list of ``cells`` (serial, single level), returning a
    fresh interpolated ``DMPlex`` with boundary/region labels transferred.

    Counterpart to :func:`sbr_refine` but with a **bounded conforming closure**, so
    a marked cell deep in a uniform patch adds O(1) cells locally instead of
    draining the longest-edge path to the patch edge — the property that lets
    successive levels *grade* (see :mod:`underworld3.utilities.nvb`).

    Single-shot: builds an :class:`~underworld3.utilities.nvb.NVBMesh` from ``dm``
    (longest-edge seed), refines, and emits the DM. For a **multi-level** graded
    adapt, drive a *persistent* ``NVBMesh`` across levels instead (so the
    refinement-edge labelling — hence the similarity-class / shape-regularity bound
    — propagates parent→child); ``Mesh._adapt_nested(engine="nvb")`` does this.

    ``boundaries`` / ``regions`` are ``(name, value)`` iterables of labels to carry.
    """
    from underworld3.utilities.nvb import NVBMesh
    nvb = NVBMesh.from_dm(dm, boundaries=boundaries, regions=regions)
    nvb.refine(set(int(c) for c in cells))
    return nvb.to_dm(boundaries=boundaries, regions=regions, comm=dm.comm)


# --------------------------------------------------------------------------- #
#  Coordinate helpers
# --------------------------------------------------------------------------- #
def _to_petsc_aij(csr):
    csr = csr.tocsr()
    # Match the PETSc build's integer width for the CSR index arrays: a hard int32 cast
    # can overflow / mis-address entries on a 64-bit PetscInt build with large meshes.
    M = PETSc.Mat().createAIJ(
        size=csr.shape,
        csr=(csr.indptr.astype(PETSc.IntType),
             csr.indices.astype(PETSc.IntType), csr.data),
    )
    M.assemble()
    return M


def _reduce_to_global(dm, full_coords):
    """Reduce full local DOF coordinates to the solver's global (BC-eliminated,
    MG-ordered) layout by scattering each component through ``localToGlobal``
    (which drops constrained boundary DOFs and reorders to global indices)."""
    lvec = dm.getLocalVec()
    gvec = dm.getGlobalVec()
    cdim = full_coords.shape[1]
    if lvec.getLocalSize() != full_coords.shape[0]:
        dm.restoreLocalVec(lvec)
        dm.restoreGlobalVec(gvec)
        raise RuntimeError(
            f"custom_mg: local DOF count {lvec.getLocalSize()} != coord count "
            f"{full_coords.shape[0]} — degree/continuity mismatch")
    out = np.zeros((gvec.getSize(), cdim))
    for c in range(cdim):
        lvec.array[:] = full_coords[:, c]
        dm.localToGlobal(lvec, gvec, addv=False)
        out[:, c] = gvec.array
    dm.restoreLocalVec(lvec)
    dm.restoreGlobalVec(gvec)
    return out


# --------------------------------------------------------------------------- #
#  Layer 1 — generalized FMG hierarchy (BC-reduced, adapter-agnostic)
#
#  INVARIANT (load-bearing): essential BCs are applied at EVERY level, so every
#  transfer maps reduced->reduced. PETSc native FMG does this automatically via
#  each level's constrained PetscSection; custom-P built from raw coordinates must
#  replicate it. Omitting it is fatal on an EXACTLY-nested hierarchy: a coarse
#  boundary DOF coincides with a BC-removed fine DOF -> zero column -> singular
#  Galerkin coarse operator. (See docs/developer/design/
#  GENERALIZED_FMG_HIERARCHY_AND_ADAPT.md.)
# --------------------------------------------------------------------------- #
def _field_subdm(dm, field_id):
    """Return the sub-DM for ``field_id`` (whole dm if single-field / None)."""
    if field_id is None:
        return dm
    try:
        if dm.getNumFields() <= 1:
            return dm
    except PETSc.Error:
        # Sanctioned: a DM whose DS is not created yet cannot report its
        # field count; treat it as single-field rather than mask a real
        # multi-field mistake with a broad swallow.
        return dm
    _iset, sub = dm.createSubDM(field_id)
    return sub


def _reduced_map(dm, field_id=None):
    """Reduced->full DOF map for a field on a BUILT DM (serial). Index ``k`` of
    the returned array is the local DOF that maps to global/MG index ``k`` —
    i.e. the BC-eliminated, MG-ordered layout the operator lives on.

    NOTE: serial-correct (uses local indices). Parallel correctness — mapping to
    *global* numbering across ranks — is a Phase-3 item (the supported parallel
    path keeps levels co-partitioned so this stays rank-local)."""
    sub = _field_subdm(dm, field_id)
    lv = sub.getLocalVec()
    n_full = lv.getLocalSize()
    lv.array[:] = np.arange(n_full)
    gv = sub.getGlobalVec()
    sub.localToGlobal(lv, gv, addv=False)
    r2f = gv.array.astype(int).copy()
    sub.restoreLocalVec(lv)
    sub.restoreGlobalVec(gv)
    return r2f, n_full


def _clone_dm_with_solver_discretisation(solver, coarse_mesh):
    """Clone a coarse mesh DM carrying the finest ``solver``'s discretisation.

    Copy the (built) solver's fields + DS onto the clone. The DS carries UW's
    exact essential-BC definitions (the custom essential-field boundaries), and
    is a topology-independent discretisation spec, so ``createDS`` constrains
    the matching boundary DOFs on ANY coarse mesh that carries the same
    boundary labels (nested or not). The resulting global section gives the
    same reduced map a full same-discretisation solver would — validated
    identical to the old ``level_solver_factory`` path. Leak-free: DM ops
    only, no SNES / JIT."""
    cdm = coarse_mesh.dm.clone()
    solver.dm.copyFields(cdm)
    solver.dm.copyDS(cdm)
    cdm.createDS()
    return cdm


def _coarse_reduced_map(solver, coarse_mesh, field_id=None):
    """A COARSE level's BC-constrained reduced map, with NO throwaway solver
    (see :func:`_clone_dm_with_solver_discretisation` for the copyDS trick).

    NOTE: serial, like ``_reduced_map`` (uses local indices); parallel correctness
    is a Phase-3 item."""
    cdm = _clone_dm_with_solver_discretisation(solver, coarse_mesh)
    return _reduced_map(cdm, field_id)


def _reduced_from_node_transfer(Pn, r2f_c, r2f_f, ncomp):
    """Interleave ``ncomp`` components of a node-level scalar ``Pn`` and drop the
    BC rows/columns — the half of :func:`_reduced_transfer` that does not care
    where the node weights came from."""
    import scipy.sparse as sp
    Pv = sp.kron(Pn, sp.eye(ncomp), format="csr")          # interleaved full vector
    return Pv.tocsr()[r2f_f, :][:, r2f_c]                  # reduced -> reduced


def _reduced_transfer(coarse_coords, fine_coords, r2f_c, r2f_f, ncomp, builder):
    """Build one prolongation reduced(coarse) -> reduced(fine):
    node-level scalar P -> interleave ``ncomp`` components -> drop BC rows/cols."""
    Pn = builder(coarse_coords, fine_coords)               # (n_f_nodes, n_c_nodes)
    return _reduced_from_node_transfer(Pn, r2f_c, r2f_f, ncomp)


# --------------------------------------------------------------------------- #
#  Parallel (np>1) — nested co-partitioned, rank-local P + MPIAIJ
#
#  Supported path: the levels are co-partitioned (uniform refine() / on-rank SBR
#  keep a fine cell's parent coarse cell on the same rank), so each rank builds
#  its block of P from its LOCAL (ghost-inclusive) coarse coords — point-location
#  is rank-local. The reduced global numbering rides the DM global section.
# --------------------------------------------------------------------------- #
class LevelLayout(NamedTuple):
    """Parallel DOF layout of one MG level.

    ``l2g[i]`` is the GLOBAL reduced index of local DOF ``i`` — ghost-resolved
    to the owner's global index, and ``-1`` for a BC-constrained DOF.
    ``[rstart, rend)`` is this rank's owned global range; ``n_full`` is the
    local (ghost-inclusive) full DOF count."""

    l2g: np.ndarray
    rstart: int
    rend: int
    n_full: int


def _level_dof_layout(dm, field_id=None):
    """Parallel DOF layout for one level (a :class:`LevelLayout`).

    ``l2g`` is built by scattering each owned global index out to the local
    (incl. ghost) layout via ``globalToLocal`` (constrained local DOFs have no
    global source, so they keep the pre-set ``-1``)."""
    sub = _field_subdm(dm, field_id)
    gv = sub.getGlobalVec()
    lv = sub.getLocalVec()
    rstart, rend = gv.getOwnershipRange()
    gv.array[:] = np.arange(rstart, rend, dtype=float)
    lv.set(-1.0)
    sub.globalToLocal(gv, lv, addv=PETSc.InsertMode.INSERT_VALUES)
    l2g = np.rint(lv.array).astype(np.int64).copy()
    n_full = lv.getLocalSize()
    sub.restoreGlobalVec(gv)
    sub.restoreLocalVec(lv)
    return LevelLayout(l2g, rstart, rend, n_full)


def _coarse_dof_layout(solver, coarse_mesh, field_id=None):
    """Parallel coarse-level DOF layout, no throwaway solver — same copyDS trick
    as :func:`_coarse_reduced_map` but returning the parallel layout."""
    cdm = _clone_dm_with_solver_discretisation(solver, coarse_mesh)
    return _level_dof_layout(cdm, field_id)


def _build_parallel_transfer(coarse_coords, fine_coords, coarse_layout,
                             fine_layout, ncomp, builder, comm):
    """One reduced->reduced prolongation as an MPIAIJ matrix.

    Node-level barycentric/RBF weights are built rank-locally (coarse LOCAL coords
    incl. ghosts -> every owned fine node lands in a local coarse simplex). Each
    fine OWNED DOF becomes a global row; its coarse contributions map through the
    coarse ``l2g`` to global columns (off-rank columns are fine for MPIAIJ).
    Constrained coarse DOFs (``l2g == -1``) drop out -> reduced->reduced."""
    l2g_c = coarse_layout.l2g
    l2g_f, fstart, fend = fine_layout.l2g, fine_layout.rstart, fine_layout.rend
    Pn = builder(coarse_coords, fine_coords).tocsr()   # (n_f_nodes, n_c_nodes), local
    nloc_f = fend - fstart
    nloc_c = coarse_layout.rend - coarse_layout.rstart

    P = PETSc.Mat().create(comm=comm)
    P.setSizes(((nloc_f, None), (nloc_c, None)))
    P.setType("aij")
    P.setUp()
    for i in range(Pn.shape[0]):                 # fine local node
        js = Pn.indices[Pn.indptr[i]:Pn.indptr[i + 1]]
        ws = Pn.data[Pn.indptr[i]:Pn.indptr[i + 1]]
        for c in range(ncomp):
            grow = int(l2g_f[i * ncomp + c])
            if grow < fstart or grow >= fend:    # set OWNED rows only
                continue
            gcols, vals = [], []
            for jj, w in zip(js.tolist(), ws.tolist()):
                gcol = int(l2g_c[jj * ncomp + c])
                if gcol >= 0:                    # skip constrained coarse DOFs
                    gcols.append(gcol)
                    vals.append(w)
            if gcols:
                P.setValues([grow], gcols, vals, addv=PETSc.InsertMode.INSERT_VALUES)
    P.assemble()
    return P


def _gather_coarse_cloud(coarse_coords, coarse_layout, ncomp, comm):
    """All-gather the (small) coarse node cloud across ranks and deduplicate.

    Returns ``(coords_u, cols_u)``: ``coords_u`` is the FULL coarse node cloud
    (Nu, dim) — every coarse node in the mesh, on every rank — and ``cols_u`` is
    (Nu, ncomp) the GLOBAL reduced column index of each node/component (``-1`` for a
    BC-constrained DOF). Ghost copies are bit-identical and dedup by rounded
    coordinate; constrained nodes stay in the cloud as barycentric vertices but
    carry ``-1`` so they are dropped from the transfer columns."""
    m4 = comm.tompi4py()
    ncn = coarse_coords.shape[0]
    cols_local = np.asarray(coarse_layout.l2g, dtype=np.int64).reshape(ncn, ncomp)
    cc_all = np.vstack(m4.allgather(
        np.ascontiguousarray(coarse_coords, dtype=float)))
    cols_all = np.vstack(m4.allgather(np.ascontiguousarray(cols_local)))
    _key, uidx = np.unique(np.round(cc_all, 9), axis=0, return_index=True)
    uidx = np.sort(uidx)
    return cc_all[uidx], cols_all[uidx]


def _build_crosspart_transfer(coarse_coords, fine_coords, coarse_layout,
                              fine_layout, ncomp, builder, comm):
    """One reduced->reduced prolongation for a NON-NESTED (independently
    partitioned) coarse level.

    The co-partitioned builder locates each rank's fine nodes only in that rank's
    LOCAL coarse coords; when the coarse and fine partitions differ a fine leaf on
    rank ``r`` can sit in a coarse cell owned by rank ``s`` -> missed (nearest-DOF
    fallback, wrong) or a coarse DOF with no fine image (zero column). Here every
    rank locates its OWNED fine nodes against the FULL coarse cloud (all-gathered,
    small — that is what makes a coarse MG level coarse), so point location spans
    partitions. Columns are the coarse GLOBAL reduced indices (off-rank columns are
    fine for MPIAIJ); constrained coarse DOFs (col < 0) drop out."""
    l2g_f, fstart, fend = fine_layout.l2g, fine_layout.rstart, fine_layout.rend
    coords_u, cols_u = _gather_coarse_cloud(coarse_coords, coarse_layout,
                                            ncomp, comm)

    Pn = builder(coords_u, fine_coords).tocsr()  # (n_f_nodes_local, Nu)
    nloc_f = fend - fstart
    nloc_c = coarse_layout.rend - coarse_layout.rstart

    P = PETSc.Mat().create(comm=comm)
    P.setSizes(((nloc_f, None), (nloc_c, None)))
    P.setType("aij")
    P.setUp()
    for i in range(Pn.shape[0]):                 # fine local node
        js = Pn.indices[Pn.indptr[i]:Pn.indptr[i + 1]]
        ws = Pn.data[Pn.indptr[i]:Pn.indptr[i + 1]]
        for c in range(ncomp):
            grow = int(l2g_f[i * ncomp + c])
            if grow < fstart or grow >= fend:    # set OWNED rows only
                continue
            gcols, vals = [], []
            for jj, w in zip(js.tolist(), ws.tolist()):
                gcol = int(cols_u[jj, c])
                if gcol >= 0:                    # skip constrained coarse DOFs
                    gcols.append(gcol)
                    vals.append(w)
            if gcols:
                P.setValues([grow], gcols, vals, addv=PETSc.InsertMode.INSERT_VALUES)
    P.assemble()
    return P


def _count_zero_columns_parallel(P, comm):
    """Number of empty columns of ``P`` across all ranks (coarse DOFs with no fine
    image). Column sums via P^T·1 (barycentric/RBF weights are positive, so a zero
    sum is an empty column). Used to auto-detect a cross-partition point-location
    miss (non-nested coarse level) and, separately, as the hard guard below."""
    ones_f = P.createVecLeft()
    ones_f.set(1.0)
    colsum = P.createVecRight()
    P.multTranspose(ones_f, colsum)
    nzero_local = int((colsum.array == 0.0).sum())
    nzero = comm.tompi4py().allreduce(nzero_local)
    ones_f.destroy()
    colsum.destroy()
    return nzero


def _assert_no_zero_columns_parallel(P, comm):
    """Parallel zero-column guard: a coarse DOF with no fine image -> singular
    Galerkin coarse operator."""
    nzero = _count_zero_columns_parallel(P, comm)
    if nzero:
        raise RuntimeError(
            f"parallel transfer has {nzero} zero columns (coarse DOFs with no fine "
            f"image) — BC-per-level reduction failed; coarse operator would be singular.")


def _assert_no_zero_columns_serial(P_csr, level):
    """Serial zero-column guard — same physics as the parallel guard above: a
    coarse DOF with no fine image makes the Galerkin coarse operator singular."""
    nzero = int((np.asarray((P_csr != 0).sum(axis=0)).ravel() == 0).sum())
    if nzero:
        raise RuntimeError(
            f"transfer {level - 1}->{level} has {nzero} zero columns (coarse DOFs "
            f"with no fine image) — BC-per-level reduction failed; coarse "
            f"operator would be singular.")


def _configure_pcmg(pc, Ps):
    """Reconfigure ``pc`` as a fresh PCMG (FMG F-cycle) driven by the supplied
    reduced->reduced prolongations ``Ps``, Galerkin RAP for coarse operators.

    Writes the MG bundle into the options DB under the PC's OWN options prefix
    (``pc.getOptionsPrefix()`` — e.g. ``Solver_N_`` for the scalar top-level PC,
    ``Solver_N_fieldsplit_velocity_`` for the Stokes velocity sub-PC) and removes
    any gamg keys BEFORE ``setFromOptions`` — otherwise ``setFromOptions`` re-reads
    a lingering ``pc_type=gamg`` and reverts ``setType("mg")``. ``setMGInterpolation``
    persists through ``setFromOptions``; the first ``PCSetUp`` builds the coarse
    operators from our P (no ``MatProductReplaceMats`` shape bug, since the PCMG is
    fresh and P's size is fixed)."""
    nlev = len(Ps) + 1
    prefix = pc.getOptionsPrefix() or ""
    opts = PETSc.Options()
    opts.setValue(prefix + "pc_type", "mg")
    opts.setValue(prefix + "pc_mg_type", "full")
    opts.setValue(prefix + "pc_mg_galerkin", "both")
    opts.setValue(prefix + "mg_levels_ksp_type", "richardson")
    opts.setValue(prefix + "mg_levels_pc_type", "sor")
    opts.setValue(prefix + "mg_coarse_pc_type", "redundant")
    opts.setValue(prefix + "mg_coarse_redundant_pc_type", "lu")
    for key in ("pc_gamg_type", "pc_gamg_repartition", "pc_gamg_agg_nsmooths"):
        opts.delValue(prefix + key)
    pc.setType("mg")
    pc.setMGLevels(nlev)
    pc.setMGType(PETSc.PC.MGType.FULL)
    for l in range(1, nlev):
        pc.setMGInterpolation(l, Ps[l - 1])
    pc.setFromOptions()


def _install_transfers(solver, Ps, verbose=False):
    """Configure the managed PCMG block to use the supplied prolongations.

    Two paths, keyed by ``solver._pc_option_prefix``:

    * **scalar / single-field vector** (top-level PC, prefix ``""``): build-time
      injection *before* the first ``PCSetUp`` — the first Galerkin assembly is
      built from our P directly.
    * **Stokes velocity block** (prefix ``"fieldsplit_velocity_"``): the velocity
      sub-PC is unreachable until the monolithic Jacobian is assembled
      (``PCFieldSplit`` forms ``A_vv`` via ``MatCreateSubMatrix``; ``snes.setUp``
      builds structure only -> err73). So force a Jacobian assembly, reach the
      velocity sub-PC, ``reset`` it and rebuild a fresh PCMG from our P, then
      re-attach the coupled Stokes nullspace.

    Either way ``KSPSetDMActive(OPERATOR, False)`` stops PETSc re-deriving the
    interpolation from the DM hierarchy."""
    nlev = len(Ps) + 1
    pfx = solver._pc_option_prefix or ""

    if pfx == "":
        ksp = solver.snes.getKSP()
        solver.snes.setUp()
        ksp.setDMActive(PETSc.KSP.DMActive.OPERATOR, False)
        _configure_pcmg(ksp.getPC(), Ps)
        if verbose:
            from underworld3 import mpi
            mpi.pprint(f"[{solver.name}] custom FMG installed: {nlev} levels, "
                       f"P sizes {[tuple(P.getSize()) for P in Ps]}")
        return

    if pfx != "fieldsplit_velocity_":
        raise NotImplementedError(
            f"custom_mg install: unsupported PC prefix '{pfx}'.")

    _install_velocity_block_transfers(solver, Ps, verbose=verbose)


def _install_velocity_block_transfers(solver, Ps, verbose=False):
    """Stokes velocity-block install (mechanism A: reset + fresh PCMG).

    Preconditions: ``solver._build`` + ``setFromOptions`` + ``_attach_stokes_nullspace``
    have run (the call site in ``SNES_Stokes_SaddlePt.solve`` guarantees this), so
    the SNES / DM exist but the Jacobian VALUES are not yet assembled."""
    snes = solver.snes
    snes.setUp()

    # 1. force monolithic Jacobian assembly so the fieldsplit can form A_vv
    x0 = solver.dm.getGlobalVec()
    x0.set(0.0)
    J = snes.getJacobian()[0]
    Pmat = snes.getJacobian()[1]
    try:
        f = J.createVecLeft()
        snes.computeFunction(x0, f)
        snes.computeJacobian(x0, J, Pmat)
    except PETSc.Error:
        # fallback: throwaway max_it=0 solve assembles + splits the operator
        saved = (solver.petsc_options.getString("snes_max_it")
                 if solver.petsc_options.hasName("snes_max_it") else None)
        solver.petsc_options["snes_max_it"] = 0
        snes.setFromOptions()
        snes.solve(None, x0)
        if saved is not None:
            solver.petsc_options["snes_max_it"] = saved
        snes.setFromOptions()
    solver.dm.restoreGlobalVec(x0)

    # Wire the freshly-assembled Jacobian into the KSP/outer-PC. SNESSolve does
    # this lazily, but we reach the fieldsplit BEFORE the solve — without it the
    # outer PC can carry an unassembled operator and PCSetUp fails with
    # "Matrix must be set first" (err73) for some configurations.
    snes.getKSP().setOperators(J, Pmat)

    # 2. split -> reach the velocity sub-KSP / sub-PC (field 0)
    outer_pc = snes.getKSP().getPC()
    outer_pc.setUp()
    vel_ksp = outer_pc.getFieldSplitSubKSP()[0]
    vel_pc = vel_ksp.getPC()
    A_vv, P_vv = vel_pc.getOperators()        # capture before reset (reset drops them)

    # 3. fresh PCMG on the velocity sub-block from our Ps
    vel_ksp.setDMActive(PETSc.KSP.DMActive.OPERATOR, False)
    vel_pc.reset()
    vel_pc.setOperators(A_vv, P_vv)
    _configure_pcmg(vel_pc, Ps)
    vel_pc.setUp()

    # 4. re-attach the coupled Stokes nullspace (operator state was touched)
    solver._attach_stokes_nullspace()

    if verbose:
        from underworld3 import mpi
        vel_prefix = vel_pc.getOptionsPrefix() or "fieldsplit_velocity_"
        mpi.pprint(f"[{solver.name}] custom FMG installed on velocity block: "
                   f"{len(Ps) + 1} levels, sub-prefix {vel_prefix!r}, "
                   f"P sizes {[tuple(P.getSize()) for P in Ps]}")


class CustomMGHierarchy:
    """A generalized FMG hierarchy: a sequence of level meshes (coarsest..finest)
    whose transfers are built by a pluggable builder, with BCs applied at every
    level. Adapter-agnostic — it consumes meshes + a way to get each level's
    BC-reduced DOF map; it does not know how the levels were produced.

    Parameters
    ----------
    level_meshes : list of Mesh
        Coarsest-first; the LAST entry must be the solver's own mesh.
    builder : {"barycentric", "rbf"}
        Per-level node prolongation builder.
    field_id : int or None
        Field index for multi-field solvers (e.g. 0 = velocity); None = single field.
    cross_partition : {"auto", True, False}
        Parallel (np>1) transfer strategy. ``False`` = rank-local point location
        (the co-partitioned nested / adapt-child fast path; each rank uses only its
        LOCAL coarse coords). ``True`` = all-gather the coarse cloud so every rank
        locates its fine nodes against the FULL coarse mesh (required when coarse
        and fine are partitioned independently — non-nested coarse tails). ``"auto"``
        (default) uses the fast path and, if it produces a zero-column transfer
        (the signature of a cross-partition point-location miss), rebuilds that
        level cross-partition. Serial builds ignore this.
    """

    def __init__(self, level_meshes, builder="barycentric", field_id=None,
                 cross_partition="auto"):
        if builder not in _BUILDERS:
            raise ValueError("builder must be 'barycentric' or 'rbf'")
        if len(level_meshes) < 2:
            raise ValueError("need at least 2 levels (>=1 coarse + finest)")
        if cross_partition not in ("auto", True, False):
            raise ValueError("cross_partition must be 'auto', True or False")
        self.level_meshes = list(level_meshes)
        self.builder = _BUILDERS[builder]
        self.builder_name = builder
        self.field_id = field_id
        self.cross_partition = cross_partition
        self.transfers = None

    def _recorded_node_transfer(self, level, nlev, degree, n_coarse, n_fine):
        """The EXACT nested prolongation recorded by ``mesh.adapt`` for this
        transfer, or ``None`` to fall back to point location.

        ``adapt`` maintains the parent/child relation, so for a bisection
        hierarchy the coarse-to-fine embedding is known exactly — every fine
        vertex is an inherited coarse vertex (weight 1) or an edge midpoint
        (1/2, 1/2). Using it avoids re-deriving an approximation by Delaunay
        point location, and it is **structurally full rank**: no coarse DOF can
        be left without a fine image, so the zero-column failure (#424) cannot
        arise on this path.

        Restricted to ``degree == 1``: the recorded relation is vertex-level,
        and a higher-degree field also has edge/face DOFs that it says nothing
        about. Those still go through the geometric builder — see #425 for the
        any-degree extension (record the parent CELL, then evaluate the coarse
        basis at the fine DOF reference coordinates).

        The recorded list covers the ADAPT generations only; the uniform coarse
        tail beneath them is not a bisection hierarchy, so its transfers keep
        using the builder. Shapes are checked rather than assumed — a mismatch
        means the levels do not line up as expected and it is safer to fall
        back than to install a silently wrong transfer.
        """
        import scipy.sparse as sp
        if degree != 1:
            return None
        recorded = getattr(self.level_meshes[-1], "_adapt_prolongation", None)
        if not recorded:
            return None
        idx = level - (nlev - len(recorded))
        if idx < 0 or idx >= len(recorded) or recorded[idx] is None:
            return None
        rows, cols, vals = recorded[idx]
        if (rows.size == 0 or int(rows.max()) >= n_fine
                or int(cols.max()) >= n_coarse):
            return None
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_fine, n_coarse))

    def build(self, solver):
        """Build the BC-reduced prolongations. ``solver`` is the (built) finest
        solver. Each COARSE level's BC-constrained reduced map is derived directly
        from the coarse mesh DM by copying the finest solver's fields + DS onto it
        (no throwaway solver); the finest level reads its map from ``solver.dm``.

        Serial: scipy reduced->reduced CSR. Parallel (np>1): rank-local node-level
        weights assembled into MPIAIJ transfers with global-section reduced
        numbering (nested co-partitioned path)."""
        from underworld3 import mpi
        var = solver.Unknowns.u
        degree = var.degree
        continuous = getattr(var, "continuous", True)
        nlev = len(self.level_meshes)
        parallel = mpi.size > 1

        # Operator-faithful finest level: finalize the DM section and assemble the
        # operator BEFORE reading the finest reduced map. The finest transfer's row
        # space must be the space the operator's PCMG will Galerkin against; the DM
        # global section is that space only once the SNES is set up (an adapt()
        # child can otherwise carry a not-yet-finalized / auxiliary section that
        # disagrees with the assembled operator -> rectangular finest transfer ->
        # cryptic PETSc error 60 in the PtAP). setUp is idempotent (the install
        # paths call it again).
        try:
            solver.snes.setUp()
        except Exception:
            # Sanctioned swallow: setUp can fail on a not-yet-fully-configured
            # SNES (pre-solve injection). The install paths call setUp again;
            # the finest map then reads the DM's current global section.
            pass

        coords, maps, ncomp = [], [], []
        for k, mesh in enumerate(self.level_meshes):
            c = np.asarray(mesh._get_coords_for_basis(degree, continuous))
            finest = (k == nlev - 1)
            if parallel:
                # ``maps`` holds a LevelLayout per level in parallel, and a
                # bare reduced->full index array per level in serial.
                lay = (_level_dof_layout(solver.dm, self.field_id) if finest
                       else _coarse_dof_layout(solver, mesh, self.field_id))
                nfull = lay.n_full
                maps.append(lay)
            else:
                rmap, nfull = (_reduced_map(solver.dm, self.field_id) if finest
                               else _coarse_reduced_map(solver, mesh, self.field_id))
                maps.append(rmap)
            nc = nfull // c.shape[0]
            if nfull % c.shape[0] != 0:
                raise RuntimeError(
                    f"level {k}: full DOFs {nfull} not divisible by nodes {c.shape[0]}")
            coords.append(c)
            ncomp.append(nc)

        if len(set(ncomp)) != 1:
            raise RuntimeError(f"inconsistent component counts across levels: {ncomp}")
        nc = ncomp[0]

        # Operator-faithful check: the finest reduced map must span exactly the
        # assembled operator's rows. Checkable directly for the monolithic
        # single-field operator (field_id is None — scalar / single-field vector,
        # e.g. Poisson, Projection, semi-Lagrangian AdvDiffusion on an adapt child).
        # Fail here with an actionable message rather than deep inside PETSc's PtAP.
        if self.field_id is None:
            self._assert_finest_matches_operator(solver, maps[-1], parallel)

        Ps = []
        comm = solver.dm.comm
        for l in range(1, nlev):
            if parallel:
                args = (coords[l - 1], coords[l], maps[l - 1], maps[l], nc,
                        self.builder, comm)
                if self.cross_partition is True:
                    P = _build_crosspart_transfer(*args)
                else:
                    P = _build_parallel_transfer(*args)
                    # "auto": a zero-column transfer means the coarse level is NOT
                    # co-partitioned with the fine level (a fine leaf sits in an
                    # off-rank coarse cell). Rebuild it spanning partitions.
                    if (self.cross_partition == "auto"
                            and _count_zero_columns_parallel(P, comm) > 0):
                        P = _build_crosspart_transfer(*args)
                _assert_no_zero_columns_parallel(P, comm)
                Ps.append(P)
            else:
                Pn = self._recorded_node_transfer(
                    l, nlev, degree, coords[l - 1].shape[0], coords[l].shape[0])
                if Pn is not None:
                    Pr = _reduced_from_node_transfer(Pn, maps[l - 1], maps[l], nc)
                else:
                    Pr = _reduced_transfer(coords[l - 1], coords[l], maps[l - 1],
                                           maps[l], nc, self.builder)
                _assert_no_zero_columns_serial(Pr, l)
                Ps.append(_to_petsc_aij(Pr))
        self.transfers = Ps
        return Ps

    @staticmethod
    def _assert_finest_matches_operator(solver, finest_map, parallel):
        """Guarantee the finest reduced map spans the assembled operator's rows.

        The finest transfer is Galerkin-multiplied against the solver's real
        operator (``PtAP``); if the row space disagrees the product is rectangular
        and PETSc aborts with a bare error 60. On a plain mesh the DM global section
        and the operator always agree; the guard matters for adapt() children whose
        DM section could be stale relative to the freshly assembled operator."""
        try:
            op_n = int(solver.snes.getJacobian()[0].getSize()[0])
        except Exception:
            return                                   # can't read operator -> skip
        if op_n <= 0:
            return
        if parallel:
            red_n = int(solver.dm.comm.tompi4py().allreduce(
                int(finest_map.rend - finest_map.rstart)))
        else:
            red_n = int(len(finest_map))             # r2f length = reduced global size
        if red_n != op_n:
            raise RuntimeError(
                f"custom_mg: finest reduced-map size {red_n} != assembled operator "
                f"size {op_n}. The DM global section disagrees with the operator "
                f"(an adapt-child section inconsistency); the finest transfer would "
                f"be rectangular and the Galerkin PtAP would abort (PETSc error 60). "
                f"Rebuild the solver so its DM section matches the operator before "
                f"installing custom-P.")

    def install(self, solver, verbose=False):
        if self.transfers is None:
            raise RuntimeError("call build() before install()")
        _install_transfers(solver, self.transfers, verbose=verbose)


# --------------------------------------------------------------------------- #
#  Entry points
# --------------------------------------------------------------------------- #
def set_custom_fmg(solver, coarse_meshes, *, builder="barycentric",
                   field_id=None, cross_partition="auto", verbose=False):
    """Generalized custom-P FMG with BC-per-level reduction (the correct path).

    Registers a :class:`CustomMGHierarchy` on the solver so that the next
    ``solve()`` builds and installs it (build-time injection). The hierarchy is
    ``[*coarse_meshes, solver.mesh]``; each coarse level's BC-constrained reduced
    map is derived directly from its DM by copying the solver's fields + DS
    (``_coarse_reduced_map``), so ``coarse_meshes`` need only carry the same
    boundary labels as the solver's mesh. For a saddle-point (Stokes) solver pass
    ``field_id=0`` to target the velocity sub-block.

    ``cross_partition`` selects the parallel (np>1) transfer strategy (see
    :class:`CustomMGHierarchy`); the default ``"auto"`` handles both nested and
    non-nested coarse tails."""
    solver._custom_mg = {
        "mode": "hierarchy",
        "hierarchy": CustomMGHierarchy(list(coarse_meshes) + [solver.mesh],
                                       builder=builder, field_id=field_id,
                                       cross_partition=cross_partition),
        "verbose": verbose,
    }
    solver.is_setup = False


def auto_inject_custom_mg(solver, field_id=None):
    """Solve-hook entry: inject custom-P FMG from either a solver-set hierarchy
    (``set_custom_fmg``) or a **mesh-owned** one (``mesh.adapt`` refinement child).

    A refinement child carries ``mesh._custom_mg_coarse_meshes`` (the static
    coarse tail). The first time a solver on such a mesh solves, we lazily build a
    :class:`CustomMGHierarchy` ``[*coarse, solver.mesh]`` targeting ``field_id``
    (0 for the Stokes velocity block, None for scalar/vector) and register it on
    the solver — so every solver on an adapted mesh drives geometric MG with no
    per-solver call. A solver-set hierarchy (if present) always wins.
    """
    # Solver-set hierarchy (set_custom_fmg): the user asked for it explicitly —
    # build + install directly and let any error surface.
    if solver._custom_mg is not None:
        inject_custom_mg(solver)
        return

    # Mesh-owned hierarchy (adapt() child): OPPORTUNISTIC auto-pickup. It must never
    # crash a solve, so build the transfers (which now validate the finest reduced
    # map against the assembled operator — see CustomMGHierarchy.build) inside a
    # try/except and fall back to the solver's default preconditioner on any failure.
    # The finest map is derived from the DM section AFTER snes.setUp() finalizes it,
    # so it is faithful to the operator on adapt children too — including scalar
    # semi-Lagrangian advection-diffusion (which earlier had to be skipped).
    coarse = getattr(solver.mesh, "_custom_mg_coarse_meshes", None)
    if coarse is None:
        return                              # nothing to inject

    builder = getattr(solver.mesh, "_custom_mg_builder", "barycentric")
    # Retry with the RBF builder before abandoning geometric MG. The
    # barycentric builder has LOCAL support: it re-triangulates the coarse
    # DOF cloud and locates each fine DOF in one simplex, so a coarse DOF
    # is only reached if some fine DOF lands in a simplex touching it. Move
    # the fine coordinates — which is exactly what mesh.relax() does — and a
    # coarse DOF can lose every fine image, giving a zero column and a
    # singular Galerkin coarse operator. The RBF builder has GLOBAL support
    # (every coarse DOF is reached through the RBF solve), so it does not
    # have that failure mode: measured on a relaxed 3D adapt child,
    # barycentric fell back to GAMG at 23 its while RBF kept pc=mg at 2.
    # Falling back to GAMG loses the hierarchy entirely, so try the cheaper
    # degradation first. See #424.
    _attempts = [builder] + (["rbf"] if builder != "rbf" else [])
    h = Ps = None
    for _i, _b in enumerate(_attempts):
        h = CustomMGHierarchy(list(coarse) + [solver.mesh], builder=_b,
                              field_id=field_id)
        try:
            Ps = h.build(solver)
            break
        except Exception as exc:            # pragma: no cover - defensive
            import warnings
            if _i + 1 < len(_attempts):
                warnings.warn(
                    f"custom_mg: {_b} transfer build failed ({exc}); "
                    f"retrying with the '{_attempts[_i + 1]}' builder, which "
                    f"has global support and cannot leave a coarse DOF "
                    f"without a fine image.")
                continue
            warnings.warn(
                f"custom_mg: mesh-owned FMG build failed ({exc}); using the "
                "solver's default preconditioner.")
            return

    # Dimensional guard (checkable for the monolithic operator, field_id is None):
    # the finest transfer must chain to the operator PCMG will Galerkin against.
    if field_id is None and len(Ps):
        try:
            solver.snes.setUp()
            op_n = int(solver.snes.getJacobian()[0].getSize()[0])
            pr, pc = (int(v) for v in Ps[-1].getSize())
            # Only a genuine size mismatch disqualifies the transfer. A
            # SQUARE finest transfer (pc == pr) is legitimate and common on
            # boundary-focused refinement: a generation whose new vertices
            # all land on a Dirichlet boundary adds only CONSTRAINED dofs,
            # so the free-dof counts of the last two levels coincide — the
            # level is redundant but correct, and rejecting it silently
            # downgraded every curved-domain boundary-layer adapt to the
            # default preconditioner (round-3b annulus finding, 2026-07).
            if op_n > 0 and pr != op_n:
                import warnings
                warnings.warn(
                    "custom_mg: mesh-owned adapt-mesh FMG transfer is incompatible "
                    f"with this solver's operator (transfer {pr}x{pc}, operator {op_n}); "
                    "skipping the auto-pickup (using the default preconditioner). "
                    "set_custom_fmg() an explicit hierarchy to override.")
                return
        except Exception:
            pass                            # can't check -> don't block working cases

    h.install(solver, verbose=False)
    solver._custom_mg = {"mode": "hierarchy", "hierarchy": h, "verbose": False}


def inject_custom_mg(solver):
    """Build + install the custom-P FMG. Called from ``solve()`` (after ``_build``,
    before the SNES solve) when ``solver._custom_mg`` is set. Dispatches:
    - ``mode == "hierarchy"`` -> BC-per-level reduced path (correct, general);
    - legacy dict ``{coarse_meshes, kind}`` -> finest-only reduction (kept for
      back-compat; valid only when coarse levels are non-nested / unconstrained)."""
    cfg = solver._custom_mg

    if isinstance(cfg, dict) and cfg.get("mode") == "hierarchy":
        h = cfg["hierarchy"]
        h.build(solver)              # parallel-capable (nested co-partitioned)
        h.install(solver, verbose=cfg.get("verbose", False))
        return

    # ---- legacy finest-only path (back-compat, serial only) -----------------
    # TODO(deprecate): remove together with SolverBaseClass.set_custom_mg and
    # test_1015's legacy cases. The one behavioural difference from the
    # hierarchy path: NO BC-per-level reduction — transfers are built on the
    # full node cloud, which is only valid when the coarse levels are
    # non-nested / unconstrained (a nested coarse boundary DOF coinciding with
    # a BC-removed fine DOF would give a zero column -> singular coarse op).
    _require_serial("legacy custom_mg (set_custom_mg)")
    coarse_meshes = cfg["coarse_meshes"]
    builder = _BUILDERS[cfg["kind"]]
    var = solver.Unknowns.u
    degree = var.degree
    continuous = getattr(var, "continuous", True)
    if solver._pc_option_prefix not in ("", None):
        raise NotImplementedError("legacy custom_mg path is scalar/single-field only")
    fine = _reduce_to_global(solver.dm,
                             solver.mesh._get_coords_for_basis(degree, continuous))
    levels = [m._get_coords_for_basis(degree, continuous) for m in coarse_meshes]
    levels.append(fine)
    Ps = [_to_petsc_aij(builder(levels[l - 1], levels[l])) for l in range(1, len(levels))]
    _install_transfers(solver, Ps, verbose=cfg.get("verbose", False))
