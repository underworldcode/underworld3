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

import os
from typing import NamedTuple

import numpy as np
from petsc4py import PETSc

from underworld3.utilities import multigrid_options

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
    """RBF prolongation via the STANDARD local interpolator (#429).

    The sparse, linear-exact, kd-tree-based local RBF the rest of the code
    uses (``kdtree.interpolation_matrix``, ``order=1``: polyharmonic
    r² log r kernel + affine tail solved per target over its nearest
    neighbours — see ``docs/developer/subsystems/interpolation.md``).
    Replaces the original GLOBAL builder, which assembled and solved the
    dense coarse-cloud kernel matrix and returned a transfer with
    ``nnz/row == n_coarse`` — a rescue whose Galerkin coarse operators
    were dense too, and which had no conditioning path on large clouds
    (#429). Same kernel, same reproduction guarantees (constants and
    linears to machine precision), sparse support.

    A row-wise kNN builder guarantees nonzeros per ROW, never per COLUMN
    (#424): a coarse DOF outside every fine stencil still yields an empty
    column and a singular PᵀAP. The build loop's zero-column repair
    (nearest-fine-DOF injection) is the counterpart, for this builder and
    the barycentric one alike. ``smooth`` is accepted for signature
    compatibility and unused — locality is the conditioning here.
    """
    from underworld3 import kdtree

    kdt = kdtree.KDTree(np.ascontiguousarray(coarse_coords, dtype=float))
    return kdt.interpolation_matrix(
        np.ascontiguousarray(fine_coords, dtype=float), order=1)


def _drop_structural_zeros(P_csr, tol=1e-12):
    """Remove numerically-zero transfer weights that are STRUCTURALLY nonzero.

    A fine node coincident with a coarse node gets barycentric weights
    ``[1, ~1e-16, ~1e-16, ~1e-16]`` — an identity row in VALUE but a 4-entry
    row in STRUCTURE, and Galerkin RAP fills by structure. On a composite
    (placed/overlay) hierarchy the background is all such rows, so the junk
    entries fatten every coarse operator's background block level over level
    (measured 90 -> 265 -> 481 nnz/row down a 4-level tail, #629). Weights
    are O(1) partition-of-unity values, so ``tol`` cuts only float noise; a
    row cannot empty (its weights sum to 1)."""
    P = P_csr.tocsr()
    P.data[np.abs(P.data) < tol] = 0.0
    P.eliminate_zeros()
    return P


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
    return _reduced_from_node_transfer(_drop_structural_zeros(Pn),
                                       r2f_c, r2f_f, ncomp)


def _is_native_refine_pair(coarse_mesh, fine_mesh):
    """Are these two levels consecutive members of one native ``refine()``
    hierarchy? Decided by the ``_refine_slot`` tags ``_coarse_level_meshes``
    (and the requested-native arm of :func:`build_transfers`) stamp on the
    family — token identity plus consecutive level indices. Untagged levels
    (placed meshes, adapt generations, moved bases) answer False and keep
    their existing transfer routes."""
    sa = getattr(coarse_mesh, "_refine_slot", None)
    sb = getattr(fine_mesh, "_refine_slot", None)
    return (sa is not None and sb is not None
            and sa[0] is sb[0] and sb[1] == sa[1] + 1)


def nested_refine_pair_prolongation(coarse_mesh, fine_mesh, degree, continuous,
                                    coarse_coords=None, fine_coords=None):
    """EXACT node-level prolongation for a native ``refine()`` pair, at ANY
    polynomial degree — the FE embedding of the coarse Lagrange space in the
    fine one. Returns a scipy CSR ``(n_fine_nodes, n_coarse_nodes)`` matrix,
    or ``None`` to decline (the caller falls back to the geometric builder).

    Why: the DOF clouds of degree >= 2 spaces do NOT nest even where the
    meshes do (an L1 edge node is no L0 node), so the point-located builders
    return scattered transfers whose Galerkin product runs 3-5x fatter per row
    than a native coarse operator (481 vs ~90-150 nnz/row measured, #629).
    PETSc's own ``DMCreateInterpolation`` general path is NOT the embedding
    either (measured: row sums to 1.375, quadratic reproduction error 1e-2).

    Construction (#425, the dual-basis identity): a Lagrange basis is dual to
    its nodal points, so with ``M[i,m] = mu_m(x_i)`` over a parent cell's own
    ``n`` nodes and ``B[t,m] = mu_m(x_t)`` at the fine nodes inside it, the
    weight block is ``W = B M^-1`` — no reference element, no per-degree
    formulae, no point location. Coordinates are pulled back through the
    parent's affine map first: raw monomials on a cell of size ``h`` give
    ``cond(M) ~ h^-k``, while under the pullback conditioning depends only on
    (degree, dim) — and the pulled-back coordinate is the barycentric vector,
    so the "is this fine node inside its parent?" guard is free.

    The parent relation itself is recovered from the two DMs: every fine
    vertex of a uniform refinement is an inherited coarse vertex or an exact
    edge midpoint (identified by bit equality, so a snapped/relaxed hierarchy
    declines rather than guesses), and the parent cell follows topologically
    (:func:`~underworld3.utilities.nvb.nested_cell_parents`). Structurally
    full rank: every coarse node is inherited into the fine level with weight
    1, so the zero-column failure (#424) cannot arise here.
    """
    import itertools

    import scipy.sparse as sp
    from underworld3.utilities.nvb import (nested_cell_parents,
                                           nested_prolongation_from_dms)

    cdm, fdm = coarse_mesh.dm, fine_mesh.dm
    dim = cdm.getDimension()
    if cdm.getCoordinateDim() != dim:      # embedded surface: no affine pullback
        return None
    vP = nested_prolongation_from_dms(cdm, fdm)
    if vP is None:
        return None
    parents = nested_cell_parents(cdm, fdm, vP)
    if parents is None:
        return None
    try:
        cn = coarse_mesh._cell_node_indices(degree, continuous)
        fn = fine_mesh._cell_node_indices(degree, continuous)
    except (NotImplementedError, AttributeError):
        return None

    Xc = (np.asarray(coarse_coords) if coarse_coords is not None
          else np.asarray(coarse_mesh._get_coords_for_basis(degree, continuous)))
    Xf = (np.asarray(fine_coords) if fine_coords is not None
          else np.asarray(fine_mesh._get_coords_for_basis(degree, continuous)))
    n_c, n_f = Xc.shape[0], Xf.shape[0]

    # One containing fine cell per fine node — any one: the parent interpolant
    # is continuous across parent faces, so the weights agree either way.
    owner = np.full(n_f, -1, dtype=np.int64)
    for k in range(fn.shape[0]):
        owner[fn[k]] = k
    if (owner < 0).any():
        return None

    ccS, ccE = cdm.getHeightStratum(0)
    cvS, cvE = cdm.getDepthStratum(0)
    vxy = np.ascontiguousarray(
        cdm.getCoordinatesLocal().array.reshape(-1, dim))
    cell_verts = np.empty((ccE - ccS, dim + 1), dtype=np.int64)
    for c in range(ccS, ccE):
        vv = [q - cvS for q in cdm.getTransitiveClosure(c)[0]
              if cvS <= q < cvE]
        if len(vv) != dim + 1:
            return None
        cell_verts[c - ccS] = vv

    # Monomials of total degree <= degree: exactly nodes-per-cell many on a
    # simplex, so M is square (guaranteed by _cell_node_indices' own check).
    E = np.asarray([e for e in itertools.product(range(degree + 1), repeat=dim)
                    if sum(e) <= degree])
    if E.shape[0] != cn.shape[1]:
        return None

    def vander(L):
        return np.prod(L[:, None, :] ** E[None, :, :], axis=2)

    parent_of = parents[owner] - ccS           # coarse cell index per fine node
    order = np.argsort(parent_of, kind="stable")
    runs = np.flatnonzero(np.diff(parent_of[order])) + 1
    rows, cols, vals = [], [], []
    for grp in np.split(order, runs):
        c = int(parent_of[grp[0]])
        verts = vxy[cell_verts[c]]
        A = (verts[1:] - verts[0]).T
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return None
        Lc = (Xc[cn[c]] - verts[0]) @ Ainv.T
        Lf = (Xf[grp] - verts[0]) @ Ainv.T
        # The pullback is the barycentric vector: outside the parent means the
        # attribution (or the geometry) is off — decline, never extrapolate.
        if Lf.min() < -1e-8 or (1.0 - Lf.sum(axis=1)).min() < -1e-8:
            return None
        try:
            W = np.linalg.solve(vander(Lc).T, vander(Lf).T).T   # B M^-1
        except np.linalg.LinAlgError:
            return None
        nodes_c = cn[c]
        for t_i, t in enumerate(grp):
            keep = np.flatnonzero(np.abs(W[t_i]) > 1e-12)
            rows.extend([int(t)] * keep.size)
            cols.extend(int(nodes_c[j]) for j in keep)
            vals.extend(float(W[t_i, j]) for j in keep)

    P = sp.csr_matrix((vals, (rows, cols)), shape=(n_f, n_c))
    # Lagrange partition of unity: every row must sum to 1 exactly.
    if np.abs(np.asarray(P.sum(axis=1)).ravel() - 1.0).max() > 1e-8:
        return None
    return P


def _fac_patch_split(P_csr, coords_c, coords_f, map_c, map_f, nc,
                     w_tol=1e-8, x_tol=1e-9, cover_max=0.75):
    """FAC patch/background split of one level's reduced fine DOFs (#629).

    A locally-refined hierarchy is a COMPOSITE grid: levels differ only in
    the refined patch, and the background falls through unchanged. Smoothing
    the whole level is the measured pathology (a V-cycle smooths ~175% of the
    fine level's nonzeros); the FAC/MLAT answer is to smooth each level only
    on its patch plus an interface halo, with the background smoothed once at
    the level that owns it. This function finds that patch algebraically from
    the level's own transfer — builder-agnostic, so placed levels work too.

    A fine reduced DOF is BACKGROUND when its transfer row is an identity row
    (one effective weight-1 entry) AND its node coincides with the referenced
    coarse node (within ``x_tol``): the node falls through the pair. Anything
    else is PATCH. Split-duplicated nodes — two fine nodes onto one coarse
    node, a fault slit — count as patch even though each row is identity: the
    pair changes topology there. The HALO is the one-coarse-cell layer of
    background whose coarse node is referenced by some patch row. Membership
    is decided per NODE (all components together), so the split is invariant
    under the rotated path's per-node Q rotation.

    Returns ``(owned_rows, subdomain_rows)`` — reduced fine row indices of
    the patch and of patch+halo — or ``None`` when patch+halo covers more
    than ``cover_max`` of the level: a uniform pair refines everywhere, and
    whole-level smoothing is then the right configuration.
    """
    P = P_csr.tocsr()
    n = P.shape[0]
    if n == 0:
        return None
    # TODO(MEASURE): #629 campaign knob for the whole-level gate; remove
    # once the cover threshold is settled.
    cover_max = float(os.environ.get("UW_FAC_COVER", cover_max))
    map_f = np.asarray(map_f, dtype=np.int64)
    map_c = np.asarray(map_c, dtype=np.int64)
    node_f = map_f // nc
    indptr, cols, data = P.indptr, P.indices, P.data

    bg_col = np.full(n, -1, dtype=np.int64)     # coarse NODE of background rows
    patch_row = np.zeros(n, dtype=bool)
    for r in range(n):
        sl = slice(indptr[r], indptr[r + 1])
        w = data[sl]
        if w.size == 0:
            patch_row[r] = True
            continue
        aw = np.abs(w)
        j = int(np.argmax(aw))
        if abs(w[j] - 1.0) > w_tol or (aw.sum() - aw[j]) > w_tol:
            patch_row[r] = True
            continue
        cnode = int(map_c[cols[sl][j]] // nc)
        if np.max(np.abs(coords_f[int(node_f[r])] - coords_c[cnode])) > x_tol:
            patch_row[r] = True
            continue
        bg_col[r] = cnode

    nn = int(node_f.max()) + 1
    node_is_patch = np.zeros(nn, dtype=bool)
    node_is_patch[node_f[patch_row]] = True
    # split duplicates: >1 distinct background fine node onto one coarse node
    bgr = np.flatnonzero(bg_col >= 0)
    if bgr.size:
        pairs = np.unique(np.stack([bg_col[bgr], node_f[bgr]], axis=1), axis=0)
        counts = np.bincount(pairs[:, 0])
        dup = np.flatnonzero(counts > 1)
        if dup.size:
            node_is_patch[pairs[np.isin(pairs[:, 0], dup), 1]] = True

    row_is_patch = node_is_patch[node_f]
    # halo: background rows whose coarse node is referenced by a patch row
    entry_rows = np.repeat(np.arange(n), np.diff(indptr))
    csel = np.zeros(coords_c.shape[0], dtype=bool)
    csel[map_c[cols[row_is_patch[entry_rows]]] // nc] = True
    halo_rows = (~row_is_patch) & (bg_col >= 0) & csel[np.clip(bg_col, 0, None)]
    node_in_sub = node_is_patch.copy()
    node_in_sub[node_f[halo_rows]] = True

    sub = np.flatnonzero(node_in_sub[node_f])
    if sub.size > cover_max * n:
        return None
    owned = np.flatnonzero(row_is_patch)
    return owned.astype(np.int64), sub.astype(np.int64)


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
    Pn = _drop_structural_zeros(
        builder(coarse_coords, fine_coords))           # (n_f_nodes, n_c_nodes), local
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

    Pn = _drop_structural_zeros(builder(coords_u, fine_coords))  # (n_f_local, Nu)
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


def _repair_zero_columns_serial(P_csr, coords_c, coords_f, map_c, map_f,
                                nc, level):
    """Give each unreached coarse DOF a nearest-fine-DOF entry (serial).

    The barycentric builder has LOCAL support, so on NON-NESTED level pairs
    — two independently PLACED meshes (#626), a relaxed child (#424) — a
    coarse DOF can lose every fine image and its Galerkin column goes
    singular. The dense-RBF rescue fixes that globally at a performance
    cliff; when only a handful of columns are empty, a surgical injection
    entry (weight 1 at the nearest fine DOF of the same component) makes
    the RAP nonsingular at zero cost. These are preconditioner transfers,
    not the discretisation — a few injected rows cost iterations at worst,
    never correctness. Returns the (possibly repaired) matrix and the
    repair count; a column it cannot repair is left empty for the guard
    to refuse loudly.
    """
    colsum = np.asarray((P_csr != 0).sum(axis=0)).ravel()
    zero = np.flatnonzero(colsum == 0)
    if not len(zero):
        return P_csr, 0
    from scipy.spatial import cKDTree

    n_full_f = coords_f.shape[0] * nc
    inv_f = -np.ones(n_full_f, dtype=np.int64)
    inv_f[np.asarray(map_f, dtype=np.int64)] = np.arange(len(map_f))
    tree = cKDTree(coords_f)
    P = P_csr.tolil()
    repaired = 0
    for j in zero:
        full_c = int(map_c[j])
        node_c, comp = divmod(full_c, nc)
        k = min(8, coords_f.shape[0])
        for i in np.atleast_1d(tree.query(coords_c[node_c], k=k)[1]):
            full_f = int(i) * nc + comp
            if inv_f[full_f] >= 0:
                P[int(inv_f[full_f]), int(j)] = 1.0
                repaired += 1
                break
    return P.tocsr(), repaired


def _configure_pcmg(pc, Ps, coarse="redundant", smoother="robust", owned=None,
                    ksp=None, patch_rows=None):
    """Reconfigure ``pc`` as a fresh PCMG (FMG F-cycle) driven by the supplied
    reduced->reduced prolongations ``Ps``, Galerkin RAP for coarse operators.

    The option VALUES come from :func:`multigrid_options.geometric_mg_bundle` —
    the same bundle the native (DMPlex-refinement) route applies — so custom-P
    and native multigrid cannot be configured differently (#468). ``coarse``
    selects the coarse-solve variant: ``"svd"`` on the rotated path, whose
    Galerkin-coarsened operator inherits the rigid-rotation null space. ``smoother``
    is the variant ``solver.strategy`` asks for — pass
    ``solver._mg_smoother_variant``, or the strategy is honoured on the native route
    and silently ignored here, which is the drift this module exists to prevent.
    ``owned``
    is the solver's record of the option values UW3 has written, so a key the USER
    set is left alone (see :meth:`multigrid_options.MGSettings.apply`); ``None`` writes
    unconditionally, which is right for the rotated path's per-solve prefix.

    The bundle is written into the options DB under the PC's OWN options prefix
    (``pc.getOptionsPrefix()`` — e.g. ``Solver_N_`` for the scalar top-level PC,
    ``Solver_N_fieldsplit_velocity_`` for the Stokes velocity sub-PC) BEFORE
    ``setFromOptions``, and it clears the gamg keys — otherwise ``setFromOptions``
    re-reads a lingering ``pc_type=gamg`` and reverts ``setType("mg")``.
    ``setMGInterpolation`` persists through ``setFromOptions``; the first
    ``PCSetUp`` builds the coarse operators from our P (no
    ``MatProductReplaceMats`` shape bug, since the PCMG is fresh and P's size is
    fixed).

    ``patch_rows`` is the FAC configuration (#629): per PCMG level, ``None``
    (whole-level smoothing, the classical setup) or ``(owned, subdomain)``
    reduced-row index arrays from :func:`_fac_patch_split`. A level with a
    patch gets its smoother PC switched to ASM with that ONE user subdomain
    — the smoother relaxes only the refined patch plus its interface halo,
    the background falls through to the level that owns it, and the V-cycle's
    smoothing cost follows the patch sizes' geometric series instead of
    levels-times-whole-mesh. Residual and transfer work stay global. The
    per-level ``mg_levels_<l>_pc_type`` keys written here are returned so a
    per-solve caller (the rotated path) can drop them from the DB."""
    nlev = len(Ps) + 1
    prefix = pc.getOptionsPrefix() or ""
    opts = PETSc.Options()
    multigrid_options.geometric_mg_bundle(coarse=coarse, smoother=smoother).apply(
        opts, prefix, owned=owned)
    # TODO(MEASURE): #629 contrast-campaign knob — smoother iteration count
    # override (the bundle's gmres/4 vs /8 discriminator); remove when settled.
    _sm_its = os.environ.get("UW_MG_SMOOTH_ITS")
    if _sm_its:
        opts[prefix + "mg_levels_ksp_max_it"] = _sm_its
    # Per-level override BEFORE setFromOptions: the numbered key beats the
    # generic mg_levels_pc_type from the bundle, and having it in the DB keeps
    # any later setFromOptions from reverting the live setType below.
    fac_keys = []
    _only = os.environ.get("UW_FAC_LEVELS")            # TODO(MEASURE): #629 knob
    _only = {int(t) for t in _only.split(",")} if _only else None
    if patch_rows:
        for l in range(1, nlev):
            if (l < len(patch_rows) and patch_rows[l] is not None
                    and (_only is None or l in _only)):
                key = f"mg_levels_{l}_pc_type"
                opts[prefix + key] = "asm"
                fac_keys.append(key)
    # ``ksp_type`` is in the bundle (#514: a Krylov smoother makes this PC vary
    # between applications, so its KSP must judge convergence flexibly), but on
    # the top-level path the KSP consumed its options long before this
    # injection runs, so a database write alone never takes effect. Apply the
    # RESOLVED value — the bundle's, or the user's own where the ownership
    # latch left it alone — to the live object. The fieldsplit velocity
    # sub-KSP does not need this: its ``setFromOptions`` runs at the parent's
    # ``PCSetUp``, after the write.
    if ksp is not None:
        ksp.setType(opts.getString(prefix + "ksp_type", ksp.getType()))
    pc.setType("mg")
    pc.setMGLevels(nlev)
    pc.setMGType(PETSc.PC.MGType.FULL)
    for l in range(1, nlev):
        pc.setMGInterpolation(l, Ps[l - 1])
    pc.setFromOptions()
    # FAC subdomains go on the LIVE smoother PCs — an IS cannot ride the
    # options DB. setFromOptions above has already pushed the bundle (and the
    # per-level asm keys) into the level KSPs, so the objects are stable now.
    if patch_rows:
        # TODO(MEASURE): #629 campaign knobs — ASM variant and subdomain
        # solver for the FAC smoother; remove once the configuration settles.
        # BASIC, not restricted, ASM: the correction must land on the halo
        # too. Measured (banded Poisson, 4-level tail): restrict stalls the
        # outer KSP at 80-375 iterations where basic runs 6 against a
        # whole-level baseline of 4 — discarding the subdomain solve's halo
        # correction breaks the interface error systematically.
        _asm_type = os.environ.get("UW_FAC_ASM_TYPE", "basic")
        _sub_pc = os.environ.get("UW_FAC_SUB_PC")
        _whole = os.environ.get("UW_FAC_WHOLE")        # asm WITHOUT subdomain
        for l in range(1, nlev):
            entry = patch_rows[l] if l < len(patch_rows) else None
            if entry is None or (_only is not None and l not in _only):
                continue
            owned_rows, sub_rows = entry
            if _whole:
                pc.getMGSmoother(l).getPC().setType("asm")
                continue
            sm = pc.getMGSmoother(l)
            spc = sm.getPC()
            spc.setType("asm")
            is_sub = PETSc.IS().createGeneral(
                np.asarray(sub_rows, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
            is_own = PETSc.IS().createGeneral(
                np.asarray(owned_rows, dtype=PETSc.IntType),
                comm=PETSc.COMM_SELF)
            # One subdomain = patch + halo; corrections land on the patch
            # (restricted ASM). Overlap 0: the halo IS the overlap, one
            # coarse-cell layer through the transfer graph, so no algebraic
            # extension at PCSetUp.
            if _asm_type == "basic":
                spc.setASMType(PETSc.PC.ASMType.BASIC)
                spc.setASMLocalSubdomains(1, [is_sub])
            else:
                spc.setASMLocalSubdomains(1, [is_sub], [is_own])
            # Extra operator-sparsity overlap layers on top of the transfer
            # halo (PCASM extends via MatIncreaseOverlap at PCSetUp).
            spc.setASMOverlap(int(os.environ.get("UW_FAC_OVERLAP", "0")))
            # Subdomain solver: SOR, the patch-restricted twin of the
            # whole-level smoother — no factorization, so no pivot to hit.
            # PCASM's default sub-solve (ILU-0) takes a NUMERIC_ZEROPIVOT on
            # the rotated Galerkin patch block (measured: min |diag| 8e-5 near
            # the constraint; PC_FAILED -11 before the first iteration).
            opts[prefix + f"mg_levels_{l}_sub_pc_type"] = _sub_pc or "sor"
            fac_keys.append(f"mg_levels_{l}_sub_pc_type")
            if (_sub_pc or "sor") in ("lu", "ilu", "cholesky"):
                # The rotated Galerkin patch block carries near-zero pivots
                # (constraint-zeroed transfer rows leave weakly-attached
                # coarse DOFs, min diag ~1e-5); an unshifted factorization
                # takes NUMERIC_ZEROPIVOT even as exact LU.
                key = f"mg_levels_{l}_sub_pc_factor_shift_type"
                opts[prefix + key] = "nonzero"
                fac_keys.append(key)
            is_sub.destroy()             # the PC holds its own references
            is_own.destroy()
    return fac_keys


def _install_transfers(solver, Ps, verbose=False, patch_rows=None):
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
        _configure_pcmg(ksp.getPC(), Ps,
                        smoother=solver._mg_smoother_variant,
                        owned=solver._managed_pc_options, ksp=ksp,
                        patch_rows=patch_rows)
        if verbose:
            from underworld3 import mpi
            mpi.pprint(f"[{solver.name}] custom FMG installed: {nlev} levels, "
                       f"P sizes {[tuple(P.getSize()) for P in Ps]}")
        return

    if pfx != "fieldsplit_velocity_":
        raise NotImplementedError(
            f"custom_mg install: unsupported PC prefix '{pfx}'.")

    _install_velocity_block_transfers(solver, Ps, verbose=verbose,
                                      patch_rows=patch_rows)


def _install_velocity_block_transfers(solver, Ps, verbose=False,
                                      patch_rows=None):
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
        solver._record_pc_fallback(
            "custom_mg.velocity_block_assembly",
            requested="direct Jacobian assembly (computeFunction/computeJacobian)",
            installed="throwaway max_it=0 assembly route; same PC installed",
            reason="build_failed",
            detail="snes.computeJacobian raised; the operator is assembled by "
                   "a zero-iteration solve instead")
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
    _configure_pcmg(vel_pc, Ps, smoother=solver._mg_smoother_variant,
                    owned=solver._managed_pc_options, patch_rows=patch_rows)
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
            # the finest map then reads the DM's current global section. The
            # skip is recorded so "the section was finalized" is checkable.
            solver._record_pc_fallback(
                "custom_mg.presolve_setup",
                requested="pre-build snes.setUp() (finalize the DM section)",
                installed="deferred to install-time setUp",
                reason="check_skipped",
                detail="setUp raised on the not-yet-fully-configured SNES")

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
        # FAC patch smoothing (#629): one entry per PCMG level; level ``l``'s
        # patch is read off its own transfer ``Ps[l-1]``. Serial-only for now,
        # like the rest of the custom-P specifics; parallel leaves every entry
        # None, which _configure_pcmg reads as whole-level smoothing.
        self.level_patch_rows = [None] * nlev
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
                # A native refine() pair gets the EXACT nested embedding at the
                # field's own degree (#629 item 1): the point-located builders
                # scatter across the non-nested >P1 node clouds and fatten the
                # Galerkin product 3-5x. Declines (None) fall through to the
                # recorded/geometric routes unchanged.
                Pn = None
                # TODO(MEASURE): the env flag is an A/B affordance for the #629
                # benchmark campaign (nested-native vs geometric on the base
                # ladder); remove once the comparison is settled.
                if (not os.environ.get("UW_CUSTOM_MG_DISABLE_NESTED_NATIVE")
                        and _is_native_refine_pair(self.level_meshes[l - 1],
                                                   self.level_meshes[l])):
                    Pn = nested_refine_pair_prolongation(
                        self.level_meshes[l - 1], self.level_meshes[l],
                        degree, continuous,
                        coarse_coords=coords[l - 1], fine_coords=coords[l])
                if Pn is None:
                    Pn = self._recorded_node_transfer(
                        l, nlev, degree, coords[l - 1].shape[0],
                        coords[l].shape[0])
                if Pn is not None:
                    Pr = _reduced_from_node_transfer(Pn, maps[l - 1], maps[l], nc)
                else:
                    Pr = _reduced_transfer(coords[l - 1], coords[l], maps[l - 1],
                                           maps[l], nc, self.builder)
                Pr, n_rep = _repair_zero_columns_serial(
                    Pr, coords[l - 1], coords[l], maps[l - 1], maps[l],
                    nc, l)
                if n_rep:
                    import warnings
                    warnings.warn(
                        f"custom_mg: transfer {l - 1}->{l} had {n_rep} coarse "
                        f"DOF(s) with no fine image (non-nested levels); "
                        f"repaired by nearest-fine-DOF injection. Costs "
                        f"iterations at worst, never correctness.")
                _assert_no_zero_columns_serial(Pr, l)
                # TODO(MEASURE): A/B affordance for the #629 campaign, like
                # the nested-native flag above; remove when settled.
                if not os.environ.get("UW_CUSTOM_MG_DISABLE_FAC"):
                    self.level_patch_rows[l] = _fac_patch_split(
                        Pr, coords[l - 1], coords[l], maps[l - 1], maps[l], nc)
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
            # Sanctioned: no readable operator to check against — record the
            # skipped guard rather than silently waiving it (#484).
            solver._record_pc_fallback(
                "custom_mg.finest_operator_check",
                requested="finest reduced-map vs operator span check",
                installed="unchecked",
                reason="check_skipped",
                detail="could not read the assembled operator")
            return
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
        _install_transfers(solver, self.transfers, verbose=verbose,
                           patch_rows=getattr(self, "level_patch_rows", None))
        # Record what is live on this solver's PC, so a repeat solve can
        # skip the re-install (see _pcmg_still_installed).
        solver._custom_mg_live = {"h": id(self),
                                  "nlev": len(self.transfers) + 1}


class _DMLevelView:
    """A mesh-shaped view over one DM of a native ``refine()`` hierarchy.

    ``CustomMGHierarchy`` consumes *meshes*, but the requested-native source in
    :func:`build_transfers` (#478) has only the raw ``mesh.dm_hierarchy`` DMs.
    This adapter provides exactly what a coarse level is asked for: ``.dm``
    (used by ``_clone_dm_with_solver_discretisation`` — the hierarchy DMs carry
    the boundary labels ``refine()`` propagates, which is all copyDS needs)
    plus ``_get_coords_for_basis``, delegated UNBOUND to
    ``discretisation.Mesh`` so there is one implementation. That method reads
    only ``self.dm`` and the four scalars copied here (``dim``, ``cdim``,
    ``isSimplex``, ``qdegree`` — shared by every level of a uniform
    refinement), which is what makes the unbound call safe. No full ``Mesh``
    is built: a coarse MG level needs no variables, caches, or registries.
    """

    def __init__(self, dm, fine_mesh):
        # A CLONE, never the hierarchy DM itself: the base (gmsh-imported)
        # level may need its coordinate field repaired below, and the shared
        # hierarchy DMs also feed the native Stokes FMG route.
        self.dm = dm.clone()
        self.dim = fine_mesh.dim
        self.cdim = fine_mesh.cdim
        self.isSimplex = fine_mesh.isSimplex
        self.qdegree = fine_mesh.qdegree

        # The gmsh-imported BASE level carries section-only coordinates (its
        # coordinate field is a PetscContainer, no PetscFE), and
        # DMCreateInterpolation from such a source silently returns a ZERO
        # matrix (measured) — _get_coords_for_basis would then hand every
        # level-0 node the coordinate (0,0) and the transfer build collapses.
        # refine() gives the child levels an FE coordinate space; give the
        # clone's base the same: P1 Lagrange on the identical vertex layout.
        cdm = self.dm.getCoordinateDM()
        field = cdm.getField(0)
        fobj = field[0] if isinstance(field, tuple) else field
        if not isinstance(fobj, PETSc.FE):
            fe = PETSc.FE().createLagrange(self.dim, self.cdim, self.isSimplex,
                                           1, self.qdegree, comm=PETSc.COMM_SELF)
            cdm.setField(0, fe)
            cdm.createDS()

    def _get_coords_for_basis(self, degree, continuous):
        from underworld3.discretisation import Mesh
        return Mesh._get_coords_for_basis(self, degree, continuous)

    def _basis_coordinate_dm(self, degree, continuous):
        from underworld3.discretisation import Mesh
        return Mesh._basis_coordinate_dm(self, degree, continuous)

    def _cell_node_indices(self, degree, continuous):
        # Same unbound-delegation pattern as _get_coords_for_basis; the cache
        # dict a full Mesh initialises in _setup_ds is created on demand here.
        from underworld3.discretisation import Mesh
        if not hasattr(self, "_cell_node_array"):
            self._cell_node_array = {}
        return Mesh._cell_node_indices(self, degree, continuous)


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


def build_transfers(solver, field_id=None):
    """The custom-P prolongations this solver should drive, built and ready to
    install — from a solver-set hierarchy (``set_custom_fmg``), a **mesh-owned**
    one (a ``mesh.adapt`` refinement child), or a **requested-native** one
    (explicit ``preconditioner="fmg"`` on a single-field solver, #478).

    This is the shared resolution rule for every route that can drive custom-P
    multigrid: the standard solve path via :func:`auto_inject_custom_mg`, and the
    rotated free-slip path, which builds its own IS-based fieldsplit inside
    ``utilities.rotated_bc`` and so never reaches the standard injection hook
    (#467). Both must answer "which hierarchy does this solver get?" the same way.

    Resolution order: **solver-set > mesh-owned > requested-native.** The first
    is a DEMAND (build errors raise — the user registered it explicitly); the
    other two are PREFERENCES, built through the same opportunistic arm
    (barycentric, RBF retry, degrade to the solver's default preconditioner —
    every step recorded in ``solver.pc_fallbacks``).

    A refinement child carries ``mesh._custom_mg_coarse_meshes`` (the static
    coarse tail), so a :class:`CustomMGHierarchy` ``[*coarse, solver.mesh]``
    targeting ``field_id`` (0 for the Stokes velocity block, None for
    scalar/vector) is built lazily on first solve — every solver on an adapted
    mesh drives geometric MG with no per-solver call. The requested-native
    source instead wraps the mesh's own ``dm_hierarchy`` tail in
    :class:`_DMLevelView` adapters — the same coarse levels native FMG would
    use, driven through injection-free custom-P transfers.

    Parameters
    ----------
    solver : SolverBaseClass
        Built solver (``_build`` has run) whose hierarchy is wanted.
    field_id : int or None
        Field index for multi-field solvers (0 = Stokes velocity), None = single
        field.

    Returns
    -------
    (CustomMGHierarchy, list of Mat) or (None, None)
        ``(None, None)`` means "no hierarchy here" — either none is registered or
        an OPPORTUNISTIC mesh-owned build failed and the caller should keep its
        default preconditioner. A build failure on a solver-set hierarchy raises
        instead: the user asked for it explicitly.
    """
    cfg = getattr(solver, "_custom_mg", None)
    if cfg is not None:
        if not (isinstance(cfg, dict) and cfg.get("mode") == "hierarchy"):
            raise NotImplementedError(
                "the legacy set_custom_mg registration has no hierarchy to "
                "resolve; use set_custom_fmg().")
        h = cfg["hierarchy"]
        return h, h.build(solver)           # explicit request: errors surface

    # Mesh-owned hierarchy (adapt() child): OPPORTUNISTIC auto-pickup. It must never
    # crash a solve, so build the transfers (which validate the finest reduced
    # map against the assembled operator — see CustomMGHierarchy.build) inside a
    # try/except and fall back to the solver's default preconditioner on any failure.
    # The finest map is derived from the DM section AFTER snes.setUp() finalizes it,
    # so it is faithful to the operator on adapt children too — including scalar
    # semi-Lagrangian advection-diffusion (which earlier had to be skipped).
    coarse = getattr(solver.mesh, "_custom_mg_coarse_meshes", None)
    if coarse is not None:
        # An EXPLICIT preconditioner choice beats the opportunistic pickup. Before
        # this guard, `solver.preconditioner = "gamg"` on an adapt child was
        # silently clobbered back to the custom-P PCMG at solve time (measured:
        # both arms of test_0842's fmg-vs-gamg comparison ran pc_type=mg), so a
        # user could not opt out and any FMG-vs-GAMG comparison was vacuous.
        # `_pc_user_override` is the same statement in the other spelling: the
        # solver's option manager has latched "the user owns this block's pc_type"
        # (they wrote a pc_type of their own into petsc_options), and an
        # opportunistic pickup must stand down for exactly the same reason.
        # "auto" (the default) still picks up the mesh-owned hierarchy.
        # NOTE the arity: this function returns a 2-tuple, never bare None — a bare
        # `return` here is what turned the gate into a TypeError at the call site
        # when this hunk migrated from auto_inject_custom_mg (which returns nothing)
        # during the #488 x #471 merge.
        if (getattr(solver, "_preconditioner", "auto") == "gamg"
                or getattr(solver, "_pc_user_override", False)):
            return None, None
        level_tail = list(coarse)
        builder = getattr(solver.mesh, "_custom_mg_builder", "barycentric")
    elif getattr(solver, "_pc_single_field_geo_requested", False):
        # Requested-native source (#478): an explicit `preconditioner="fmg"`
        # on a single-field solver. The gate in _apply_preconditioner_options
        # set the flag only when the mesh reported a hierarchy, but re-check
        # here — a remesh between build and solve can collapse it, and this
        # arm must degrade readably, never crash a solve.
        hierarchy_dms = list(getattr(solver.mesh, "dm_hierarchy", []) or [])
        if len(hierarchy_dms) < 2:
            solver._record_pc_fallback(
                "custom_mg.requested_native",
                requested="custom-P geometric MG over mesh.dm_hierarchy",
                installed="default preconditioner",
                reason="unavailable",
                detail="the refinement hierarchy is gone (collapsed by a "
                       "remesh between build and solve)")
            return None, None
        level_tail = [_DMLevelView(dm, solver.mesh) for dm in hierarchy_dms[:-1]]
        # Stamp the refine-family slots so consecutive pairs (including the
        # finest pair, whose fine level is the solver's own mesh) take the
        # exact nested transfer. Reuse the mesh's token if _coarse_level_meshes
        # already stamped one — the level indices coincide by construction.
        _slot = getattr(solver.mesh, "_refine_slot", None)
        _token = _slot[0] if _slot is not None else object()
        for _k, _v in enumerate(level_tail):
            _v._refine_slot = (_token, _k)
        if _slot is None:
            solver.mesh._refine_slot = (_token, len(hierarchy_dms) - 1)
        builder = "barycentric"
    else:
        return None, None                   # nothing to inject
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
        h = CustomMGHierarchy(level_tail + [solver.mesh], builder=_b,
                              field_id=field_id)
        try:
            Ps = h.build(solver)
            break
        except Exception as exc:            # pragma: no cover - defensive
            import warnings
            if _i + 1 < len(_attempts):
                solver._record_pc_fallback(
                    "custom_mg.transfer_builder",
                    requested=_b,
                    installed=f"{_attempts[_i + 1]} (DENSE transfer)",
                    reason="build_failed",
                    detail=f"{exc}; the RBF rescue is a performance cliff — "
                           f"its transfer is dense (nnz/row == n_coarse), see #424")
                warnings.warn(
                    f"custom_mg: {_b} transfer build failed ({exc}); "
                    f"retrying with the '{_attempts[_i + 1]}' builder, which "
                    f"has global support and cannot leave a coarse DOF "
                    f"without a fine image. NOTE the RBF transfer is DENSE "
                    f"(nnz/row == n_coarse), so the Galerkin coarse operators "
                    f"are dense too — this rescues correctness but does not "
                    f"scale. If it fires on a production-sized problem, treat "
                    f"it as a performance cliff and fix the cause, not the "
                    f"symptom (#424).")
                continue
            solver._record_pc_fallback(
                "custom_mg.build",
                requested=f"custom-P geometric MG ({' -> '.join(_attempts)})",
                installed="default preconditioner",
                reason="build_failed",
                detail=str(exc))
            warnings.warn(
                f"custom_mg: opportunistic custom-P FMG build failed ({exc}); "
                "using the solver's default preconditioner.")
            return None, None

    return h, Ps


def auto_inject_custom_mg(solver, field_id=None):
    """Solve-hook entry on the STANDARD path: resolve this solver's custom-P
    hierarchy (:func:`build_transfers`) and install it on the managed PC.

    The rotated free-slip path does not come through here — it builds its own
    IS-based fieldsplit and calls :func:`build_transfers` directly.
    """
    # Solver-set hierarchy (set_custom_fmg, or the deprecated set_custom_mg):
    # the user asked for it explicitly — build + install directly and let any
    # error surface. inject_custom_mg is the one place that still understands the
    # legacy registration.
    if solver._custom_mg is not None:
        inject_custom_mg(solver)
        return

    # build_transfers' contract is a 2-tuple, but a "no hierarchy" answer has
    # been written as a bare `return` before (the #488 x #471 merge shipped
    # exactly that inside the explicit-gamg gate): a None here must mean
    # "nothing to inject", never a TypeError mid-solve.
    resolved = build_transfers(solver, field_id=field_id)
    h, Ps = resolved if resolved is not None else (None, None)
    if h is None or Ps is None:
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
                solver._record_pc_fallback(
                    "custom_mg.dimensional_guard",
                    requested="custom-P geometric MG hierarchy",
                    installed="default preconditioner",
                    reason="unavailable",
                    detail=f"finest transfer {pr}x{pc} is incompatible with the "
                           f"operator (size {op_n}); set_custom_fmg() an explicit "
                           f"hierarchy to override")
                warnings.warn(
                    "custom_mg: mesh-owned adapt-mesh FMG transfer is incompatible "
                    f"with this solver's operator (transfer {pr}x{pc}, operator {op_n}); "
                    "skipping the auto-pickup (using the default preconditioner). "
                    "set_custom_fmg() an explicit hierarchy to override.")
                return
        except Exception:
            # Sanctioned: an unreadable operator must not block working cases —
            # but the skipped guard is on the record.
            solver._record_pc_fallback(
                "custom_mg.dimensional_guard",
                requested="finest-transfer vs operator size check",
                installed="unchecked (hierarchy installed anyway)",
                reason="check_skipped",
                detail="could not read the assembled operator to check the "
                       "finest transfer against it")

    h.install(solver, verbose=False)
    # auto_cached marks this as a RESOLUTION product (auto/fmg install), not a
    # user registration: the preconditioner setter drops it so a later explicit
    # choice re-resolves instead of re-injecting this hierarchy unconditionally.
    solver._custom_mg = {"mode": "hierarchy", "hierarchy": h, "verbose": False,
                         "auto_cached": True}


def _pcmg_still_installed(solver, h):
    """Is THIS hierarchy still live on the solver's managed PC block?

    The marker written by :meth:`CustomMGHierarchy.install` says an install
    happened; it cannot say the PC still carries it — a rebuilt SNES, an
    explicit ``preconditioner=`` change, or anything that reset the PC leaves
    the marker stale. So the marker is only the cheap first test, and the
    verdict comes from the LIVE object: the managed block must exist, be a
    PCMG, and have the hierarchy's level count. Any doubt (unreachable PC,
    un-set-up fieldsplit) answers False — the cost of a wrong False is one
    redundant install, the cost of a wrong True is a solve on a stale PC.
    """
    mark = getattr(solver, "_custom_mg_live", None)
    if not mark or mark.get("h") != id(h):
        return False
    try:
        pfx = solver._pc_option_prefix or ""
        if pfx == "":
            pc = solver.snes.getKSP().getPC()
        elif pfx == "fieldsplit_velocity_":
            outer = solver.snes.getKSP().getPC()
            if outer.getType() != "fieldsplit":
                return False
            pc = outer.getFieldSplitSubKSP()[0].getPC()
        else:
            return False
        return (pc.getType() == "mg"
                and pc.getMGLevels() == mark.get("nlev"))
    except PETSc.Error:
        return False


def inject_custom_mg(solver):
    """Build + install the custom-P FMG. Called from ``solve()`` (after ``_build``,
    before the SNES solve) when ``solver._custom_mg`` is set. Dispatches:
    - ``mode == "hierarchy"`` -> BC-per-level reduced path (correct, general);
    - legacy dict ``{coarse_meshes, kind}`` -> finest-only reduction (kept for
      back-compat; valid only when coarse levels are non-nested / unconstrained)."""
    cfg = solver._custom_mg

    if isinstance(cfg, dict) and cfg.get("mode") == "hierarchy":
        h = cfg["hierarchy"]
        if _pcmg_still_installed(solver, h):
            # The hierarchy is already live on this solver's PC. PETSc
            # re-Galerkins the coarse operators when the fine operator's
            # values change at the next PCSetUp, so a repeat solve needs
            # NO re-install — and the install is expensive: measured
            # (#622) at 55 s of a 61 s warm Stokes solve on an 85k-cell
            # cut child, 49.5 s of it a DUPLICATE Jacobian assembly done
            # only to make the fieldsplit reachable, 5.8 s rebuilding
            # transfers that depend only on the meshes.
            return
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
