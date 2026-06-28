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

import numpy as np
from petsc4py import PETSc

__all__ = ["barycentric_prolongation", "rbf_prolongation", "inject_custom_mg",
           "CustomMGHierarchy", "set_custom_fmg", "sbr_refine", "sbr_refine_where"]


# --------------------------------------------------------------------------- #
#  Prolongation builders (return scipy CSR matrices)
# --------------------------------------------------------------------------- #
def barycentric_prolongation(coarse_coords, fine_coords):
    """FE-exact prolongation: each fine DOF interpolated by the coarse element
    that contains it (barycentric weights). Partition of unity (row sums = 1);
    fine points outside the coarse hull fall back to the nearest coarse DOF."""
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
            rows.append(i); cols.append(int(tree.query(fine_coords[i])[1])); vals.append(1.0)
            continue
        verts = tri.simplices[s]
        T = tri.transform[s]
        bary = T[:dim].dot(fine_coords[i] - T[dim])
        w = np.append(bary, 1.0 - bary.sum())
        for k in range(dim + 1):
            rows.append(i); cols.append(int(verts[k])); vals.append(float(w[k]))
    return sp.csr_matrix((vals, (rows, cols)),
                         shape=(fine_coords.shape[0], coarse_coords.shape[0]))


def rbf_prolongation(coarse_coords, fine_coords, smooth=0.0):
    """RBF prolongation: polyharmonic (r² log r) kernel + affine polynomial tail
    (reproduces linear fields), Shepard row-normalised to a partition of unity.
    Works for arbitrary (non-nested) point sets; software-equivalent to the
    barycentric builder as an MG transfer operator."""
    import scipy.sparse as sp
    from scipy.spatial.distance import cdist

    def phi(r):
        r = np.where(r == 0.0, 1e-30, r)
        return r ** 2 * np.log(r)

    nc, dim = coarse_coords.shape
    Pc = np.hstack([np.ones((nc, 1)), coarse_coords])          # affine tail
    Acc = phi(cdist(coarse_coords, coarse_coords)) + smooth * np.eye(nc)
    M = np.block([[Acc, Pc], [Pc.T, np.zeros((dim + 1, dim + 1))]])
    Minv = np.linalg.inv(M)
    B = np.hstack([phi(cdist(fine_coords, coarse_coords)),
                   np.ones((fine_coords.shape[0], 1)), fine_coords])
    Praw = (B @ Minv)[:, :nc]
    rs = Praw.sum(axis=1, keepdims=True)
    rs[np.abs(rs) < 1e-12] = 1.0
    return sp.csr_matrix(Praw / rs)


_BUILDERS = {"barycentric": barycentric_prolongation, "rbf": rbf_prolongation}


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
#  Coordinate helpers
# --------------------------------------------------------------------------- #
def _to_petsc_aij(csr):
    csr = csr.tocsr()
    M = PETSc.Mat().createAIJ(
        size=csr.shape,
        csr=(csr.indptr.astype("int32"), csr.indices.astype("int32"), csr.data),
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
        dm.restoreLocalVec(lvec); dm.restoreGlobalVec(gvec)
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
    except Exception:
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


def _reduced_transfer(coarse_coords, fine_coords, r2f_c, r2f_f, ncomp, builder):
    """Build one prolongation reduced(coarse) -> reduced(fine):
    node-level scalar P -> interleave ``ncomp`` components -> drop BC rows/cols."""
    import scipy.sparse as sp
    Pn = builder(coarse_coords, fine_coords)               # (n_f_nodes, n_c_nodes)
    Pv = sp.kron(Pn, sp.eye(ncomp), format="csr")          # interleaved full vector
    Pr = Pv.tocsr()[r2f_f, :][:, r2f_c]                    # reduced -> reduced
    return Pr


def _install_transfers(solver, Ps, verbose=False):
    """Configure the managed PCMG block to use the supplied prolongations.
    Build-time injection (before first PCSetUp): set DMActive(OPERATOR,False) so
    PETSc does not re-derive interpolation from the DM, Galerkin RAP for coarse
    operators. Scalar / single-field-vector (top-level PC: ``_pc_option_prefix``
    is ``""``); the Stokes velocity-block path is Phase 2."""
    nlev = len(Ps) + 1
    opts = solver.petsc_options
    pfx = solver._pc_option_prefix or ""
    if pfx not in ("",):
        raise NotImplementedError(
            "Layer-1 install supports top-level PC (scalar / single-field vector). "
            f"PC prefix '{pfx}' (e.g. Stokes velocity block) is Phase 2.")
    opts[pfx + "pc_type"] = "mg"
    opts[pfx + "pc_mg_type"] = "full"
    opts[pfx + "pc_mg_galerkin"] = "both"
    opts[pfx + "mg_levels_ksp_type"] = "richardson"
    opts[pfx + "mg_levels_pc_type"] = "sor"
    opts[pfx + "mg_coarse_pc_type"] = "redundant"
    opts[pfx + "mg_coarse_redundant_pc_type"] = "lu"
    for key in ("pc_gamg_type", "pc_gamg_repartition", "pc_gamg_agg_nsmooths"):
        opts.delValue(pfx + key)

    solver.snes.setUp()
    ksp = solver.snes.getKSP()
    ksp.setDMActive(PETSc.KSP.DMActive.OPERATOR, False)
    pc = ksp.getPC()
    pc.setType("mg")
    pc.setMGLevels(nlev)
    pc.setMGType(PETSc.PC.MGType.FULL)
    for l in range(1, nlev):
        pc.setMGInterpolation(l, Ps[l - 1])
    pc.setFromOptions()
    if verbose:
        from underworld3 import mpi
        mpi.pprint(f"[{solver.name}] custom FMG installed: {nlev} levels, "
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
    """

    def __init__(self, level_meshes, builder="barycentric", field_id=None):
        if builder not in _BUILDERS:
            raise ValueError("builder must be 'barycentric' or 'rbf'")
        if len(level_meshes) < 2:
            raise ValueError("need at least 2 levels (>=1 coarse + finest)")
        self.level_meshes = list(level_meshes)
        self.builder = _BUILDERS[builder]
        self.builder_name = builder
        self.field_id = field_id
        self.transfers = None

    def build(self, solver, level_solver_factory):
        """Build the BC-reduced prolongations. ``solver`` is the (built) finest
        solver; ``level_solver_factory(mesh) -> built solver`` provides a
        same-discretisation solver on each COARSE level so its constrained
        section gives that level's reduced map. The finest level uses ``solver``."""
        var = solver.Unknowns.u
        degree = var.degree
        continuous = getattr(var, "continuous", True)
        nlev = len(self.level_meshes)

        coords, r2f, ncomp = [], [], []
        for k, mesh in enumerate(self.level_meshes):
            c = np.asarray(mesh._get_coords_for_basis(degree, continuous))
            if k == nlev - 1:
                rmap, nfull = _reduced_map(solver.dm, self.field_id)
            else:
                tmp = level_solver_factory(mesh)
                tmp._build(False, False, None)
                tmp.snes.setUp()
                rmap, nfull = _reduced_map(tmp.dm, self.field_id)
            nc = nfull // c.shape[0]
            if nfull % c.shape[0] != 0:
                raise RuntimeError(
                    f"level {k}: full DOFs {nfull} not divisible by nodes {c.shape[0]}")
            coords.append(c); r2f.append(rmap); ncomp.append(nc)

        if len(set(ncomp)) != 1:
            raise RuntimeError(f"inconsistent component counts across levels: {ncomp}")
        nc = ncomp[0]

        Ps = []
        for l in range(1, nlev):
            Pr = _reduced_transfer(coords[l - 1], coords[l], r2f[l - 1], r2f[l],
                                   nc, self.builder)
            zc = int((np.asarray((Pr != 0).sum(axis=0)).ravel() == 0).sum())
            if zc:
                raise RuntimeError(
                    f"transfer {l-1}->{l} has {zc} zero columns (coarse DOFs with no "
                    f"fine image) — BC-per-level reduction failed; coarse operator "
                    f"would be singular.")
            Ps.append(_to_petsc_aij(Pr))
        self.transfers = Ps
        return Ps

    def install(self, solver, verbose=False):
        if self.transfers is None:
            raise RuntimeError("call build() before install()")
        _install_transfers(solver, self.transfers, verbose=verbose)


# --------------------------------------------------------------------------- #
#  Entry points
# --------------------------------------------------------------------------- #
def set_custom_fmg(solver, coarse_meshes, *, level_solver_factory,
                   builder="barycentric", field_id=None, verbose=False):
    """Generalized custom-P FMG with BC-per-level reduction (the correct path).

    Registers a :class:`CustomMGHierarchy` on the solver so that the next
    ``solve()`` builds and installs it (build-time injection). The hierarchy is
    ``[*coarse_meshes, solver.mesh]``; ``level_solver_factory(mesh)`` must return
    a same-discretisation solver on a coarse level (used only to read its
    BC-constrained section)."""
    solver._custom_mg = {
        "mode": "hierarchy",
        "hierarchy": CustomMGHierarchy(list(coarse_meshes) + [solver.mesh],
                                       builder=builder, field_id=field_id),
        "factory": level_solver_factory,
        "verbose": verbose,
    }
    solver.is_setup = False


def inject_custom_mg(solver):
    """Build + install the custom-P FMG. Called from ``solve()`` (after ``_build``,
    before the SNES solve) when ``solver._custom_mg`` is set. Dispatches:
    - ``mode == "hierarchy"`` -> BC-per-level reduced path (correct, general);
    - legacy dict ``{coarse_meshes, kind}`` -> finest-only reduction (kept for
      back-compat; valid only when coarse levels are non-nested / unconstrained)."""
    cfg = solver._custom_mg

    if isinstance(cfg, dict) and cfg.get("mode") == "hierarchy":
        h = cfg["hierarchy"]
        h.build(solver, cfg["factory"])
        h.install(solver, verbose=cfg.get("verbose", False))
        return

    # ---- legacy finest-only path (back-compat) ------------------------------
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
