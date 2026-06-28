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

__all__ = ["barycentric_prolongation", "rbf_prolongation", "inject_custom_mg"]


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
#  Injection
# --------------------------------------------------------------------------- #
def inject_custom_mg(solver):
    """Configure ``solver``'s PCMG to use our custom prolongation hierarchy.

    Called from the solver's ``solve()`` (after ``_build``, before the SNES
    solve) when ``solver._custom_mg`` is set via ``set_custom_mg``. Scalar /
    single-field solvers only for now (the Stokes velocity-block path is a
    separate increment).
    """
    cfg = solver._custom_mg
    coarse_meshes = cfg["coarse_meshes"]
    kind = cfg["kind"]
    builder = _BUILDERS[kind]

    var = solver.Unknowns.u
    degree = var.degree
    continuous = getattr(var, "continuous", True)

    if solver._pc_option_prefix not in ("", None):
        raise NotImplementedError(
            "custom_mg currently supports single-field (scalar/vector) solvers; "
            f"solver uses PC prefix '{solver._pc_option_prefix}' (e.g. Stokes "
            "velocity block) which is a separate increment.")

    # --- per-level DOF coordinates --------------------------------------- #
    fine = _reduce_to_global(solver.dm,
                             solver.mesh._get_coords_for_basis(degree, continuous))
    levels = [m._get_coords_for_basis(degree, continuous) for m in coarse_meshes]
    levels.append(fine)
    nlev = len(levels)
    if nlev < 2:
        raise ValueError("custom_mg needs at least one coarse mesh")

    # --- build prolongations P[l]: level l-1 -> level l ------------------ #
    Ps = [None] + [_to_petsc_aij(builder(levels[l - 1], levels[l]))
                   for l in range(1, nlev)]

    # --- configure the geometric MG bundle on the managed block ---------- #
    opts = solver.petsc_options
    pfx = solver._pc_option_prefix or ""
    opts[pfx + "pc_type"] = "mg"
    opts[pfx + "pc_mg_type"] = "full"
    opts[pfx + "pc_mg_galerkin"] = "both"          # coarse operators = PᵀAP
    opts[pfx + "mg_levels_ksp_type"] = "richardson"
    opts[pfx + "mg_levels_pc_type"] = "sor"
    opts[pfx + "mg_coarse_pc_type"] = "redundant"
    opts[pfx + "mg_coarse_redundant_pc_type"] = "lu"
    for key in ("pc_gamg_type", "pc_gamg_repartition", "pc_gamg_agg_nsmooths"):
        opts.delValue(pfx + key)

    # --- inject before the first PCSetUp --------------------------------- #
    solver.snes.setUp()
    ksp = solver.snes.getKSP()
    # Stop PETSc re-deriving interpolation/operators from the DM hierarchy.
    ksp.setDMActive(PETSc.KSP.DMActive.OPERATOR, False)
    pc = ksp.getPC()
    pc.setType("mg")
    pc.setMGLevels(nlev)
    pc.setMGType(PETSc.PC.MGType.FULL)
    for l in range(1, nlev):
        pc.setMGInterpolation(l, Ps[l])
    pc.setFromOptions()

    if cfg.get("verbose"):
        from underworld3 import mpi
        mpi.pprint(f"[{solver.name}] custom_mg ({kind}): levels "
                   f"{[lv.shape[0] for lv in levels]}")
