r"""Geometry-aware multigrid interpolation for mover-adapted meshes.

When the finest mesh level is relocated by a node mover (anisotropic metric
adaptation, free-surface ALE, ...), PETSc's geometric-multigrid level transfers
become coordinate-blind. The nested transfer is built once from the refinement
topology and assumes each fine node still sits at its *refinement* position
(an edge bisects its coarse edge, etc.). After anisotropic adaptation a fine
node can sit much closer to one end of its coarse element than the bisection
assumes — the true interpolation weights swing heavily toward the near vertex.
Measured on a fault-adapted annulus, the true barycentric interpolant differs
from PETSc's nested transfer by ~100% in action, and the multigrid iteration
count climbs as the field (and the operator) sharpen.

This module rebuilds the **finest-level** prolongation as the *true barycentric*
geometric interpolant on every solver setup: each moved fine node is located in
the fixed coarse element it actually occupies and the coarse P2 basis is
evaluated there. Only the finest pair needs this — a node mover deforms only
``mesh.dm``; the coarser levels keep their uniform-refinement positions where
PETSc's transfer is already correct.

Injection without the Galerkin stale-product trap
-------------------------------------------------
The geometric transfer has *fresh sparsity* (a node's true coarse cell is
generally not the cell frozen into the nested matrix). Replacing the
interpolation Mat object trips PETSc's cached Galerkin ``PtAP`` (operator and
transfer roles swap → dimension error). So we turn **Galerkin off** on the
multigrid sub-PC and supply the coarse operators explicitly, computed by
``A_{L-1} = I_L^{T} A_L I_L`` (``MatPtAP``) from the finest operator down,
using the geometric transfer at the finest pair and PETSc's nested transfers
below. No global option or ``mesh.dm`` mutation is touched, so the mesh's own
point-location (SLCN advection, boundary integrals) is unaffected, and the
override is rebuilt each setup so it survives the per-adapt SNES/PC teardown.

Usage
-----
.. code-block:: python

    from underworld3.utilities.gmg_geometric_interpolation import (
        geometric_mg_interpolation,
    )

    stokes._pre_solve_hook = geometric_mg_interpolation()

The default locates the multigrid PC automatically (the velocity fieldsplit
sub-PC of a saddle-point solve, else the main PC when it is type ``mg``). It is
a no-op unless that PC is multigrid.

.. note::
   Currently validated for **serial** runs. In parallel the distributed DOF
   orderings require an explicit coordinate scatter that is not yet
   implemented; the hook detects ``comm size > 1`` and leaves PETSc's nested
   transfer in place.
"""

import numpy as np

import underworld3 as uw

__all__ = ["geometric_mg_interpolation", "GeometricMGInterpolator"]


# ----------------------------------------------------------------------------
# Coarse P2 cell structure (read once; the coarse level never moves).
# ----------------------------------------------------------------------------


def coarse_cell_structure(dm, dim=2):
    """Per coarse cell P2 structure for geometric location.

    Returns a dict: ``V`` (ncell,3 vertex node ids), ``Vxy`` (their coords),
    ``E`` (3 edge-midpoint node ids), ``Evl`` (local vertex pair each edge
    joins), centroid KD-tree helpers ``e1``/``e2``/``det`` (affine inverse) and
    ``cent``. Node id = section offset // dim, matching the coarse interpolation
    column ordering.
    """
    sec = dm.getLocalSection()
    vc = dm.getCoordinatesLocal().array.reshape(-1, dim)
    cdm = dm.getCoordinateDM()
    csec = cdm.getLocalSection()
    vcoord = lambda vtx: vc[csec.getOffset(vtx) // dim]
    cS, cE = dm.getHeightStratum(0)
    ncell = cE - cS
    V = np.zeros((ncell, 3), np.int64)
    Vxy = np.zeros((ncell, 3, dim))
    E = np.zeros((ncell, 3), np.int64)
    Evl = np.zeros((ncell, 3, 2), np.int64)
    for ci, c in enumerate(range(cS, cE)):
        edges = dm.getCone(c)
        verts = list(dict.fromkeys(np.concatenate([dm.getCone(e) for e in edges])))
        loc = {vtx: k for k, vtx in enumerate(verts)}
        for k, vtx in enumerate(verts):
            V[ci, k] = sec.getOffset(vtx) // dim
            Vxy[ci, k] = vcoord(vtx)
        for k, e in enumerate(edges):
            E[ci, k] = sec.getOffset(e) // dim
            a, b = dm.getCone(e)
            Evl[ci, k] = (loc[a], loc[b])
    ncoarse = sec.getStorageSize() // dim
    e1 = Vxy[:, 1] - Vxy[:, 0]
    e2 = Vxy[:, 2] - Vxy[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    cent = Vxy.mean(axis=1)
    return dict(V=V, Vxy=Vxy, E=E, Evl=Evl, e1=e1, e2=e2, det=det,
                cent=cent, ncell=ncell, ncoarse=ncoarse)


def build_true_barycentric_P(cells, fine_xy, dim=2, knn=12):
    """Build the finest-pair geometric P2 prolongation (coarse -> fine) as a
    fresh PETSc AIJ matrix.

    Each fine node is located in the coarse element it occupies (KD-tree over
    cell centroids + barycentric test, clamp-to-simplex for nodes just outside
    the coarse polygon at curved boundaries) and the coarse P2 basis is
    evaluated there. Component blocks are interleaved (``dof = dim*node+comp``)
    to match PETSc's interpolation layout. Serial.
    """
    from petsc4py import PETSc
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    fine_xy = np.asarray(fine_xy)
    V, Vxy, E, Evl = cells["V"], cells["Vxy"], cells["E"], cells["Evl"]
    e1, e2, det, cent = cells["e1"], cells["e2"], cells["det"], cells["cent"]
    ncoarse = cells["ncoarse"]
    tree = cKDTree(cent)
    nf = fine_xy.shape[0]
    rows = np.empty(nf * 6, np.int32)
    cols = np.empty(nf * 6, np.int32)
    vals = np.empty(nf * 6)
    for i in range(nf):
        P = fine_xy[i]
        _, cand = tree.query(P, k=knn)
        best = None
        bestpen = 1e30
        for c in np.atleast_1d(cand):
            d = det[c]
            if abs(d) < 1e-30:
                continue
            rp = P - Vxy[c, 0]
            l1 = (rp[0] * e2[c, 1] - rp[1] * e2[c, 0]) / d
            l2 = (e1[c, 0] * rp[1] - e1[c, 1] * rp[0]) / d
            l0 = 1.0 - l1 - l2
            pen = max(0.0, -l0) + max(0.0, -l1) + max(0.0, -l2)
            if pen < bestpen:
                bestpen = pen
                best = (c, l0, l1, l2)
            if pen == 0.0:
                break
        c, l0, l1, l2 = best
        lam = np.array([l0, l1, l2])
        if bestpen > 0:                          # clamp into the simplex
            lam = np.clip(lam, 0.0, None)
            lam /= lam.sum()
            l0, l1, l2 = lam
        b = i * 6
        for k in range(3):                       # vertex basis lam_k(2 lam_k-1)
            rows[b + k] = i
            cols[b + k] = V[c, k]
            vals[b + k] = lam[k] * (2.0 * lam[k] - 1.0)
        for k in range(3):                       # edge basis 4 lam_a lam_b
            a, bb = Evl[c, k]
            rows[b + 3 + k] = i
            cols[b + 3 + k] = E[c, k]
            vals[b + 3 + k] = 4.0 * lam[a] * lam[bb]
    R = np.repeat(rows, dim) * dim + np.tile(np.arange(dim), len(rows))
    C = np.repeat(cols, dim) * dim + np.tile(np.arange(dim), len(cols))
    Vv = np.repeat(vals, dim)
    Pcsr = sp.csr_matrix((Vv, (R, C)), shape=(dim * nf, dim * ncoarse))
    M = PETSc.Mat().createAIJ(
        Pcsr.shape,
        csr=(Pcsr.indptr.astype(np.int32), Pcsr.indices.astype(np.int32), Pcsr.data),
    )
    M.assemble()
    return M


def _set_mg_galerkin_none(pc):
    """Turn off Galerkin coarse-operator assembly on a PCMG.

    petsc4py does not expose ``PCMGSetGalerkin``; reach it through ctypes so we
    can supply explicit coarse operators instead (avoids the cached-PtAP
    operator/transfer swap when the finest transfer is replaced with one of a
    different sparsity).
    """
    import ctypes
    import petsc4py
    import os

    cfg = petsc4py.get_config()
    libname = "libpetsc.dylib"
    libpath = os.path.join(cfg["PETSC_DIR"], cfg["PETSC_ARCH"], "lib", libname)
    if not os.path.exists(libpath):
        libpath = os.path.join(cfg["PETSC_DIR"], cfg["PETSC_ARCH"], "lib", "libpetsc.so")
    lib = ctypes.CDLL(libpath)
    PC_MG_GALERKIN_NONE = 3
    lib.PCMGSetGalerkin(ctypes.c_void_p(pc.handle), ctypes.c_int(PC_MG_GALERKIN_NONE))


def inject_geometric_transfer(pc, cells, fine_xy, dim=2):
    """Replace the finest multigrid prolongation with the geometric P2 transfer
    and supply explicit coarse operators (Galerkin off).

    Coarse operators are formed top-down ``A_{L-1} = I_L^T A_L I_L`` with the
    geometric transfer at the finest pair and PETSc's nested transfers below.
    Returns the finest geometric Mat (kept alive by the caller).
    """
    nl = pc.getMGLevels()
    PA = build_true_barycentric_P(cells, fine_xy, dim)

    _set_mg_galerkin_none(pc)
    # interpolation into each level (finest = geometric, below = nested)
    interp = {nl - 1: PA}
    for L in range(1, nl - 1):
        interp[L] = pc.getMGInterpolation(L)
    # cascade the coarse operators down from the finest operator
    A = pc.getMGSmoother(nl - 1).getOperators()[0]
    keep = [PA]
    for L in range(nl - 1, 0, -1):
        A = A.PtAP(interp[L])
        pc.getMGSmoother(L - 1).setOperators(A, A)
        keep.append(A)
    pc.setMGInterpolation(nl - 1, PA)
    pc.setUp()
    return keep


# ----------------------------------------------------------------------------
# Pre-solve hook.
# ----------------------------------------------------------------------------


def _default_locate_mg_pc(solver):
    """Return the PCMG to override, or ``None``.

    Saddle-point (Stokes) solves keep velocity in the first fieldsplit block;
    its sub-PC carries the geometric-multigrid hierarchy. Scalar / vector solves
    use the main PC directly when it is type ``mg``. ``getFieldSplitSubKSP``
    raises (PETSc error 73) before the first solve has set up the fieldsplit;
    that is caught and reported as "not ready" (``None``).
    """
    from petsc4py import PETSc

    ksp = solver.snes.getKSP()
    pc = ksp.getPC()
    if pc.getType() == PETSc.PC.Type.FIELDSPLIT:
        try:
            sub = pc.getFieldSplitSubKSP()
        except Exception:
            return None
        if not sub:
            return None
        vpc = sub[0].getPC()
        return vpc if vpc.getType() == PETSc.PC.Type.MG else None
    return pc if pc.getType() == PETSc.PC.Type.MG else None


class GeometricMGInterpolator:
    """Callable pre-solve hook that replaces the finest multigrid prolongation
    with the true barycentric geometric interpolant at the current node
    positions (see the module docstring).

    Assign an instance to ``solver._pre_solve_hook``.

    Parameters
    ----------
    locate_mg_pc : callable, optional
        ``locate_mg_pc(solver) -> petsc4py.PETSc.PC`` returning the PCMG to
        override (or ``None`` to skip). Defaults to the velocity fieldsplit
        sub-PC / main PC autodetection.
    verbose : bool, default False
        Log injection events on rank 0.
    """

    def __init__(self, locate_mg_pc=None, verbose=False):
        self._locate = locate_mg_pc or _default_locate_mg_pc
        self._verbose = verbose
        self._cells = None          # cached coarse cell structure (never moves)
        self._keep = None           # keep injected mats alive
        self._warned_parallel = False
        self._calls = 0

    def _log(self, msg):
        if self._verbose and uw.mpi.rank == 0:
            print(f"[geometric-mg] {msg}", flush=True)

    def __call__(self, solver):
        from petsc4py import PETSc

        if uw.mpi.size > 1:
            if not self._warned_parallel:
                self._log("comm size > 1: not yet implemented; nested transfer kept")
                self._warned_parallel = True
            return

        # The first solve is on the unmoved mesh; the fieldsplit sub-KSPs are
        # not built yet (probing raises and poisons the subsequent Galerkin), so
        # let PETSc's nested transfer run and begin overriding from the second.
        self._calls += 1
        if self._calls == 1:
            self._log("first solve: nested transfer (mesh assumed unmoved)")
            return

        pc = self._locate(solver)
        if pc is None or pc.getType() != PETSc.PC.Type.MG:
            return
        try:
            nl = pc.getMGLevels()
        except Exception:
            return
        if nl < 2:
            return

        dim = solver.mesh.dim
        if self._cells is None:
            cdm = pc.getMGSmoother(nl - 2).getDM()
            if cdm is None or cdm.getType() != PETSc.DM.Type.PLEX:
                self._log("coarse level DM not yet a plex; nested this solve")
                return
            self._cells = coarse_cell_structure(cdm, dim)
            self._log(
                f"cached coarse cell structure: {self._cells['ncoarse']} nodes, "
                f"{self._cells['ncell']} cells"
            )

        fine_xy = np.asarray(solver.u.coords)
        try:
            self._keep = inject_geometric_transfer(pc, self._cells, fine_xy, dim)
        except Exception as exc:
            self._log(f"injection failed ({type(exc).__name__}: {exc}); nested kept")
            return
        self._log("injected true-barycentric finest transfer (galerkin off, explicit coarse ops)")


def geometric_mg_interpolation(locate_mg_pc=None, verbose=False):
    """Build a pre-solve hook that replaces the finest multigrid prolongation
    with the true barycentric geometric interpolant rebuilt from current node
    positions each setup (geometry-aware GMG on mover-adapted meshes).

    See :class:`GeometricMGInterpolator`. Returns a callable suitable for
    ``solver._pre_solve_hook``.
    """
    return GeometricMGInterpolator(locate_mg_pc=locate_mg_pc, verbose=verbose)
