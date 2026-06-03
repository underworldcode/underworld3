r"""Geometry-aware multigrid interpolation for mover-adapted meshes.

When the finest mesh level is relocated by a node mover (anisotropic metric
adaptation, free-surface ALE, ...), PETSc's geometric-multigrid level transfers
become coordinate-blind. They are constructed once from the refinement topology
and assume the fine nodes still sit at their refinement positions relative to
the coarse cells. After the mover moves them, the finest prolongation
interpolates from the wrong place and the multigrid iteration count climbs as
the operator stiffens (e.g. a sharpening viscosity gradient in convection).

This module re-targets the **finest-level** prolongation to the *current* node
positions on every solver setup, keeping the multigrid cycle iteration-flat.
Only the finest pair needs this: a node mover deforms only ``mesh.dm`` (the
finest level); the coarser levels keep their uniform-refinement positions where
PETSc's transfer is already correct.

Design — recompute-nested-values (in place)
-------------------------------------------
PETSc builds its multigrid hierarchy normally (correct level DMs, vector sizes,
communicators, and a nested transfer at the finest pair). On a P2 velocity
field that transfer is block-diagonal in component and, per fine DOF row,
reproduces constants and linears exactly at the node's *refinement* position:

.. math::  \sum_c w_c = 1, \qquad \sum_c w_c\, X_c = x_i^{\text{nested}} .

After the mover, the fine node sits at a new position :math:`x_i`, but the
nested weights still point at :math:`x_i^{\text{nested}}`. We overwrite **only
the values** of the existing interpolation matrix (its sparsity, ordering and
the PETSc Mat object are untouched) with the minimal weight correction that
re-satisfies linear reproduction at the new position:

.. math::  w = w_0 + A^{\mathsf T}(A A^{\mathsf T})^{-1}\,(b - A w_0),
   \quad A=\begin{bmatrix}1\cdots\\ X_c^{\mathsf T}\end{bmatrix},\;
   b=\begin{bmatrix}1\\ x_i\end{bmatrix}.

This is a small ``(dim+1)`` solve per row. It keeps the proven nested smoothing
structure where the node did not move (:math:`b=Aw_0\Rightarrow w=w_0`) and
shifts it geometrically where it did. Reusing the *same* Mat object is essential
— replacing it would make PETSc's cached Galerkin product swap operator/transfer
roles and fail the ``PtAP``; an in-place value update lets the Galerkin coarse
operators (``pc_mg_galerkin``) recompute cleanly from the corrected transfer.

Nothing here mutates ``mesh.dm`` (coordinates, sections or refinement flags), so
the mesh's own point-location (SLCN advection, boundary integrals) is untouched.
The override lives entirely on the multigrid sub-PC and is rebuilt each setup,
surviving the per-adapt SNES/PC teardown.

Usage
-----
.. code-block:: python

    from underworld3.utilities.gmg_geometric_interpolation import (
        geometric_mg_interpolation,
    )

    # velocity-block GMG on a per-step-adapted annulus Stokes solve
    stokes._pre_solve_hook = geometric_mg_interpolation()

The default locates the multigrid PC automatically (the velocity fieldsplit
sub-PC of a saddle-point solve, else the main PC when it is type ``mg``). It is
a no-op unless that PC is multigrid, so it is safe to leave attached.

.. note::
   Currently validated for **serial** runs. In parallel the fine-row /
   coarse-column DOF orderings of the distributed transfer require an explicit
   coordinate scatter that is not yet implemented; the hook detects ``comm
   size > 1`` and falls back to PETSc's nested transfer (still correct, only
   the iteration-flatness benefit is forgone).
"""

import numpy as np

import underworld3 as uw

__all__ = ["geometric_mg_interpolation", "GeometricMGInterpolator"]


def coarse_node_coords(dm, dim=2):
    """P2 DOF *node* coordinates of ``dm`` in its block-vector ordering.

    Coarse DOF ``d`` belongs to node ``d // dim``; node ``i`` occupies vector
    indices ``dim*i .. dim*i+dim-1``. Vertices carry their own coordinate; edge
    nodes are the midpoint of the edge's two vertices. Returns ``(Nnode, dim)``.

    The coarse level never moves under a finest-level mover, so this is read
    once and reused on every solve.
    """
    sec = dm.getLocalSection()
    vc = dm.getCoordinatesLocal().array.reshape(-1, dim)
    cdm = dm.getCoordinateDM()
    csec = cdm.getLocalSection()
    vS, vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    vcoord = {vtx: vc[csec.getOffset(vtx) // dim] for vtx in range(vS, vE)}
    npt = sec.getStorageSize() // dim
    out = np.zeros((npt, dim))
    for vtx in range(vS, vE):
        if sec.getDof(vtx):
            out[sec.getOffset(vtx) // dim] = vcoord[vtx]
    for e in range(eS, eE):
        if sec.getDof(e):
            c = dm.getCone(e)
            out[sec.getOffset(e) // dim] = 0.5 * (vcoord[c[0]] + vcoord[c[1]])
    return out


def retarget_interpolation_values(P, coarse_xy, fine_xy, dim=2):
    """Overwrite the values of interpolation Mat ``P`` (coarse -> fine) in place
    so each fine row reproduces constants and linears at the *current* fine node
    position, via the minimal correction to the existing (nested) weights.

    The Mat object, sparsity and ordering are preserved (only numerical values
    change), so a cached Galerkin product recomputes cleanly. Returns the worst
    reproduction residual (≈ machine epsilon when well posed) for diagnostics.

    Parameters
    ----------
    P : petsc4py.PETSc.Mat
        The finest-level interpolation, already built by PETSc.
    coarse_xy : (Ncoarse_node, dim) array
        Coarse P2 node coordinates indexed so coarse DOF ``d`` -> node
        ``d // dim`` (see :func:`coarse_node_coords`).
    fine_xy : (Nfine_node, dim) array
        Current fine velocity node coordinates (``solver.u.coords``); fine DOF
        ``r`` -> node ``r // dim``.
    """
    ai, aj, av = P.getValuesCSR()
    av = av.copy()
    nrows = len(ai) - 1
    worst = 0.0
    for r in range(nrows):
        s, e = ai[r], ai[r + 1]
        cols = aj[s:e]
        comp = r % dim
        node_i = r // dim
        same = (cols % dim) == comp
        Xc = coarse_xy[cols[same] // dim]            # (k, dim)
        w0 = av[s:e][same]
        k = Xc.shape[0]
        if k == 0:
            continue
        A = np.vstack([np.ones(k), Xc.T])            # (dim+1, k)
        M = A @ A.T                                   # (dim+1, dim+1)
        b = np.empty(dim + 1)
        b[0] = 1.0
        b[1:] = fine_xy[node_i]
        resid = b - A @ w0
        try:
            wnew = w0 + A.T @ np.linalg.solve(M, resid)
        except np.linalg.LinAlgError:
            continue                                  # keep nested row as-is
        block = av[s:e]
        block[same] = wnew
        block[~same] = 0.0
        av[s:e] = block
        worst = max(
            worst,
            abs(wnew.sum() - 1.0),
            float(np.max(np.abs(Xc.T @ wnew - fine_xy[node_i]))),
        )
    P.setValuesCSR(ai, aj, av)
    P.assemble()
    return worst


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
    """Callable pre-solve hook that re-targets the finest-level multigrid
    prolongation to the current node positions (see the module docstring).

    Assign an instance to ``solver._pre_solve_hook``. It is invoked once per
    solve (after the operator and nullspaces are attached, before
    ``snes.solve``), so it is re-applied automatically after the per-adapt
    SNES/PC teardown.

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
        self._coarse_xy = None  # cached coarse P2 node coords (never move)
        self._warned_parallel = False
        self._calls = 0

    def _log(self, msg):
        if self._verbose and uw.mpi.rank == 0:
            print(f"[geometric-mg] {msg}", flush=True)

    def __call__(self, solver):
        from petsc4py import PETSc

        # Parallel transfer ordering not yet handled -> nested fallback.
        if uw.mpi.size > 1:
            if not self._warned_parallel:
                self._log(
                    "comm size > 1: parallel DOF ordering unimplemented; "
                    "using PETSc nested transfer"
                )
                self._warned_parallel = True
            return

        # Skip the FIRST solve entirely without touching PETSc. Before any solve
        # the fieldsplit sub-KSPs are not built and probing them raises PETSc
        # error 73, whose raised state then breaks the subsequent Galerkin PtAP.
        # The first solve is on the unmoved mesh anyway, where the nested
        # transfer is correct, so we let it run untouched and begin retargeting
        # from the second solve — by then the sub-PC, MG levels and coarse plex
        # DM are all available with no setup call (verified).
        self._calls += 1
        if self._calls == 1:
            self._log("first solve: nested transfer (mesh assumed unmoved)")
            return

        # IMPORTANT: never call ksp.setUp() here — forcing setup early builds a
        # degenerate finest interpolation before the coarse DM exists and breaks
        # the real solve. Query the already-set-up sub-PC instead.
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
        if self._coarse_xy is None:
            cdm = pc.getMGSmoother(nl - 2).getDM()
            if cdm is None or cdm.getType() != PETSc.DM.Type.PLEX:
                self._log("coarse level DM not yet a plex; nested this solve")
                return
            self._coarse_xy = coarse_node_coords(cdm, dim)
            self._log(f"cached coarse P2 node coords: {self._coarse_xy.shape[0]} nodes")

        P = pc.getMGInterpolation(nl - 1)
        fine_xy = np.asarray(solver.u.coords)
        if P.getSize()[1] != self._coarse_xy.shape[0] * dim:
            self._log(
                f"coarse size {P.getSize()[1]} != cached "
                f"{self._coarse_xy.shape[0] * dim}; skipping"
            )
            return

        worst = retarget_interpolation_values(P, self._coarse_xy, fine_xy, dim)
        self._log(f"retargeted finest interpolation (reproduction resid {worst:.1e})")


def geometric_mg_interpolation(locate_mg_pc=None, verbose=False):
    """Build a pre-solve hook that re-targets the finest-level multigrid
    prolongation to current node positions each setup (geometry-aware GMG on
    mover-adapted meshes).

    See :class:`GeometricMGInterpolator`. Returns a callable suitable for
    ``solver._pre_solve_hook``.
    """
    return GeometricMGInterpolator(locate_mg_pc=locate_mg_pc, verbose=verbose)
