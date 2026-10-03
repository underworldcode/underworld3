r"""Global L2 projection of scattered, weighted values onto a continuous P1 field.

Given points :math:`\mathbf{x}_p` with values :math:`v_p` and weights :math:`w_p`
(each point's share of the domain), the continuous linear field
:math:`u = \sum_i u_i \phi_i` that best fits them in the weighted least-squares
sense solves

.. math::

    \Big(\sum_p w_p\,\phi_i(\mathbf{x}_p)\,\phi_j(\mathbf{x}_p)
         + \varepsilon M_{ij} + \alpha K_{ij}\Big)\,u_j
    = \sum_p w_p\,\phi_i(\mathbf{x}_p)\,v_p + \varepsilon M_{ij}\,u^{\mathrm{old}}_j .

The first term is the finite-element mass matrix with the points as its
quadrature rule: when the points are the integration points of the mesh, or
those points moved by an incompressible flow, the weights are a quadrature rule
and the system is the ordinary L2 projection. There is no per-cell fit: every
nodal value is set by all the points in the patch of cells around the node, so
the node is interpolated from data on every side of it rather than extrapolated
from one cell's points.

A node whose patch received no points has an empty row. The term
:math:`\varepsilon M (u - u^{\mathrm{old}})` (:math:`M` the finite-element mass
matrix) keeps such a node at its previous value and is negligible where the
patch is sampled; :math:`\alpha K` (:math:`K` the stiffness matrix, :math:`\alpha`
per cell, length squared) is the optional gradient penalty of the history
projections' ``store_smoothing``.

Simplex meshes. In parallel each point is contributed by the rank that owns its
cell (the mesh is distributed without overlap); the shared nodes on a partition
seam sum their contributions through PETSc's local-to-global ADD, so the system,
and to the solver tolerance the answer, does not depend on the partition.
"""

import numpy as np
from petsc4py import PETSc

import underworld3 as uw


class ParticleL2Projector:
    """Weighted least-squares projection of scattered values onto continuous P1.

    Parameters
    ----------
    mesh : Mesh
        A simplex mesh.
    rtol : float
        Relative tolerance of the conjugate-gradient solve.
    """

    instances = 0

    def __init__(self, mesh, rtol=1.0e-12):
        ParticleL2Projector.instances += 1
        self.mesh = mesh
        self.rtol = rtol
        # the scalar layout the matrix and vectors live on; a P1 variable of any
        # shape keeps its rows in the same vertex order (rows = section offsets)
        self._var = uw.discretisation.MeshVariable(
            f"_pl2_{ParticleL2Projector.instances}", mesh, 1, degree=1, continuous=True)
        self._dm = None

    def _build(self):
        """The cell-to-row map, the element matrices and the solver, for the mesh
        DM as it is now (adding a variable rebuilds the DM, so this is redone
        whenever the DM changes)."""
        mesh = self.mesh
        dm = mesh.dm
        d = mesh.dim
        c0, c1 = dm.getHeightStratum(0)
        v0, v1 = dm.getDepthStratum(0)
        _, self._sub = dm.createSubDM(self._var.field_id)
        sec = self._sub.getLocalSection()
        csec = dm.getCoordinateSection()
        coords = dm.getCoordinatesLocal().array.reshape(-1, mesh.cdim)
        cells = []
        for c in range(c0, c1):
            verts = [p for p in dm.getTransitiveClosure(c)[0] if v0 <= p < v1]
            if len(verts) != d + 1:
                raise NotImplementedError("ParticleL2Projector needs a simplex mesh")
            cells.append(verts)
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, d + 1)
        self._rows = np.array([[sec.getOffset(int(p)) for p in row] for row in cells],
                              dtype=np.int32).reshape(-1, d + 1)
        Xv = np.array([[coords[csec.getOffset(int(p)) // mesh.cdim, :d] for p in row] for row in cells])
        self._Xv = Xv
        self._x0 = Xv[:, 0, :] if cells.size else np.zeros((0, d))
        # barycentric coordinates: lambda_{1..d} = Tinv (x - x0), lambda_0 = 1 - sum
        T = np.transpose(Xv[:, 1:, :] - Xv[:, :1, :], (0, 2, 1)) if cells.size else np.zeros((0, d, d))
        self._Tinv = np.linalg.inv(T) if cells.size else T
        measure = np.abs(np.linalg.det(T)) / (1.0 if d == 1 else 2.0 if d == 2 else 6.0) if cells.size else np.zeros(0)
        self.cell_measure = measure
        # element matrices: consistent mass and stiffness of the linear simplex
        nv = d + 1
        base = (np.ones((nv, nv)) + np.eye(nv)) / ((d + 1) * (d + 2))
        self._Me = measure[:, None, None] * base[None, :, :]
        G = np.concatenate([-self._Tinv.sum(axis=1, keepdims=True), self._Tinv], axis=1)   # grad lambda, (nc, nv, d)
        self._Ke = measure[:, None, None] * np.einsum("cid,cjd->cij", G, G)
        self._A = self._sub.createMatrix()
        self._A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self._ksp = PETSc.KSP().create(comm=dm.comm)
        self._ksp.setType("cg")
        self._ksp.getPC().setType("jacobi")
        self._ksp.setTolerances(rtol=self.rtol, atol=0.0, max_it=10000)
        self._lb = self._sub.createLocalVector()
        self._gb = self._sub.createGlobalVector()
        self._gx = self._sub.createGlobalVector()
        self._lx = self._sub.createLocalVector()
        self.n_local_rows = self._lb.getSize()
        self._dm = dm

    def barycentric(self, X, cell):
        """Barycentric coordinates of points ``X`` in their cells, shape (n, d+1)."""
        if self._dm is not self.mesh.dm:
            self._build()
        d = self.mesh.dim
        lam = np.einsum("nij,nj->ni", self._Tinv[cell], X[:, :d] - self._x0[cell])
        return np.concatenate([1.0 - lam.sum(axis=1, keepdims=True), lam], axis=1)

    def project(self, X, values, weights, cell, old=None, eps=1.0e-8, alpha=None):
        """The projected field at the local rows, shape (n_local_rows, ncomponents).

        ``X`` (n, d), ``values`` (n, k), ``weights`` (n,) and ``cell`` (n,), the
        owning local cell of each point; ``old`` (n_local_rows, k), the previous
        field, which unreached nodes keep; ``eps`` scales the pull towards it
        relative to the finite-element mass matrix; ``alpha`` (ncell,) the
        gradient penalty per cell."""
        if self._dm is not self.mesh.dm:
            self._build()
        values = np.asarray(values, dtype=float)
        # no points at all (a rank whose patches nothing reached): the component
        # count comes from the previous field
        k = values.shape[1] if values.ndim == 2 else (np.asarray(old).shape[1] if old is not None else 1)
        values = values.reshape(len(X), k)
        ncell = self._rows.shape[0]
        lam = self.barycentric(np.asarray(X, dtype=float), cell)
        Me = np.zeros_like(self._Me)
        Re = np.zeros((ncell, self._rows.shape[1], k))
        w = np.asarray(weights, dtype=float)
        np.add.at(Me, cell, w[:, None, None] * lam[:, :, None] * lam[:, None, :])
        np.add.at(Re, cell, w[:, None, None] * lam[:, :, None] * values[:, None, :])
        if eps > 0.0:
            Me = Me + eps * self._Me
            if old is not None:
                Re = Re + eps * np.einsum("cij,cjk->cik", self._Me, np.asarray(old)[self._rows])
        if alpha is not None:
            Me = Me + np.asarray(alpha, dtype=float).reshape(-1)[:, None, None] * self._Ke
        A = self._A
        A.zeroEntries()
        for c in range(ncell):
            A.setValuesLocal(self._rows[c], self._rows[c], Me[c], addv=PETSc.InsertMode.ADD_VALUES)
        A.assemble()
        self._ksp.setOperators(A)
        out = np.zeros((self.n_local_rows, k))
        for j in range(k):
            self._lb.zeroEntries()
            b = self._lb.getArray()
            np.add.at(b, self._rows.ravel(), Re[:, :, j].ravel())
            self._lb.setArray(b)
            self._gb.zeroEntries()
            self._sub.localToGlobal(self._lb, self._gb, addv=PETSc.InsertMode.ADD_VALUES)
            self._gx.zeroEntries()
            self._ksp.solve(self._gb, self._gx)
            self._sub.globalToLocal(self._gx, self._lx)
            out[:, j] = self._lx.getArray()
        return out
