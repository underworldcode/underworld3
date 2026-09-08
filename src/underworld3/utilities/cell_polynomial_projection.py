"""Cell-by-cell polynomial fit of particle data.

This is the reconstruction behind ``SwarmVariable(proxy_location="cells")``:
every cell of a discontinuous mesh variable receives the least-squares
polynomial through the particles it holds. The result is exact for
polynomial particle fields up to the proxy degree, is a polynomial on the
mesh cell (so the default integration rule integrates it exactly, no
oversampling guard), keeps a material step sharp at cell edges, has a
gradient, and needs no neighbour search across ranks: a rank fits its own
cells from its own particles.

A cell with too few particles for a well-posed fit (fewer than the basis
size plus two) is fitted instead to the particles nearest its centroid,
which is what the RBF proxy's neighbourhood does. That cell is then
consistent but not a cell-local moment fit. Dense cells and light swarms
therefore share one code path and degrade gracefully together.

Why least squares rather than moment matching: the conservative
particle-to-mesh transfer (rule mass on the left, particle moments on the
right, as PETSc's ``DMSwarmProjectFields`` does) conserves the particle
sums exactly but its nodal values carry the Monte-Carlo error of the
particle "quadrature", which scales with the field value over the square
root of the particle count. Measured on a linear field with 21 jittered
particles per cell its error was 25% of the field; the least-squares fit
was exact (`~/+Simulations/integration_point_proxy`, 2026-09-08).
"""

from __future__ import annotations

import numpy as np

import underworld3 as uw
from underworld3.cython.petsc_quadrature_fe import cell_affine_maps, tabulate


class CellPolynomialProjector:
    """Least-squares fit of particle values onto a discontinuous mesh variable, cell by cell.

    Parameters
    ----------
    meshVar :
        A discontinuous (``continuous=False``) mesh variable of any degree and
        component count. Its element and the mesh's affine cell maps are
        tabulated once here; rebuild the projector when the mesh moves
        (``mesh_version`` records the mesh version it was built for).
    """

    def __init__(self, meshVar):
        mesh = meshVar.mesh
        if getattr(meshVar, "continuous", True) or getattr(meshVar, "is_integration_point", False):
            raise ValueError("CellPolynomialProjector needs a discontinuous (cell-local) mesh variable")
        self.mesh = mesh
        self.var = meshVar
        self.mesh_version = mesh._mesh_version
        self.dim = mesh.dim
        self.num_components = meshVar.num_components
        self.fe = mesh.dm.getField(meshVar.field_id)[0]
        self.v0, self.invJ, self.detJ = cell_affine_maps(mesh.dm)
        self.ncells = self.detJ.shape[0]
        probe = tabulate(self.fe, np.zeros((1, self.dim)))
        self.Nb = probe.shape[1] // self.num_components     # scalar basis size
        # Reference-cell centroid, mapped: xi_c + 1 = 2 / (dim + 1) on every axis.
        J = np.linalg.inv(self.invJ) if self.ncells else self.invJ
        self.centroids = self.v0 + np.einsum(
            "cij,j->ci", J, np.full(self.dim, 2.0 / (self.dim + 1))
        )
        self._check_layout()

    # -- geometry -----------------------------------------------------------

    def _scalar_basis(self, xi):
        """Scalar basis values at reference points, shape (Np, Nb)."""
        if xi.shape[0] == 0:
            return np.zeros((0, self.Nb))
        T = tabulate(self.fe, xi)                      # (Np, Nb*Nc, Nc), interleaved
        return T[:, 0 :: self.num_components, 0]

    def _check_layout(self):
        """The discontinuous local vector is cell-major with ``Nb`` dofs per cell in basis order."""
        X = np.asarray(self.var.coords_nd)
        if X.shape[0] != self.ncells * self.Nb:
            raise RuntimeError(
                f"unexpected dof count for {self.var.clean_name}: {X.shape[0]} != {self.ncells} x {self.Nb}"
            )
        if self.ncells == 0:
            return
        cells = np.repeat(np.arange(self.ncells), self.Nb)
        xi = self.reference_coords(X, cells)
        B = self._scalar_basis(xi).reshape(self.ncells, self.Nb, self.Nb)
        if not np.allclose(B, np.eye(self.Nb)[None], atol=1e-9):
            raise RuntimeError("discontinuous dof layout is not cell-major; cannot fit cell by cell")

    def reference_coords(self, coords, cells):
        """Reference coordinates (PETSc's [-1, 1] frame) of points in their cells."""
        return np.einsum("cij,cj->ci", self.invJ[cells], coords - self.v0[cells]) - 1.0

    def locate(self, coords):
        """Owning local cell of each point (-1 when not on this rank) and its reference coordinates."""
        coords = np.asarray(coords, dtype=np.float64)
        cells = np.asarray(self.mesh._robust_owning_cells(coords), dtype=np.int64)
        ok = cells >= 0
        xi = np.zeros_like(coords)
        if ok.any():
            xi[ok] = self.reference_coords(coords[ok], cells[ok])
        return cells, ok, xi

    # -- the fit ------------------------------------------------------------

    def fit(self, coords, values, nmin=None, patch_nnn=None):
        """Fit every cell; returns nodal values shaped like ``meshVar.data``.

        Parameters
        ----------
        coords : (N, dim) particle coordinates (non-dimensional).
        values : (N, num_components) particle values.
        nmin : particles a cell needs for its own fit (default basis size + 2).
        patch_nnn : particles nearest the centroid used for a thin cell
            (default twice the basis size + 2).
        """
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, self.dim)
        values = np.asarray(values, dtype=np.float64).reshape(coords.shape[0], -1)
        nc = values.shape[1]
        cells, ok, xi = self.locate(coords)
        c = cells[ok]
        B = self._scalar_basis(xi[ok])                                  # (Np, Nb)
        psi = values[ok]                                               # (Np, nc)
        npc = np.bincount(c, minlength=self.ncells)

        G = np.zeros((self.ncells, self.Nb, self.Nb))
        R = np.zeros((self.ncells, self.Nb, nc))
        np.add.at(G, c, B[:, :, None] * B[:, None, :])
        np.add.at(R, c, B[:, :, None] * psi[:, None, :])

        U = np.zeros((self.ncells, self.Nb, nc))
        nmin = nmin or self.Nb + 2
        dense = npc >= nmin
        if dense.any():
            ridge = 1e-10 * np.trace(G[dense], axis1=1, axis2=2)[:, None, None] / self.Nb
            U[dense] = np.linalg.solve(G[dense] + ridge * np.eye(self.Nb)[None], R[dense])

        thin = np.nonzero(~dense)[0]
        self.n_thin = int(thin.shape[0])
        self.n_empty = int((npc == 0).sum())
        if thin.shape[0] > 0 and c.shape[0] > 0:
            Xp = coords[ok]
            nnn = min(patch_nnn or 2 * self.Nb + 2, Xp.shape[0])
            tree = uw.kdtree.KDTree(Xp)
            _, idx = tree.query(self.centroids[thin], k=nnn)
            idx = np.asarray(idx).reshape(thin.shape[0], nnn)
            xp = Xp[idx]                                               # (nthin, nnn, dim)
            xit = np.einsum("cij,cpj->cpi", self.invJ[thin], xp - self.v0[thin][:, None, :]) - 1.0
            Bt = self._scalar_basis(xit.reshape(-1, self.dim)).reshape(thin.shape[0], nnn, self.Nb)
            Gt = np.einsum("cpb,cpd->cbd", Bt, Bt)
            Rt = np.einsum("cpb,cpk->cbk", Bt, psi[idx])
            ridge = 1e-10 * np.trace(Gt, axis1=1, axis2=2)[:, None, None] / self.Nb + 1e-30
            U[thin] = np.linalg.solve(Gt + ridge * np.eye(self.Nb)[None], Rt)

        return U.reshape(self.ncells * self.Nb, nc)

    def interpolate(self, U, coords):
        """The fitted polynomials evaluated at points (NaN off-rank): the FLIP read-back."""
        U = np.asarray(U).reshape(self.ncells, self.Nb, -1)
        cells, ok, xi = self.locate(coords)
        out = np.full((coords.shape[0], U.shape[2]), np.nan)
        B = self._scalar_basis(xi[ok])
        out[ok] = np.einsum("pb,pbk->pk", B, U[cells[ok]])
        return out
