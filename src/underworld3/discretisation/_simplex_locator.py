"""Exact point location on a straight-sided simplex mesh: barycentric coordinates
and a face-crossing walk.

A point is in a simplex when all its barycentric coordinates are non-negative.
That test is exact up to rounding for any straight-sided simplex, so no geometric
tolerance is needed: the one guard, ``ROUNDOFF``, bounds the rounding in the
barycentric coordinates themselves, which are dimensionless. The face a point
fails worst (its most negative coordinate) is the face to cross, so a walk from
any start cell reaches the containing cell in about the number of cells between
them. On a non-convex boundary the walk crosses the most violated face that has a
neighbour, so it follows the boundary instead of leaving through it.

A point that several cells contain (on a shared face, edge or vertex) is given
the lowest-numbered local cell in the star of that entity, so the answer depends
on the mesh and not on where the walk started.

This is the locator behind ``Mesh._robust_owning_cells`` for simplex volume
meshes with straight-sided cells. Quadrilaterals, hexahedra, curved cells and
2-manifolds keep the cell-wall test until they have an inverse map.
"""
import numpy as np


class SimplexLocator:
    """Barycentric locator over one DM's local cells.

    Parameters
    ----------
    dm : PETSc.DMPlex
        The DM whose cell numbers are returned.
    vertex_coords : callable
        ``vertex_coords(vertex_points) -> (n, dim)`` coordinates of vertices.
    """

    # Rounding guard on a barycentric coordinate (dimensionless). A coordinate at
    # or above -ROUNDOFF counts as non-negative. It bounds the rounding of the
    # affine inverse, not any geometry: it is the only tolerance in the locator.
    ROUNDOFF = 1.0e-12

    def __init__(self, dm, vertex_coords):
        from underworld3.utilities._plex_topology import closure_points, supports

        self.dim = dm.getDimension()
        d = self.dim
        self.cStart, cEnd = dm.getHeightStratum(0)
        cells = np.arange(self.cStart, cEnd, dtype=np.int64)
        self.n_cells = cells.shape[0]
        if self.n_cells == 0:
            # A rank with no cells has a locator that locates nothing, so every
            # rank takes the same path through the callers.
            self.v0 = np.zeros((0, d))
            self.Tinv = np.zeros((0, d, d))
            self.nbr = np.zeros((0, d + 1), dtype=np.int64)
            self._star_table = np.zeros((0, 1 << (d + 1)), dtype=np.int64)
            return
        verts = closure_points(dm, cells, 0)                       # (n, d+1)
        if verts.shape[1] != d + 1 or (verts < 0).any():
            raise ValueError("SimplexLocator: the cells are not all simplices")
        V = np.asarray(vertex_coords(verts.ravel()), dtype=np.float64).reshape(
            self.n_cells, d + 1, -1)
        if V.shape[2] != d:
            raise ValueError("SimplexLocator: a manifold mesh needs its tangent-plane map")
        self.v0 = V[:, 0, :].copy()
        T = np.transpose(V[:, 1:, :] - V[:, :1, :], (0, 2, 1))     # columns v_k - v_0
        self.Tinv = np.linalg.inv(T)

        # Neighbour across the face opposite each local vertex, -1 where the
        # face has one local cell (the domain boundary or a partition seam).
        faces = closure_points(dm, cells, d - 1)                   # (n, d+1)
        face_verts = closure_points(dm, faces.ravel(), 0).reshape(self.n_cells, d + 1, d)
        sup = supports(dm, faces.ravel()).reshape(self.n_cells, d + 1, -1)
        self.nbr = np.full((self.n_cells, d + 1), -1, dtype=np.int64)
        for k in range(d + 1):
            # the face that does not hold vertex k
            lacks = ~(face_verts == verts[:, k][:, None, None]).any(axis=2)   # (n, d+1)
            f = lacks.argmax(axis=1)
            s = sup[np.arange(self.n_cells), f]                     # (n, 2)
            other = np.where(s[:, 0] == cells, s[:, 1] if s.shape[1] > 1 else -1, s[:, 0])
            self.nbr[:, k] = np.where(other >= 0, other - self.cStart, -1)

        # Lowest local cell in the star of every sub-entity, by the set of a
        # cell's local vertices it spans (a bitmask over the d+1 vertices).
        self._star_min = {}
        local = cells - self.cStart
        for depth in range(d):                                      # vertices .. faces
            ents = closure_points(dm, cells, depth)                 # (n, m)
            ent_verts = closure_points(dm, ents.ravel(), 0).reshape(
                self.n_cells, ents.shape[1], -1)
            masks = np.zeros(ents.shape, dtype=np.int64)
            for k in range(d + 1):
                masks |= ((ent_verts == verts[:, k][:, None, None]).any(axis=2)
                          .astype(np.int64) << k)
            star = np.full(ents.max() + 1, np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(star, ents.ravel(), np.repeat(local, ents.shape[1]))
            for j in range(ents.shape[1]):
                self._star_min.setdefault("rows", []).append(
                    (masks[:, j], star[ents[:, j]]))
        rows = self._star_min.pop("rows")
        # star_min[cell, mask] for the masks of proper sub-entities
        self._star_table = np.full((self.n_cells, 1 << (d + 1)), -1, dtype=np.int64)
        for mask, smin in rows:
            self._star_table[np.arange(self.n_cells), mask] = smin
        full = (1 << (d + 1)) - 1
        self._star_table[:, full] = local

    def barycentric(self, points, cells):
        """``(n, dim+1)`` barycentric coordinates of ``points`` in local ``cells``."""
        lam = np.einsum("nij,nj->ni", self.Tinv[cells], points - self.v0[cells])
        return np.concatenate([1.0 - lam.sum(axis=1, keepdims=True), lam], axis=1)

    def walk(self, points, start, max_steps=None):
        """Walk from local ``start`` cells to the cell containing each point.

        Returns ``(cells, lam, exit_cell)``: the local containing cell (-1 when
        the walk left the local mesh or ran out of steps), its barycentric
        coordinates, and for a lost point the cell it left through
        (-1 otherwise).
        """
        points = np.asarray(points, dtype=np.float64)
        n = points.shape[0]
        d = self.dim
        cells = np.full(n, -1, dtype=np.int64)
        lam_out = np.zeros((n, d + 1))
        exit_cell = np.full(n, -1, dtype=np.int64)
        if n == 0 or self.n_cells == 0:
            return cells, lam_out, exit_cell
        if max_steps is None:
            max_steps = 8 + 4 * int(round(self.n_cells ** (1.0 / d)))
        active = np.arange(n)
        cur = np.asarray(start, dtype=np.int64).copy()
        for _ in range(max_steps):
            if active.shape[0] == 0:
                break
            lam = self.barycentric(points[active], cur)
            inside = lam.min(axis=1) >= -self.ROUNDOFF
            cells[active[inside]] = cur[inside]
            lam_out[active[inside]] = lam[inside]
            # Cross the most violated face that has a neighbour. On a non-convex
            # boundary the most violated face can be a boundary face although
            # the point is inside, further along the boundary (measured on
            # #1123's sheared mesh: 7 of 2e5 contained points). Only when every
            # violated face is on the local boundary has the point left the
            # local mesh.
            neighbours = self.nbr[cur]
            ranked = np.where((lam < -self.ROUNDOFF) & (neighbours >= 0), lam, np.inf)
            k = ranked.argmin(axis=1)
            can_cross = np.isfinite(ranked[np.arange(active.shape[0]), k])
            nxt = np.where(can_cross, neighbours[np.arange(active.shape[0]), k], -1)
            left = ~inside & (nxt < 0)
            exit_cell[active[left]] = cur[left]
            go = ~inside & (nxt >= 0)
            active, cur = active[go], nxt[go]
        return cells, lam_out, exit_cell

    def lowest_in_star(self, cells, lam):
        """For points located in local ``cells`` with coordinates ``lam``: the
        lowest local cell in the star of the entity the point lies on (the cell
        itself for a point strictly inside)."""
        d = self.dim
        on = lam > self.ROUNDOFF                                     # vertices spanning the entity
        mask = (on.astype(np.int64) << np.arange(d + 1)).sum(axis=1)
        out = self._star_table[cells, mask]
        return np.where(out >= 0, out, cells)

    def locate(self, points, start, restarts=(), return_exit=False):
        """Containing local cell per point (-1 if no local cell contains it).

        ``start`` gives a local start cell per point; ``restarts`` further start
        cells, each an array like ``start``, tried in turn for points still lost
        (a non-convex local mesh can put a wall between start and target).
        """
        cells, lam, exit_cell = self.walk(points, start)
        for alt in restarts:
            lost = np.flatnonzero(cells < 0)
            if lost.shape[0] == 0:
                break
            c2, l2, _ = self.walk(points[lost], np.asarray(alt)[lost])
            cells[lost], lam[lost] = c2, l2
        found = cells >= 0
        cells[found] = self.lowest_in_star(cells[found], lam[found])
        if return_exit:
            # the cell the first walk left the local mesh through (-1: none)
            exit_cell[found] = -1
            return cells, exit_cell
        return cells
