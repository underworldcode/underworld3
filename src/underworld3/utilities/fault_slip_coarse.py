r"""A coarse space for fault slip in the finest level's band smoother.

On a split fault the finest level's FAC patch is the band the welded coarse
levels cannot represent. Cutting that band into pieces, along strike or at rank
boundaries, makes its solve one-level overlapping Schwarz: nothing carries slip
from one piece to the next, and the velocity iterations grow with the number of
cuts. This module adds the second Schwarz level, whose unknowns are slip.

Each column :math:`z_q` of :math:`Z` prescribes slip :math:`\chi_q` on the slip
rows :math:`S` and carries the field that slip induces in the band, the discrete
harmonic (energy-minimising) extension

.. math::

    z_q\big|_S = \chi_q, \qquad
    z_q\big|_{I_q} = -A_{I_q I_q}^{-1} A_{I_q S}\,\chi_q, \qquad
    z_q = 0 \text{ elsewhere}.

The weights :math:`\chi_q` are a partition of unity over the band pieces
(Shepard weights reaching two piece radii), and :math:`I_q` is every band row
within a weight's support plus one band half-width, the distance over which the
field of a slip decays. The correction

.. math::

    y = Z\,(Z^T A Z)^{-1} Z^T r

is the third member of the finest smoother's multiplicative composite, after SOR
over the level and the ASM pieces. The coarse problem spans every piece of every
fault on every rank, so it couples slip across the cuts, the rank boundaries and
the segments of a network alike.

Design and measurements: ``docs/developer/design/fault-slip-coarse-space-2026-10.md``.
"""
import numpy as np
from petsc4py import PETSc

from underworld3 import kdtree
from underworld3.utilities import custom_mg, fault_contact

# Shepard weight reach in piece radii: 1.5 and 2 gave the same iteration counts.
_REACH = 2.0


def add_slip_coarse_correction(pc, solver, hierarchy, fac_keys):
    """Add the slip correction to the finest smoother of the velocity PCMG ``pc``
    of a rotated saddle-point solve, once its FAC blocks are set up
    (``fac_keys`` from :func:`custom_mg._configure_pcmg`). Collective.

    The finest level must carry the FAC patch in the composite form (whole-level
    SOR, then the ASM pieces), and the patch must hold slip rows. 2-D only."""
    if solver.mesh.cdim != 2:
        # one column per piece would prescribe slip along t1 + t2, a direction
        # that jumps wherever the node-by-node tangent frame switches axis; a
        # 3-D column needs a slip direction that is smooth along the fault
        raise NotImplementedError(
            "slip_coarse_space is built for 2-D faults only; 3-D needs one "
            "column per slip direction in a frame smooth along the fault")
    nlev = pc.getMGLevels()
    if not any(k.startswith(f"mg_levels_{nlev - 1}_") for k in fac_keys or ()):
        raise ValueError(
            "slip_coarse_space needs the FAC band patch on the finest level, "
            "and this hierarchy has none")
    smoother = pc.getMGSmoother(nlev - 1)
    level_pc = smoother.getPC()
    if level_pc.getType() != "composite":
        raise ValueError(
            "slip_coarse_space needs the composite FAC smoother on the finest "
            f"level; it is {level_pc.getType()!r} (UW_FAC_MODE)")
    A = smoother.getOperators()[0]
    correction = SlipCoarseCorrection(
        solver, A, hierarchy.level_patch_rows[nlev - 1], hierarchy.field_id)
    if correction.n_columns == 0:
        raise ValueError(
            "slip_coarse_space: the finest band patch holds no slip rows "
            "(no split fault in the patch)")
    level_pc.addCompositePCType("python")
    # the composite's members are [sor, asm] (_configure_pcmg); this is the third
    member = level_pc.getCompositePC(2)
    member.setOperators(A, A)
    member.setPythonContext(correction)
    member.setUp()
    return correction


class SlipCoarseCorrection:
    r"""The PETSc python-PC context of the slip correction,
    :math:`y = Z (Z^T A Z)^{-1} Z^T x`.

    Construction fixes the geometry, which does not change with the operator:
    each column's rows and its slip weights. :meth:`setUp` runs whenever the
    operator's values change (every Newton iterate) and rebuilds the extension,
    :math:`A_c` and its factorisation from the current operator. Every rank
    makes every collective call, including the ranks that hold no slip rows and
    so no columns.

    ``patch_rows`` is the finest level's FAC patch (``level_patch_rows``): one
    ``(owned, subdomain)`` pair or a list of them, in the velocity block's
    numbering."""

    def __init__(self, solver, A, patch_rows, field_id):
        mpi_comm = A.getComm().tompi4py()
        rstart, rend = A.getOwnershipRange()
        blocks = patch_rows if isinstance(patch_rows, list) else [patch_rows]
        pieces = [np.asarray(sub, dtype=np.int64) for _owned, sub in blocks]

        def owned(rows):
            return rows[(rows >= rstart) & (rows < rend)]

        slip = self._owned_slip_rows(solver, A)
        band = np.union1d(np.unique(np.concatenate(
            [owned(p) for p in pieces] or [np.zeros(0, dtype=np.int64)])), slip)
        xy_of = self._row_coordinates(solver, field_id, rstart, rend)
        band_xy, slip_xy = xy_of(band), xy_of(slip)

        all_slip_xy = np.concatenate(
            [c for c in mpi_comm.allgather(slip_xy) if c.size]
            or [np.zeros((0, band_xy.shape[1]))])
        slip_tree = kdtree.KDTree(all_slip_xy) if all_slip_xy.shape[0] else None
        # this rank's pieces that carry slip: the centre and radius of their
        # slip nodes, the radius at least the distance to the nearest other
        # slip node (a piece can hold a single slip node where a rank boundary
        # clips the fault, and slip nodes can coincide at a junction), and
        # the piece's band half-width, its farthest row from the fault. A
        # global half-width would take the structural rows a rank holds far
        # from any fault, and every extension would cover the whole band.
        centres, radii, widths = [], [], []
        for piece in pieces:
            own_slip = np.isin(slip, piece)
            if not own_slip.any():
                continue
            c = slip_xy[own_slip].mean(axis=0)
            R = float(np.linalg.norm(slip_xy[own_slip] - c, axis=1).max())
            if all_slip_xy.shape[0] > 1:
                near = np.ravel(slip_tree.query(
                    c[None, :], k=min(4, all_slip_xy.shape[0]),
                    sqr_dists=False)[0])
                if np.any(near > 0):
                    R = max(R, float(near[near > 0].min()))
            centres.append(c)
            radii.append(R)
            widths.append(float(slip_tree.query(xy_of(owned(piece)), k=1,
                                                sqr_dists=False)[0].max()))
        table = mpi_comm.allgather(
            (np.asarray(centres), np.asarray(radii), np.asarray(widths)))
        counts = [len(r) for _c, r, _w in table]
        self.n_columns = sum(counts)
        if self.n_columns == 0:
            return
        offset = sum(counts[:mpi_comm.rank])
        C = np.concatenate([c for c, r, _w in table if len(r)])
        R = np.concatenate([r for _c, r, _w in table if len(r)])
        W = np.concatenate([w for _c, r, w in table if len(r)])

        # Shepard weights on this rank's slip rows
        phi = np.zeros((self.n_columns, slip.size))
        for q in range(self.n_columns):
            d = np.linalg.norm(slip_xy - C[q], axis=1)
            # R stays zero only when every slip node sits on the centre, and
            # the weight is then one there
            ratio = (d / (_REACH * R[q]) if R[q] > 0
                     else np.where(d > 0, np.inf, 0.0))
            phi[q] = np.clip(1.0 - ratio, 0.0, None) ** 2
        total = phi.sum(axis=0)
        chi = np.divide(phi, total, out=np.zeros_like(phi), where=total > 0)

        # Membership M (band rows x columns), inserted by each row's owner: -1 on
        # the band rows a column extends into, the slip weight on its slip rows.
        # Its transpose hands each column's rows to the rank that owns the column.
        # The pattern of M is the pattern of Z.
        is_slip = np.isin(band, slip)
        slip_pos = np.searchsorted(slip, band)
        entries = []                                  # (rows, column, values)
        for q in range(self.n_columns if band.size else 0):
            reach = _REACH * R[q] + W[q]
            if (np.any(C[q] + reach < band_xy.min(axis=0))
                    or np.any(C[q] - reach > band_xy.max(axis=0))):
                continue
            inside = np.flatnonzero(np.linalg.norm(band_xy - C[q], axis=1) <= reach)
            if not inside.size:
                continue
            values = np.full(inside.size, -1.0)
            on_slip = is_slip[inside]
            values[on_slip] = chi[q, slip_pos[inside][on_slip]]
            entries.append((band[inside], q, values))
        nloc = rend - rstart
        d_nnz = np.zeros(nloc, dtype=PETSc.IntType)
        o_nnz = np.zeros(nloc, dtype=PETSc.IntType)
        mine = range(offset, offset + counts[mpi_comm.rank])
        for rows, q, _v in entries:
            (d_nnz if q in mine else o_nnz)[rows - rstart] += 1
        sizes = ((nloc, A.getSize()[0]), (counts[mpi_comm.rank], self.n_columns))
        M = PETSc.Mat().createAIJ(sizes, nnz=(d_nnz, o_nnz), comm=A.getComm())
        for rows, q, values in entries:
            M.setValues(rows.astype(PETSc.IntType), [q], values.reshape(-1, 1))
        M.assemble()
        self.Z = M.duplicate(copy=False)
        # a new matrix: petsc4py's transpose() without one transposes in place
        Mt = M.transpose(PETSc.Mat())
        # each owned column: its extension rows I_q and slip rows with weights
        self._columns = []
        for q in mine:
            rows, values = Mt.getRow(q)
            rows = np.asarray(rows, dtype=np.int64)
            interior = values < 0
            self._columns.append((q, rows[interior], rows[~interior],
                                  np.asarray(values[~interior])))
        Mt.destroy()
        M.destroy()
        self.Ac = None
        self._ksp = None
        self._c = self.Z.createVecRight()
        self._t = self.Z.createVecRight()
        self._to_all, self._t_all = PETSc.Scatter.toAll(self._t)
        self._c_all = self._t_all.duplicate()

    @staticmethod
    def _owned_slip_rows(solver, A):
        """This rank's slip rows in the velocity block's numbering: the
        fieldsplit's velocity index set lists the owned velocity rows of the DM
        in block order."""
        vis = np.asarray(solver._subdict["velocity"][0].getIndices(), dtype=np.int64)
        rows = fault_contact._fault_slip_rows(solver)
        if not (vis.size and rows.size):
            return np.zeros(0, dtype=np.int64)
        order = np.argsort(vis)
        pos = np.minimum(np.searchsorted(vis[order], rows), vis.size - 1)
        found = vis[order][pos] == rows
        return np.sort(A.getOwnershipRange()[0] + order[pos[found]])

    @staticmethod
    def _row_coordinates(solver, field_id, rstart, rend):
        """A map from this rank's velocity-block rows to their node coordinates,
        through the level layout (the block's numbering) as the FAC patch reads
        them."""
        var = solver.Unknowns.u
        X = np.asarray(solver.mesh._get_coords_for_basis(
            var.degree, getattr(var, "continuous", True)))
        local_of_row = custom_mg._level_dof_layout(
            solver.dm, field_id).local_of_owned_rows()

        def xy_of(rows):
            local = local_of_row[np.asarray(rows, dtype=np.int64) - rstart]
            return X[local // var.num_components]

        return xy_of

    def _extend(self, A):
        """The columns' values from the operator ``A``: each slip weight and its
        discrete harmonic extension. Collective: one submatrix extraction per
        rank, over this rank's (possibly empty) list of columns."""
        def index_set(rows):
            return PETSc.IS().createGeneral(np.asarray(rows, dtype=PETSc.IntType),
                                            comm=PETSc.COMM_SELF)

        is_I = [index_set(I) for _q, I, _S, _g in self._columns]
        is_S = [index_set(S) for _q, _I, S, _g in self._columns]
        A_II = A.createSubMatrices(is_I, is_I)
        A_IS = A.createSubMatrices(is_I, is_S)
        values = []
        for k, (q, I, S, g) in enumerate(self._columns):
            ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorShift(shift_type="nonzero")
            ksp.setOperators(A_II[k])
            rhs, z = A_IS[k].createVecLeft(), A_II[k].createVecRight()
            gv = A_IS[k].createVecRight()
            gv.setArray(g)
            A_IS[k].mult(gv, rhs)
            rhs.scale(-1.0)
            ksp.solve(rhs, z)
            values.append((q, np.concatenate([S, I]),
                           np.concatenate([g, z.getArray()])))
            for obj in (ksp, rhs, z, gv):
                obj.destroy()
        for obj in list(A_II) + list(A_IS) + is_I + is_S:
            obj.destroy()
        return values

    def setUp(self, pc):
        A = pc.getOperators()[0]
        self.Z.zeroEntries()
        for q, rows, values in self._extend(A):
            self.Z.setValues(rows.astype(PETSc.IntType), [q], values.reshape(-1, 1))
        self.Z.assemble()
        self.Ac = A.ptap(self.Z, result=self.Ac)
        # every rank factors its own copy of the small coarse operator
        everything = PETSc.IS().createStride(self.n_columns, comm=PETSc.COMM_SELF)
        Ac_copy = self.Ac.createSubMatrices(everything, everything)[0]
        everything.destroy()
        if self._ksp is None:
            self._ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            self._ksp.setType("preonly")
            self._ksp.getPC().setType("lu")
            self._ksp.getPC().setFactorShift(shift_type="nonzero")
        self._ksp.setOperators(Ac_copy)
        self._ksp.setUp()
        Ac_copy.destroy()

    def apply(self, pc, x, y):
        self.Z.multTranspose(x, self._t)
        self._to_all.scatter(self._t, self._t_all,
                             addv=PETSc.InsertMode.INSERT_VALUES,
                             mode=PETSc.ScatterMode.FORWARD)
        self._ksp.solve(self._t_all, self._c_all)
        start, end = self._c.getOwnershipRange()
        self._c.setArray(self._c_all.getArray()[start:end])
        self.Z.mult(self._c, y)
