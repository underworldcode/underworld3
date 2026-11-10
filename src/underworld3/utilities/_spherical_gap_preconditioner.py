"""Distributed norm-block preconditioner for the unchanged spherical mixed operator.

The positive velocity norm is background viscous energy plus consistent mass on
physically supported cells. Cut cells retain their full background norm; wholly
void cells contribute nothing. Unsupported velocity and pressure entries receive
identity completion without changing the physical mixed operator or its RHS.
P1 pressure mass Jacobi is spectrally equivalent to its consistent mass; the
multiplier diagonal is an explicitly supplied h-scaled surface-mass proxy.
This proxy has no established uniform H^{-1/2} equivalence. Outer convergence
must be checked against the original operator and its unpreconditioned residual.
"""

import math
import time

import numpy as np

from petsc4py import PETSc

from underworld3.utilities.multigrid_options import gamg_bundle
from underworld3.utilities._tetrahedral_gap import _tetrahedral_p2_basis


class _SphericalGapNormPreconditioner:
    """Python-PC context; all supplied distributed matrices/vectors/IS are borrowed.

    The three index sets partition the original mixed vector. H must retain
    interleaved three-component velocity nodes. Near modes are actual field-layout rigid
    motions. A verified physical nullspace is used to keep PC outputs in the
    quotient space of the singular original operator. This context performs no
    field/boundary gathers, full-system LU, rotation of equations or filtering.
    """

    def __init__(self, velocity_norm, pressure_diagonal, multiplier_diagonal,
                 velocity_is, pressure_is, multiplier_is, velocity_near_modes,
                 options_prefix, physical_nullspace):
        if velocity_norm.getSize()[0] % 3 or velocity_norm.getLocalSize()[0] % 3:
            raise ValueError("Velocity norm ownership must preserve three-component nodes.")
        for diagonal in (pressure_diagonal, multiplier_diagonal):
            if not math.isfinite(diagonal.norm()) or diagonal.min()[1] <= 0:
                raise ValueError("Norm-block diagonals must be finite and strictly positive.")
        self.h = velocity_norm
        self.physical_nullspace = physical_nullspace
        self.diagonals = (pressure_diagonal, multiplier_diagonal)
        self.index_sets = (velocity_is, pressure_is, multiplier_is)
        self.modes = []
        for original in velocity_near_modes:
            vector = original.copy()
            for previous in self.modes:
                vector.axpy(-previous.dot(vector), previous)
            length = vector.norm()
            if length < 1e-13:
                vector.destroy()
                raise ValueError("Velocity near modes must be linearly independent.")
            vector.scale(1/length)
            self.modes.append(vector)
        self.near_space = PETSc.NullSpace().create(vectors=self.modes, constant=False, comm=self.h.comm)
        self.h.setBlockSize(3)
        self.h.setNearNullSpace(self.near_space)
        self.velocity_ksp = PETSc.KSP().create(comm=self.h.comm)
        self.velocity_ksp.setOperators(self.h)
        self.velocity_ksp.setOptionsPrefix(options_prefix)
        self.options = PETSc.Options(options_prefix)
        self.keys = []
        for key, value in gamg_bundle(block_size=3).settings.items():
            if key != "mat_block_size":  # Applied explicitly to the borrowed velocity matrix.
                self.options[key] = value
                self.keys.append(key)
        self.options["ksp_type"] = "preonly"
        self.keys.append("ksp_type")
        self.velocity_ksp.setFromOptions()
        self.velocity_ksp.setUp()
        self.applications, self.closed = 0, False

    def apply(self, pc, rhs, result):
        observer = getattr(self, "observer", None)
        if observer is not None:
            observer.pc_started(self.applications+1)
        result.set(0)
        for block, index_set in enumerate(self.index_sets):
            source = rhs.getSubVector(index_set)
            target = result.getSubVector(index_set)
            try:
                if block == 0:
                    self.velocity_ksp.solve(source, target)
                    if self.velocity_ksp.getConvergedReason() <= 0:
                        raise ArithmeticError("Velocity norm GAMG application failed.")
                else:
                    target.pointwiseDivide(source, self.diagonals[block-1])
            finally:
                result.restoreSubVector(index_set, target)
                rhs.restoreSubVector(index_set, source)
        self.physical_nullspace.remove(result)
        self.applications += 1
        if observer is not None:
            observer.pc_finished(self.applications)

    def destroy(self, pc):
        if not self.closed:
            self.velocity_ksp.destroy()
            self.near_space.destroy()
            for vector in self.modes:
                vector.destroy()
            for key in self.keys:
                self.options.delValue(key)
            self.closed = True


def _velocity_norm_graph(matrix, velocity_size, local_velocity_size):
    """Copy only the distributed velocity graph into the field's ownership.

    The mixed matrix and velocity field have different row partitions. PETSc's
    general submatrix extraction consequently retains a global column index
    set on every rank. Route owned graph rows through a preallocator instead;
    temporary storage is one sparse row and no numeric mixed entries are copied.
    Explicit zero connections, including curtain couplings, are preserved.
    """
    graph = PETSc.Mat().create(comm=matrix.comm)
    try:
        sizes = ((local_velocity_size, velocity_size),) * 2
        graph.setSizes(sizes)
        graph.setType(PETSc.Mat.Type.PREALLOCATOR)
        graph.setUp()
        lower, upper = matrix.getOwnershipRange()
        for row in range(lower, min(upper, velocity_size)):
            columns, _ = matrix.getRow(row)
            columns = columns[columns < velocity_size]
            graph.setValues(row, columns, np.zeros(len(columns), dtype=PETSc.ScalarType))
        graph.assemble()
        result = PETSc.Mat().create(comm=matrix.comm)
        result.setSizes(sizes)
        result.setType(PETSc.Mat.Type.AIJ)
        graph.preallocatorPreallocate(result, fill=True)
    finally:
        graph.destroy()
    result.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    result.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
    return result


class _SphericalGapNormBlocks:
    """Own active-background norm blocks without boundary replication.

    The spherical assembler supplies only physically supported cells. Exact-zero
    H rows and pressure diagonals represent unsupported unknowns; identity
    completion gives these algebraic entries a positive norm without a penalty.
    """

    def __init__(self, velocity, matrix, nu, npres, multiplier_count, components, length):
        started = time.perf_counter()
        self.nu, self.components, self.length = nu, components, length
        field_is, field_dm = velocity.mesh.dm.createSubDM(velocity.field_id)
        layout = field_dm.createGlobalVec()
        lo, hi = layout.getOwnershipRange()
        layout.destroy(); field_dm.destroy(); field_is.destroy()
        self.velocity_is = PETSc.IS().createStride(hi-lo, first=lo, step=1, comm=matrix.comm)
        matrix.assemble()
        self.h = _velocity_norm_graph(matrix, nu, hi-lo)
        self.pressure = PETSc.Vec().createMPI(npres, comm=matrix.comm); self.pressure.set(0)
        self.multiplier = PETSc.Vec().createMPI(multiplier_count, comm=matrix.comm); self.multiplier.set(0)
        lo, hi = self.pressure.getOwnershipRange()
        self.pressure_is = PETSc.IS().createStride(hi-lo, first=nu+lo, step=1, comm=matrix.comm)
        lo, hi = self.multiplier.getOwnershipRange()
        self.multiplier_is = PETSc.IS().createStride(hi-lo, first=nu+npres+lo, step=1, comm=matrix.comm)
        self.multiplier_offset = nu+npres
        self.closed, self.assembly_seconds = False, time.perf_counter()-started
        self._completed = False

    def add_volume(self, vertices, points, weights, uids, pids, stiffness):
        if self._completed:
            raise RuntimeError("Norm contributions must precede assembly completion.")
        started = time.perf_counter()
        shape, _, bary = _tetrahedral_p2_basis(vertices, points)
        mass = np.einsum("eq,eqi,eqj->eij", weights, shape, shape)
        values = stiffness.copy()
        for component in range(3):
            values[:, component::3, component::3] += mass
        for u, value in zip(uids, values, strict=True):
            self.h.setValues(u, u, np.ascontiguousarray(value), addv=PETSc.InsertMode.ADD_VALUES)
        diagonal = np.einsum("eq,eql,eql->el", weights, bary, bary)
        self.pressure.setValues((pids-self.nu).ravel(), diagonal.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
        self.assembly_seconds += time.perf_counter()-started

    def add_surface(self, weights, multiplier, lids):
        if self._completed:
            raise RuntimeError("Norm contributions must precede assembly completion.")
        started = time.perf_counter()
        diagonal = self.length*np.einsum("eq,eql,eql->el", weights, multiplier, multiplier)
        diagonal = np.repeat(diagonal[..., None], self.components, axis=2)
        self.multiplier.setValues((lids-self.multiplier_offset).ravel(), diagonal.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
        self.assembly_seconds += time.perf_counter()-started

    def assemble(self):
        self.h.assemble()
        self.pressure.assemble()
        self.multiplier.assemble()
        if self._completed:
            return
        diagonal = self.h.getDiagonal()
        try:
            minimum = diagonal.min()[1]
            if not math.isfinite(diagonal.norm()) or minimum < 0:
                raise ArithmeticError("Active-background velocity norm has an invalid diagonal.")
            if minimum == 0:
                lower, _ = diagonal.getOwnershipRange()
                rows = np.flatnonzero(diagonal.getArray(readonly=True) == 0).astype(PETSc.IntType)+lower
                nonzero_rows = 0
                for row in rows:
                    _, values = self.h.getRow(int(row))
                    nonzero_rows += int(np.count_nonzero(values) != 0)
                # Every rank must agree before entering the collective row/column
                # completion, including ranks with no owned unsupported entries.
                if self.h.comm.tompi4py().allreduce(nonzero_rows):
                    raise ArithmeticError("A zero norm diagonal has a nonzero row.")
                self.h.zeroRowsColumns(rows, diag=1)
        finally:
            diagonal.destroy()
        minimum = self.pressure.min()[1]
        if not math.isfinite(self.pressure.norm()) or minimum < 0:
            raise ArithmeticError("Active-background pressure norm has an invalid diagonal.")
        if minimum == 0:
            lower, _ = self.pressure.getOwnershipRange()
            rows = np.flatnonzero(self.pressure.getArray(readonly=True) == 0).astype(PETSc.IntType)+lower
            self.pressure.setValues(rows, np.ones(len(rows)))
            self.pressure.assemble()
        self._completed = True

    def context(self, umap, coordinates, nodal_rotations, physical_nullspace, prefix):
        near = []
        try:
            for component in range(6):
                vector = self.h.createVecRight(); vector.set(0)
                values = np.zeros_like(coordinates)
                if component < 3:
                    values[:, component] = 1
                else:
                    values[:] = nodal_rotations[:, component-3].reshape(-1, 3)
                vector.setValues(umap, values.ravel()); vector.assemble()
                near.append(vector)
            return _SphericalGapNormPreconditioner(
                self.h, self.pressure, self.multiplier,
                self.velocity_is, self.pressure_is, self.multiplier_is,
                near, prefix, physical_nullspace,
            )
        finally:
            for vector in near:
                vector.destroy()

    def destroy(self):
        if not self.closed:
            for obj in (self.h, self.pressure, self.multiplier,
                        self.velocity_is, self.pressure_is, self.multiplier_is):
                obj.destroy()
            self.closed = True
