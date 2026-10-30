"""The slip coarse space of the fault band smoother (``slip_coarse_space``).

``set_custom_fmg(..., slip_coarse_space=True)`` cuts a split fault's structural
FAC patch at ``fac_block_rows`` and adds a coarse correction whose unknowns are
slip, ``y = Z (Z^T A Z)^{-1} Z^T r``, as the third member of the finest
smoother's composite (``utilities/fault_slip_coarse.py``; design note
``docs/developer/design/fault-slip-coarse-space-2026-10.md``). This file checks,
on the live objects after a rotated saddle-point solve:

* the finest smoother is the composite of SOR, the ASM pieces and the slip
  correction, and the patch was cut;
* the columns' slip weights are a partition of unity: ``Z`` summed over its
  columns is exactly one on every slip row;
* each column is the discrete harmonic extension of its slip weights:
  ``(A z_q)`` vanishes on the band rows the column extends into;
* the correction changes the preconditioner, not the answer: the slip matches
  the solve without it;
* the flag is refused on the standard solve path, which has no slip rows.

The fixture is test_1090's: a fault split in place over a welded base hierarchy
(1/8, 1/16), with frictionless contact. The file carries no ``mpi(min_size=2)``
mark, so it runs in the serial suite and in the parallel batches; at np 4 some
ranks hold no slip rows, which is the case where a skipped collective call hangs
the other ranks.

    python -m pytest tests/parallel/test_1091_fault_slip_coarse_space_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1091_fault_slip_coarse_space_parallel.py
"""
import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg, fault_slip_coarse
from underworld3.utilities.fault_contact import fault_slip

pytestmark = pytest.mark.level_2

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])
BLOCK_ROWS = 150


@pytest.fixture(scope="module")
def base():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=False,
        qdegree=2, refinement=1)


def _solve(base, slip_coarse_space, fault_bc=True):
    """Solve the split fixture; return the solver."""
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = base.add_fault(("Flt", TRACE))
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
    if fault_bc:
        stokes.add_fault_bc(0, "Flt")
    stokes.petsc_use_pressure_nullspace = True
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.tolerance = 1e-8
    custom_mg.set_custom_fmg(stokes, levels, field_id=0, fac_block_rows=BLOCK_ROWS,
                             slip_coarse_space=slip_coarse_space)
    stokes.solve()
    return stokes


@pytest.fixture(scope="module")
def solved(base):
    stokes = _solve(base, slip_coarse_space=True)
    info = stokes._rotated_freeslip_info
    assert info["converged"], "the rotated solve with the slip correction did not converge"
    velpc = (stokes._rotated_linear_cache["ctx"]["ksp"].getPC()
             .getFieldSplitSubKSP()[0].getPC())
    return stokes, velpc


def _finest(velpc):
    smoother = velpc.getMGSmoother(velpc.getMGLevels() - 1)
    return smoother.getPC(), smoother.getOperators()[0]


@pytest.mark.tier_b
def test_finest_smoother_carries_the_slip_correction(solved):
    _stokes, velpc = solved
    level_pc, _A = _finest(velpc)
    assert level_pc.getType() == "composite"
    members = [level_pc.getCompositePC(i) for i in range(3)]
    assert [m.getType() for m in members] == ["sor", "asm", "python"]
    correction = members[2].getPythonContext()
    assert isinstance(correction, fault_slip_coarse.SlipCoarseCorrection)
    comm = uw.mpi.comm
    blocks = comm.allreduce(len(members[1].getASMSubKSP()))
    # the cut: the uncut structural patch is one block per rank
    assert blocks > comm.size, f"the band was not cut: {blocks} blocks"
    assert correction.n_columns >= 2


@pytest.mark.tier_b
def test_slip_weights_are_a_partition_of_unity(solved):
    stokes, velpc = solved
    level_pc, A = _finest(velpc)
    correction = level_pc.getCompositePC(2).getPythonContext()
    slip = correction._owned_slip_rows(stokes, A)
    assert uw.mpi.comm.allreduce(slip.size) > 0
    ones, total = correction.Z.createVecs()
    ones.set(1.0)
    correction.Z.mult(ones, total)
    rstart = A.getOwnershipRange()[0]
    on_slip = total.getArray()[slip - rstart]
    worst = uw.mpi.comm.allreduce(
        float(np.abs(on_slip - 1.0).max()) if on_slip.size else 0.0, op=max)
    assert worst < 1e-12, f"slip weights sum to 1 within {worst:.2e}"


@pytest.mark.tier_b
def test_columns_are_discrete_harmonic_extensions(solved):
    _stokes, velpc = solved
    level_pc, A = _finest(velpc)
    correction = level_pc.getCompositePC(2).getPythonContext()
    W = A.matMult(correction.Z)

    def index_set(rows):
        return PETSc.IS().createGeneral(np.asarray(rows, dtype=PETSc.IntType),
                                        comm=PETSc.COMM_SELF)

    columns = correction._columns
    is_I = [index_set(I) for _q, I, _S, _g in columns]
    is_S = [index_set(S) for _q, _I, S, _g in columns]
    is_q = [index_set([q]) for q, _I, _S, _g in columns]
    # collective: every rank extracts once, over its own (possibly empty) columns
    on_interior = W.createSubMatrices(is_I, is_q)
    on_slip = W.createSubMatrices(is_S, is_q)
    def largest(mat):
        if mat.getSize()[0] == 0:
            return 0.0
        return float(np.abs(mat.convert("dense").getDenseArray()).max())

    worst = 0.0
    for interior, slip_part in zip(on_interior, on_slip):
        worst = max(worst, largest(interior) / largest(slip_part))
    worst = uw.mpi.comm.allreduce(worst, op=max)
    assert worst < 1e-9, f"(A z_q) on the extension rows, relative: {worst:.2e}"


@pytest.mark.tier_b
def test_the_correction_does_not_change_the_answer(base, solved):
    with_correction, _velpc = solved
    without = _solve(base, slip_coarse_space=False)
    assert without._rotated_freeslip_info["converged"]
    peaks = []
    for stokes in (with_correction, without):
        _, slip, _ = fault_slip(stokes, "Flt", stokes._rotated_freeslip_info)
        peaks.append(uw.mpi.comm.allreduce(
            float(np.abs(slip).max()) if len(slip) else 0.0, op=max))
    assert peaks[1] > 0
    assert abs(peaks[0] - peaks[1]) <= 1e-6 * peaks[1], peaks


@pytest.mark.tier_b
def test_the_standard_path_refuses_the_flag(base):
    with pytest.raises(NotImplementedError, match="slip_coarse_space"):
        _solve(base, slip_coarse_space=True, fault_bc=False)
