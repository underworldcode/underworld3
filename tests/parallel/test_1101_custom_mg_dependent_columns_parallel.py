"""Coarse unknowns with the same single fine image are dropped at install (#1101).

When the fine level moves far from coarse levels that did not move with it, two
coarse unknowns can be left whose prolongation columns each have one entry, on
the same fine row. The columns are proportional, so their Galerkin rows are,
and the coarse operator is singular. In test_1152 (a mover after #1024) this
happened on about half of the 1e-13 coordinate perturbations: SOR's inode block
over the two rows met a zero pivot. ``fault_mg.restrict_unreached`` now keeps
the largest-weight column of such a group and drops the others (the range of P
is unchanged), folds each dropped unknown's row of the level below into its
kept twin's, so the composite coarse functions are unchanged (a merge: they
still reproduce constants through the shared row), warns, and refuses a support of s > 1 fine rows shared by more
than s columns. A hierarchy with no such pair comes back untouched.

The configuration is built directly: a 1-D Laplacian, P a linear interpolation
plus two extra columns with one entry each, both on fine row 1. Rank 0 owns
that row; the two columns are owned by the last rank, so at np 2 and 3 the
shared fine row is on another rank. This file carries no ``mpi(min_size=2)``
mark: it runs serially and in the parallel batches.

    python -m pytest tests/parallel/test_1101_custom_mg_dependent_columns_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1101_custom_mg_dependent_columns_parallel.py
"""
import warnings

import numpy as np
import pytest
from petsc4py import PETSc

import underworld3  # noqa: F401  (initialises PETSc as the package does)
from underworld3.utilities.fault_mg import restrict_unreached

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(120)]

COMM = PETSc.COMM_WORLD
RANK, SIZE = COMM.getRank(), COMM.getSize()
ROWS_PER_RANK = 6                       # fine rows on each rank
N_FINE = ROWS_PER_RANK * SIZE
N_LINEAR = N_FINE // 2                  # coarse nodes at the even fine rows
SHARED_ROW = 1                          # owned by rank 0
EXTRA = [N_LINEAR, N_LINEAR + 1]        # the extra columns, owned by the last rank


def _mat(nrows, ncols, local_rows, local_cols, entries):
    """An MPIAIJ matrix; each rank sets the entries of the rows it owns."""
    A = PETSc.Mat().createAIJ(((local_rows, nrows), (local_cols, ncols)), comm=COMM)
    A.setPreallocationNNZ(8)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    rstart, rend = A.getOwnershipRange()
    for (i, j), v in entries.items():
        if rstart <= i < rend:
            A.setValue(i, j, v)
    A.assemble()
    return A


def _linear(n_fine):
    entries = {}
    for i in range(n_fine):
        j = i // 2
        if i % 2 == 0:
            entries[i, j] = 1.0
        else:
            entries[i, j] = 0.5
            if j + 1 < n_fine // 2:
                entries[i, j + 1] = 0.5
    return entries


def _coarse_layout(extra):
    cols = [N_LINEAR // SIZE] * SIZE
    cols[-1] += N_LINEAR - sum(cols) + extra
    return cols[RANK]


def _laplacian():
    entries = {}
    for i in range(N_FINE):
        entries[i, i] = 2.0
        if i > 0:
            entries[i, i - 1] = -1.0
        if i + 1 < N_FINE:
            entries[i, i + 1] = -1.0
    return _mat(N_FINE, N_FINE, ROWS_PER_RANK, ROWS_PER_RANK, entries)


def _prolongation(extra_entries):
    """Linear interpolation, then one column per extra entry ``(row, weight)``."""
    entries = _linear(N_FINE)
    for k, (row, weight) in enumerate(extra_entries):
        if isinstance(row, tuple):
            for r in row:
                entries[r, N_LINEAR + k] = weight
        else:
            entries[row, N_LINEAR + k] = weight
    n = N_LINEAR + len(extra_entries)
    return _mat(N_FINE, n, ROWS_PER_RANK, _coarse_layout(len(extra_entries)), entries)


def _dependent_pair():
    # weights 0.5 and 1.0: the second (the larger) is the one kept
    return _prolongation([(SHARED_ROW, 0.5), (SHARED_ROW, 1.0)])


def _smallest_eigenvalue(A):
    """Of a small SPD-or-singular matrix, gathered (collective)."""
    rows = []
    rstart, rend = A.getOwnershipRange()
    n = A.getSize()[1]
    for i in range(rstart, rend):
        cols, vals = A.getRow(i)
        row = np.zeros(n)
        row[cols] = vals
        rows.append(row)
    full = np.vstack([r for part in COMM.tompi4py().allgather(rows) for r in part])
    return float(np.linalg.eigvalsh(0.5 * (full + full.T)).min())


def _gathered(A):
    """Every row of a small matrix, dense, on every rank (collective)."""
    n = A.getSize()[1]
    rows = []
    for i in range(*A.getOwnershipRange()):
        cols, vals = A.getRow(i)
        row = np.zeros(n)
        row[cols] = vals
        rows.append(row)
    return np.vstack([r for part in COMM.tompi4py().allgather(rows) for r in part])


def _restrict(Ps):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = restrict_unreached(Ps, COMM)
    return out, [str(w.message) for w in caught if "#1101" in str(w.message)]


def test_the_case_is_the_one_described():
    P = _dependent_pair()
    owner = COMM.tompi4py().allgather(P.getOwnershipRangeColumn())
    on_last = [r for r, (s, e) in enumerate(owner) if s <= EXTRA[0] and EXTRA[1] < e]
    assert on_last == [SIZE - 1]
    assert P.getOwnershipRanges()[1] > SHARED_ROW          # rank 0 owns the row
    A = _laplacian()
    # unrepaired, the Galerkin operator is singular
    assert abs(_smallest_eigenvalue(A.ptap(P))) < 1e-12


def test_the_smaller_column_is_dropped_and_the_operator_is_nonsingular():
    P, A = _dependent_pair(), _laplacian()
    (T,), messages = _restrict([P])
    assert T.getSize() == (N_FINE, N_LINEAR + 1)
    # the column kept is the weight-1.0 one, on the shared row
    gathered = COMM.tompi4py().allgather(
        [(i, *T.getRow(i)) for i in range(*T.getOwnershipRange())])
    shared = [(c, v) for part in gathered for i, cs, vs in part if i == SHARED_ROW
              for c, v in zip(cs, vs)]
    assert (N_LINEAR, 1.0) in [(int(c), float(v)) for c, v in shared]
    assert _smallest_eigenvalue(A.ptap(T)) > 1e-3
    if RANK == 0:
        assert len(messages) == 1 and "1 coarse unknown(s)" in messages[0] \
            and "merged" in messages[0], messages
    else:
        assert messages == []


def test_a_three_level_hierarchy_solves_with_sor():
    """The level below the drop loses the dropped row, and PCMG with Galerkin
    coarse operators and SOR smoothing converges (on development the coarse
    factorisation met a zero pivot)."""
    P1, A = _dependent_pair(), _laplacian()
    n1 = N_LINEAR + 2
    P0 = _mat(n1, n1, _coarse_layout(2), _coarse_layout(2), {(i, i): 1.0 for i in range(n1)})
    out, _ = _restrict([P0, P1])

    prefix = "t1101_"
    options = PETSc.Options()
    settings = {"ksp_type": "cg", "ksp_rtol": 1e-10, "ksp_max_it": 50,
                "pc_type": "mg", "pc_mg_levels": 3, "pc_mg_galerkin": "both",
                "mg_levels_ksp_type": "richardson", "mg_levels_pc_type": "sor",
                "mg_coarse_ksp_type": "preonly", "mg_coarse_pc_type": "redundant"}
    for key, value in settings.items():
        options[prefix + key] = value
    ksp = PETSc.KSP().create(comm=COMM)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(A)
    ksp.setFromOptions()
    pc = ksp.getPC()
    for level, P in enumerate(out, start=1):
        pc.setMGInterpolation(level, P)
    for key in settings:
        options.delValue(prefix + key)
    b = A.createVecLeft()
    b.set(1.0)
    x = A.createVecRight()
    ksp.solve(b, x)
    assert ksp.getConvergedReason() > 0, ksp.getConvergedReason()
    r = b.duplicate()
    A.mult(x, r)
    r.axpy(-1.0, b)
    assert r.norm() < 1e-8 * b.norm()
    assert out[1].getSize() == (N_FINE, n1 - 1)
    assert out[0].getSize() == (n1 - 1, n1 - 1)


def test_the_composite_coarse_functions_are_unchanged():
    """The merge: the level below folds the dropped unknown's row into the kept
    one's, so P1' P0' is P1 P0, and constants on the coarsest level still reach
    the shared fine row at full weight (a plain drop loses the 0.5)."""
    P1 = _dependent_pair()
    n1 = N_LINEAR + 2
    rng = np.random.default_rng(1101)
    entries = {(i, j): float(rng.uniform(0.1, 1.0))
               for i in range(n1) for j in range(n1) if abs(i - j) <= 1}
    P0 = _mat(n1, n1, _coarse_layout(2), _coarse_layout(2), entries)
    out, _ = _restrict([P0, P1])
    assert out[1].getSize()[1] == n1 - 1 and out[0].getSize()[0] == n1 - 1
    np.testing.assert_allclose(_gathered(out[1].matMult(out[0])),
                               _gathered(P1.matMult(P0)), rtol=0, atol=1e-14)


def test_no_dependent_pair_leaves_the_transfer_untouched():
    P = _prolongation([(SHARED_ROW, 1.0)])      # one extra column: nothing shares
    (T,), messages = _restrict([P])
    assert T is P
    assert messages == []


def test_more_columns_than_shared_rows_is_refused_on_every_rank():
    P = _prolongation([((1, 3), 1.0), ((1, 3), 0.5), ((1, 3), 0.25)])
    raised = 0
    try:
        restrict_unreached([P], COMM)
    except RuntimeError as err:
        raised = int("#1101" in str(err))
    assert COMM.tompi4py().allgather(raised) == [1] * SIZE
