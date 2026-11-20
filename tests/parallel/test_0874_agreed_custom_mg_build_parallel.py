"""custom_mg.build agrees on failure: its set-up phase (#874).

``CustomMGHierarchy.build`` runs inside one ``agreed_build`` context
(``utilities.agreed_build``, #1218): a rank-local exception on one rank is raised
on every rank as the same ``CollectiveBuildError``, a builder failure (which the
mesh-owned path retries with the next builder), at the next agreement point or
at the build's end. This file covers the SET-UP phase, which has an agreement
point before each of its collectives: the node coordinates (a basis DM), the
level layouts (a DM clone), the allreduced node and DOF counts, the move check,
the operator check and the FAC kill check. The per-level transfer loop is the
next phase.

The hierarchy and the solver are those of a placed fault ribbon's solve (the
fixture of test_1147); ``build`` is called again with one failure forced on the
last rank, at a rank-local step just after a collective has returned:

* the first level's coordinates (as the build reads them);
* a coarse level's layout (its field read, between the layout and the
  allreduce);
* the move check's result;
* the FAC kill check's result, the last step of the set-up phase.

And the per-level LOOP's own body, which starts each level at an agreement point
and has one before each collective it makes:

* after a node-cloud transfer has been built (the placed ribbon's pairs);
* after the zero-column count;
* after the FAC patch blocks are gathered;
* after a split pair's cell-located transfer (an adapt child with a fault cut
  in, whose coarse add-on levels are split).

And the coarse side's move check (#1244): a refined mesh whose coarse levels
moved with the finest reads each pair's coarse side as built, one more
collective move check per level, with an agreement point before each.

Each must be the same ``CollectiveBuildError`` on every rank, with the agreement
calls per rank paired. Before the conversion each hangs the other ranks.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg, fault_mg
from underworld3.utilities.custom_mg import CollectiveBuildError, set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d
from fault_fixtures import TRACE, adapted_2d, shear_fault_solve

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600, method="thread")]

A, B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
N_RUNGS = 80


@pytest.fixture(scope="module")
def built():
    """``(hierarchy, solver)`` of a placed fault ribbon's solve."""
    S = float(np.linalg.norm(B - A)) / N_RUNGS
    trace = A + (np.arange(N_RUNGS + 1) * S)[:, None] * (B - A) / np.linalg.norm(B - A)
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 8, regular=False, qdegree=2,
                                             refinement=2)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = place_fault_ribbon_2d(base, [("Fault", trace)], 2 * S, spines=[("Fault", trace)],
                                 margin_rings=1, clearance=0.7, split=True,
                                 mesher="ladder")[0]
    _x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = [0.0, -1.0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, boundary="Fault")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    set_custom_fmg(stokes, levels, field_id=0)
    stokes.solve()
    return stokes._custom_mg["hierarchy"], stokes


class _CountingVerdicts:
    """The verdict communicator, counting its exchanges (the agreement calls)."""

    def __init__(self, comm):
        self._comm = comm
        self.exchanges = 0

    def allreduce(self, *a, **k):
        self.exchanges += 1
        return self._comm.allreduce(*a, **k)

    def allgather(self, *a, **k):
        self.exchanges += 1
        return self._comm.allgather(*a, **k)

    def __getattr__(self, name):
        return getattr(self._comm, name)


def _on_last_rank(message):
    if uw.mpi.rank == uw.mpi.size - 1:
        raise IndexError(f"{message} (forced on one rank)")


def _build_raises_everywhere(hierarchy, solver, message, monkeypatch):
    counter = _CountingVerdicts(uw.mpi.verdict_comm)
    monkeypatch.setattr(uw.mpi, "verdict_comm", counter)
    caught = None
    try:
        hierarchy.build(solver)
    except RuntimeError as err:
        assert isinstance(err, CollectiveBuildError), type(err)
        assert err.builder_failure, "a build failure must be retryable"
        caught = str(err)
    finally:
        monkeypatch.setattr(uw.mpi, "verdict_comm", counter._comm)
    messages = uw.mpi.comm.allgather(caught)
    counts = uw.mpi.comm.allgather(counter.exchanges)
    assert all(m is not None and message in m for m in messages), messages
    assert len(set(messages)) == 1, messages
    assert len(set(counts)) == 1, counts


class _RaisingArray:
    """A result that raises on the last rank when the build reads it."""

    def __init__(self, value, message):
        self._value, self._message = value, message

    def __array__(self, dtype=None, copy=None):
        _on_last_rank(self._message)
        return np.asarray(self._value)


def test_the_build_still_gives_the_same_transfers(built, monkeypatch):
    hierarchy, solver = built
    before = [P.getSize() for P in hierarchy.transfers]
    counter = _CountingVerdicts(uw.mpi.verdict_comm)
    monkeypatch.setattr(uw.mpi, "verdict_comm", counter)
    try:
        Ps = hierarchy.build(solver)
    finally:
        monkeypatch.setattr(uw.mpi, "verdict_comm", counter._comm)
    assert [P.getSize() for P in Ps] == before
    counts = uw.mpi.comm.allgather(counter.exchanges)
    assert len(set(counts)) == 1 and counts[0] > 0, counts


def test_a_failure_reading_the_coordinates(built, monkeypatch):
    hierarchy, solver = built
    mesh = hierarchy.level_meshes[0]
    original = mesh._get_coords_for_basis
    monkeypatch.setattr(mesh, "_get_coords_for_basis",
                        lambda *a, **k: _RaisingArray(original(*a, **k), "Coordinates"))
    _build_raises_everywhere(hierarchy, solver, "Coordinates", monkeypatch)


def test_a_failure_reading_a_coarse_layout(built, monkeypatch):
    hierarchy, solver = built
    original = custom_mg._coarse_dof_layout

    class _Layout:
        def __init__(self, lay):
            self._lay = lay

        @property
        def n_full(self):
            _on_last_rank("Coarse layout")
            return self._lay.n_full

        def __getattr__(self, name):
            return getattr(self._lay, name)

    monkeypatch.setattr(custom_mg, "_coarse_dof_layout",
                        lambda *a, **k: _Layout(original(*a, **k)))
    _build_raises_everywhere(hierarchy, solver, "Coarse layout", monkeypatch)


def test_a_failure_after_the_move_check(built, monkeypatch):
    hierarchy, solver = built
    original = fault_mg.as_built_nodes

    def failing(*a, **k):
        out = original(*a, **k)             # the collective has returned
        _on_last_rank("Move check")
        return out

    monkeypatch.setattr(fault_mg, "as_built_nodes", failing)
    _build_raises_everywhere(hierarchy, solver, "Move check", monkeypatch)


def test_a_failure_at_the_end_of_the_set_up_phase(built, monkeypatch):
    hierarchy, solver = built
    original = type(hierarchy)._fac_patch_killed

    def failing(self, solver_):
        out = original(self, solver_)
        _on_last_rank("FAC kill check")
        return out

    monkeypatch.setattr(type(hierarchy), "_fac_patch_killed", failing)
    _build_raises_everywhere(hierarchy, solver, "FAC kill check", monkeypatch)



@pytest.fixture(scope="module")
def split_built():
    """``(hierarchy, solver)`` of an adapt child with a fault cut in: its coarse
    add-on levels are split, so its pairs take the cell-located transfer."""
    mesh = adapted_2d(2).add_fault([("Flt", TRACE)])
    stokes, info, _slip = shear_fault_solve(mesh)
    return stokes._custom_mg["hierarchy"], stokes


def _after(module, name, message, monkeypatch):
    """Make ``module.name`` raise on the last rank after it has returned."""
    original = getattr(module, name)

    def failing(*a, **k):
        out = original(*a, **k)
        _on_last_rank(message)
        return out

    monkeypatch.setattr(module, name, failing)


def test_a_failure_after_a_node_cloud_transfer(built, monkeypatch):
    hierarchy, solver = built
    _after(custom_mg, "_build_parallel_transfer", "Node-cloud transfer", monkeypatch)
    _build_raises_everywhere(hierarchy, solver, "Node-cloud transfer", monkeypatch)


def test_a_failure_after_the_zero_column_count(built, monkeypatch):
    hierarchy, solver = built
    _after(custom_mg, "_count_zero_columns_parallel", "Zero columns", monkeypatch)
    _build_raises_everywhere(hierarchy, solver, "Zero columns", monkeypatch)


def test_a_failure_after_the_patch_blocks(built, monkeypatch):
    hierarchy, solver = built
    _after(custom_mg, "_gathered_patch_blocks", "Patch blocks", monkeypatch)
    _build_raises_everywhere(hierarchy, solver, "Patch blocks", monkeypatch)


def test_a_failure_after_a_split_pair_transfer(split_built, monkeypatch):
    hierarchy, solver = split_built
    assert hierarchy.has_split_levels
    _after(fault_mg, "cell_located_transfer_parallel", "Split pair", monkeypatch)
    _build_raises_everywhere(hierarchy, solver, "Split pair", monkeypatch)


# --------------------------------------------------------------------------- #
#  Phase 3: the node-cloud builders' insides
# --------------------------------------------------------------------------- #
def test_a_node_cloud_builder_failure_on_one_rank(built, monkeypatch):
    """The builder itself fails on one rank: _agreed_node_transfer's verdict is
    now the library's agreement point, raised on every rank."""
    hierarchy, solver = built

    def failing(*a, **k):
        _on_last_rank("Node builder")
        return original(*a, **k)

    original = custom_mg._node_transfer_no_repair
    monkeypatch.setattr(custom_mg, "_node_transfer_no_repair", failing)
    _build_raises_everywhere(hierarchy, solver, "Node builder", monkeypatch)


def test_a_failure_assembling_the_rows(built, monkeypatch):
    """Inside _assemble_node_transfer, before its allreduce: the node transfer
    it is handed raises when the rows are read."""
    hierarchy, solver = built
    original = custom_mg._drop_structural_zeros

    class _Rows:
        def __init__(self, Pn):
            self._Pn = Pn

        def tocsr(self):
            _on_last_rank("Assembling rows")
            return self._Pn.tocsr()

        def __getattr__(self, name):
            return getattr(self._Pn, name)

    monkeypatch.setattr(custom_mg, "_drop_structural_zeros",
                        lambda Pn: _Rows(original(Pn)))
    _build_raises_everywhere(hierarchy, solver, "Assembling rows", monkeypatch)


def test_a_failure_before_the_matrix_is_created(built, monkeypatch):
    """Inside _assemble_node_transfer, between its allreduce and the collective
    createAIJ: the local CSR raises."""
    import sys
    import scipy.sparse as sp
    hierarchy, solver = built
    original = sp.csr_matrix

    def failing(*a, **k):
        if sys._getframe(1).f_code.co_name == "_assemble_node_transfer":
            _on_last_rank("Local CSR")
        return original(*a, **k)

    monkeypatch.setattr(sp, "csr_matrix", failing)
    _build_raises_everywhere(hierarchy, solver, "Local CSR", monkeypatch)


# --------------------------------------------------------------------------- #
#  The coarse side's move check (#1244)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def moved_built():
    """``(hierarchy, solver)`` of a refined triangle box whose top wall has
    risen: its coarse levels moved with the finest, so the set-up phase reads
    each pair's coarse side as built (a second move check per level)."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25, qdegree=2, refinement=2)
    coarse = list(mesh.multigrid_levels[:-1])
    c = np.asarray(mesh.X.coords).copy()
    c[:, -1] *= 1 + 0.1 * np.sin(np.pi * c[:, 0])
    mesh.deform(c)
    x, _y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    for wall in ("Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.bodyforce = [0.0, x]
    stokes.tolerance = 1.0e-6
    set_custom_fmg(stokes, coarse, field_id=0)
    stokes.solve()
    return stokes._custom_mg["hierarchy"], stokes


def _move_checks_from(first_failing, original):
    """``as_built_nodes``, raising on the last rank from its call
    ``first_failing`` on (counted from 0, after the collective has returned)."""
    calls = []

    def counted(*a, **k):
        out = original(*a, **k)
        calls.append(1)
        if len(calls) > first_failing:
            _on_last_rank("Coarse-side move check")
        return out

    return counted, calls


def test_a_failure_after_a_coarse_side_move_check(moved_built, monkeypatch):
    """The coarse-side loop's move check: the first loop makes one per fine
    level, so the failure is forced at the first coarse-side call, and the
    other ranks meet the next coarse-side call's agreement point."""
    hierarchy, solver = moved_built
    nfine = len(hierarchy.level_meshes) - 1
    assert nfine >= 2, "the coarse-side loop needs a next level to hang in"

    original = fault_mg.as_built_nodes

    # the fixture does read the coarse side as built: more calls than fine levels
    counted, calls = _move_checks_from(10**9, original)
    monkeypatch.setattr(fault_mg, "as_built_nodes", counted)
    hierarchy.build(solver)
    assert len(calls) > nfine, len(calls)

    failing, _calls = _move_checks_from(nfine, original)
    monkeypatch.setattr(fault_mg, "as_built_nodes", failing)
    _build_raises_everywhere(hierarchy, solver, "Coarse-side move check", monkeypatch)
