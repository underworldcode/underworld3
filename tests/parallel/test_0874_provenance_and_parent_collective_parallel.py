"""Rank-local failures in the provenance transfer and the add-on partition's
parent relation are collective (#874, the remainder).

* ``fault_mg.provenance_transfer_parallel`` weights each owned fine node in its
  recorded parent (``_node_weights``) and, when the record is on another mesh,
  locates the fine cells in it. A failure on one rank raised there alone while
  the other ranks waited in the next allreduce.
* ``addon_partition._parent_cells_local`` locates each fine cell's centroid in
  the level below; ``inherit_provenance`` locates a surgery's cells (and, after
  a move, its vertices) in the mesh it was made from;
  ``_base_parent_addresses`` locates a level's cells in the mesh its record is
  on. A failure on one rank raised there alone while the other ranks waited in
  the allreduce of the unlocated count.

Each is now raised on every rank, the same error, through
``fault_contact._collective_raise``: a ``CollectiveBuildError`` in the
transfer, as the checks beside it raise, so the build retries or degrades on
every rank together; a ``RuntimeError`` in ``addon_partition``, as its checks
raise. Each test forces the failure on the last rank only. Before the guard,
the other ranks hang until the timeout.

Two of the sites are not reached by today's routes: every surgery inherits
the record, so the transfer's record is on the fine mesh itself, and a level's
record is on the level. They are reached here through a mesh that hands every
attribute to the real one and hides its record, so the location path runs on
the real meshes; it is checked to give exactly what the record gives.
"""
import numpy as np
import pytest

import underworld3 as uw
import sys

from fault_fixtures import TRACE, adapted_2d, shear_fault_solve
from underworld3.utilities.custom_mg import CollectiveBuildError
from underworld3.utilities.place_surface import place_fault_ribbon_2d
from underworld3.utilities import addon_partition as ap
from underworld3.utilities import fault_mg
from underworld3.utilities.fault_mg import addon_level_indices

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600, method="thread")]

K = 2


def _fail_on_last_rank(name, message, when=None):
    """Replace ``fault_mg.<name>`` with a version that raises on the last rank,
    for the calls whose calling frame ``when`` accepts (all, by default)."""
    original = getattr(fault_mg, name)

    def failing(*a, **k):
        if uw.mpi.rank == uw.mpi.size - 1 and (when is None or when(sys._getframe(1))):
            raise np.linalg.LinAlgError(f"{message} (forced on one rank)")
        return original(*a, **k)

    setattr(fault_mg, name, failing)
    return original


def _same_error_everywhere(call, name, message, when=None, error=RuntimeError):
    original = _fail_on_last_rank(name, message, when)
    try:
        caught = None
        try:
            call()
        except RuntimeError as err:
            assert isinstance(err, error), type(err)
            # another builder cannot change a failed location or weight: the
            # build degrades at once instead of retrying (#1165 review)
            assert not getattr(err, "builder_failure", False)
            caught = str(err)
    finally:
        setattr(fault_mg, name, original)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and message in m for m in messages), messages
    assert len(set(messages)) == 1, messages


class _Hidden:
    """A mesh that hands every attribute to ``mesh`` except ``overrides``."""

    def __init__(self, mesh, **overrides):
        self.__dict__.update(_mesh=mesh, _overrides=overrides)

    def __getattr__(self, key):
        if key in self._overrides:
            return self._overrides[key]
        return getattr(self._mesh, key)


def _captured_provenance_call():
    calls = []
    original = fault_mg.provenance_transfer_parallel

    def spy(*args, **kwargs):
        if not calls:
            calls.append((args, kwargs))
        return original(*args, **kwargs)

    fault_mg.provenance_transfer_parallel = spy
    try:
        mesh = adapted_2d(K).add_fault([("Flt", TRACE)])
        shear_fault_solve(mesh)
    finally:
        fault_mg.provenance_transfer_parallel = original
    assert uw.mpi.comm.allreduce(len(calls)) == uw.mpi.size, "no provenance transfer built"
    return calls[0]


def _moved_box():
    """A box refined twice and moved by mesh.deform (as test_1173)."""
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 8, regular=True, qdegree=2,
                                             refinement=2)
    X0 = np.asarray(base.X.coords).copy()
    w = (1 / 32) * np.sin(np.pi * X0[:, 0]) * np.sin(np.pi * X0[:, 1])
    base.deform(X0 + np.column_stack([w, w]))
    return base


def _ladder(base):
    length = float(np.linalg.norm(TRACE[1] - TRACE[0]))
    spacing = length / 80
    trace = TRACE[0] + (np.arange(81) * spacing)[:, None] * (TRACE[1] - TRACE[0]) / length
    return place_fault_ribbon_2d(base, [("Flt", trace)], 2 * spacing, margin_rings=1,
                                 clearance=0.7, split=True, mesher="ladder")[0]


def _caller(name, reference=None):
    """Accept calls from function ``name``; with ``reference`` set, only those
    on (True) or off (False) its moved-host branch, which has ``taken_back``."""
    def when(frame):
        if frame.f_code.co_name != name:
            return False
        return reference is None or (("taken_back" in frame.f_locals) == reference)
    return when


def test_parent_location_failure_is_raised_on_every_rank():
    child = adapted_2d(K)
    tail = child._custom_mg_coarse_meshes
    levels = [tail[i] for i in addon_level_indices(tail)] + [child]
    assert len(levels) >= 2
    _same_error_everywhere(lambda: ap._parent_cells_local(levels[0], levels[1]),
                           "locate_in_cells_exhaustive", "No parent located")


def test_provenance_weight_failure_is_raised_on_every_rank():
    args, kwargs = _captured_provenance_call()
    _same_error_everywhere(lambda: fault_mg.provenance_transfer_parallel(*args, **kwargs),
                           "_node_weights", "Singular matrix", error=CollectiveBuildError)


def test_provenance_location_failure_is_raised_on_every_rank():
    args, kwargs = _captured_provenance_call()
    coarse, fine, pre = args[:3]
    assert pre is fine
    # the record on another mesh: the fine cells are located in it
    hidden = (coarse, fine, _Hidden(fine)) + tuple(args[3:])
    located = fault_mg.provenance_transfer_parallel(*hidden, **kwargs)
    recorded = fault_mg.provenance_transfer_parallel(*args, **kwargs)
    located.axpy(-1.0, recorded)
    assert located.norm() == 0.0
    _same_error_everywhere(lambda: fault_mg.provenance_transfer_parallel(*hidden, **kwargs),
                           "locate_in_cells_exhaustive", "No cell located",
                           error=CollectiveBuildError)


def test_inherited_record_location_failure_is_raised_on_every_rank():
    child = adapted_2d(K)
    _same_error_everywhere(lambda: child.add_fault([("Flt", TRACE)]),
                           "locate_in_cells_exhaustive", "No host cell",
                           when=_caller("inherit_provenance", reference=False))


def test_inherited_reference_location_failure_is_raised_on_every_rank():
    base = _moved_box()
    _same_error_everywhere(lambda: _ladder(base), "locate_in_cells_exhaustive",
                           "No moved host cell",
                           when=_caller("inherit_provenance", reference=True))


def test_base_parent_location_failure_is_raised_on_every_rank():
    out = ap.partition_addon(adapted_2d(K), ap.mark_polylines([TRACE]), layers=1,
                             tolerance=0.1)
    below = out._addon_below
    comm = out.dm.getComm().tompi4py()
    # the level with its record hidden: the record is read from its parent
    hidden = _Hidden(out, _addon_below=None, _addon_parent=None, parent=out)
    assert fault_mg.provenance_source(below, hidden) is out
    located = ap._base_parent_addresses(below, hidden, comm)
    assert np.array_equal(located, np.asarray(out._addon_parent))
    _same_error_everywhere(lambda: ap._base_parent_addresses(below, hidden, comm),
                           "locate_in_cells_exhaustive", "No recorded cell")


# --------------------------------------------------------------------------- #
#  The record itself (#874, finding 4 of the #1165 review)
# --------------------------------------------------------------------------- #
def _with_record(args, record_on_last_rank):
    """The captured provenance call with ``pre``'s record replaced on the last
    rank only (every other rank keeps its own)."""
    coarse, fine, pre = args[:3]
    record = np.asarray(pre._addon_parent)
    if uw.mpi.rank == uw.mpi.size - 1:
        record = record_on_last_rank(record)
    return (coarse, fine, _Hidden(pre, _addon_parent=record)) + tuple(args[3:])


def _raised_everywhere(call, message):
    caught = None
    try:
        call()
    except CollectiveBuildError as err:
        caught = str(err)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and message in m for m in messages), messages
    assert len(set(messages)) == 1, messages


@pytest.mark.parametrize("damage, message", [
    (lambda r: r[: max(len(r) // 2, 0)], "does not describe this mesh"),
    (lambda r: np.column_stack([np.full(len(r), 10 ** 6), r[:, 1]]),
     "name a rank or a cell that does not exist"),
    (lambda r: np.column_stack([r[:, 0], np.full(len(r), 10 ** 9)]),
     "name a rank or a cell that does not exist"),
], ids=["short", "no-such-rank", "no-such-cell"])
def test_a_damaged_record_on_one_rank_is_refused_on_every_rank(damage, message):
    args, kwargs = _captured_provenance_call()
    damaged = _with_record(args, damage)
    _raised_everywhere(lambda: fault_mg.provenance_transfer_parallel(*damaged, **kwargs),
                       message)


def test_a_cell_table_failure_before_the_parent_location_is_raised_on_every_rank():
    child = adapted_2d(K)
    tail = child._custom_mg_coarse_meshes
    levels = [tail[i] for i in addon_level_indices(tail)] + [child]
    original = ap._cell_centroids

    def failing(mesh, **kwargs):
        if uw.mpi.rank == uw.mpi.size - 1:
            raise IndexError("cell table (forced on one rank)")
        return original(mesh, **kwargs)

    ap._cell_centroids = failing
    try:
        caught = None
        try:
            ap._parent_cells_local(levels[0], levels[1])
        except RuntimeError as err:
            caught = str(err)
    finally:
        ap._cell_centroids = original
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and "cell table" in m for m in messages), messages
    assert len(set(messages)) == 1, messages


# --------------------------------------------------------------------------- #
#  The transfer's opening reads, and _resolve_parents' records (#874, the
#  follow-ups to #1201's review)
# --------------------------------------------------------------------------- #
def test_an_opening_read_failure_in_the_provenance_transfer_is_raised_on_every_rank():
    args, kwargs = _captured_provenance_call()
    _same_error_everywhere(lambda: fault_mg.provenance_transfer_parallel(*args, **kwargs),
                           "_cell_vertex_table", "Level table",
                           when=_caller("provenance_transfer_parallel"),
                           error=CollectiveBuildError)


def _captured_resolve_call():
    calls = []
    original = ap._resolve_parents

    def spy(below, lev):
        calls.append((below, lev, np.asarray(lev._addon_parent_old).copy()))
        return original(below, lev)

    ap._resolve_parents = spy
    try:
        ap.partition_addon(adapted_2d(K), ap.mark_polylines([TRACE]), layers=1,
                           tolerance=0.1)
    finally:
        ap._resolve_parents = original
    moved = [c for c in calls if getattr(c[0], "_addon_self_old", None) is not None]
    assert uw.mpi.comm.allreduce(len(moved), op=__import__("mpi4py").MPI.MIN) > 0, \
        "no level below moved: _resolve_parents never reached its star forests"
    return moved[0]


@pytest.mark.parametrize("damage, message", [
    (lambda r: r[: max(len(r) // 2, 0)], "have shape"),
    (lambda r: np.column_stack([np.full(len(r), 10 ** 6), r[:, 1]]),
     "name a rank or a cell that does not exist"),
    (lambda r: np.column_stack([r[:, 0], np.full(len(r), 10 ** 9)]),
     "name a rank or a cell that does not exist"),
], ids=["short", "no-such-rank", "no-such-cell"])
@pytest.mark.parametrize("which", ["_addon_parent_old", "_addon_self_old"])
def test_a_damaged_old_address_is_refused_on_every_rank(damage, message, which):
    below, lev, _parent_old = _captured_resolve_call()
    owner = lev if which == "_addon_parent_old" else below
    record = np.asarray(getattr(owner, which))
    if uw.mpi.rank == uw.mpi.size - 1:
        record = damage(record)
    if which == "_addon_parent_old":
        lev = _Hidden(lev, _addon_parent_old=record)
    else:
        below = _Hidden(below, _addon_self_old=record)
    caught = None
    try:
        ap._resolve_parents(below, lev)
    except RuntimeError as err:
        caught = str(err)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and message in m for m in messages), messages
    assert len(set(messages)) == 1, messages


# --------------------------------------------------------------------------- #
#  The sweep (#874): every rank-local step of the remaining builders is raised
#  on every rank. Each case forces one failure on the last rank only.
# --------------------------------------------------------------------------- #
def test_a_failure_reading_the_reference_vertices_is_raised_on_every_rank():
    # The reference is read once, before the level tables (#1244): the coarse
    # side's move check depends on it. Unguarded there, a failure on one rank
    # left the others waiting at the tables' agreement point.
    args, kwargs = _captured_provenance_call()
    _same_error_everywhere(lambda: fault_mg.provenance_transfer_parallel(*args, **kwargs),
                           "reference_vertices", "Reference vertices",
                           when=_caller("provenance_transfer_parallel"),
                           error=CollectiveBuildError)


def _captured_cell_located_call():
    calls = []
    original = fault_mg.cell_located_transfer_parallel

    def spy(*args, **kwargs):
        if not calls:
            calls.append((args, kwargs))
        return original(*args, **kwargs)

    fault_mg.cell_located_transfer_parallel = spy
    try:
        shear_fault_solve(adapted_2d(K).add_fault([("Flt", TRACE)]))
    finally:
        fault_mg.cell_located_transfer_parallel = original
    assert uw.mpi.comm.allreduce(len(calls)) == uw.mpi.size
    return calls[0]


def test_an_opening_read_failure_in_the_cell_located_transfer_is_raised_on_every_rank():
    args, kwargs = _captured_cell_located_call()
    _same_error_everywhere(lambda: fault_mg.cell_located_transfer_parallel(*args, **kwargs),
                           "_cell_vertex_table", "Level table",
                           when=_caller("cell_located_transfer_parallel"),
                           error=CollectiveBuildError)


def _captured_inheritance():
    """``(child, host)`` of the cut's provenance inheritance in a partitioned
    add_fault (the host carries the add-on record)."""
    calls = []
    original = ap.inherit_provenance

    def spy(child, host):
        if getattr(host, "_addon_parent", None) is not None and not calls:
            calls.append((child, host))
        return original(child, host)

    ap.inherit_provenance = spy
    try:
        adapted_2d(K).add_fault([("Flt", TRACE)])
    finally:
        ap.inherit_provenance = original
    assert uw.mpi.comm.allreduce(len(calls), op=__import__("mpi4py").MPI.MIN) > 0
    return calls[0]


def test_a_short_host_record_in_the_inheritance_is_refused_on_every_rank():
    child, host = _captured_inheritance()
    record = np.asarray(host._addon_parent)
    if uw.mpi.rank == uw.mpi.size - 1:
        record = record[: len(record) // 2]
    hidden = _Hidden(host, _addon_parent=record)
    caught = None
    try:
        ap.inherit_provenance(_Hidden(child), hidden)
    except RuntimeError as err:
        caught = str(err)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and "host's record" in m for m in messages), messages
    assert len(set(messages)) == 1, messages


def test_a_short_record_in_the_base_parent_addresses_is_refused_on_every_rank():
    out = ap.partition_addon(adapted_2d(K), ap.mark_polylines([TRACE]), layers=1,
                             tolerance=0.1)
    below = out._addon_below
    comm = out.dm.getComm().tompi4py()
    record = np.asarray(out._addon_parent)
    if uw.mpi.rank == uw.mpi.size - 1:
        record = record[: len(record) // 2]
    pre = _Hidden(out, _addon_parent=record)
    lev = _Hidden(out, _addon_below=None, _addon_parent=None, parent=pre)
    caught = None
    try:
        ap._base_parent_addresses(below, lev, comm)
    except RuntimeError as err:
        caught = str(err)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and "provenance is on holds" in m for m in messages), messages
    assert len(set(messages)) == 1, messages


def test_a_failure_reading_the_carried_reference_is_raised_on_every_rank():
    calls = []
    original = ap._carried_reference

    def spy(moved, carried, comm):
        calls.append((moved, carried, comm))
        return original(moved, carried, comm)

    ap._carried_reference = spy
    try:
        ap.partition_addon(_moved_box(), ap.mark_polylines([TRACE]), layers=1,
                           tolerance=0.1)
    finally:
        ap._carried_reference = original
    assert uw.mpi.comm.allreduce(len(calls), op=__import__("mpi4py").MPI.MIN) > 0
    moved, carried, comm = calls[0]
    _same_error_everywhere(lambda: ap._carried_reference(moved, carried, comm),
                           "_cell_vertex_table", "Vertex table",
                           when=_caller("_carried_reference"))


def _captured_level_rotation():
    """A split coarse level's rotation call from a solve with a fault and a
    rotated free-slip wall, so both the pair and the wall rotations run."""
    calls = []
    original = fault_mg.level_rotation

    def spy(level, solver, field_id):
        if fault_mg.is_split_level(level) and not calls:
            calls.append((level, solver, field_id))
        return original(level, solver, field_id)

    fault_mg.level_rotation = spy
    try:
        mesh = adapted_2d(K).add_fault([("Flt", TRACE)])
        x, y = mesh.X
        s = uw.systems.Stokes(mesh)
        s.constitutive_model = uw.constitutive_models.ViscousFlowModel
        s.constitutive_model.Parameters.shear_viscosity_0 = 1
        s.petsc_use_pressure_nullspace = True
        for wall in ("Top", "Bottom", "Right"):
            s.add_dirichlet_bc((y - 0.5, 0.0), wall)
        s.add_rotated_freeslip_bc(0.0, "Left")
        s.add_fault_bc(0, "Flt")
        s.solve()
    finally:
        fault_mg.level_rotation = original
    assert uw.mpi.comm.allreduce(len(calls), op=__import__("mpi4py").MPI.MIN) > 0
    return calls[0]


def _csr_failing_on_last_rank(message):
    import scipy.sparse as real_sp
    from types import SimpleNamespace

    def failing(*a, **k):
        if uw.mpi.rank == uw.mpi.size - 1:
            raise IndexError(f"{message} (forced on one rank)")
        return real_sp.csr_matrix(*a, **k)

    names = {n: getattr(real_sp, n) for n in dir(real_sp) if not n.startswith("__")}
    names["csr_matrix"] = failing
    return SimpleNamespace(**names)


@pytest.mark.parametrize("builder", ["level_pair_rotation", "level_wall_rotation"])
def test_a_failure_building_a_level_rotation_is_raised_on_every_rank(builder):
    level, solver, field_id = _captured_level_rotation()
    real_sp = fault_mg.sp
    fault_mg.sp = _csr_failing_on_last_rank("Rotation blocks")
    try:
        caught = None
        try:
            getattr(fault_mg, builder)(level, solver, field_id)
        except RuntimeError as err:
            assert isinstance(err, CollectiveBuildError), type(err)
            caught = str(err)
    finally:
        fault_mg.sp = real_sp
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and "Rotation blocks" in m for m in messages), messages
    assert len(set(messages)) == 1, messages
