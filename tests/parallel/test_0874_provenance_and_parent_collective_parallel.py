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
