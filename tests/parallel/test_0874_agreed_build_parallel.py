"""An agreed build raises a rank-local failure on every rank (#874).

``utilities.agreed_build`` makes a parallel builder's collectives agree on
failure instead of guarding each rank-local step. Inside ``agreed_build``, an
exception on one rank is caught at the context's exit, where that rank makes one
agreement call; the other ranks meet it at their next agreement point (``agree``,
placed before each collective) and every rank raises the same
``CollectiveBuildError``, carrying the failing rank's message.

The proof is ``custom_mg._fac_patch_split_parallel``, the FAC patch split, with
its helper ``_append_remote_coarse_nodes``: one context over the function and
an agreement point before each collective, and no per-statement guard. The call
is captured from a placed fault ribbon's solve (test_1147's fixture), and one failure is forced on the last rank
at three points:

* before the helper's first collective (the helper itself raises);
* inside the helper, between its two collectives (the scatter raises);
* after the function's last collective (``_patch_links`` raises).

Before the conversion each of these hangs the other ranks (or, after the last
collective, raises on one rank alone).
"""
from types import SimpleNamespace

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg
from underworld3.utilities.custom_mg import CollectiveBuildError, set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600, method="thread")]

A, B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
N_RUNGS = 80


def _ribbon_solve(empty_last_rank=False, explicit=True, bodyforce=(0.0, 0.0)):
    """A placed fault ribbon with the custom hierarchy set, whose finest level
    carries the structural FAC patch (the fixture of test_1147, one fault)."""
    S = float(np.linalg.norm(B - A)) / N_RUNGS
    trace = A + (np.arange(N_RUNGS + 1) * S)[:, None] * (B - A) / np.linalg.norm(B - A)
    dmod = uw.discretisation.discretisation_mesh
    original = dmod._from_plexh5
    if empty_last_rank:
        from petsc4py import PETSc

        def load(*a, **k):
            res = original(*a, **k)
            if not isinstance(res, tuple):
                return res
            sf0, dm = res
            cS, cE = dm.getHeightStratum(0)
            # every cell on the ranks before the last: the last holds none
            owner = np.arange(cE - cS) % max(uw.mpi.size - 1, 1)
            part = dm.getPartitioner()
            part.setType(PETSc.Partitioner.Type.SHELL)
            part.setShellPartition(uw.mpi.size,
                                   sizes=np.bincount(owner, minlength=uw.mpi.size).astype(np.int32),
                                   points=np.argsort(owner, kind="stable").astype(np.int32))
            return sf0, dm
        dmod._from_plexh5 = load
    try:
        base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                                 cellSize=1 / 8, regular=False, qdegree=2,
                                                 refinement=2)
    finally:
        dmod._from_plexh5 = original
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
    stokes.bodyforce = list(bodyforce)
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, boundary="Fault")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    if explicit:
        set_custom_fmg(stokes, levels, field_id=0)
    stokes.solve()
    return stokes


def _captured_split_call(empty_last_rank=False):
    calls = []
    original = custom_mg._fac_patch_split_parallel

    def spy(*args, **kwargs):
        if not calls:
            calls.append((args, dict(kwargs)))
        return original(*args, **kwargs)

    custom_mg._fac_patch_split_parallel = spy
    try:
        _ribbon_solve(empty_last_rank)
    finally:
        custom_mg._fac_patch_split_parallel = original
    assert uw.mpi.comm.allreduce(len(calls)) == uw.mpi.size, "no FAC patch split built"
    args, kwargs = calls[0]
    kwargs["gate"] = {}
    return args, kwargs


class _CountingVerdicts:
    """The verdict communicator, counting its exchanges (the agreement calls)."""

    def __init__(self, comm):
        self._comm = comm
        self.allgathers = 0          # every exchange: the flag and the messages

    def allreduce(self, value, **kwargs):
        self.allgathers += 1
        return self._comm.allreduce(value, **kwargs)

    def allgather(self, value):
        self.allgathers += 1
        return self._comm.allgather(value)

    def __getattr__(self, name):
        return getattr(self._comm, name)


class _CheckedComm:
    """The build's mpi4py communicator, checking at run time that every
    collective made inside an active agreed_build follows an agreement (W1's
    review: the static rule cannot see a collective inside a helper)."""

    def __init__(self, comm, verdicts):
        self._comm = comm
        self._verdicts = verdicts
        self._seen = -1
        self.unagreed = []

    def _check(self, name):
        from underworld3.utilities import agreed_build
        if agreed_build._active:
            if self._verdicts.allgathers <= self._seen:
                self.unagreed.append(name)
            self._seen = self._verdicts.allgathers

    def allreduce(self, *a, **k):
        self._check("allreduce")
        return self._comm.allreduce(*a, **k)

    def allgather(self, *a, **k):
        self._check("allgather")
        return self._comm.allgather(*a, **k)

    def bcast(self, *a, **k):
        self._check("bcast")
        return self._comm.bcast(*a, **k)

    def __getattr__(self, name):
        return getattr(self._comm, name)


def _counting(call, monkeypatch):
    """Run ``call``; return (the error message or None, the agreement calls
    made on each rank, gathered)."""
    counter = _CountingVerdicts(uw.mpi.verdict_comm)
    monkeypatch.setattr(uw.mpi, "verdict_comm", counter)
    caught = None
    try:
        call()
    except RuntimeError as err:
        assert isinstance(err, CollectiveBuildError), type(err)
        caught = str(err)
    finally:
        monkeypatch.setattr(uw.mpi, "verdict_comm", counter._comm)
    return caught, uw.mpi.comm.allgather(counter.allgathers)


def _raises_everywhere(call, message, monkeypatch):
    caught, counts = _counting(call, monkeypatch)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and message in m for m in messages), messages
    assert len(set(messages)) == 1, messages
    # every rank made the same number of agreement calls: the failed rank's
    # one exit agreement paired with the others' next agreement point
    assert len(set(counts)) == 1, counts


def _on_last_rank(message):
    if uw.mpi.rank == uw.mpi.size - 1:
        raise IndexError(f"{message} (forced on one rank)")


def test_the_split_still_gives_the_same_answer(monkeypatch):
    args, kwargs = _captured_split_call()
    counter = _CountingVerdicts(uw.mpi.verdict_comm)
    checked = _CheckedComm(args[6], counter)
    monkeypatch.setattr(uw.mpi, "verdict_comm", counter)
    try:
        custom_mg._fac_patch_split_parallel(*(args[:6] + (checked,) + args[7:]), **kwargs)
    finally:
        monkeypatch.setattr(uw.mpi, "verdict_comm", counter._comm)
    counts = uw.mpi.comm.allgather(counter.allgathers)
    # the healthy path: paired agreement counts, and every collective of the
    # build's communicator inside the context came after an agreement
    assert len(set(counts)) == 1 and counts[0] > 0, counts
    assert not checked.unagreed, checked.unagreed
    first = custom_mg._fac_patch_split_parallel(*args, **kwargs)
    second = custom_mg._fac_patch_split_parallel(*args, **kwargs)
    assert (first is None) == (second is None)
    if first is not None:
        assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_a_failure_before_the_helper_is_raised_on_every_rank(monkeypatch):
    args, kwargs = _captured_split_call()
    original = custom_mg._append_remote_coarse_nodes

    def failing(*a, **k):
        _on_last_rank("Before the fetch")
        return original(*a, **k)

    monkeypatch.setattr(custom_mg, "_append_remote_coarse_nodes", failing)
    _raises_everywhere(lambda: custom_mg._fac_patch_split_parallel(*args, **kwargs),
                       "Before the fetch", monkeypatch)


def test_a_failure_between_the_helper_s_collectives_is_raised_on_every_rank(monkeypatch):
    args, kwargs = _captured_split_call()
    real = custom_mg.PETSc

    class _IS:
        # rank-local (COMM_SELF), between the helper's two collectives and
        # before the agreement point in front of the scatter
        def createGeneral(self, *a, **k):
            _on_last_rank("Inside the fetch")
            return real.IS().createGeneral(*a, **k)

    proxy = SimpleNamespace(**{n: getattr(real, n) for n in dir(real) if not n.startswith("__")})
    proxy.IS = _IS
    monkeypatch.setattr(custom_mg, "PETSc", proxy)
    _raises_everywhere(lambda: custom_mg._fac_patch_split_parallel(*args, **kwargs),
                       "Inside the fetch", monkeypatch)


def test_a_failure_after_the_last_collective_is_raised_on_every_rank(monkeypatch):
    args, kwargs = _captured_split_call()
    kwargs["cover_max"] = 1.0e9            # take the patch, so _patch_links runs
    original = custom_mg._patch_links

    def failing(*a, **k):
        _on_last_rank("After the gate")
        return original(*a, **k)

    monkeypatch.setattr(custom_mg, "_patch_links", failing)
    _raises_everywhere(lambda: custom_mg._fac_patch_split_parallel(*args, **kwargs),
                       "After the gate", monkeypatch)


@pytest.mark.skipif(uw.mpi.size < 3, reason=(
    "at np 2 the placed ribbon refuses a base with all its cells on one rank "
    "(addon_partition: not co-partitioned), collectively, before the split"))
def test_a_failure_on_a_rank_with_no_cells_is_raised_on_every_rank(monkeypatch):
    """The base is dealt to every rank but the last, so the last holds no cell
    of the base (a placed level may give it some); the failure is forced there,
    before the fetch."""
    args, kwargs = _captured_split_call(empty_last_rank=True)
    original = custom_mg._append_remote_coarse_nodes

    def failing(*a, **k):
        _on_last_rank("On the empty rank")
        return original(*a, **k)

    monkeypatch.setattr(custom_mg, "_append_remote_coarse_nodes", failing)
    _raises_everywhere(lambda: custom_mg._fac_patch_split_parallel(*args, **kwargs),
                       "On the empty rank", monkeypatch)


# --------------------------------------------------------------------------- #
#  H1's round 2: the retry, the swallows, the envelope
# --------------------------------------------------------------------------- #
def test_a_failure_inside_the_build_takes_the_rbf_retry(monkeypatch):
    """A failure in the FAC split, on one rank and once, is a builder failure:
    the mesh-owned build is retried with the RBF builder
    (custom_mg.build_transfers), on every rank, rather than degrading to the
    default preconditioner."""
    original = custom_mg._patch_links
    fired = []

    def failing_once(*a, **k):
        if not fired:
            fired.append(True)
            _on_last_rank("Once in the build")
        return original(*a, **k)

    monkeypatch.setattr(custom_mg, "_patch_links", failing_once)
    # the mesh-owned hierarchy (no set_custom_fmg): an explicit registration
    # surfaces a build error by design, the automatic pickup retries
    # a body force, so the retried hierarchy has work to do (H1, round 3: the
    # walls' shear alone converged at the first iteration)
    with pytest.warns(UserWarning, match="retrying with the 'rbf' builder"):
        stokes = _ribbon_solve(explicit=False, bodyforce=(0.0, -1.0))
    assert stokes.snes.getConvergedReason() > 0
    # the fault takes the rotated path, whose own KSP the solver reports
    info = stokes._rotated_freeslip_info
    assert info is not None and max(info["vel_its_last"]) > 0, info
    record = stokes.pc_fallbacks.get("custom_mg.transfer_builder")
    assert record is not None and "Once in the build" in record["detail"], record


def test_an_agreed_error_is_not_swallowed_by_the_build(monkeypatch):
    """build's sanctioned swallow around snes.setUp() re-raises an error every
    rank raised together: swallowed there, the healthy ranks would go on while
    a failed rank had left. The setUp is a stand-in that raises an agreed
    CollectiveBuildError on every rank."""
    calls = []
    original = custom_mg.CustomMGHierarchy.build

    def spy(self, solver):
        if not calls:
            calls.append((self, solver))
        return original(self, solver)

    monkeypatch.setattr(custom_mg.CustomMGHierarchy, "build", spy)
    _ribbon_solve()
    monkeypatch.setattr(custom_mg.CustomMGHierarchy, "build", original)
    hierarchy, solver = calls[0]

    class _Snes:
        def __init__(self, snes):
            self._snes = snes

        def setUp(self):
            err = CollectiveBuildError("Agreed in setUp")
            err.agreed = True
            raise err

        def __getattr__(self, name):
            return getattr(self._snes, name)

    class _Solver:
        def __init__(self, s):
            self.__dict__["_s"] = s
            self.__dict__["snes"] = _Snes(s.snes)

        def __getattr__(self, name):
            return getattr(self._s, name)

    caught = None
    try:
        hierarchy.build(_Solver(solver))
    except CollectiveBuildError as err:
        caught = str(err)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m == "Agreed in setUp" for m in messages), messages


def test_two_kinds_of_verdict_that_pair_are_refused_on_every_rank():
    """One rank at a _collective_raise, the others at a snapshot verdict: the
    envelope's site tells them apart, and every rank raises the same mismatch
    error rather than reading one payload as the other."""
    caught = None
    try:
        if uw.mpi.rank == 0:
            uw.mpi.first_verdict("a failure", site="fault_contact._collective_raise")
        else:
            uw.mpi.first_verdict(None, site="snapshot._agreed_problem")
    except RuntimeError as err:
        caught = str(err)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and "verdict exchange mismatch" in m for m in messages), messages
    # one text on every rank, carrying the failing rank's message
    assert len(set(messages)) == 1 and "a failure" in messages[0], messages


def test_reraise_agreed_passes_only_agreed_errors():
    from underworld3.utilities.agreed_build import reraise_agreed
    plain = ValueError("rank-local")
    reraise_agreed(plain)                         # not agreed: returns
    agreed = CollectiveBuildError("everyone")
    agreed.agreed = True
    with pytest.raises(CollectiveBuildError):
        reraise_agreed(agreed)


def test_an_all_rank_check_keeps_its_own_builder_failure():
    """A CollectiveBuildError every rank raises from a check on agreed counts
    (not through an agreement point) keeps builder_failure=False through the
    context's exit: a check another builder cannot change must not trigger an
    RBF rebuild (H1, round 3)."""
    from underworld3.utilities.agreed_build import agreed_build
    caught = None
    try:
        with agreed_build("a check"):
            raise CollectiveBuildError("counts disagree", builder_failure=False)
    except CollectiveBuildError as err:
        caught = (str(err), err.builder_failure)
    gathered = uw.mpi.comm.allgather(caught)
    assert all(g is not None and g[1] is False and "counts disagree" in g[0]
               for g in gathered), gathered
    assert len(set(gathered)) == 1, gathered
