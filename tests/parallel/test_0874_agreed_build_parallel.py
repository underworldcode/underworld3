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


def _ribbon_solve():
    """A placed fault ribbon with the custom hierarchy set, whose finest level
    carries the structural FAC patch (the fixture of test_1147, one fault)."""
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
    stokes.bodyforce = [0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, boundary="Fault")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    set_custom_fmg(stokes, levels, field_id=0)
    stokes.solve()


def _captured_split_call():
    calls = []
    original = custom_mg._fac_patch_split_parallel

    def spy(*args, **kwargs):
        if not calls:
            calls.append((args, dict(kwargs)))
        return original(*args, **kwargs)

    custom_mg._fac_patch_split_parallel = spy
    try:
        _ribbon_solve()
    finally:
        custom_mg._fac_patch_split_parallel = original
    assert uw.mpi.comm.allreduce(len(calls)) == uw.mpi.size, "no FAC patch split built"
    args, kwargs = calls[0]
    kwargs["gate"] = {}
    return args, kwargs


def _raises_everywhere(call, message):
    caught = None
    try:
        call()
    except RuntimeError as err:
        assert isinstance(err, CollectiveBuildError), type(err)
        caught = str(err)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and message in m for m in messages), messages
    assert len(set(messages)) == 1, messages


def _on_last_rank(message):
    if uw.mpi.rank == uw.mpi.size - 1:
        raise IndexError(f"{message} (forced on one rank)")


def test_the_split_still_gives_the_same_answer():
    args, kwargs = _captured_split_call()
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
                       "Before the fetch")


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
                       "Inside the fetch")


def test_a_failure_after_the_last_collective_is_raised_on_every_rank(monkeypatch):
    args, kwargs = _captured_split_call()
    kwargs["cover_max"] = 1.0e9            # take the patch, so _patch_links runs
    original = custom_mg._patch_links

    def failing(*a, **k):
        _on_last_rank("After the gate")
        return original(*a, **k)

    monkeypatch.setattr(custom_mg, "_patch_links", failing)
    _raises_everywhere(lambda: custom_mg._fac_patch_split_parallel(*args, **kwargs),
                       "After the gate")
