"""DM rebuild lifecycle across MeshVariable creation (issue #492).

Creating a MeshVariable on a mesh that already has fields rebuilds ``mesh.dm``
(the finalized PETSc Section cannot be extended in place through petsc4py).
Before the fix, ``_setup_ds`` eagerly ``destroy()``-ed the old DM and the old
variable vectors. petsc4py's ``destroy()`` zeroes the handle of the wrapper
object itself — and ``mesh.dm`` is a plain attribute, so a user-captured
handle IS that wrapper. The next call on it was a NULL-handle dereference:
a hard SIGSEGV on optimized PETSc, verified as subprocess exit code -11 by
the issue-492 design probes (probe3_stale_holders.py). That crash cannot be
asserted in CI without segfaulting the test process, so these tests assert
the observable precondition instead: the held wrapper must keep a non-zero
handle and keep answering queries. On the unfixed build the handle is zeroed,
so the asserts below fail cleanly.

The fix drops the reference instead of destroying: the old DM/Vecs die with
their last holder (PETSc refcounting), so a held handle is stale-but-valid.
Measured RSS over repeated rebuild+solve cycles is identical with and without
the eager destroy (design note, scratchpad i492), which the accumulation test
bounds here.
"""
import gc
import resource

import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _box(cellSize=0.3, **kwargs):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cellSize,
        regular=False, qdegree=2, **kwargs)


def _solve_poisson(mesh, u, preconditioner=None):
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.petsc_options["ksp_rtol"] = 1e-8
    if preconditioner is not None:
        poisson.preconditioner = preconditioner
    poisson.solve()
    return poisson


def test_held_dm_handle_survives_variable_creation():
    """The issue #492 reproducer: a captured ``mesh.dm`` handle must remain
    valid (stale, but safe to query) after a second variable rebuilds the DM,
    and the mesh must work on the new DM."""
    mesh = _box()
    held = mesh.dm
    cells = held.getHeightStratum(0)[1]

    v = uw.discretisation.MeshVariable("v1", mesh, mesh.dim, degree=2)
    # first variable takes the addField fast path — no rebuild
    assert mesh.dm is held

    p = uw.discretisation.MeshVariable("p1", mesh, 1, degree=1)
    # second variable rebuilds: mesh.dm is a NEW wrapper ...
    assert mesh.dm is not held
    assert mesh.dm.getNumFields() == 2
    # ... and the held handle is ALIVE (pre-fix: handle == 0, then SIGSEGV
    # on any call — see module docstring)
    assert held.handle != 0
    assert held.getDimension() == 2
    assert held.getHeightStratum(0)[1] == cells

    # the mesh is fully functional on the rebuilt DM
    poisson = _solve_poisson(mesh, p)
    err = np.linalg.norm(p.data[:, 0] - p.coords[:, 1])
    assert err / (np.linalg.norm(p.coords[:, 1]) + 1e-30) < 1e-8


def test_old_dm_released_to_last_holder_no_accumulation():
    """The mesh must hand the old DM to its remaining holders (refcount 1 =
    the captured wrapper is the last one), and unheld rebuild cycles must not
    accumulate memory: dropping the eager destroy is leak-free because the
    wrapper's own dealloc frees the object when nobody else kept it."""
    mesh = _box()
    u0 = uw.discretisation.MeshVariable("u0", mesh, 1, degree=1)

    held = mesh.dm
    u1 = uw.discretisation.MeshVariable("u1", mesh, 1, degree=1)
    assert held.handle != 0
    # the mesh released its reference: the captured wrapper is the last holder
    assert held.getRefCount() == 1
    del held
    gc.collect()

    # unheld cycles: each creation rebuilds the DM; nothing may accumulate.
    # Bound set from measurement on this loop (see test docstring note below);
    # a leaked DM per cycle would blow through it immediately.
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    for i in range(20):
        uw.discretisation.MeshVariable(f"w{i}", mesh, 1, degree=1)
        gc.collect()
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    assert rss1 - rss0 < 50.0, f"RSS grew {rss1 - rss0:.1f} MB over 20 rebuilds"


def test_data_views_are_refreshed_after_rebuild():
    """UW3 must never hand back a view of the released vectors: after a
    rebuild, ``.data`` returns a fresh buffer carrying the preserved values.
    (A RAW numpy view captured by user code before the rebuild cannot be
    reached — data-access.md documents that it does not survive variable
    creation and must be re-read.)"""
    mesh = _box()
    u1 = uw.discretisation.MeshVariable("u1", mesh, 1, degree=1)
    u1.data[:, 0] = 42.0
    stale_view = np.asarray(u1.data)

    uw.discretisation.MeshVariable("u2", mesh, 1, degree=1)

    fresh = u1.data
    # Object identity only — never np.shares_memory here: the released buffer
    # is freed before the replacement allocates, so the allocator may recycle
    # the same block and make an address comparison fail spuriously on
    # exactly the platform this guards (#536 review). Dereferencing
    # stale_view is likewise out: it dangles by the documented contract.
    assert fresh is not stale_view
    assert np.all(np.asarray(fresh)[:, 0] == 42.0)  # data preserved across rebuild
    # round-trip on the new buffer
    u1.data[:, 0] = 7.0
    assert np.all(np.asarray(u1.data)[:, 0] == 7.0)


@pytest.mark.level_2
def test_adapt_child_second_variable_after_solve():
    """The PR #488 CI detonation shape (test_0842-shaped, 2-D for speed):
    on an adapt child, variable -> solve -> SECOND variable -> second solve,
    then full teardown. Pre-fix this armed a use-after-free that crashed
    Linux CI two test files later; post-fix the old DM survives as long as
    anything holds it and teardown is clean."""
    pytest.importorskip(
        "underworld3.utilities._nvb_transform",
        reason="native uwnvb transform not built (needs the custom-PETSc/amr env)")

    mesh = _box(cellSize=0.3, refinement=1)

    def metric(centroids):
        r = np.linalg.norm(np.asarray(centroids) - 0.5, axis=1)
        h = np.where(r < 0.18, 0.04,
                     np.minimum(0.04 + (0.3 - 0.04) * (r - 0.18) / 0.25, 0.3))
        return 1.0 / h**2

    child = mesh.adapt(metric, max_levels=1)

    u1 = uw.discretisation.MeshVariable("u1", child, 1, degree=1)
    # fmg builds the custom-P coarse chain — the original detonation had the
    # FMG hierarchy live on the child when the rebuild fired (#536 review:
    # under "auto" a single-field solver declines to GAMG and the chain this
    # test exists to exercise is never constructed).
    _solve_poisson(child, u1, preconditioner="fmg")

    held = child.dm  # captured post-solve, pre-rebuild (the arming step)
    u2 = uw.discretisation.MeshVariable("u2", child, 1, degree=1)
    assert child.dm is not held
    assert held.handle != 0 and held.getDimension() == 2

    poisson2 = _solve_poisson(child, u2)
    err = np.linalg.norm(u2.data[:, 0] - u2.coords[:, 1])
    assert err / (np.linalg.norm(u2.coords[:, 1]) + 1e-30) < 1e-8

    # teardown ordering from the CI story: solver, then meshes, then gc
    del poisson2, held, u1, u2, child, mesh
    gc.collect()
