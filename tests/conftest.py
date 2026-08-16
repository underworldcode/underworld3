import os

# ==============================================================================
# VISUALIZATION BACKENDS - must be set before any visualization imports
# ==============================================================================
# Use non-interactive backends for pytest runs to prevent GUI windows
# These imports are wrapped in try/except for CI environments where packages
# may not be installed yet.

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

try:
    import pyvista
    pyvista.OFF_SCREEN = True
except ImportError:
    pass


# ==============================================================================
# TEST ISOLATION FIXTURES
# ==============================================================================
# Ensure tests don't pollute each other with global state (model, units mode).
# The global Model singleton and scaling coefficients cause test pollution
# if not reset between tests.
# ==============================================================================

import pytest


# ==============================================================================
# MESH FILES - one directory per xdist worker
# ==============================================================================
# Generated mesh files are named from the mesh PARAMETERS, so two workers
# building the same geometry choose the same name and one can read what the
# other is still writing (issue #563). The writes are atomic, which makes that
# safe; giving each worker its own directory also stops them doing the identical
# work twice. Set at import, before any test builds a mesh.
#
# `PYTEST_XDIST_WORKER` is absent in a serial run, which correctly leaves the
# default `.meshes/` in place.
_xdist_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _xdist_worker:
    os.environ.setdefault("UW_MESH_CACHE_DIR", f".meshes/{_xdist_worker}")


# ==============================================================================
# COLLECTION-TIME GLOBAL STATE GUARD (#575)
# ==============================================================================
# pytest imports a test module in order to collect it, so anything a module does
# at import time runs BEFORE any test, any fixture and any isolation the
# fixtures below provide. Two defects have reached `development` that way:
#
#   #567  a module switched the units system on process-wide at import, and the
#         first module-scoped fixture in that worker built its mesh under
#         dimensional coordinates;
#   #505  a module ran two Stokes solves at import, so `--collect-only` sat
#         inside SNESSolve for 20+ minutes looking like a silent death.
#
# The fixtures below fix the consequence. This guard fixes the practice: it
# fingerprints the process state around each module's import and fails the run,
# naming the module and what moved, when a module writes to it.
#
# The check is skipped when underworld3 is not importable — conftest.py is
# loaded before the package is necessarily installed in CI.


def _global_state_fingerprint():
    """Process-global state that importing a test module must leave alone."""

    try:
        import underworld3 as uw
        from underworld3 import model as _model
        from underworld3.utilities._api_tools import uw_object
    except ImportError:
        return None

    active_model = _model._default_model
    reference_quantities = ()
    if active_model is not None:
        reference_quantities = tuple(
            sorted(getattr(active_model, "_reference_quantities", None) or {})
        )

    return {
        "uw objects created": uw_object.uw_object_counter(),
        "units reference quantities": reference_quantities,
        "strict units": uw.is_strict_units_active(),
    }


# Modules that already do this, measured on `development` at the time the guard
# was written. They are exempted so the guard can be turned on today; the list
# is a ratchet, not an approval — nothing may be added to it, and each entry is
# a module whose module-level work belongs in a fixture (#587).
#
# `test_0601_mesh_vector_calc.py` is the one to fix first: alone in this list it
# moves the units state (`use_strict_units(False)` at import), which is the #567
# mechanism itself and reaches every module collected after it.
_KNOWN_COLLECTION_TIME_WORK = (
    "parallel/test_0765_internal_boundary_integral_mpi.py",
    "test_0004_pointwise_fns.py",
    "test_0005_IndexSwarmVariable.py",
    "test_0501_integrals.py",
    "test_0502_boundary_integrals.py",
    "test_0504_projections.py",
    "test_0601_mesh_vector_calc.py",
    "test_0810_amr_swarm_migration_regression.py",
    "test_0830_mesh_adapt_variable_transfer.py",
    "test_1000_poissonCart.py",
    "test_1000_poissonNaturalBC.py",
    "test_1001_poissonSph.py",
    "test_1004_DarcyCartesian.py",
    "test_1010_stokesCart.py",
    "test_1011_stokesSph.py",
    "test_1014_stokes_multigrid.py",
    "test_1014_stokes_shell_nullspace.py",
    "test_1050_VEstokesCart.py",
)


def _is_known_offender(nodeid):
    path = nodeid.replace(os.sep, "/")
    return any(path.endswith(known) for known in _KNOWN_COLLECTION_TIME_WORK)


_state_before_collect = {}
_collection_offenders = []


def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        _state_before_collect[collector.nodeid] = _global_state_fingerprint()


def pytest_collectreport(report):
    before = _state_before_collect.pop(report.nodeid, None)
    if before is None:
        return

    after = _global_state_fingerprint()
    if after is None:
        return

    moved = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    if moved and not _is_known_offender(report.nodeid):
        _collection_offenders.append((report.nodeid, moved))


def pytest_collection_finish(session):
    if not _collection_offenders:
        return

    lines = [
        "Test modules changed global state while being COLLECTED.",
        "",
        "Work belongs inside a test function or a fixture. Code at module level",
        "runs at import, before the isolation fixtures in tests/conftest.py can",
        "act, and it runs even under --collect-only (issues #567, #505).",
        "",
    ]
    for nodeid, moved in _collection_offenders:
        lines.append(f"  {nodeid}")
        for key, (before, after) in moved.items():
            lines.append(f"      {key}: {before} -> {after}")

    raise pytest.UsageError("\n".join(lines))


@pytest.fixture(scope="module", autouse=True)
def isolate_module_state():
    """Reset the global model BEFORE a module's own fixtures are built.

    ``isolate_test_state`` below runs per test, and a per-test fixture cannot
    protect a module-scoped one: pytest builds higher-scoped fixtures first, so
    a ``scope="module"`` fixture that creates a mesh is set up BEFORE the first
    test's function-scoped reset ever runs. Anything the process did earlier —
    a previous module's last test, or module-level code executed during
    COLLECTION — is therefore still in force while that mesh is built.

    That is #567. ``test_0741_expression_arithmetic_units.py`` used to call
    ``set_reference_quantities`` at import time, which switches the units
    system on globally. ``test_0761_point_locator.py`` builds its mesh and P1
    variable in a module-scoped fixture, and with units active ``var.coords``
    returns DIMENSIONAL coordinates (0..2.9e6 m rather than the mesh's 0..1),
    so the fixture wrote nodal values sampled at the wrong points and every
    later evaluation disagreed with the closed form by O(1). It only bit when
    that file happened to be the FIRST in its process — which under
    ``--dist loadfile`` is decided by the worker count, and never happens in a
    serial run because an earlier file's per-test reset has already cleaned up.

    Resetting at module scope closes that window for every module-scoped
    fixture in the suite, not just the one that exposed it.
    """
    import underworld3 as uw

    uw.reset_default_model()
    yield
    uw.reset_default_model()


@pytest.fixture(scope="function", autouse=True)
def isolate_test_state(request):
    """
    Isolate tests from global state pollution.

    - Reset the global model between tests (prevents reference quantities leaking)
    - Disable strict units mode for legacy tests
    - Tests in test_0814_strict_units_enforcement.py manage their own state
    """
    # Import inside fixture to defer loading until tests run
    # (conftest.py is loaded before underworld3 may be installed in CI)
    import underworld3 as uw

    test_file = request.node.fspath.basename

    # Reset model to prevent pollution from previous tests
    uw.reset_default_model()

    # Only keep strict mode ON for the strict enforcement tests
    if test_file == "test_0814_strict_units_enforcement.py":
        # These tests manage their own strict mode state
        yield
    else:
        # All other tests: disable strict mode for backward compatibility
        original_state = uw.is_strict_units_active()
        uw.use_strict_units(False)
        yield
        uw.use_strict_units(original_state)
