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
