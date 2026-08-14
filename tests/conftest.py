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
