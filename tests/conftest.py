# ==============================================================================
# VISUALIZATION BACKENDS - must be set before any visualization imports
# ==============================================================================
# Use non-interactive backends for pytest runs to prevent GUI windows

# Matplotlib: use non-interactive Agg backend
import matplotlib
matplotlib.use('Agg')

# PyVista: enable offscreen rendering (no windows)
import pyvista
pyvista.OFF_SCREEN = True

# import underworld3 as uw
# import pytest


# @pytest.fixture(scope="module")
# def load_2d_simplices_mesh():
#     resX, resY = 10, 10
#     minX, maxX = -5.0, 5.0
#     minY, maxY = -5.0, 5.0

#     mesh = uw.discretisation.Box(elementRes=(resX, resY), minCoords=(minX, minY), maxCoords=(maxX, maxY), simplex=True)
#     return mesh


# @pytest.fixture(scope="module")
# def load_2d_quads_mesh():
#     resX, resY = 10, 10
#     minX, maxX = -5.0, 5.0
#     minY, maxY = -5.0, 5.0

#     mesh = uw.discretisation.Box(elementRes=(resX, resY), minCoords=(minX, minY), maxCoords=(maxX, maxY), simplex=False)
#     return mesh


# @pytest.fixture(scope="module")
# def load_3d_quads_mesh():
#     resX, resY, resZ = 10, 10, 10
#     minX, maxX = -5.0, 5.0
#     minY, maxY = -5.0, 5.0
#     minZ, maxZ = -5.0, 5.0

#     mesh = uw.discretisation.Box(
#         elementRes=(resX, resY, resZ), minCoords=(minX, minY, minZ), maxCoords=(maxX, maxY, maxZ), simplex=False
#     )
#     return mesh


# @pytest.fixture(scope="module")
# def load_3d_simplices_mesh():
#     resX, resY, resZ = 10, 10, 10
#     minX, maxX = -5.0, 5.0
#     minY, maxY = -5.0, 5.0
#     minZ, maxZ = -5.0, 5.0

#     mesh = uw.discretisation.Box(
#         elementRes=(resX, resY, resZ), minCoords=(minX, minY, minZ), maxCoords=(maxX, maxY, maxZ), simplex=True
#     )
#     return mesh


# @pytest.fixture(scope="module", params=["2DQuads", "3DQuads", "2DSimplices", "3DSimplices"])
# def load_multi_meshes(request, load_2d_quads_mesh, load_3d_quads_mesh, load_2d_simplices_mesh, load_3d_simplices_mesh):
#     mesh_dict = {
#         "2DQuads": load_2d_quads_mesh,
#         "3DQuads": load_3d_quads_mesh,
#         "2DSimplices": load_2d_simplices_mesh,
#         "3DSimplices": load_3d_simplices_mesh,
#     }
#     mesh_type = request.param
#     return mesh_dict[mesh_type]


# ==============================================================================
# TEST ISOLATION FIXTURES
# ==============================================================================
# Ensure tests don't pollute each other with global state (model, units mode).
# The global Model singleton and scaling coefficients cause test pollution
# if not reset between tests.
# ==============================================================================

import pytest
import underworld3 as uw


@pytest.fixture(scope="function", autouse=True)
def isolate_test_state(request):
    """
    Isolate tests from global state pollution.

    - Reset the global model between tests (prevents reference quantities leaking)
    - Disable strict units mode for legacy tests
    - Tests in test_0814_strict_units_enforcement.py manage their own state
    """
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
