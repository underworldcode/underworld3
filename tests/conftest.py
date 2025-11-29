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
# STRICT UNITS MODE FIXTURE FOR LEGACY TESTS
# ==============================================================================
# Most existing tests were written before strict units mode was implemented.
# They test various features (units, solvers, etc.) but don't set reference
# quantities first. Disable strict mode for all tests EXCEPT the strict
# enforcement tests themselves.
# ==============================================================================

import pytest
import underworld3 as uw


@pytest.fixture(scope="function", autouse=True)
def manage_strict_units_mode(request):
    """
    Manage strict units mode for tests.

    - Disable for all tests EXCEPT test_0814_strict_units_enforcement.py
    - This allows legacy tests to work while enforcing strict mode for new code
    """
    test_file = request.node.fspath.basename

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
