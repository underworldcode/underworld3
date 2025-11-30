"""
Test swarm variable creation constraints.

Tests that variables cannot be added to swarms after they are populated,
ensuring users get clear error messages instead of PETSc crashes.
"""

import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
import underworld3 as uw


def test_swarm_variable_before_population():
    """Test that variables can be created before population (correct usage)"""

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5)
    swarm = uw.swarm.Swarm(mesh)

    # Should work fine - variables before population
    material = swarm.add_variable("material", 1, dtype=int)
    temperature = swarm.add_variable("temperature", 1)

    assert material is not None
    assert temperature is not None
    assert swarm.local_size <= 0  # Not populated yet (may be -1)

    # Population should work
    swarm.populate(fill_param=2)
    assert swarm.local_size > 0  # Now populated


def test_swarm_variable_after_population_fails():
    """Test that variables cannot be added after population (incorrect usage)"""

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5)
    swarm = uw.swarm.Swarm(mesh)

    # Populate first
    swarm.populate(fill_param=2)
    assert swarm.local_size > 0

    # Now trying to add variable should fail with clear error
    with pytest.raises(RuntimeError) as exc_info:
        swarm.add_variable("velocity", 2)

    error_msg = str(exc_info.value)
    assert "already populated" in error_msg
    assert "Variables must be created before" in error_msg
    assert "Correct usage:" in error_msg
    assert "swarm.add_variable" in error_msg
    assert "swarm.populate" in error_msg


def test_swarm_variable_constructor_after_population_fails():
    """Test that SwarmVariable constructor also prevents post-population creation"""

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5)
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=2)

    # Direct constructor should also fail
    with pytest.raises(RuntimeError) as exc_info:
        uw.swarm.SwarmVariable("test_var", swarm, 1)

    error_msg = str(exc_info.value)
    assert "already populated" in error_msg


def test_multiple_swarms_independent():
    """Test that swarm population state is independent between swarms"""

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5)

    # First swarm - populate then try to add variable (should fail)
    swarm1 = uw.swarm.Swarm(mesh)
    swarm1.populate(fill_param=2)

    with pytest.raises(RuntimeError):
        swarm1.add_variable("var1", 1)

    # Second swarm - should still allow variable creation
    swarm2 = uw.swarm.Swarm(mesh)
    var2 = swarm2.add_variable("var2", 1)  # Should work
    assert var2 is not None

    # But after populating swarm2, it should also prevent new variables
    swarm2.populate(fill_param=2)
    with pytest.raises(RuntimeError):
        swarm2.add_variable("var3", 1)


def test_error_message_contains_variable_name():
    """Test that error message includes the specific variable name being added"""

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5)
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=2)

    variable_name = "my_custom_variable"

    with pytest.raises(RuntimeError) as exc_info:
        swarm.add_variable(variable_name, 1)

    error_msg = str(exc_info.value)
    assert variable_name in error_msg


def test_error_message_shows_particle_count():
    """Test that error message shows the current particle count"""

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5)
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=2)

    particle_count = swarm.local_size
    assert particle_count > 0

    with pytest.raises(RuntimeError) as exc_info:
        swarm.add_variable("test_var", 1)

    error_msg = str(exc_info.value)
    assert str(particle_count) in error_msg
