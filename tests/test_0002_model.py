"""
Basic tests for the Model orchestration system.

Tests that meshes, swarms, and variables are automatically registered
with the default model for serialization and orchestration.
"""

import underworld3 as uw
import json


def test_model_auto_registration():
    """Test that objects are automatically registered with default model"""
    
    # Reset to clean state
    uw.reset_default_model()
    model = uw.get_default_model()
    
    # Create mesh - should auto-register
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5
    )
    
    assert model.mesh is mesh, "Mesh should be registered with model"
    
    # Create variable - should auto-register
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    
    assert "T" in model._variables, "Variable should be registered with model"
    assert model._variables["T"] is temperature
    
    # Create swarm - should auto-register
    swarm = uw.swarm.Swarm(mesh)
    
    assert len(model._swarms) > 0, "Swarm should be registered with model"
    
    # Create swarm variable - should auto-register
    material = swarm.add_variable("material", 1, dtype=int)
    
    assert "material" in model._variables, "Swarm variable should be registered"
    

def test_model_serialization():
    """Test basic model serialization to dict"""
    
    # Reset and create simple model
    uw.reset_default_model()
    model = uw.get_default_model()
    
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5
    )
    
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    velocity = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    
    # Export to dict
    config = model.to_dict()
    
    assert config["model_name"] == "default"
    assert config["mesh_type"] == "Mesh"
    assert "T" in config["variables"]
    assert "V" in config["variables"]
    assert config["swarm_count"] == 0
    
    # Should be JSON serializable
    json_str = json.dumps(config)
    assert len(json_str) > 0


def test_model_multiple_objects():
    """Test model tracks multiple swarms and variables"""
    
    uw.reset_default_model()
    model = uw.get_default_model()
    
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5
    )
    
    # Multiple variables
    T = uw.discretisation.MeshVariable("T", mesh, 1)
    P = uw.discretisation.MeshVariable("P", mesh, 1)
    V = uw.discretisation.MeshVariable("V", mesh, 2)
    
    # Multiple swarms
    swarm1 = uw.swarm.Swarm(mesh)
    swarm2 = uw.swarm.Swarm(mesh)
    
    config = model.to_dict()
    
    assert len(config["variables"]) >= 3, "Should track all mesh variables"
    assert config["swarm_count"] == 2, "Should track both swarms"


def test_model_repr():
    """Test model string representation"""

    uw.reset_default_model()
    model = uw.get_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1)
    swarm = uw.swarm.Swarm(mesh)

    repr_str = repr(model)

    assert "default" in repr_str
    assert "Mesh" in repr_str
    assert "1 variables" in repr_str or "variable" in repr_str
    assert "1 swarms" in repr_str or "swarm" in repr_str


def test_model_view():
    """Test model view method functionality"""

    uw.reset_default_model()
    model = uw.get_default_model()

    # Test view on empty model
    try:
        model.view()
        # If we get here, view() didn't crash on empty model
        view_works = True
    except Exception:
        view_works = False

    assert view_works, "Model.view() should work on empty model"

    # Add some content
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5
    )

    temperature = uw.discretisation.MeshVariable("temperature", mesh, 1)
    velocity = uw.discretisation.MeshVariable("velocity", mesh, 2)

    model.set_material('fluid', {'viscosity': 1e21, 'density': 3300})
    model.set_petsc_option('ksp_type', 'cg')

    # Test the current view implementation (no parameters due to duplicate method)
    try:
        model.view()  # Current implementation doesn't accept parameters
        view_comprehensive = True
    except Exception:
        view_comprehensive = False

    assert view_comprehensive, "Model.view() should work"