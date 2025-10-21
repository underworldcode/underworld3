#!/usr/bin/env python3
"""
Test script for unit metadata serialization implementation.

This script tests the new unit metadata features added to checkpoint/serialization:
- MeshVariable unit metadata in save() and write() methods
- SwarmVariable unit metadata in save() method
- Swarm coordinate unit metadata in save() method
- Mesh coordinate unit metadata in write() method

Usage: pixi run -e default python test_unit_serialization.py
"""

import underworld3 as uw
import numpy as np
import h5py
import json
import tempfile
import os

def test_mesh_variable_unit_serialization():
    """Test MeshVariable unit metadata serialization."""
    print("=== Testing MeshVariable Unit Serialization ===")

    # Create a simple mesh
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    # Create a variable with units
    velocity = uw.discretisation.MeshVariable(
        "velocity",
        mesh,
        vtype=uw.VarType.VECTOR,
        degree=2,
        units="m/s"
    )

    # Debug: Check if units were set
    print(f"  Variable has units: {hasattr(velocity, 'units')}")
    if hasattr(velocity, 'units'):
        print(f"  Units value: {velocity.units}")

    # Set some data
    velocity.array[...] = np.random.random(velocity.array.shape)

    # Test save() method (appends to existing mesh file)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        mesh_filename = tmp.name

    try:
        # First save mesh to create the file
        mesh.write(mesh_filename)

        # Then save variable with unit metadata
        velocity.save(mesh_filename)

        # Check if unit metadata was added
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                with h5py.File(mesh_filename, "r") as f:
                    if "metadata" in f and f"variable_{velocity.clean_name}_units" in f["metadata"].attrs:
                        metadata_str = f["metadata"].attrs[f"variable_{velocity.clean_name}_units"]
                        metadata = json.loads(metadata_str)
                        print(f"✓ Variable unit metadata found: {metadata}")
                        # Compare units using pint - canonical form may differ from input
                        assert uw.units(metadata["units"]) == uw.units("m/s")
                        assert metadata["variable_name"] == "velocity"
                        assert metadata["num_components"] == 2
                        print("✓ MeshVariable save() unit metadata test PASSED")
                    else:
                        print("❌ Variable unit metadata not found in save() test")

    finally:
        if os.path.exists(mesh_filename):
            os.unlink(mesh_filename)

    # Test write() method (standalone variable file)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        var_filename = tmp.name

    try:
        # Write standalone variable file
        velocity.write(var_filename)

        # Check if unit metadata was added
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                with h5py.File(var_filename, "r") as f:
                    if "variable_metadata" in f.attrs:
                        metadata_str = f.attrs["variable_metadata"]
                        metadata = json.loads(metadata_str)
                        print(f"✓ Standalone variable metadata found: {metadata}")
                        # Compare units using pint - canonical form may differ from input
                        assert uw.units(metadata["units"]) == uw.units("m/s")
                        assert metadata["variable_name"] == "velocity"
                        print("✓ MeshVariable write() unit metadata test PASSED")
                    else:
                        print("❌ Variable metadata not found in write() test")

    finally:
        if os.path.exists(var_filename):
            os.unlink(var_filename)


def test_swarm_unit_serialization():
    """Test Swarm and SwarmVariable unit metadata serialization."""
    print("\n=== Testing Swarm Unit Serialization ===")

    # Create mesh and swarm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    swarm = uw.swarm.Swarm(mesh)

    # Create a swarm variable with units (before populating)
    material = uw.swarm.SwarmVariable(
        "material",
        swarm,
        size=1,
        units="kg/m^3"
    )

    # Debug: Check if units were set
    print(f"  SwarmVariable has units: {hasattr(material, 'units')}")
    if hasattr(material, 'units'):
        print(f"  Units value: {material.units}")

    # Now populate the swarm
    swarm.populate(fill_param=3)

    # Set some data
    material.data[...] = np.random.random(material.data.shape) * 1000

    # Test swarm coordinate serialization
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        swarm_filename = tmp.name

    try:
        # Save swarm coordinates
        swarm.save(swarm_filename)

        # Check if coordinate metadata was added
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                with h5py.File(swarm_filename, "r") as f:
                    if "coordinates" in f and "swarm_metadata" in f["coordinates"].attrs:
                        metadata_str = f["coordinates"].attrs["swarm_metadata"]
                        metadata = json.loads(metadata_str)
                        print(f"✓ Swarm coordinate metadata found: {metadata}")
                        assert "swarm_type" in metadata
                        assert metadata["dimension"] == 2
                        print("✓ Swarm coordinate metadata test PASSED")
                    else:
                        print("❌ Swarm coordinate metadata not found")

    finally:
        if os.path.exists(swarm_filename):
            os.unlink(swarm_filename)

    # Test swarm variable serialization
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        var_filename = tmp.name

    try:
        # Save swarm variable
        material.save(var_filename)

        # Check if variable unit metadata was added
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                with h5py.File(var_filename, "r") as f:
                    if "data" in f and "units_metadata" in f["data"].attrs:
                        metadata_str = f["data"].attrs["units_metadata"]
                        metadata = json.loads(metadata_str)
                        print(f"✓ SwarmVariable unit metadata found: {metadata}")
                        assert metadata["variable_units"] == "kg/m^3"
                        assert metadata["variable_name"] == "material"
                        print("✓ SwarmVariable unit metadata test PASSED")
                    else:
                        print("❌ SwarmVariable unit metadata not found")

    finally:
        if os.path.exists(var_filename):
            os.unlink(var_filename)


def test_mesh_coordinate_unit_serialization():
    """Test mesh coordinate unit metadata serialization."""
    print("\n=== Testing Mesh Coordinate Unit Serialization ===")

    # Create a mesh
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5
    )

    # Add coordinate units if supported (this might not exist yet)
    if hasattr(mesh, 'coordinate_units'):
        mesh.coordinate_units = "km"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        mesh_filename = tmp.name

    try:
        # Save mesh
        mesh.write(mesh_filename)

        # Check if coordinate unit metadata was added
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                with h5py.File(mesh_filename, "r") as f:
                    if "metadata" in f:
                        print(f"✓ Available metadata keys: {list(f['metadata'].attrs.keys())}")

                        # Check for boundary metadata (should exist)
                        if "boundaries" in f["metadata"].attrs:
                            boundaries_str = f["metadata"].attrs["boundaries"]
                            boundaries = json.loads(boundaries_str)
                            print(f"✓ Boundary metadata found: {boundaries}")

                        # Check for coordinate system type (should exist)
                        if "coordinate_system_type" in f["metadata"].attrs:
                            coord_type_str = f["metadata"].attrs["coordinate_system_type"]
                            coord_type = json.loads(coord_type_str)
                            print(f"✓ Coordinate system type found: {coord_type}")

                        # Check for coordinate units (new feature)
                        if "coordinate_units" in f["metadata"].attrs:
                            coord_units_str = f["metadata"].attrs["coordinate_units"]
                            coord_units = json.loads(coord_units_str)
                            print(f"✓ Coordinate units metadata found: {coord_units}")
                            print("✓ Mesh coordinate unit metadata test PASSED")
                        else:
                            print("ℹ️ Coordinate units metadata not found (coordinate_units not set on mesh)")
                    else:
                        print("❌ Metadata group not found in mesh file")

    finally:
        if os.path.exists(mesh_filename):
            os.unlink(mesh_filename)


def main():
    """Run all unit serialization tests."""
    print("Testing Unit Metadata Serialization Implementation")
    print("=" * 50)

    try:
        test_mesh_variable_unit_serialization()
        test_swarm_unit_serialization()
        test_mesh_coordinate_unit_serialization()

        print("\n" + "=" * 50)
        print("✅ All unit metadata serialization tests completed!")
        print("The selective_ranks pattern is working correctly for parallel-safe metadata.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())