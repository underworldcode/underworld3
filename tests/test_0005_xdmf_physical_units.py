"""Dimensional /fields output and optional native PETSc checkpoints."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import underworld3 as uw


def _set_reference_scales():
    model = uw.get_default_model()
    model.set_scaling_mode("exact")
    model.set_reference_quantities(
        length=uw.quantity(10, "km"),
        velocity=uw.quantity(5, "mm/year"),
        pressure=uw.quantity(2, "MPa"),
    )


@pytest.mark.level_1
@pytest.mark.tier_b
def test_fields_are_dimensional_and_checkpoint_is_native(tmp_path):
    """Analysis sees physical values while exact reload restores solver values."""
    _set_reference_scales()
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    velocity = uw.discretisation.MeshVariable("velocity", mesh, mesh.dim, degree=2, units="mm/year")
    pressure = uw.discretisation.MeshVariable(
        "pressure", mesh, 1, degree=0, continuous=False, units="MPa"
    )
    velocity.array[:, 0, 0] = uw.quantity(10.0, "mm/year")
    velocity.array[:, 0, 1] = uw.quantity(15.0, "mm/year")
    pressure.array[:, 0, 0] = uw.quantity(8.0, "MPa")
    expected_velocity = np.array(velocity.array)
    expected_pressure = np.array(pressure.array)

    directory = Path(uw.mpi.comm.bcast(str(tmp_path), root=0))
    mesh.write_timestep(
        "physical",
        0,
        outputPath=str(directory),
        meshVars=[velocity, pressure],
        petsc_reload=True,
    )

    velocity_file = directory / "physical.mesh.velocity.00000.h5"
    pressure_file = directory / "physical.mesh.pressure.00000.h5"
    velocity_restart = uw.discretisation.MeshVariable(
        "velocity_restart", mesh, mesh.dim, degree=2, units="mm/year"
    )
    pressure_restart = uw.discretisation.MeshVariable(
        "pressure_restart", mesh, 1, degree=0, continuous=False, units="MPa"
    )
    velocity_restart.read_checkpoint(str(velocity_file), data_name="velocity")
    pressure_restart.read_checkpoint(str(pressure_file), data_name="pressure")
    np.testing.assert_allclose(np.array(velocity_restart.array), expected_velocity)
    np.testing.assert_allclose(np.array(pressure_restart.array), expected_pressure)

    velocity_remap = uw.discretisation.MeshVariable(
        "velocity_remap", mesh, mesh.dim, degree=2, units="mm/year"
    )
    velocity_remap.read_timestep("physical", "velocity", 0, outputPath=str(directory))
    np.testing.assert_allclose(np.array(velocity_remap.array), expected_velocity)

    if uw.mpi.rank != 0:
        return
    with h5py.File(directory / "physical.mesh.00000.h5", "r") as handle:
        native_coordinates = handle["geometry/vertices"][:]
        physical_coordinates = handle["viz/geometry/vertices"][:]
        np.testing.assert_allclose(physical_coordinates, native_coordinates * 10.0)
        assert handle["viz/geometry/vertices"].attrs["units"] == "kilometer"

    with h5py.File(velocity_file, "r") as handle:
        physical_velocity = handle["fields/velocity"][:]
        np.testing.assert_allclose(physical_velocity[:, 0], 10.0)
        np.testing.assert_allclose(physical_velocity[:, 1], 15.0)
        assert np.isclose(handle["fields/coordinates"][:].max(), 10.0)
        assert handle["fields/velocity"].attrs["units"] == "millimeter / year"
        assert handle["fields/coordinates"].attrs["units"] == "kilometer"
        assert "visualization" not in handle
        assert "vertex_fields" not in handle
        assert "restart/petsc/topologies/uw_mesh/dms/velocity/vecs/velocity/velocity" in handle

    with h5py.File(pressure_file, "r") as handle:
        np.testing.assert_allclose(handle["fields/pressure"][:], 8.0)
        assert handle["fields/pressure"].attrs["units"] == "megapascal"
        assert "cell_fields" not in handle

    text = (directory / "physical.mesh.00000.xdmf").read_text()
    assert "&velocity_Data;:/fields/velocity" in text
    assert "&pressure_Data;:/fields/pressure" in text
    assert 'Information Name="Units" Value="millimeter / year"' in text
    assert 'Information Name="Units" Value="megapascal"' in text
    assert 'Information Name="Units" Value="kilometer"' in text


@pytest.mark.level_1
@pytest.mark.tier_b
def test_dg1_fields_are_physical_at_element_corners(tmp_path):
    """DG1 native reload data and corner visualization use physical units."""
    _set_reference_scales()
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    pressure = uw.discretisation.MeshVariable(
        "dg_pressure", mesh, 1, degree=1, continuous=False, units="MPa"
    )
    pressure.array[:, 0, 0] = uw.quantity(8.0, "MPa")
    directory = Path(uw.mpi.comm.bcast(str(tmp_path), root=0))
    mesh.write_timestep("dg", 0, outputPath=str(directory), meshVars=[pressure])

    remapped = uw.discretisation.MeshVariable(
        "dg_pressure_remapped",
        mesh,
        1,
        degree=1,
        continuous=False,
        units="MPa",
    )
    remapped.read_timestep("dg", "dg_pressure", 0, outputPath=str(directory))
    np.testing.assert_allclose(np.array(remapped.array), np.array(pressure.array))

    if uw.mpi.rank == 0:
        with h5py.File(directory / "dg.mesh.dg_pressure.00000.h5", "r") as handle:
            np.testing.assert_allclose(handle["fields/dg_pressure"][:], 8.0)
            np.testing.assert_allclose(handle["visualization/dg_pressure"][:], 8.0)
            assert np.isclose(handle["visualization/coordinates"][:].max(), 10.0)
            assert handle["fields/dg_pressure"].attrs["units"] == "megapascal"
            assert handle["fields/coordinates"].attrs["units"] == "kilometer"
            assert handle["visualization/dg_pressure"].attrs["units"] == "megapascal"
            assert handle["visualization/coordinates"].attrs["units"] == "kilometer"
            assert "dg1" not in handle
