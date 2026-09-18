"""Physical XDMF output keeps native checkpoint arrays nondimensional."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import underworld3 as uw


def _set_reference_scales():
    """Use exact scales with easy-to-check physical conversions."""
    orchestration_model = uw.get_default_model()
    orchestration_model.set_scaling_mode("exact")
    orchestration_model.set_reference_quantities(
        length=uw.quantity(10, "km"),
        velocity=uw.quantity(5, "mm/year"),
        pressure=uw.quantity(2, "MPa"),
    )


@pytest.mark.level_1
@pytest.mark.tier_b
def test_xdmf_uses_declared_physical_units(tmp_path):
    """Visualisation copies are physical while restart data stays native."""
    _set_reference_scales()

    mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
    velocity = uw.discretisation.MeshVariable(
        "velocity", mesh, mesh.dim, degree=2, units="mm/year"
    )
    pressure = uw.discretisation.MeshVariable(
        "pressure", mesh, 1, degree=0, continuous=False, units="MPa"
    )
    surface = uw.meshing.Surface(
        "unit_test",
        mesh,
        control_points=uw.quantity([[0.0, 0.0], [10.0, 0.0]], "km"),
    )
    surface.discretize()
    distance = surface.abs_distance
    velocity.data[:, 0] = 2.0
    velocity.data[:, 1] = 3.0
    pressure.data[:, 0] = 4.0

    directory = Path(uw.mpi.comm.bcast(str(tmp_path), root=0))
    mesh.write_timestep(
        "physical",
        index=0,
        outputPath=str(directory),
        meshVars=[velocity, pressure, distance],
        petsc_reload=True,
    )

    with uw.selective_ranks(0) as should_execute:
        if not should_execute:
            return

    mesh_file = directory / "physical.mesh.00000.h5"
    velocity_file = directory / "physical.mesh.velocity.00000.h5"
    pressure_file = directory / "physical.mesh.pressure.00000.h5"
    distance_file = directory / "physical.mesh.surf_unit_test_absdistance.00000.h5"

    with h5py.File(mesh_file, "r") as handle:
        native_coordinates = handle["geometry/vertices"][:]
        physical_coordinates = handle["viz/geometry/vertices"][:]
        np.testing.assert_allclose(physical_coordinates, native_coordinates * 10.0)
        assert handle["geometry/vertices"].attrs["units"] == "nondimensional"
        assert handle["viz/geometry/vertices"].attrs["units"] == "kilometer"

    with h5py.File(velocity_file, "r") as handle:
        native = handle["fields/velocity"][:].reshape(-1, mesh.dim)
        physical = handle["vertex_fields/velocity_velocity"][:].reshape(-1, mesh.dim)
        np.testing.assert_allclose(native, [[2.0, 3.0]] * len(native))
        np.testing.assert_allclose(physical, [[10.0, 15.0]] * len(physical))
        np.testing.assert_allclose(
            handle["vertex_fields/coordinates"][:].reshape(-1, mesh.dim),
            native_coordinates * 10.0,
        )
        assert handle["fields/velocity"].attrs["units"] == "nondimensional"
        assert (
            handle["vertex_fields/velocity_velocity"].attrs["units"]
            == "millimeter / year"
        )
        assert handle["vertex_fields/coordinates"].attrs["units"] == "kilometer"
        np.testing.assert_allclose(
            handle["uw_checkpoint/velocity"][:].reshape(-1, mesh.dim), native
        )

    with h5py.File(pressure_file, "r") as handle:
        native = handle["fields/pressure"][:].reshape(-1)
        physical = handle["cell_fields/pressure_pressure"][:].reshape(-1)
        np.testing.assert_allclose(native, 4.0)
        np.testing.assert_allclose(physical, 8.0)
        assert handle["cell_fields/pressure_pressure"].attrs["units"] == "megapascal"

    with h5py.File(distance_file, "r") as handle:
        native = handle["fields/surf_unit_test_absdistance"][:].reshape(-1)
        physical = handle[
            "vertex_fields/surf_unit_test_absdistance_surf_unit_test_absdistance"
        ][:].reshape(-1)
        np.testing.assert_allclose(physical, native * 10.0)
        assert distance.units == uw.units("km").units
        assert (
            handle[
                "vertex_fields/surf_unit_test_absdistance_surf_unit_test_absdistance"
            ].attrs["units"]
            == "kilometer"
        )

    xdmf = (directory / "physical.mesh.00000.xdmf").read_text()
    assert "&MeshData;:/viz/geometry/vertices" in xdmf
    assert 'Name="velocity"' in xdmf
    assert 'Information Name="Units" Value="millimeter / year"' in xdmf
    assert 'Information Name="Units" Value="kilometer"' in xdmf


@pytest.mark.level_1
@pytest.mark.tier_b
def test_dg1_xdmf_uses_native_interpolation_and_physical_output(tmp_path):
    """DG1 interpolation stays native before its disconnected grid is scaled."""
    _set_reference_scales()
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    pressure = uw.discretisation.MeshVariable(
        "dg_pressure", mesh, 1, degree=1, continuous=False, units="MPa"
    )
    pressure.data[:, 0] = 4.0

    directory = Path(uw.mpi.comm.bcast(str(tmp_path), root=0))
    mesh.write_timestep(
        "dg_physical",
        index=0,
        outputPath=str(directory),
        meshVars=[pressure],
        petsc_reload=True,
    )

    with uw.selective_ranks(0) as should_execute:
        if not should_execute:
            return

    field_file = directory / "dg_physical.mesh.dg_pressure.00000.h5"
    with h5py.File(field_file, "r") as handle:
        np.testing.assert_allclose(handle["fields/dg_pressure"][:], 4.0)
        np.testing.assert_allclose(handle["dg1/values"][:], 8.0)
        assert np.isclose(handle["dg1/vertices"][:].max(), 10.0)
        assert handle["dg1/vertices"].attrs["units"] == "kilometer"
        assert handle["dg1/values"].attrs["units"] == "megapascal"

    xdmf = (directory / "dg_physical.mesh.00000.xdmf").read_text()
    assert 'Information Name="Units" Value="megapascal"' in xdmf
    assert 'Information Name="Units" Value="kilometer"' in xdmf
