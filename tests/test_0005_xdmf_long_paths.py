"""Long output paths reach native HDF5 I/O only as basenames (#645)."""

import os
from pathlib import Path

import numpy as np
import pytest
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities._io import _short_io_path


pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
@pytest.mark.parametrize("relative", [False, True])
def test_short_io_path_restores_directory_on_error(tmp_path, relative):
    original_directory = Path.cwd()
    directory = tmp_path / "nested"
    directory.mkdir()
    target = directory / "field.h5"
    path = os.path.relpath(target, original_directory) if relative else target
    with pytest.raises(RuntimeError, match="test failure"):
        with _short_io_path(path) as short_path:
            assert short_path == "field.h5"
            assert Path.cwd() == directory
            raise RuntimeError("test failure")
    assert Path.cwd() == original_directory


@pytest.mark.tier_b
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("create_xdmf,petsc_reload", [(True, True), (True, False), (False, True)])
def test_long_timestep_paths_and_reload(tmp_path, monkeypatch, relative, create_xdmf, petsc_reload):
    """Cover mesh, P2, DG1, reduction, physical geometry and restart I/O under MPI.

    Recording the filename passed to PETSc catches regression even on stacks
    that accept long paths. The generated files retain their original location.
    """
    orchestration_model = uw.get_default_model()
    orchestration_model.set_scaling_mode("exact")
    orchestration_model.set_reference_quantities(
        length=uw.quantity(10, "km"),
        velocity=uw.quantity(5, "mm/year"),
        pressure=uw.quantity(2, "MPa"),
    )
    original_directory = Path.cwd()
    directory = Path(uw.mpi.comm.bcast(str(tmp_path), root=0)) / ("long_" + "a" * 90) / ("b" * 90)
    if uw.mpi.rank == 0:
        directory.mkdir(parents=True)
    uw.mpi.barrier()
    output_path = os.path.relpath(directory, original_directory) if relative else str(directory)
    assert len(str(directory / "fields.mesh.p2.00000.h5")) > 254

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    specs = [("p2", 2, True), ("dg1", 1, False), ("p3", 3, True), ("dg0", 0, False)]
    variables = []
    expected = {}
    for name, degree, continuous in specs:
        var = uw.discretisation.MeshVariable(
            name, mesh, 1, degree=degree, continuous=continuous, units="MPa"
        )
        values = 2 + var.coords_nd[:, 0] + 3 * var.coords_nd[:, 1]
        var.array[:, 0, 0] = uw.quantity(2 * values, "MPa")
        variables.append(var)
        expected[name] = np.asarray(var.array).copy()

    original_viewer = PETSc.ViewerHDF5
    native_paths = []

    class ViewerProbe:
        def create(self, filename, mode, comm):
            assert filename == Path(filename).name, filename
            assert Path.cwd() == directory
            native_paths.append((filename, mode))
            return original_viewer().create(filename, mode, comm=comm)

    monkeypatch.setattr(PETSc, "ViewerHDF5", ViewerProbe)
    mesh.write_timestep(
        "fields",
        0,
        outputPath=output_path,
        meshVars=variables,
        create_xdmf=create_xdmf,
        petsc_reload=petsc_reload,
    )
    assert Path.cwd() == original_directory

    if petsc_reload:
        restored_mesh = uw.discretisation.Mesh(str(directory / "fields.mesh.00000.h5"))
        for (name, degree, continuous), source in zip(specs, variables):
            source.array[:] = uw.quantity(0, "MPa")
            source.read_checkpoint(
                str(directory / f"fields.mesh.{name}.00000.h5"), data_name=name, same_layout=True
            )
            np.testing.assert_allclose(np.asarray(source.array), expected[name], atol=1.0e-12)
            restored = uw.discretisation.MeshVariable(
                name, restored_mesh, 1, degree=degree, continuous=continuous, units="MPa"
            )
            restored.read_checkpoint(
                str(directory / f"fields.mesh.{name}.00000.h5"), data_name=name
            )
            np.testing.assert_allclose(
                np.asarray(restored.array)[:, 0, 0],
                2 * (2 + restored.coords_nd[:, 0] + 3 * restored.coords_nd[:, 1]),
                atol=1.0e-12,
            )

    if create_xdmf:
        for name, degree, continuous in specs:
            remapped = uw.discretisation.MeshVariable(
                f"remapped_{name}", mesh, 1, degree=degree, continuous=continuous, units="MPa"
            )
            remapped.read_timestep("fields", name, 0, outputPath=output_path)
            np.testing.assert_allclose(np.asarray(remapped.array), expected[name], atol=1.0e-12)
        assert (directory / "fields.mesh.00000.xdmf").exists()
    assert native_paths
    assert any(mode == "w" for _, mode in native_paths)
    if petsc_reload:
        assert any(mode == "r" for _, mode in native_paths)
    assert Path.cwd() == original_directory
