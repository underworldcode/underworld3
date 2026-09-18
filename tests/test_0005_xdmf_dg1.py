"""Direct DG1 XDMF output preserves element-local affine traces."""

from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

import underworld3 as uw


@pytest.mark.level_1
@pytest.mark.tier_b
@pytest.mark.parametrize("dim", [2, 3])
def test_dg1_simplex_output(tmp_path, dim):
    directory = Path(uw.mpi.comm.bcast(str(tmp_path), root=0))
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim,
        maxCoords=(1.0,) * dim,
        cellSize=0.5,
        regular=True,
        qdegree=3,
    )
    scalar = uw.discretisation.MeshVariable("dg_scalar", mesh, 1, degree=1, continuous=False)
    tensor = uw.discretisation.MeshVariable(
        "dg_tensor", mesh, degree=1, continuous=False, vtype=uw.VarType.TENSOR
    )
    pressure = uw.discretisation.MeshVariable("pressure", mesh, 1, degree=1)
    rows = mesh._cell_node_indices(1, False).reshape(-1, dim + 1)
    coords = scalar.coords
    offsets = np.floor(coords[rows].mean(axis=1)[:, 0] * 7 + 1.0e-8)
    scalar.array[rows, 0, 0] = 1 + coords[rows, 0] + 2 * coords[rows, 1] + offsets[:, None]
    tensor.array[:] = 0
    tensor.array[:, 0, 0] = scalar.array[:, 0, 0]
    tensor.array[:, 0, 1] = 3 + coords[:, 0]
    tensor.array[:, 1, 0] = -2 + coords[:, 1]
    tensor.array[:, 1, 1] = 5
    pressure.array[:, 0, 0] = pressure.coords[:, 0]
    mesh.write_timestep(
        "fields",
        0,
        outputPath=str(directory),
        meshVars=[pressure, scalar, tensor],
        petsc_reload=True,
    )
    reloaded_mesh = uw.discretisation.Mesh(str(directory / "fields.mesh.00000.h5"))
    restored = uw.discretisation.MeshVariable(
        "dg_scalar", reloaded_mesh, 1, degree=1, continuous=False
    )
    restored.read_checkpoint(
        str(directory / "fields.mesh.dg_scalar.00000.h5"), data_name="dg_scalar"
    )
    restored_rows = reloaded_mesh._cell_node_indices(1, False).reshape(-1, dim + 1)
    restored_coords = restored.coords
    restored_offsets = np.floor(restored_coords[restored_rows].mean(axis=1)[:, 0] * 7 + 1.0e-8)
    expected_restored = (
        1
        + restored_coords[restored_rows, 0]
        + 2 * restored_coords[restored_rows, 1]
        + restored_offsets[:, None]
    )
    np.testing.assert_allclose(
        restored.array[restored_rows, 0, 0],
        expected_restored,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    remapped = uw.discretisation.MeshVariable("remapped", mesh, 1, degree=1, continuous=False)
    with pytest.raises(RuntimeError, match="DG1 corner basis conversion"):
        remapped.read_timestep("fields", "dg_scalar", 0, outputPath=str(directory))

    if uw.mpi.rank != 0:
        return
    with h5py.File(directory / "fields.mesh.dg_scalar.00000.h5", "r") as handle:
        points = handle["fields/coordinates"][:]
        cells = handle["fields/cells"][:]
        values = handle["fields/dg_scalar"][:].reshape(-1)
        assert handle["fields/dg_scalar"].attrs["representation"] == "basis_conversion"
        assert len(points) == len(cells) * (dim + 1)
        assert len(np.unique(cells)) == len(points)
        assert "dg1" not in handle
        assert "visualization" not in handle
    with h5py.File(directory / "fields.mesh.00000.h5", "r") as handle:
        assert len(cells) == len(handle["viz/topology/cells"])

    corners = points[cells]
    determinants = np.linalg.det((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1))
    assert np.all(determinants > 0)
    centers = corners.mean(axis=1)
    offsets = np.repeat(np.floor(centers[:, 0] * 7 + 1.0e-8), dim + 1)
    expected = 1 + points[:, 0] + 2 * points[:, 1] + offsets
    np.testing.assert_allclose(values, expected, rtol=1.0e-12, atol=1.0e-12)

    _, inverse = np.unique(np.round(points, 10), axis=0, return_inverse=True)
    low = np.full(inverse.max() + 1, np.inf)
    high = np.full(inverse.max() + 1, -np.inf)
    np.minimum.at(low, inverse, values)
    np.maximum.at(high, inverse, values)
    assert np.max(high - low) >= 1

    with h5py.File(directory / "fields.mesh.dg_tensor.00000.h5", "r") as handle:
        tensor_values = handle["fields/dg_tensor"][:]
        assert tensor_values.shape == (len(points), dim * dim)
        np.testing.assert_allclose(tensor_values[:, 0], expected)
        np.testing.assert_allclose(tensor_values[:, 1], 3 + points[:, 0])
        np.testing.assert_allclose(tensor_values[:, dim], -2 + points[:, 1], atol=1.0e-12)
        np.testing.assert_allclose(tensor_values[:, dim + 1], 5)

    tree = ET.parse(directory / "fields.mesh.00000.xdmf")
    grids = tree.findall(".//Grid[@GridType='Uniform']")
    assert {grid.get("Name") for grid in grids} == {"domain", "dg_scalar", "dg_tensor"}
    for name in ("dg_scalar", "dg_tensor"):
        grid = next(grid for grid in grids if grid.get("Name") == name)
        attribute = grid.find("Attribute")
        assert attribute.get("Name") == name
        assert attribute.get("Center") == "Node"


@pytest.mark.level_1
@pytest.mark.tier_b
def test_unsupported_dg_layout_uses_dg0_visualization_reduction(tmp_path):
    """Non-simplex DG1 remains exact in /fields and gets one DG0 XDMF array."""
    directory = Path(uw.mpi.comm.bcast(str(tmp_path), root=0))
    mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
    dg = uw.discretisation.MeshVariable("dg", mesh, 1, degree=1, continuous=False)
    dg.array[:, 0, 0] = 4.0
    mesh.write_timestep("structured", 0, outputPath=str(directory), meshVars=[dg])
    if uw.mpi.rank == 0:
        with h5py.File(directory / "structured.mesh.dg.00000.h5", "r") as handle:
            assert "fields/dg" in handle
            assert "visualization/dg" in handle
            np.testing.assert_allclose(handle["visualization/dg"][:], 4.0)
            assert handle["visualization/dg"].attrs["visualization_degree"] == 0
