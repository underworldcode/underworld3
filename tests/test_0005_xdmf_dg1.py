"""DG1 visualization preserves affine fields and jumps, including MPI ownership."""

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
        "dg_tensor",
        mesh,
        degree=1,
        continuous=False,
        vtype=uw.VarType.TENSOR,
    )
    pressure = uw.discretisation.MeshVariable("pressure", mesh, 1, degree=1)
    vector = uw.discretisation.MeshVariable("dg_vector", mesh, dim, degree=1, continuous=False)
    symmetric = uw.discretisation.MeshVariable(
        "dg_symmetric", mesh, degree=1, continuous=False, vtype=uw.VarType.SYM_TENSOR
    )
    rows = mesh._cell_node_indices(1, False).reshape(-1, dim + 1)
    coords = scalar.coords
    offset = np.floor(coords[rows].mean(axis=1)[:, 0] * 7 + 1e-8)
    scalar.array[rows, 0, 0] = 1 + coords[rows, 0] + 2 * coords[rows, 1] + offset[:, None]
    tensor.array[:] = 0
    tensor.array[:, 0, 0] = scalar.array[:, 0, 0]
    tensor.array[:, 0, 1] = 3 + coords[:, 0]
    tensor.array[:, 1, 0] = -2 + coords[:, 1]
    tensor.array[:, 1, 1] = 5
    pressure.array[:, 0, 0] = pressure.coords[:, 0]
    vector.array[:, 0, :] = coords
    symmetric.array[:] = 0
    symmetric.array[:, 0, 0] = 2
    symmetric.array[:, 1, 1] = 3
    symmetric.array[:, 0, 1] = coords[:, 0]
    original = np.array(scalar.array)
    mesh.write_timestep(
        "fields",
        index=0,
        outputPath=str(directory),
        meshVars=[pressure, scalar, tensor, vector, symmetric],
        petsc_reload=True,
    )
    restored = uw.discretisation.MeshVariable("restored", mesh, 1, degree=1, continuous=False)
    restored.read_checkpoint(
        str(directory / "fields.mesh.dg_scalar.00000.h5"), data_name="dg_scalar", same_layout=True
    )
    np.testing.assert_allclose(restored.array, original, rtol=1e-12, atol=1e-12)
    if uw.mpi.rank != 0:
        return
    with h5py.File(directory / "fields.mesh.dg_scalar.00000.h5", "r") as handle:
        points = handle["dg1/vertices"][:]
        cells = handle["dg1/cells"][:]
        values = handle["dg1/values"][:].reshape(-1)
        native = handle["fields/dg_scalar"][:]
        assert len(points) == len(cells) * (dim + 1)
        assert len(np.unique(cells)) == len(points)
        assert native.size == len(points)
    with h5py.File(directory / "fields.mesh.00000.h5", "r") as handle:
        assert len(cells) == len(handle["viz/topology/cells"])
    corners = points[cells]
    assert np.all(np.linalg.det((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1)) > 0)
    centers = corners.mean(axis=1)
    assert len(np.unique(np.round(centers, 10), axis=0)) == len(cells)
    offset = np.repeat(np.floor(centers[:, 0] * 7 + 1e-8), dim + 1)
    expected = 1 + points[:, 0] + 2 * points[:, 1] + offset
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)
    # Repeated positions can have different values: these traces must not merge.
    _, inverse = np.unique(np.round(points, 10), axis=0, return_inverse=True)
    low = np.full(inverse.max() + 1, np.inf)
    high = np.full(inverse.max() + 1, -np.inf)
    np.minimum.at(low, inverse, values)
    np.maximum.at(high, inverse, values)
    assert np.max(high - low) >= 1
    with h5py.File(directory / "fields.mesh.dg_tensor.00000.h5", "r") as handle:
        tensor_values = handle["dg1/values"][:]
        np.testing.assert_allclose(handle["dg1/vertices"][:], points)
        assert tensor_values.shape == (len(points), 9)
        np.testing.assert_allclose(tensor_values[:, 0], expected)
        np.testing.assert_allclose(tensor_values[:, 1], 3 + points[:, 0])
        np.testing.assert_allclose(tensor_values[:, 3], -2 + points[:, 1], atol=1e-12)
        np.testing.assert_allclose(tensor_values[:, 4], 5)
    tree = ET.parse(directory / "fields.mesh.00000.xdmf")
    with h5py.File(directory / "fields.mesh.dg_vector.00000.h5", "r") as handle:
        np.testing.assert_allclose(handle["dg1/values"][:], points, atol=1e-12)
    with h5py.File(directory / "fields.mesh.dg_symmetric.00000.h5", "r") as handle:
        sym_values = handle["dg1/values"][:]
        assert sym_values.shape == (len(points), 9)
        np.testing.assert_allclose(sym_values[:, 0], 2)
        np.testing.assert_allclose(sym_values[:, 4], 3)
        np.testing.assert_allclose(sym_values[:, 1], points[:, 0], atol=1e-12)
        np.testing.assert_allclose(sym_values[:, 3], points[:, 0], atol=1e-12)
    grids = tree.findall(".//Grid[@GridType='Uniform']")
    assert len(grids) == 2
    dg = next(grid for grid in grids if grid.get("Name") == "DG1")
    assert {a.get("Name") for a in dg.findall("Attribute")} == {
        "dg_scalar",
        "dg_tensor",
        "dg_vector",
        "dg_symmetric",
    }
    assert all(a.get("Center") == "Node" for a in dg.findall("Attribute"))
    for item in tree.findall(".//DataItem[@Format='HDF']"):
        filename, dataset = item.text.strip().split(":", 1)
        with h5py.File(directory / filename, "r") as handle:
            assert tuple(map(int, item.get("Dimensions").split())) == handle[dataset].shape


@pytest.mark.level_1
@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [1, 2])
def test_unsupported_dg_layout_fails_before_writing(tmp_path, degree):
    directory = Path(uw.mpi.comm.bcast(str(tmp_path), root=0))
    mesh = (
        uw.meshing.StructuredQuadBox(elementRes=(2, 2))
        if degree == 1
        else uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    )
    dg = uw.discretisation.MeshVariable("dg", mesh, 1, degree=degree, continuous=False)
    with pytest.raises(NotImplementedError, match="DG.*XDMF"):
        mesh.write_timestep("unsupported", index=0, outputPath=str(directory), meshVars=[dg])
    assert not (directory / "unsupported.mesh.00000.h5").exists()
    # Unsupported visualization must not prevent native-only checkpoints.
    mesh.write_timestep(
        "native", index=0, outputPath=str(directory), meshVars=[dg], create_xdmf=False
    )
