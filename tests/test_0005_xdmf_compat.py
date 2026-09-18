"""Compact XDMF/HDF5 layouts for continuous and cell fields."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

import underworld3 as uw


_OLD_GROUPS = ("vertex_fields", "cell_fields", "dg1")


def _shared_path(tmp_path):
    return Path(uw.mpi.comm.bcast(str(tmp_path), root=0))


def _assert_xdmf_references_exist(xdmf_path):
    """Check every HDF reference and advertised shape in an XDMF file."""
    text = xdmf_path.read_text()
    entities = dict(re.findall(r'<!ENTITY\s+(\w+)\s+"([^"]+\.h5)"\s*>', text))
    tree = ET.parse(xdmf_path)
    for item in tree.findall(".//DataItem[@Format='HDF']"):
        reference, dataset = item.text.strip().split(":", 1)
        entity = reference.removeprefix("&").removesuffix(";")
        h5_name = entities.get(entity, reference)
        with h5py.File(xdmf_path.parent / h5_name, "r") as handle:
            assert dataset in handle
            dimensions = item.get("Dimensions")
            if dimensions:
                assert tuple(map(int, dimensions.split())) == handle[dataset].shape


def _assert_no_compatibility_copies(handle):
    for group in _OLD_GROUPS:
        assert group not in handle


@pytest.mark.level_1
@pytest.mark.tier_b
def test_direct_p1_p2_and_dg0_use_fields_only(tmp_path):
    """P1, triangular P2, and DG0 are visualized directly from /fields."""
    directory = _shared_path(tmp_path)
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.4, regular=True, qdegree=3)
    p1 = uw.discretisation.MeshVariable("p1", mesh, 1, degree=1)
    p2 = uw.discretisation.MeshVariable("p2", mesh, 2, degree=2)
    dg0 = uw.discretisation.MeshVariable("dg0", mesh, 1, degree=0, continuous=False)

    p1.array[:, 0, 0] = p1.coords[:, 0] + 2.0 * p1.coords[:, 1]
    p2.array[:, 0, 0] = p2.coords[:, 0]
    p2.array[:, 0, 1] = p2.coords[:, 1]
    dg0.array[:, 0, 0] = 7.0
    mesh.write_timestep("compact", 0, outputPath=str(directory), meshVars=[p1, p2, dg0])

    if uw.mpi.rank == 0:
        p1_file = directory / "compact.mesh.p1.00000.h5"
        p2_file = directory / "compact.mesh.p2.00000.h5"
        dg0_file = directory / "compact.mesh.dg0.00000.h5"
        with h5py.File(p1_file, "r") as handle:
            assert set(handle) == {"fields"}
            assert set(handle["fields"]) == {"coordinates", "p1"}
            assert "visualization" not in handle
            _assert_no_compatibility_copies(handle)
        with h5py.File(dg0_file, "r") as handle:
            assert set(handle) == {"fields"}
            assert handle["fields/dg0"].shape[0] == handle["fields/coordinates"].shape[0]
            assert "visualization" not in handle
            _assert_no_compatibility_copies(handle)
        with h5py.File(p2_file, "r") as handle:
            assert set(handle) == {"fields"}
            coordinates = handle["fields/coordinates"][:]
            cells = handle["fields/cells"][:]
            values = handle["fields/p2"][:]
            assert cells.shape[1] == 6
            assert values.shape == coordinates.shape
            np.testing.assert_allclose(values, coordinates, atol=1.0e-12)
            points = coordinates[cells]
            np.testing.assert_allclose(points[:, 3], 0.5 * (points[:, 0] + points[:, 1]))
            np.testing.assert_allclose(points[:, 4], 0.5 * (points[:, 1] + points[:, 2]))
            np.testing.assert_allclose(points[:, 5], 0.5 * (points[:, 2] + points[:, 0]))
            _assert_no_compatibility_copies(handle)

        xdmf = directory / "compact.mesh.00000.xdmf"
        text = xdmf.read_text()
        assert 'TopologyType="Triangle_6"' in text
        assert "&p1_Data;:/fields/p1" in text
        assert "&p2_Data;:/fields/p2" in text
        assert "&dg0_Data;:/fields/dg0" in text
        _assert_xdmf_references_exist(xdmf)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_higher_order_fields_keep_exact_data_and_one_compact_reduction(tmp_path):
    """P3+ uses P1 and DG2+ uses DG0 only for the XDMF view."""
    directory = _shared_path(tmp_path)
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.4, regular=True, qdegree=5)
    continuous = [
        uw.discretisation.MeshVariable(f"p{degree}", mesh, 1, degree=degree) for degree in (3, 4)
    ]
    discontinuous = [
        uw.discretisation.MeshVariable(f"dg{degree}", mesh, 1, degree=degree, continuous=False)
        for degree in (2, 3, 4)
    ]
    for field in continuous:
        field.array[:, 0, 0] = 2.0
    for field in discontinuous:
        field.array[:, 0, 0] = 3.0
    mesh.write_timestep("high", 0, outputPath=str(directory), meshVars=continuous + discontinuous)

    if uw.mpi.rank == 0:
        mesh_file = directory / "high.mesh.00000.h5"
        with h5py.File(mesh_file, "r") as handle:
            vertex_count = handle["geometry/vertices"].shape[0]
            cell_count = handle["viz/topology/cells"].shape[0]
        for field in continuous:
            with h5py.File(directory / f"high.mesh.{field.clean_name}.00000.h5", "r") as handle:
                assert handle[f"fields/{field.clean_name}"].shape[0] > vertex_count
                assert handle[f"visualization/{field.clean_name}"].shape[0] == vertex_count
                np.testing.assert_allclose(handle[f"visualization/{field.clean_name}"][:], 2.0)
                assert (
                    handle[f"visualization/{field.clean_name}"].attrs["visualization_degree"] == 1
                )
                _assert_no_compatibility_copies(handle)
        for field in discontinuous:
            with h5py.File(directory / f"high.mesh.{field.clean_name}.00000.h5", "r") as handle:
                assert handle[f"fields/{field.clean_name}"].shape[0] > cell_count
                assert handle[f"visualization/{field.clean_name}"].shape[0] == cell_count
                np.testing.assert_allclose(handle[f"visualization/{field.clean_name}"][:], 3.0)
                assert (
                    handle[f"visualization/{field.clean_name}"].attrs["visualization_degree"] == 0
                )
                _assert_no_compatibility_copies(handle)

        xdmf = directory / "high.mesh.00000.xdmf"
        text = xdmf.read_text()
        for field in continuous + discontinuous:
            assert f"&{field.clean_name}_Data;:/visualization/{field.clean_name}" in text
        _assert_xdmf_references_exist(xdmf)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_compact_tensor_keeps_native_component_count(tmp_path):
    """The authoritative field avoids a second nine-component tensor copy."""
    directory = _shared_path(tmp_path)
    mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
    tensor = uw.discretisation.MeshVariable("tensor", mesh, degree=1, vtype=uw.VarType.TENSOR)
    tensor.array[:, 0, 0] = 1.0
    tensor.array[:, 0, 1] = 2.0
    tensor.array[:, 1, 0] = 3.0
    tensor.array[:, 1, 1] = 4.0
    mesh.write_timestep("tensor", 0, outputPath=str(directory), meshVars=[tensor])

    if uw.mpi.rank == 0:
        with h5py.File(directory / "tensor.mesh.tensor.00000.h5", "r") as handle:
            assert handle["fields/tensor"].shape[1] == 4
            assert "visualization" not in handle
            _assert_no_compatibility_copies(handle)
        text = (directory / "tensor.mesh.00000.xdmf").read_text()
        assert 'Type="Matrix"' in text


@pytest.mark.level_1
@pytest.mark.tier_b
def test_timestep_and_optional_checkpoint_roundtrip(tmp_path):
    """Dimensional fields support remap; /uw_checkpoint supports exact reload."""
    directory = _shared_path(tmp_path)
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.4, regular=True)
    source = uw.discretisation.MeshVariable("source", mesh, 1, degree=2)
    source.array[:, 0, 0] = source.coords[:, 0] + 2.0 * source.coords[:, 1]
    same_layout_expected = np.array(source.array)
    mesh.write_timestep(
        "roundtrip", 0, outputPath=str(directory), meshVars=[source], petsc_reload=True
    )

    field_file = directory / "roundtrip.mesh.source.00000.h5"
    source.array[:, 0, 0] = 0.0
    source.read_checkpoint(str(field_file), data_name="source", same_layout=True)
    np.testing.assert_allclose(source.array, same_layout_expected, atol=1.0e-12)

    reloaded_mesh = uw.discretisation.Mesh(str(directory / "roundtrip.mesh.00000.h5"))
    restored = uw.discretisation.MeshVariable("source", reloaded_mesh, 1, degree=2)
    restored.read_checkpoint(str(field_file), data_name="source")
    expected = restored.coords[:, 0] + 2.0 * restored.coords[:, 1]
    np.testing.assert_allclose(restored.array[:, 0, 0], expected, atol=1.0e-12)

    remapped = uw.discretisation.MeshVariable("remapped", mesh, 1, degree=2)
    remapped.read_timestep("roundtrip", "source", 0, outputPath=str(directory))
    expected = remapped.coords[:, 0] + 2.0 * remapped.coords[:, 1]
    np.testing.assert_allclose(remapped.array[:, 0, 0], expected, atol=1.0e-12)

    if uw.mpi.rank == 0:
        with h5py.File(field_file, "r") as handle:
            assert set(handle) == {"fields", "uw_checkpoint"}
            assert "uw_checkpoint/topologies/uw_mesh/dms/source/vecs/source/source" in handle
            assert set(handle["uw_checkpoint/topologies/uw_mesh/dms"]) == {
                "source",
                "uw_mesh",
            }


@pytest.mark.level_1
@pytest.mark.tier_b
def test_create_xdmf_false_preserves_native_writer(tmp_path):
    """Disabling XDMF leaves the established native field file untouched."""
    directory = _shared_path(tmp_path)
    mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
    field = uw.discretisation.MeshVariable("field", mesh, 1, degree=1)
    field.array[:, 0, 0] = 1.0
    mesh.write_timestep("native", 0, outputPath=str(directory), meshVars=[field], create_xdmf=False)
    if uw.mpi.rank == 0:
        assert not (directory / "native.mesh.00000.xdmf").exists()
        with h5py.File(directory / "native.mesh.field.00000.h5", "r") as handle:
            assert "fields/field" in handle
            assert "visualization" not in handle


@pytest.mark.level_1
@pytest.mark.tier_b
@pytest.mark.parametrize("dim", [2, 3])
def test_mesh_xdmf_topology_is_direct_and_bounded(tmp_path, dim):
    """The base mesh XDMF uses valid direct cell-to-vertex connectivity."""
    directory = _shared_path(tmp_path)
    mesh = uw.meshing.StructuredQuadBox(elementRes=(2,) * dim)
    mesh.write_timestep("topology", 0, outputPath=str(directory))
    if uw.mpi.rank == 0:
        with h5py.File(directory / "topology.mesh.00000.h5", "r") as handle:
            cells = handle["viz/topology/cells"][:]
            vertices = handle["geometry/vertices"][:]
            assert cells.ndim == 2
            assert cells.min() >= 0
            assert cells.max() < len(vertices)
        text = (directory / "topology.mesh.00000.xdmf").read_text()
        assert "&MeshData;:/viz/topology/cells" in text
        assert "&MeshData;:/topology/cells" not in text
