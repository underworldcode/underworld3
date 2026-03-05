"""Test XDMF/HDF5 compatibility groups written by write_timestep.

Validates that /vertex_fields/ and /cell_fields/ groups are created
correctly, with proper component counts and data values.
"""

import os
import re

import h5py
import numpy as np
import pytest

import underworld3 as uw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_h5_group_exists(h5_path, group_path):
    """Return True if group_path exists in the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        return group_path in f


def _read_h5_dataset(h5_path, dataset_path):
    """Read and return an HDF5 dataset as a numpy array."""
    with h5py.File(h5_path, "r") as f:
        return f[dataset_path][:]


def _check_xdmf_refs(xdmf_path, tmp_dir):
    """Verify all XDMF entity references point to real HDF5 datasets."""
    with open(xdmf_path, "r") as f:
        content = f.read()

    doctype_match = re.search(r"<!DOCTYPE\s+Xdmf.*?\[(.*?)\]>", content, re.DOTALL)
    assert doctype_match, "No DOCTYPE entity block found in XDMF file"

    entity_block = doctype_match.group(1)
    entities = dict(re.findall(r'<!ENTITY\s+(\w+)\s+"([^"]+\.h5)"\s*>', entity_block))

    # Check both vertex_fields and cell_fields references
    refs = re.findall(r"&(\w+);:(/(vertex_fields|cell_fields)/[A-Za-z0-9_]+)", content)
    errors = []
    for entity_name, dataset_path, _ in refs:
        h5_file = entities.get(entity_name)
        if not h5_file:
            errors.append(f"Entity {entity_name} not found")
            continue
        h5_full = os.path.join(tmp_dir, h5_file)
        if not os.path.exists(h5_full):
            errors.append(f"File {h5_file} not found")
            continue
        with h5py.File(h5_full, "r") as f:
            if dataset_path.lstrip("/") not in f:
                errors.append(f"{h5_file}: {dataset_path} missing")

    assert not errors, "XDMF reference errors:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# Test: P1 scalar + P2 vector (2D)
# ---------------------------------------------------------------------------


def test_xdmf_compat_2d(tmp_path):
    """write_timestep creates correct compat groups for 2D mesh variables."""

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # P1 scalar, P2 vector
    p_var = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    v_var = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)

    # Initialise with known values
    x, y = mesh.X
    p_var.data[:, 0] = mesh._coords[:, 0]  # p = x coordinate
    v_var.data[:, 0] = 1.0
    v_var.data[:, 1] = 2.0

    mesh.write_timestep(
        "test", index=0, outputPath=str(tmp_path), meshVars=[p_var, v_var]
    )

    # Check XDMF file was created
    xdmf_file = os.path.join(str(tmp_path), "test.mesh.00000.xdmf")
    assert os.path.exists(xdmf_file), "XDMF file not created"

    # Check P1 scalar: /vertex_fields/p_p should exist with correct shape
    p_h5 = os.path.join(str(tmp_path), "test.mesh.p.00000.h5")
    assert _check_h5_group_exists(p_h5, "vertex_fields/p_p")
    assert _check_h5_group_exists(p_h5, "vertex_fields/coordinates")

    p_compat = _read_h5_dataset(p_h5, "vertex_fields/p_p")
    # PETSc writes 1-component scalars as 1D (N,); accept both (N,) and (N,1)
    effective_ncomp = p_compat.shape[1] if p_compat.ndim == 2 else 1
    assert effective_ncomp == 1, f"P1 scalar should have 1 component, got {effective_ncomp}"

    # P1 scalar: compat values should match var.data exactly
    p_original = _read_h5_dataset(p_h5, "fields/p")
    np.testing.assert_allclose(
        p_compat.ravel(), p_original.ravel(), atol=1e-10,
        err_msg="P1 compat data should match original field data exactly"
    )

    # Check P2 vector: /vertex_fields/v_v should exist with dim components
    v_h5 = os.path.join(str(tmp_path), "test.mesh.v.00000.h5")
    assert _check_h5_group_exists(v_h5, "vertex_fields/v_v")

    v_compat = _read_h5_dataset(v_h5, "vertex_fields/v_v")
    # Standalone Vec writes as 1D — infer components from total size / vertex count
    n_verts = mesh._coords.shape[0]
    v_total = v_compat.size
    v_ncomp = v_total // n_verts if n_verts > 0 else 0
    assert v_ncomp == mesh.dim, (
        f"P2 vector should have {mesh.dim} components, got {v_ncomp}"
    )

    # P2 vector: vertex count should match mesh vertices, not P2 DOFs
    coords_compat = _read_h5_dataset(v_h5, "vertex_fields/coordinates")
    coords_n_verts = coords_compat.size // mesh.dim
    assert coords_n_verts == mesh._coords.shape[0], (
        "Vertex count mismatch between compat coords and mesh"
    )

    # Verify XDMF references are valid
    _check_xdmf_refs(xdmf_file, str(tmp_path))

    del mesh


# ---------------------------------------------------------------------------
# Test: 3D mesh
# ---------------------------------------------------------------------------


def test_xdmf_compat_3d(tmp_path):
    """write_timestep creates correct compat groups for 3D mesh."""

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(3, 3, 3),
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
    )

    s_var = uw.discretisation.MeshVariable("s", mesh, 1, degree=1)
    s_var.data[:, 0] = mesh._coords[:, 2]  # s = z

    mesh.write_timestep(
        "test3d", index=0, outputPath=str(tmp_path), meshVars=[s_var]
    )

    s_h5 = os.path.join(str(tmp_path), "test3d.mesh.s.00000.h5")
    assert _check_h5_group_exists(s_h5, "vertex_fields/s_s")

    s_compat = _read_h5_dataset(s_h5, "vertex_fields/s_s")
    s_original = _read_h5_dataset(s_h5, "fields/s")
    np.testing.assert_allclose(s_compat.ravel(), s_original.ravel(), atol=1e-10)

    # Verify XDMF
    xdmf_file = os.path.join(str(tmp_path), "test3d.mesh.00000.xdmf")
    _check_xdmf_refs(xdmf_file, str(tmp_path))

    del mesh


# ---------------------------------------------------------------------------
# Test: Cell (discontinuous) variable
# ---------------------------------------------------------------------------


def test_xdmf_compat_cell_variable(tmp_path):
    """Cell variables go to /cell_fields/ group."""

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # Discontinuous (cell-centred) variable
    c_var = uw.discretisation.MeshVariable(
        "c", mesh, 1, degree=0, continuous=False
    )
    c_var.data[:, 0] = 42.0

    mesh.write_timestep(
        "testcell", index=0, outputPath=str(tmp_path), meshVars=[c_var]
    )

    c_h5 = os.path.join(str(tmp_path), "testcell.mesh.c.00000.h5")
    assert _check_h5_group_exists(c_h5, "cell_fields/c_c"), (
        "/cell_fields/c_c group should exist for discontinuous variable"
    )

    c_compat = _read_h5_dataset(c_h5, "cell_fields/c_c")
    c_effective_ncomp = c_compat.shape[1] if c_compat.ndim == 2 else 1
    assert c_effective_ncomp == 1

    del mesh


# ---------------------------------------------------------------------------
# Test: read_timestep round-trip still works (checkpoint integrity)
# ---------------------------------------------------------------------------


def test_xdmf_checkpoint_roundtrip(tmp_path):
    """Compat groups don't corrupt the /fields/ data used by read_timestep."""

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    p1 = uw.discretisation.MeshVariable("p1", mesh, 1, degree=1)
    p1.data[:, 0] = mesh._coords[:, 0] + mesh._coords[:, 1]

    mesh.write_timestep(
        "roundtrip", index=0, outputPath=str(tmp_path), meshVars=[p1]
    )

    # Read back
    p1_check = uw.discretisation.MeshVariable("p1check", mesh, 1, degree=1)
    p1_check.read_timestep("roundtrip", "p1", 0, outputPath=str(tmp_path))

    np.testing.assert_allclose(p1.data, p1_check.data, atol=1e-10)

    del mesh


# ---------------------------------------------------------------------------
# Test: create_xdmf=False skips compat groups
# ---------------------------------------------------------------------------


def test_no_xdmf_when_disabled(tmp_path):
    """create_xdmf=False should not create compat groups or XDMF file."""

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    s = uw.discretisation.MeshVariable("s", mesh, 1, degree=1)
    s.data[:, 0] = 1.0  # initialise so gvec exists

    mesh.write_timestep(
        "noxdmf", index=0, outputPath=str(tmp_path),
        meshVars=[s], create_xdmf=False,
    )

    s_h5 = os.path.join(str(tmp_path), "noxdmf.mesh.s.00000.h5")
    # Check that our compat dataset was NOT written.  We check the specific
    # dataset rather than the group because some PETSc versions create
    # /vertex_fields/ as a side effect of DM-associated writes.
    assert not _check_h5_group_exists(s_h5, "vertex_fields/s_s"), (
        "vertex_fields/s_s should not exist when create_xdmf=False"
    )

    xdmf_file = os.path.join(str(tmp_path), "noxdmf.mesh.00000.xdmf")
    assert not os.path.exists(xdmf_file), "XDMF file should not exist"

    del mesh


# ---------------------------------------------------------------------------
# Test: Tensor variable repacking
# ---------------------------------------------------------------------------


def test_tensor_variable_repacking(tmp_path):
    """Tensor variables are repacked to 9 components (3x3) in vertex_fields."""

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # 2D full tensor (4 components internally)
    T = uw.discretisation.MeshVariable(
        "T", mesh, mesh.dim, degree=1, vtype=uw.VarType.TENSOR
    )
    T.data[:, 0] = 1.0  # xx
    T.data[:, 1] = 0.1  # xy
    T.data[:, 2] = 0.2  # yx
    T.data[:, 3] = 2.0  # yy

    mesh.write_timestep(
        "tensor", index=0, outputPath=str(tmp_path), meshVars=[T]
    )

    t_h5 = os.path.join(str(tmp_path), "tensor.mesh.T.00000.h5")
    assert _check_h5_group_exists(t_h5, "vertex_fields/T_T")

    t_compat = _read_h5_dataset(t_h5, "vertex_fields/T_T")
    n_verts = mesh._coords.shape[0]
    t_ncomp = t_compat.size // n_verts
    assert t_ncomp == 9, f"Tensor should be repacked to 9 components, got {t_ncomp}"

    # Verify XDMF
    xdmf_file = os.path.join(str(tmp_path), "tensor.mesh.00000.xdmf")
    _check_xdmf_refs(xdmf_file, str(tmp_path))

    del mesh
