import pytest
import sympy
import underworld3 as uw

import re
import h5py
import glob

import os
import numpy as np

# +
structured_quad_box = uw.meshing.StructuredQuadBox(elementRes=(3,) * 2)

unstructured_quad_box_irregular = uw.meshing.UnstructuredSimplexBox(
    cellSize=0.2, regular=False, qdegree=2, refinement=1
)
unstructured_quad_box_regular = uw.meshing.UnstructuredSimplexBox(
    cellSize=0.2, regular=True, qdegree=2, refinement=2
)

unstructured_quad_box_irregular_3D = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0, 0.0),
    maxCoords=(1.0, 1.0, 1.0),
    cellSize=0.25,
    regular=False,
    qdegree=2,
)


# -


def check_xdmf_vertex_fields_exist_in_h5(xdmf_filename, tmp_path=""):
    errors = []
    with open(xdmf_filename, "r") as f:
        content = f.read()
    doctype_match = re.search(r"<!DOCTYPE\s+Xdmf.*?\[(.*?)\]>", content, re.DOTALL)
    if not doctype_match:
        raise AssertionError("No DOCTYPE entity block found in XDMF file.")
    entity_block = doctype_match.group(1)
    entities = dict(re.findall(r'<!ENTITY\s+(\w+)\s+"([^"]+\.h5)"\s*>', entity_block))
    refs = re.findall(r"&(\w+);:(/vertex_fields/[A-Za-z0-9_]+)", content)
    print("Checking vertex field dataset references in XDMF:")
    for entity_name, dataset_path in refs:
        h5_file = entities.get(entity_name)
        if not h5_file:
            err = f"[ENTITY NOT FOUND] {entity_name}: {dataset_path}"
            print(err)
            errors.append(err)
            continue
        h5_full_path = os.path.join(tmp_path, h5_file)
        try:
            with h5py.File(h5_full_path, "r") as f:
                h5_path = dataset_path.lstrip("/")
                if h5_path in f:
                    print(f"[OK] {h5_file}: {dataset_path} found")
                else:
                    err = f"[MISSING] {h5_file}: {dataset_path} not found"
                    print(err)
                    errors.append(err)
        except OSError as e:
            err = f"[ERROR] Cannot open {h5_file}: {e}"
            print(err)
            errors.append(err)
    if errors:
        raise AssertionError("Missing or inaccessible vertex fields:\n" + "\n".join(errors))


def remove_test_mesh_files(directory="."):
    """Delete all files starting with test.mesh. in the specified directory."""
    pattern = os.path.join(directory, "test.mesh.*")
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Could not remove {file_path}: {e}")


@pytest.mark.parametrize(
    "mesh",
    [
        structured_quad_box,
        unstructured_quad_box_irregular,
        unstructured_quad_box_regular,
        unstructured_quad_box_irregular_3D,
    ],
)
def test_stokes_boxmesh(mesh, tmp_path):
    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")
    mesh.dm.view()

    if mesh.dim == 2:
        x, y = mesh.X
    else:
        x, y, z = mesh.X

    u = uw.discretisation.MeshVariable("u", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    u2 = uw.discretisation.MeshVariable("u2", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2)
    p2 = uw.discretisation.MeshVariable("p2", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["ksp_monitor"] = None
    stokes.petsc_options["snes_monitor"] = None
    stokes.tolerance = 1.0e-3

    # stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

    stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

    if mesh.dim == 2:
        stokes.bodyforce = 1.0e6 * sympy.Matrix([0, x])

        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, None), "Top")

        stokes.add_dirichlet_bc((0.0, None), "Left")
        stokes.add_condition(conds=(0.0, None), label="Right", f_id=0, c_type="dirichlet")
    else:
        stokes.bodyforce = 1.0e6 * sympy.Matrix([0, x, 0])

        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Top")

        stokes.add_dirichlet_bc((0.0, None, sympy.oo), "Left")
        stokes.add_dirichlet_bc((0.0, sympy.oo, None), "Right")

        stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Front")
        stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Back")

    stokes.solve()

    print(f"Mesh dimensions {mesh.dim}", flush=True)
    # stokes.dm.ds.view()

    assert stokes.snes.getConvergedReason() > 0

    mesh.write_timestep("test", meshUpdates=False, meshVars=[u, p], outputPath=tmp_path, index=0)

    # Call XDMF/HDF5 checker (assume xdmf file is named "test.mesh.xdmf" and written in tmp_path)
    xdmf_filename = os.path.join(tmp_path, "test.mesh.00000.xdmf")
    check_xdmf_vertex_fields_exist_in_h5(xdmf_filename, tmp_path=str(tmp_path))

    u2.read_timestep("test", "u", 0, outputPath=tmp_path)
    p2.read_timestep("test", "p", 0, outputPath=tmp_path)

    with mesh.access():
        assert np.allclose(u.data, u2.data)
        assert np.allclose(p.data, p2.data)

    remove_test_mesh_files(directory=tmp_path)

    del mesh
    del stokes
    print(
        "----------------------------------------------------------------------------------------"
    )
    return
