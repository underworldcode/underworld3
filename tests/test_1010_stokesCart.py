import pytest
import sympy
import underworld3 as uw

# These are tested by test_001_meshes.py

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

# Maybe lower and upper would work better for the names of the box mesh boundaries too.


@pytest.mark.parametrize(
    "mesh",
    [
        structured_quad_box,
        unstructured_quad_box_irregular,
        unstructured_quad_box_irregular_3D,
    ],
)
def test_stokes_boxmesh(mesh):
    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")
    mesh.dm.view()

    if mesh.dim == 2:
        x, y = mesh.X
    else:
        x, y, z = mesh.X

    u = uw.discretisation.MeshVariable(
        r"mathbf{u}", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2
    )
    p = uw.discretisation.MeshVariable(
        r"mathbf{p}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
    )

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
        stokes.add_condition(
            conds=(0.0, None), label="Right", f_id=0, c_type="dirichlet"
        )
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
    stokes.dm.ds.view()

    assert stokes.snes.getConvergedReason() > 0

    return


def test_stokes_boxmesh_bc_failure():
    mesh = uw.meshing.UnstructuredSimplexBox(
        cellSize=0.2, regular=False, qdegree=2, refinement=1
    )

    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")
    mesh.dm.view()

    if mesh.dim == 2:
        x, y = mesh.X
    else:
        x, y, z = mesh.X

    u = uw.discretisation.MeshVariable(
        r"mathbf{u}", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2
    )
    p = uw.discretisation.MeshVariable(
        r"mathbf{p}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
    )

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    stokes.petsc_options["snes_monitor"] = None
    stokes.petsc_options["ksp_monitor"] = None

    # stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

    stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

    if mesh.dim == 2:
        stokes.bodyforce = 1.0e6 * sympy.Matrix([0, x])

        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0), "Top")

        stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
        stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
    else:
        stokes.bodyforce = 1.0e6 * sympy.Matrix([0, x, 0])

        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Top")

        stokes.add_dirichlet_bc((0.0, sympy.oo, sympy.oo), "Left")
        stokes.add_dirichlet_bc((0.0, sympy.oo, sympy.oo), "Right")

        stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Front")
        stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Back")

    stokes.solve()

    print(f"Mesh dimensions {mesh.dim}", flush=True)
    stokes.dm.ds.view()

    assert stokes.snes.getConvergedReason() > 0


def test_stokes_viscosity_with_fn():
    mesh = uw.meshing.UnstructuredSimplexBox(
        cellSize=0.21, regular=False, qdegree=2, refinement=1
    )

    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")
    mesh.dm.view()

    if mesh.dim == 2:
        x, y = mesh.X
    else:
        x, y, z = mesh.X

    u = uw.discretisation.MeshVariable(
        r"mathbf{u}", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2
    )
    p = uw.discretisation.MeshVariable(
        r"mathbf{p}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
    )

    viscosity_fn = 1 + sympy.sin(x) ** 2

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    stokes.petsc_options["snes_monitor"] = None
    stokes.petsc_options["ksp_monitor"] = None

    # stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

    stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

    if mesh.dim == 2:
        stokes.bodyforce = 1.0e6 * sympy.Matrix([0, x])

        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0), "Top")

        stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
        stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
    else:
        stokes.bodyforce = 1.0e6 * sympy.Matrix([0, x, 0])

        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Top")

        stokes.add_dirichlet_bc((0.0, sympy.oo, sympy.oo), "Left")
        stokes.add_dirichlet_bc((0.0, sympy.oo, sympy.oo), "Right")

        stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Front")
        stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Back")

    stokes.solve()

    print(f"Mesh dimensions {mesh.dim}", flush=True)
    stokes.dm.ds.view()

    assert stokes.snes.getConvergedReason() > 0

    return


def test_stokes_viscosity_with_variable_fn():
    mesh = uw.meshing.UnstructuredSimplexBox(
        cellSize=0.21, regular=False, qdegree=2, refinement=1
    )

    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")
    mesh.dm.view()

    if mesh.dim == 2:
        x, y = mesh.X
    else:
        x, y, z = mesh.X

    u = uw.discretisation.MeshVariable(
        r"mathbf{u}", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2
    )
    p = uw.discretisation.MeshVariable(
        r"mathbf{p}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
    )

    s = uw.discretisation.MeshVariable(
        r"mathbf{s}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
    )

    try:
        print(f"u.array access: {u.array is not None}")
        print(f"p.array access: {p.array is not None}")
        print(f"s.array access: {s.array is not None}")
        print("✓ All direct access works")
    except Exception as e:
        print(f"✗ Direct access failed: {e}")

    s.array[...] = uw.function.evaluate(x + y, s.coords, rbf=False)

    viscosity_fn = sympy.Piecewise(
        (sympy.sympify(1) / 100, s.sym[0, 0] < 0.05),
        (
            1,
            True,
        ),
    )

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    stokes.petsc_options["snes_monitor"] = None
    stokes.petsc_options["ksp_monitor"] = None

    # stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

    stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

    if mesh.dim == 2:
        stokes.bodyforce = 1.0e6 * sympy.Matrix([0, x])

        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0), "Top")

        stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
        stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
    else:
        stokes.bodyforce = 1.0e6 * sympy.Matrix([0, x, 0])

        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Top")

        stokes.add_dirichlet_bc((0.0, sympy.oo, sympy.oo), "Left")
        stokes.add_dirichlet_bc((0.0, sympy.oo, sympy.oo), "Right")

        stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Front")
        stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Back")

    stokes.solve()

    print(f"Mesh dimensions {mesh.dim}", flush=True)
    stokes.dm.ds.view()

    assert stokes.snes.getConvergedReason() > 0

    return
