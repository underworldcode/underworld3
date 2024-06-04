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
        unstructured_quad_box_regular,
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
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.constitutive_model.Parameters.shear_modulus = 1
    stokes.constitutive_model.Parameters.dt_elastic = sympy.sympify(1) / 10

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

    del mesh
    del stokes

    return


## Note this one fails because the corner boundary condition is not applied
## correctly when the regular simplex mesh is used.
## Mark as xfail for now


@pytest.mark.parametrize(
    "mesh",
    [
        unstructured_quad_box_regular,
    ],
)
def test_stokes_boxmesh_bc_failure(mesh):
    # mesh = unstructured_quad_box_regular

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
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.constitutive_model.Parameters.shear_modulus = 1
    stokes.constitutive_model.Parameters.dt_elastic = sympy.sympify(1) / 10

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
        stokes.add_dirichlet_bc((0.0, None), "Top")

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

    del mesh
    del stokes

    return


del structured_quad_box
del unstructured_quad_box_regular
del unstructured_quad_box_irregular
del unstructured_quad_box_irregular_3D
