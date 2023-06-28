import pytest
import sympy
import underworld3 as uw


annulus = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1, qdegree=2)


spherical_shell = uw.meshing.SphericalShell(
    radiusOuter=1.0, radiusInner=0.5, cellSize=0.5, qdegree=2
)

# cubed_sphere = uw.meshing.CubedSphere(
#     radiusOuter=1.0, radiusInner=0.5, numElements=3, qdegree=2
# )


@pytest.mark.parametrize("mesh", [annulus, spherical_shell])
def test_stokes_sphere(mesh):
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
    stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(u)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

    stokes.tolerance = 1.0e-2
    stokes.petsc_options["ksp_monitor"] = None
    stokes.penalty = 1.0

    if mesh.dim == 2:
        stokes.bodyforce = sympy.Matrix([0, x])
        stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))
        stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
    else:
        stokes.bodyforce = sympy.Matrix([0, x, 0])
        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Lower", (0, 1, 2))
        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Upper", (0, 1, 2))

    stokes.solve()

    assert stokes.snes.getConvergedReason() > 0

    del stokes
    del mesh

    return


del annulus
del spherical_shell
