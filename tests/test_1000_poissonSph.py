import pytest
import underworld3 as uw


annulus = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1, qdegree=2)


spherical_shell = uw.meshing.SphericalShell(
    radiusOuter=1.0, radiusInner=0.5, cellSize=0.5, qdegree=2
)

# cubed_sphere = uw.meshing.CubedSphere(
#     radiusOuter=1.0, radiusInner=0.5, numElements=3, qdegree=2
# )

# Maybe lower and upper would work better for the names of the box mesh boundaries too.


@pytest.mark.parametrize("mesh", [annulus, spherical_shell])
def test_poisson_sphere(mesh):
    u = uw.discretisation.MeshVariable(
        r"mathbf{u}", mesh, 1, vtype=uw.VarType.SCALAR, degree=2
    )

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.systems.constitutive_models.DiffusionModel(u)
    poisson.constitutive_model.Parameters.diffusivity = 1

    poisson.f = 0.0
    poisson.add_dirichlet_bc(1.0, "Lower", 0)
    poisson.add_dirichlet_bc(0.0, "Upper", 0)
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    del poisson
    del mesh
