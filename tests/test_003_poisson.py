import pytest
import underworld3 as uw

structured_quad_box = uw.meshing.StructuredQuadBox(elementRes=(5,)*2)
unstructured_quad_box_irregular = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, regular=False)
unstructured_quad_box_regular = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, regular=True) 
unstructured_quad_box_regular_3D = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0,0.0), 
                                                                     maxCoords=(1.0,1.0,1.0),
                                                                     cellSize=0.1, 
                                                                     regular=True) 

spherical_shell = uw.meshing.SphericalShell()
annulus = uw.meshing.Annulus()
cubed_sphere = uw.meshing.CubedSphere()


# Maybe lower and upper would work better for the names of the box mesh boundaries too.

@pytest.mark.parametrize("mesh", [structured_quad_box, unstructured_quad_box_irregular, 
                                  unstructured_quad_box_regular, unstructured_quad_box_regular_3D])
def test_poisson_boxmesh(mesh):

    u = uw.discretisation.MeshVariable(r"\mathbf{u}", mesh, 1, vtype=uw.VarType.SCALAR, degree=2)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
    poisson.constitutive_model.material_properties = poisson.constitutive_model.Parameters(diffusivity = 1)
    poisson.f = 0.0
    poisson.add_dirichlet_bc( 1., "Bottom")
    poisson.add_dirichlet_bc( 0., "Top")
    poisson.solve()

@pytest.mark.parametrize("mesh", [spherical_shell, annulus, cubed_sphere])
def test_poisson_sphere(mesh):
    u = uw.discretisation.MeshVariable(r"\mathbf{u}", mesh, 1, vtype=uw.VarType.SCALAR, degree=2)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
    poisson.constitutive_model.material_properties = poisson.constitutive_model.Parameters(diffusivity = 1)

    poisson.f = 0.0
    poisson.add_dirichlet_bc( 1., "Lower")
    poisson.add_dirichlet_bc( 0., "Upper")
    poisson.solve()

