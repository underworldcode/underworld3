import pytest
import underworld3 as uw

structured_quad_box = uw.util_mesh.StructuredQuadBox(elementRes=(5,)*2)
unstructured_quad_box_irregular = uw.util_mesh.UnstructuredSimplexBox(cellSize=0.1, regular=False)
unstructured_quad_box_regular = uw.util_mesh.UnstructuredSimplexBox(cellSize=0.1, regular=True) 
spherical_shell = uw.util_mesh.SphericalShell()
annulus = uw.util_mesh.Annulus()
cubic_sphere = uw.util_mesh.CubicSphere()


@pytest.mark.parametrize("mesh", [structured_quad_box, unstructured_quad_box_irregular, unstructured_quad_box_regular])
def test_poisson_boxmesh(mesh):
    poisson = uw.systems.Poisson(mesh)
    poisson.k = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc( 1., "Bottom")
    poisson.add_dirichlet_bc( 0., "Top")
    poisson.solve()

@pytest.mark.parametrize("mesh", [spherical_shell, annulus, cubic_sphere])
def test_poisson_sphere(mesh):
    poisson = uw.systems.Poisson(mesh)
    poisson.k = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc( 1., "Lower")
    poisson.add_dirichlet_bc( 0., "Upper")
    poisson.solve()

