import pytest
import underworld3 as uw
import sympy
import numpy as np

'''
Unit test for Natural BCs in a Poisson (scalar) problem. 
'''

res   = 16

width   = 1
height  = 1

minX, maxX = 0, width
minY, maxY = 0, height

mesh_simp_reg = uw.meshing.UnstructuredSimplexBox(  minCoords   = (minX, minY), 
                                                    maxCoords   = (maxX, maxY), 
                                                    cellSize    = 1 / res, 
                                                    qdegree     = 3,
                                                    regular = True)
mesh_simp_irreg = uw.meshing.UnstructuredSimplexBox(minCoords   = (minX, minY), 
                                                    maxCoords   =(maxX, maxY), 
                                                    cellSize    = 1 / res, 
                                                    regular     = False)
mesh_quad = uw.meshing.StructuredQuadBox(minCoords  = (minX, minY), 
                                        maxCoords   = (maxX, maxY), 
                                        elementRes  = (res, res))

@pytest.mark.parametrize("mesh", 
                         [mesh_simp_reg,
                          mesh_simp_irreg,
                          mesh_quad]
)

def test_poisson_natural_bc(mesh):

    T_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree = 2)

    poisson = uw.systems.Poisson(mesh = mesh, 
                                u_Field = T_soln, 
                                degree = 2,
                                verbose = False,
                                )
    
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.petsc_options.delValue("ksp_monitor")

    # set the source based on analytical solution
    x, y = mesh.N.x, mesh.N.y
    ana_soln = (x**2) * y 

    #poisson.tolerance = 1e-6 # increase tolerance to decrease atol in allclose
    poisson.f = -2 * y

    poisson.add_natural_bc(0.0, "Left")
    poisson.add_natural_bc([2*y], "Right")
    poisson.add_natural_bc([x**2], "Bottom")
    poisson.add_natural_bc([x**2], "Top")

    poisson.add_dirichlet_bc(0.0, "Left")
    poisson.add_dirichlet_bc([y], "Right")
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc([x**2], "Top")

    poisson.solve()

    with mesh.access():
        num = T_soln.data[:].squeeze()
        ana = uw.function.evaluate(ana_soln, T_soln.coords)

    assert np.allclose(ana, num, atol = 1e-4), "Numerical and analytical solutions differ!"

    del poisson 
    del mesh