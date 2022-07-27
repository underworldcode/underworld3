import os
# DISABLE SYMPY CACHE, AS IT GETS IN THE WAY FOR IDENTICALLY NAMED VARIABLES.
# NEED TO FIX.
os.environ["SYMPY_USE_CACHE"]="no"
import underworld3 as uw
import underworld3.function as fn
import numpy as np
import sympy 

n = 10
x = np.linspace(0.1,0.9,n)
y = np.linspace(0.2,0.8,n)
xv, yv = np.meshgrid(x, y, sparse=True)
coords = np.vstack((xv[0,:],yv[:,0])).T

# Python function which generates a polynomial space spanning function of the required degree.
# For example for degree 2:
# tensor_product(2,x,y) = 1 + x + y + x**2*y + x*y**2 + x**2*y**2
def tensor_product(order, val1, val2):
    sum = 0.
    order+=1
    for i in range(order):
        for j in range(order):
            sum+= val1**i*val2**j
    return sum

def test_non_uw_variable_constant():
    result = fn.evaluate(sympy.sympify(1.5), coords)
    assert np.allclose(1.5, result, rtol=1e-05, atol=1e-08)

def test_non_uw_variable_linear():
    mesh = uw.meshing.StructuredQuadBox()
    result = fn.evaluate(mesh.r[0],coords)
    assert np.allclose(x, result, rtol=1e-05, atol=1e-08)

def test_non_uw_variable_sine():
    mesh = uw.meshing.StructuredQuadBox()
    result = fn.evaluate(sympy.sin(mesh.r[1]),coords)
    assert np.allclose(np.sin(y), result, rtol=1e-05, atol=1e-08)

def test_single_scalar_variable():
    mesh = uw.meshing.StructuredQuadBox()
    var  = uw.discretisation.MeshVariable(name="var", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR )
    with mesh.access(var):
        var.data[:]=1.1
    result = fn.evaluate(var.fn,coords)
    assert np.allclose(1.1, result, rtol=1e-05, atol=1e-08)

def test_single_vector_variable():
    mesh = uw.meshing.StructuredQuadBox()
    var  = uw.discretisation.MeshVariable(name="var", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR )
    with mesh.access(var):
        var.data[:]=(1.1,1.2)
    result = uw.function.evaluate(var.fn,coords)
    assert np.allclose(np.array(((1.1,1.2),)), result, rtol=1e-05, atol=1e-08)
   
def test_scalar_vector_mult():
    mesh = uw.meshing.StructuredQuadBox()
    var_scalar  = uw.discretisation.MeshVariable(name="var_scalar", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR )
    var_vector  = uw.discretisation.MeshVariable(name="var_vector", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR )
    with mesh.access(var_scalar, var_vector):
        var_scalar.data[:]=3.
        var_vector.data[:]=(4.,5.)
    result = uw.function.evaluate(var_scalar.fn*var_vector.fn,coords)
    assert np.allclose(np.array(((12.,15),)), result, rtol=1e-05, atol=1e-08)

def test_vector_dot_product():
    mesh = uw.meshing.StructuredQuadBox()
    var_vector1  = uw.discretisation.MeshVariable(name="var_vector1", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR )
    var_vector2  = uw.discretisation.MeshVariable(name="var_vector2", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR )
    with mesh.access(var_vector1, var_vector2):
        var_vector1.data[:]=(1.,2.)
        var_vector2.data[:]=(3.,4.)
    result = uw.function.evaluate(var_vector1.fn.dot(var_vector2.fn),coords)
    assert np.allclose(11., result, rtol=1e-05, atol=1e-08)

def test_many_many_scalar_mult_var():
    mesh = uw.meshing.StructuredQuadBox()
    # Note that this test fails for n>~15. Something something subdm segfault. 
    # Must investigate.
    nn=15
    vars = []
    for i in range(nn):
        vars.append( uw.discretisation.MeshVariable(name=f"var_{i}", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR) )
    factorial = 1.
    with mesh.access(*vars):
        for i, var in enumerate(vars):
            var.data[:] = float(i)
            factorial*=float(i)
    multexpr = vars[0].fn
    for var in vars[1:]:
        multexpr*=var.fn
    result = uw.function.evaluate(multexpr,coords)
    assert np.allclose(factorial, result, rtol=1e-05, atol=1e-08)


# that test needs to be able to take degree as a parameter...
def test_polynomial_mesh_var_degree():

    mesh = uw.meshing.StructuredQuadBox()
    maxdegree = 10
    vars = []

    # Create required vars of different degree.
    for degree in range(maxdegree+1):
        vars.append( uw.discretisation.MeshVariable(name="var"+str(degree), mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR, degree=degree ) )

    # Set variable data to represent polynomial function.
    with mesh.access(*vars):
        for var in vars:
            vcoords = var.coords
            var.data[:,0] = tensor_product(var.degree, vcoords[:,0], vcoords[:,1])

    # Test that interpolated variables reproduce exactly polymial function of associated degree.
    for var in vars:
        result = uw.function.evaluate(var.fn,coords)
        assert np.allclose(tensor_product(var.degree, coords[:,0], coords[:,1]), result, rtol=1e-05, atol=1e-08)


# Let's now do the same, but instead do it Sympy wise.
# We don't really need any UW infrastructure for this test, but it's useful
# to stress our `evaluate()` function. It should however simply reduce
# to Sympy's `lambdify` routine. 
def test_polynomial_sympy():
    degree = 20
    mesh = uw.meshing.StructuredQuadBox()
    assert np.allclose(tensor_product(degree, coords[:,0], coords[:,1]), uw.function.evaluate( tensor_product(degree, mesh.r[0], mesh.r[1]) , coords ), rtol=1e-05, atol=1e-08)


# Now we'll do something similar but involve UW variables.
# Instead of using the Sympy symbols for (x,y), we'll set the 
# coordinate locations on the var data itself.
# For a cartesian mesh, linear elements will suffice. 
# We'll also do it twice, once using (xvar,yvar), and
# another time using (xyvar[0], xyvar[1]).
def test_polynomial_mesh_var_sympy():
    mesh = uw.meshing.StructuredQuadBox()
    xvar = uw.discretisation.MeshVariable(name="xvar", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR )
    yvar = uw.discretisation.MeshVariable(name="yvar", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR )
    xyvar = uw.discretisation.MeshVariable(name="xyvar", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR )
    with mesh.access(xvar,yvar,xyvar):
        # Note that all the `coords` arrays should actually reduce to an identical array,
        # as all vars have identical degree and layout.
        xvar.data[:,0] = xvar.coords[:,0]
        yvar.data[:,0] = yvar.coords[:,1]
        xyvar.data[:] = xyvar.coords[:]
    degree = 10 
    assert np.allclose(tensor_product(degree, coords[:,0], coords[:,1]), uw.function.evaluate( tensor_product(degree, xvar.fn, yvar.fn) , coords ), rtol=1e-05, atol=1e-08)
    assert np.allclose(tensor_product(degree, coords[:,0], coords[:,1]), uw.function.evaluate( tensor_product(degree, xyvar.fn.dot(mesh.N.i), 
                                                                                                                         xyvar.fn.dot(mesh.N.j)) , coords ), rtol=1e-05, atol=1e-08)

# NOTE THAT WE NEEDED TO DISABLE MESH HASHING FOR 3D MESH FOR SOME REASON.
# CHECK `DMInterpolationSetUp_UW()` FOR DETAILS.
def test_3d_cross_product():
    
    # Create a set of evaluation coords in 3d
    n=10
    x = np.linspace(0.1,0.9,n)
    y = np.linspace(0.2,0.8,n)
    z = np.linspace(0.3,0.7,n)
    xv, yv, zv = np.meshgrid(x, y, z, sparse=True)
    coords = np.vstack((xv[0,:,0],yv[:,0,0],zv[0,0,:])).T
    
    # Now mesh and vars etc. 
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4,)*3)
    name = "vector cross product test"
    var_vector1  = uw.discretisation.MeshVariable(name="var_vector1", mesh=mesh, num_components=3, vtype=uw.VarType.VECTOR )
    var_vector2  = uw.discretisation.MeshVariable(name="var_vector2", mesh=mesh, num_components=3, vtype=uw.VarType.VECTOR )
    with mesh.access(var_vector1, var_vector2):
        var_vector1.data[:]=(1.,2.,3.)
        var_vector2.data[:]=(4.,5.,6.)
    result = uw.function.evaluate(var_vector1.fn.cross(var_vector2.fn),coords)
    assert np.allclose(np.array(((-3,6,-3),)), result, rtol=1e-05, atol=1e-08)