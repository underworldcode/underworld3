# %%
import os
# DISABLE SYMPY CACHE, AS IT GETS IN THE WAY FOR IDENTICALLY NAMED VARIABLES.
# NEED TO FIX.
os.environ["SYMPY_USE_CACHE"]="no"
import underworld3 as uw
import underworld3.function
import numpy as np
import sympy 

def test(name, expected, encountered, rtol=1e-05, atol=1e-08):
    if not np.allclose(expected, encountered, rtol, atol):
        raise RuntimeError(f"Test '{name}' failed.\nExpected    = {expected}\nEncountered = {encountered}")

n = 10
x = np.linspace(0.1,0.9,n)
y = np.linspace(0.2,0.8,n)
xv, yv = np.meshgrid(x, y, sparse=True)
coords = np.vstack((xv[0,:],yv[:,0])).T

# %%
name = "non uw variable constant test"
result = uw.function.evaluate(sympy.sympify(1.5),coords)
test(name, 1.5, result)

# %%
name = "non uw variable linear test"
mesh = uw.mesh.Mesh()
result = uw.function.evaluate(mesh.r[0],coords)
test(name, x, result)

# %%
name = "non uw variable sine test"
mesh = uw.mesh.Mesh()
result = uw.function.evaluate(sympy.sin(mesh.r[1]),coords)
test(name, np.sin(y), result)

# %%
name = "single scalar variable test"
mesh = uw.mesh.Mesh()
var  = uw.mesh.MeshVariable(name="var", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR )
with mesh.access(var):
    var.data[:]=1.1
result = uw.function.evaluate(var.fn,coords)
test(name, 1.1, result)

# %%
name = "single vector variable test"
mesh = uw.mesh.Mesh()
var  = uw.mesh.MeshVariable(name="var", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR )
with mesh.access(var):
    var.data[:]=(1.1,1.2)
result = uw.function.evaluate(var.fn,coords)
test(name, np.array(((1.1,1.2),)), result)

# %%
name = "scalar*vector mult test"
mesh = uw.mesh.Mesh()
var_scalar  = uw.mesh.MeshVariable(name="var_scalar", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR )
var_vector  = uw.mesh.MeshVariable(name="var_vector", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR )
with mesh.access(var_scalar, var_vector):
    var_scalar.data[:]=3.
    var_vector.data[:]=(4.,5.)
result = uw.function.evaluate(var_scalar.fn*var_vector.fn,coords)
test(name, np.array(((12.,15),)), result)

# %%
name = "vector dot product test"
mesh = uw.mesh.Mesh()
var_vector1  = uw.mesh.MeshVariable(name="var_vector1", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR )
var_vector2  = uw.mesh.MeshVariable(name="var_vector2", mesh=mesh, num_components=2, vtype=uw.VarType.VECTOR )
with mesh.access(var_vector1, var_vector2):
    var_vector1.data[:]=(1.,2.)
    var_vector2.data[:]=(3.,4.)
result = uw.function.evaluate(var_vector1.fn.dot(var_vector2.fn),coords)
test(name, 11., result)

# %%
name = "many many scalar mult var test"
mesh = uw.mesh.Mesh()
# Note that this test fails for n>~15. Something something subdm segfault. 
# Must investigate.
nn=15
vars = []
for i in range(nn):
    vars.append( uw.mesh.MeshVariable(name=f"var_{i}", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR) )
factorial = 1.
with mesh.access(*vars):
    for i, var in enumerate(vars):
        var.data[:] = float(i)
        factorial*=float(i)
multexpr = vars[0].fn
for var in vars[1:]:
    multexpr*=var.fn
result = uw.function.evaluate(multexpr,coords)
test(name, factorial, result)

# %%
name = "polynomial mesh var test"
mesh = uw.mesh.Mesh()
var0 = uw.mesh.MeshVariable(name="var0", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR, degree=0 )
var1 = uw.mesh.MeshVariable(name="var1", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR, degree=1 )
var2 = uw.mesh.MeshVariable(name="var2", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR, degree=2 )
var3 = uw.mesh.MeshVariable(name="var3", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR, degree=3 )
var4 = uw.mesh.MeshVariable(name="var4", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR, degree=4 )
var5 = uw.mesh.MeshVariable(name="var5", mesh=mesh, num_components=1, vtype=uw.VarType.SCALAR, degree=5 )
def tensor_product(order, v1, v2):
    sum = 0.
    order+=1
    for i in range(order):
        for j in range(order):
            sum+= v1**i*v2**j
    return sum
with mesh.access(var0,var1,var2,var3,var4,var5):
    vcoords = var0.coords
    var0.data[:,0] = tensor_product(0, vcoords[:,0], vcoords[:,1])
    vcoords = var1.coords
    var1.data[:,0] = tensor_product(1, vcoords[:,0], vcoords[:,1])
    vcoords = var2.coords
    var2.data[:,0] = tensor_product(2, vcoords[:,0], vcoords[:,1])
    vcoords = var3.coords
    var3.data[:,0] = tensor_product(3, vcoords[:,0], vcoords[:,1])
    vcoords = var4.coords
    var4.data[:,0] = tensor_product(4, vcoords[:,0], vcoords[:,1])
    vcoords = var5.coords
    var5.data[:,0] = tensor_product(5, vcoords[:,0], vcoords[:,1])

result = uw.function.evaluate(var0.fn,coords)
test(name+" degree 0", tensor_product(0, coords[:,0], coords[:,1]), result)
result = uw.function.evaluate(var1.fn,coords)
test(name+" degree 1", tensor_product(1, coords[:,0], coords[:,1]), result)
result = uw.function.evaluate(var2.fn,coords)
test(name+" degree 2", tensor_product(2, coords[:,0], coords[:,1]), result)
result = uw.function.evaluate(var3.fn,coords)
test(name+" degree 3", tensor_product(3, coords[:,0], coords[:,1]), result)
result = uw.function.evaluate(var4.fn,coords)
test(name+" degree 4", tensor_product(4, coords[:,0], coords[:,1]), result)
result = uw.function.evaluate(var5.fn,coords)
test(name+" degree 5", tensor_product(5, coords[:,0], coords[:,1]), result)


n=10
x = np.linspace(0.1,0.9,n)
y = np.linspace(0.2,0.8,n)
z = np.linspace(0.3,0.7,n)
xv, yv, zv = np.meshgrid(x, y, z, sparse=True)
coords = np.vstack((xv[0,:,0],yv[:,0,0],zv[0,0,:])).T

# %%
# NOTE THAT WE NEEDED TO DISABLE MESH HASHING FOR 3D MESH FOR SOME REASON.
# CHECK `DMInterpolationSetUp_UW()` FOR DETAILS.
name = "3d cross product test"
mesh = uw.mesh.Mesh(elementRes=(4,)*3)
name = "vector cross product test"
var_vector1  = uw.mesh.MeshVariable(name="var_vector1", mesh=mesh, num_components=3, vtype=uw.VarType.VECTOR )
var_vector2  = uw.mesh.MeshVariable(name="var_vector2", mesh=mesh, num_components=3, vtype=uw.VarType.VECTOR )
with mesh.access(var_vector1, var_vector2):
    var_vector1.data[:]=(1.,2.,3.)
    var_vector2.data[:]=(4.,5.,6.)
result = uw.function.evaluate(var_vector1.fn.cross(var_vector2.fn),coords)
test(name, np.array(((-3,6,-3),)), result)

