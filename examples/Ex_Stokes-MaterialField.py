# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np

options = PETSc.Options()
# options["help"] = None

# options["pc_type"]  = "svd"

options["ksp_rtol"] =  1.0e-6
options["ksp_monitor_short"] = None

# options["snes_type"]  = "fas"
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
options["snes_rtol"] = 1.0e-2  # set this low to force single SNES it. 
# options["snes_max_it"] = 1

# %%
n_els = 32
minCoord = (-1,-1)
maxCoord = (1,1)
mesh = uw.mesh.Mesh(elementRes=(n_els,n_els),
               minCoords=minCoord, 
               maxCoords=maxCoord )
# %%
v_degree = 1
stokes = Stokes(mesh, u_degree=v_degree, p_degree=v_degree-1 )

# %%
# add & init the mat var
matvar = uw.mesh.MeshVariable( mesh=mesh, num_components=1, name="mat", vtype=uw.VarType.SCALAR, degree=v_degree )

inside = mesh.data[:,0]**2 + mesh.data[:,1]**2 < 0.5**2
with mesh.access(matvar):
    matvar.data[:,0] = np.where(inside, 1., 0.)

# %%
# free slip.  
# note with petsc we always need to provide a vector of correct cardinality. 
bnds = mesh.boundary
stokes.add_dirichlet_bc( (0.,0.), [bnds.TOP,  bnds.BOTTOM], 1 )  # top/bottom: components, function, markers 
stokes.add_dirichlet_bc( (0.,0.), [bnds.LEFT, bnds.RIGHT],  0 )  # left/right: components, function, markers

# %%
N = mesh.N
stokes.viscosity = 1.
stokes.bodyforce =  matvar * N.j

# %%
# Solve time
stokes.solve()

# %%
import underworld as uw2
from underworld import function as fn
from underworld import visualisation as viz

mesh2 = uw2.mesh.FeMesh_Cartesian(elementRes=(n_els, n_els),
                          minCoord=minCoord,
                          maxCoord=maxCoord)

# %%
# uw2 fields
v2Field = mesh2.add_variable(nodeDofCount=2)
pField = mesh2.subMesh.add_variable(nodeDofCount=1)

# uw3 fields
fField = mesh2.add_variable(nodeDofCount=2)
vField = mesh2.add_variable(nodeDofCount=2)

# %%
with mesh.access():
    vField.data[:] = stokes.u.data[:]
fField.data[:, 1] = np.where(inside, 1., 0.)

# %%
coord = fn.input()
fn_sphere = fn.math.dot( coord, coord ) < 0.5**2
conditions = [ ( fn_sphere , (0.,1.) ), 
               ( True      , (0.,0.) ) ]
fn_buoyancy = fn.branching.conditional( conditions )

# %%
# %matplotlib inline
with mesh.access():
    import matplotlib.pyplot as plt
    imgplot = plt.imshow(stokes.p.data.reshape(mesh.elementRes), origin='lower')
    plt.colorbar(imgplot)
    plt.show()

# %%
#Underworld3 plotting prototype using lavavu
with mesh.access():
    import plot
    resolution=(500,400)
    fig = plot.Plot(rulers=True)
    fig.vector_arrows(mesh, stokes.u.data)
    fig.display(resolution)


# %%
iWalls = mesh2.specialSets["MinI_VertexSet"] + mesh2.specialSets["MaxI_VertexSet"]
jWalls = mesh2.specialSets["MinJ_VertexSet"] + mesh2.specialSets["MaxJ_VertexSet"]

freeslipBC = uw2.conditions.DirichletCondition( variable       = v2Field, 
                                               indexSetsPerDof = (iWalls, jWalls) )

stokes2 = uw2.systems.Stokes(   velocityField = v2Field, 
                               pressureField = pField, 
                               conditions    = freeslipBC,
                               fn_viscosity  = 1., 
                               fn_bodyforce  = fn_buoyancy )
solver = uw2.systems.Solver( stokes2 )

solver.solve()

# %%
# check the input
fig = viz.Figure()
fig.append(viz.objects.Surface(mesh2, fField[1] ))
fig.show()

# %%
fig = viz.Figure()
fig.append(viz.objects.VectorArrows(mesh2, vField ))
fig.show()

# %%
fig = viz.Figure()
fig.append(viz.objects.VectorArrows(mesh2, v2Field ))
fig.show()

# %%
# some analytics
vmag = fn.math.sqrt(fn.math.dot(vField, vField))
err  = vField - v2Field
l2   = fn.math.sqrt(fn.math.dot(err, err))

# %%
fig = viz.Figure()
fig.append(viz.objects.Surface(mesh2, vmag ))
fig.show()

# %%
fig = viz.Figure()
fig.append(viz.objects.Surface(mesh2, l2 ))
fig.show()

# %%
fig = viz.Figure()
fig.append(viz.objects.VectorArrows(mesh2, vField-v2Field ))
fig.show()

# %%
from numpy import linalg as LA
l2diff = LA.norm(vField.data - v2Field.data)

# %%
# was 2.79351e-2 @ nEls=16
# was 1.74721e-2 @ nEls=32
# was 1.25260e-2 @ nEls=64
if l2diff > 1.76e-2:
    raise RuntimeError("Unexpected results")