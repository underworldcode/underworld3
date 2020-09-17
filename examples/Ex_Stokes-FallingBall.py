# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.stokes import Stokes
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
mesh = uw.Mesh(elementRes=(n_els,n_els),
               minCoords=minCoord, 
               maxCoords=maxCoord )
# %%
v_degree = 1
stokes = Stokes(mesh, u_degree=v_degree, p_degree=v_degree-1 )

# %%
matField = stokes.createAux(num_components=2, isSimplex=0, degree=1)
# get the local auxiliary vector
lVec = stokes.mesh.dm.query("A")
inside = mesh.data[:,0]**2 + mesh.data[:,1]**2 < 0.5**2

f = np.zeros((len(inside),2))
f[:,1] = np.where(inside, 1., 0.)

lVec.array = f.flatten()

# %%
# free slip.  
# note with petsc we always need to provide a vector of correct cardinality. 
bnds = mesh.boundary
stokes.add_dirichlet_bc( (0.,0.), [bnds.TOP,  bnds.BOTTOM], 1 )  # top/bottom: components, function, markers 
stokes.add_dirichlet_bc( (0.,0.), [bnds.LEFT, bnds.RIGHT],  0 )  # left/right: components, function, markers

# %%
stokes.viscosity = 1.
stokes.bodyforce = matField.fn

# %%
# Solve time
stokes.solve()

# %%
uw3_v = stokes.u_local.array.reshape(-1,2)

# %%
import underworld as uw2
from underworld import function as fn
from underworld import visualisation as viz

mesh2 = uw2.mesh.FeMesh_Cartesian(elementRes= (n_els, n_els),
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
vField.data[:] = uw3_v
fField.data[:] = f

# %%
coord = fn.input()
fn_sphere = fn.math.dot( coord, coord ) < 0.5**2
conditions = [ ( fn_sphere , (0.,-1.) ), 
               ( True      , (0.,0.) ) ]
fn_buoyancy = fn.branching.conditional( conditions )

# %%
# check the inputs arrays are similar, NOTE the -1
if not np.allclose(-1*lVec.array.reshape(-1,2) - fn_buoyancy.evaluate(mesh2.data), np.zeros(mesh2.data.shape)):
    raise RuntimeError("Unexpected error: the buoyancy rhs vectors are different!")

# %%

# %% [markdown]
# ## Vis testing ...

# %%
u = stokes.u_local.array.reshape((-1,2))
p = stokes.p_local.array
res = mesh.elementRes

# %matplotlib inline
import matplotlib.pyplot as plt
imgplot = plt.imshow(p.reshape(res), origin='lower')
plt.colorbar(imgplot)
plt.show()

#Underworld3 plotting prototype using lavavu
import plot

#Create viewer
resolution=(500,400)

plot = Plot(rulers=True)
plot.nodes(mesh, pointsize=5, pointtype="sphere")
plot.display(resolution)

plot = Plot(rulers=True)
plot.edges(mesh)
plot.display(resolution)

plot = Plot(rulers=True)
faces = plot.faces(mesh, values=p, colourmap="diverge")
faces.colourbar(align="right", size=(0.865,10), position=26, outline=False)
plot.display(resolution)

plot = Plot(rulers=True)
plot.vector_arrows(mesh, u);
plot.display(resolution)


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

# %%
