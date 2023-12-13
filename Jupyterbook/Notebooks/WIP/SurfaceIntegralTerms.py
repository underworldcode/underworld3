# ## Poisseuille flow test
#
#

# +
import underworld3 as uw
import numpy as np
import sympy
import os
import sys
import petsc4py
import matplotlib.pyplot as plt

import nest_asyncio
nest_asyncio.apply()

# +
minX = -1.0
maxX = 1.0
minY = 0.0
maxY = 1.0

resX = 6
resY = 3

cell_height = maxY / resY
cell_width = maxX / resX

mesh = uw.meshing.StructuredQuadBox(
        elementRes=(resX, resY), 
        minCoords=(minX, minY), 
        maxCoords=(maxX, maxY), qdegree=3)

x, y = mesh.X

# +
stokes = uw.systems.Stokes(mesh)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

v = stokes.Unknowns.u
p = stokes.Unknowns.p

v0 = v.clone("V0", r"V^{[0]}")
v1 = v.clone("V1", r"V^{[1]}")
v2 = v.clone("V2", r"V^{[2]}")


# +
stokes.add_essential_bc( [0.,0.], "Bottom")  # no slip on the base
stokes.add_essential_bc( [0.,0.], "Top")     # no slip on the top

stokes.bodyforce = sympy.Matrix([0, sympy.sin(x*sympy.pi)])

# +
### see the SNES output
stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor_short"] = None
stokes.petsc_options["snes_atol"] = 1.0e-8
stokes.tolerance = 1.0e-4

stokes.solve()

with mesh.access(v0):
    v0.data[...] = v.data[...]

# +
# Extract the Jacobian / Preconditioner for this system

J0, P0, _ = stokes.snes.getJacobian()
ii,jj = J0.getSize()

# -




# +
## Let's take a look (Sorry, Matt, using python for this)

fig = plt.figure(figsize=(12,6), facecolor="none")
ax  = plt.subplot(131)   # 2x1 array of plots, ax refers to the 1st of them
ax2  = plt.subplot(132)   # 1x1 array of plots, ax refers to the 1st of them
ax3  = plt.subplot(133)   # 1x1 array of plots, ax refers to the 1st of them

ax.imshow(J0.getValues(range(0,ii),range(0,jj)), vmin=-1, vmax=1, cmap="coolwarm")
ax2.imshow(P0.getValues(range(0,ii),range(0,jj)), vmin=-1, vmax=1, cmap="coolwarm")
ax3.imshow((J0-P0).getValues(range(0,ii),range(0,jj)), vmin=-0.1, vmax=0.1, cmap="coolwarm")

pass
##

# +
## Now try adding a null natural bc - Expect no changes to J, P though 
## the surface terms should be active if we check the debug output

stokes.add_natural_bc(   [0.0, 0.0], "Left")  
stokes.solve(verbose=False, debug=False, zero_init_guess=True, picard=0, _force_setup=True)


# +
J1, P1, _ = stokes.snes.getJacobian()
V1, _ = stokes.snes.getFunction()

assert(J1.equal(J0))
assert(P1.equal(P0))
# +
## Now try adding a constant natural bc. J, P should be unchanged as this is 
## linear, but we should see a different RHS reflected in the SNES norms

stokes.natural_bcs = []
stokes.essential_bcs = []
stokes.dm = None
stokes._is_setup = False

stokes.add_essential_bc( [0.,0.], "Bottom")  # no slip on the base
stokes.add_essential_bc( [0.,0.], "Top")     # no slip on the top
stokes.add_natural_bc(   [1, 0], "Left")  

stokes.solve(verbose=False, debug=False, zero_init_guess=True, picard=0, _force_setup=True)



# +
J2, P2, _ = stokes.snes.getJacobian()
V2, _ = stokes.snes.getFunction()

assert(J2.equal(J0))
assert(P2.equal(P0))

# +
## Now try adding a non-linear natural bc. Now we expect to see J and P 
## have additional terms due to the non-linearity. Let's apply a force 

stokes.natural_bcs = []
stokes.essential_bcs = []
stokes.dm = None
stokes._is_setup = False

stokes.add_essential_bc( [0.,0.], "Bottom")  # no slip on the base
stokes.add_essential_bc( [0.,0.], "Top")     # no slip on the top
stokes.add_natural_bc(   [-1.0 * v.sym[0], 0.0], "Left")  
stokes.solve(verbose=False, debug=False, zero_init_guess=True, picard=0, _force_setup=True)

with mesh.access(v1):
    v1.data[...] = v.data[...]


# +
J3, P3, _ = stokes.snes.getJacobian()
V3, _ = stokes.snes.getFunction()
ii,jj = J3.getSize()

J3diff = (J3-J0).getValues(range(0,ii),range(0,jj))
P3diff = (P3-P0).getValues(range(0,ii),range(0,jj))
PJ3diff = (P3-J3).getValues(range(0,ii),range(0,jj))
PJ0diff = (P0-J0).getValues(range(0,ii),range(0,jj))

# assert(J3.equal(J0) == False) - Fails !!
assert(P3.equal(P0) == False)


# +

fig = plt.figure(figsize=(12,6), facecolor="none")
ax  = plt.subplot(131)   # 2x1 array of plots, ax refers to the 1st of them
ax2  = plt.subplot(132)   # 1x1 array of plots, ax refers to the 1st of them
ax3  = plt.subplot(133)   # 1x1 array of plots, ax refers to the 1st of them

ax.imshow(PJ3diff, vmin=-0.01, vmax=0.01, cmap="coolwarm")
ax2.imshow(J3diff, vmin=-0.01, vmax=0.01, cmap="coolwarm")
ax3.imshow(P3diff, vmin=-0.01, vmax=0.01, cmap="coolwarm")


# +
## Validation step - try this with a clean solver

stokesN = uw.systems.Stokes(mesh)
stokesN.constitutive_model = uw.constitutive_models.ViscousFlowModel
vN = stokesN.Unknowns.u

stokesN.tolerance = 1.0e-4

stokesN.add_essential_bc( [0.,0.], "Bottom")  # no slip on the base
stokesN.add_essential_bc( [0.,0.], "Top")     # no slip on the top
stokesN.add_natural_bc(   [-1 * vN.sym[0],0.0], "Left")  

stokesN.bodyforce = sympy.Matrix([0, sympy.sin(x*sympy.pi)])

stokesN.solve(verbose=False, debug=False, zero_init_guess=True, picard=0, _force_setup=True)

J4, P4, _ = stokes.snes.getJacobian()
V4, _ = stokes.snes.getFunction()
ii,jj = J4.getSize()

## Is there any difference ?
assert(J4.equal(J3))
assert(P4.equal(P3))
# -





# +
## Sigh ... without a proper Jacobian, we can do this

# +
with mesh.access(v):
    v.data[...] = 0.0

for pen in [100, 1000, 100000]:

    print(f"penalty -> {pen}")

    stokes.natural_bcs = []
    stokes.essential_bcs = []
    stokes.dm = None
    stokes._is_setup = False
    
    stokes.add_dirichlet_bc( [0.,0.], "Bottom")  # no slip on the base
    stokes.add_dirichlet_bc( [0.,0.], "Top")     # no slip on the top
    stokes.add_natural_bc(   [pen * v.sym[0], 0.0], "Left")  # pushed

    stokes.bodyforce = sympy.Matrix([0.0, sympy.sin(x*sympy.pi)])

    stokes.solve(verbose=False, debug=False, zero_init_guess=False, picard=5, _force_setup=True)
    stokes.petsc_options["snes_type"] = "newtontr"

with mesh.access(v1):
    v1.data[...] = v.data[...]
# +
## in order to get an approximate free-slip boundary 

stokes.natural_bcs = []
stokes.essential_bcs = []
stokes.dm = None
stokes._is_setup = False

stokes.add_essential_bc( [0.,0.], "Bottom")  # no slip on the base
stokes.add_essential_bc( [0.,0.], "Top")     # no slip on the top
stokes.add_essential_bc( [0.,sympy.oo], "Left")     # free slip on the Left

stokes.bodyforce = sympy.Matrix([0.0, sympy.sin(x*sympy.pi)])

stokes.solve(verbose=False, debug=False, zero_init_guess=True, picard=0, _force_setup=True)

with mesh.access(v0):
    v0.data[...] = v.data[...]


# +
# with mesh.access(v):
#     v.data[...] = 0.0

# left_boundary_mask = uw.maths.delta_function(x+1, epsilon=0.05)
# norm = sympy.sympify(1) #  / left_boundary_mask.subs(x,-1)
# left_boundary_penalty = sympy.sympify(1) * left_boundary_mask #  * norm

# for pen in [0, 1000, 100000]:

#     stokes.natural_bcs = []
#     stokes.essential_bcs = []
#     stokes.dm = None
#     stokes._is_setup = False
    
#     stokes.add_dirichlet_bc( [0.,0.], "Bottom")  # no slip on the base
#     stokes.add_dirichlet_bc( [0.,0.], "Top")     # no slip on the top
#     stokes.petsc_options["snes_type"] = "newtontr"

#     stokes.bodyforce = sympy.Matrix([-pen * left_boundary_penalty * v.sym[0], sympy.sin(x*sympy.pi)])

#     stokes.solve(verbose=False, debug=False, zero_init_guess=False)
    
# with mesh.access(v2):
#     v2.data[...] = v.data[...]
# -



# +
# Visuals


import underworld3 as uw
import pyvista as pv
import underworld3.visualisation

pl = pv.Plotter(window_size=(1000, 500))

pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)
pvmesh.point_data["V0"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, v0.sym)
pvmesh.point_data["V1"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, v1.sym)
pvmesh.point_data["V2"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, v2.sym)
pvmesh.point_data["P"] = uw.visualisation.scalar_fn_to_pv_points(pvmesh, p.sym)
pvmesh.point_data["Vmag"] = uw.visualisation.scalar_fn_to_pv_points(pvmesh, v.sym.dot(v.sym))

velocity_points = underworld3.visualisation.meshVariable_to_pv_cloud(v)
velocity_points.point_data["V0"] = uw.visualisation.vector_fn_to_pv_points(velocity_points, v0.sym)
velocity_points.point_data["V1"] = uw.visualisation.vector_fn_to_pv_points(velocity_points, v1.sym)
velocity_points.point_data["V2"] = uw.visualisation.vector_fn_to_pv_points(velocity_points, v2.sym)

pl.add_mesh(
    pvmesh,
    cmap="coolwarm",
    edge_color="Black",
    show_edges=True,
    scalars="P",
    use_transparency=False,
    opacity=1.0,
)

pl.add_arrows(velocity_points.points, velocity_points.point_data["V1"]-velocity_points.point_data["V0"], mag=100000, opacity=0.75)
pl.add_arrows(velocity_points.points+(0.01,0.0,0.0), velocity_points.point_data["V1"], mag=10, opacity=0.75)
pl.add_arrows(velocity_points.points, velocity_points.point_data["V0"], mag=10, opacity=0.75)
# pl.add_mesh(pvstream)

pl.camera.SetPosition(0.75, 0.2, 1.5)
pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
pl.camera.SetClippingRange(1.0, 8.0)

# pl.remove_scalar_bar("Omega")
# pl.remove_scalar_bar("mag")
# pl.remove_scalar_bar("V")

pl.show()

# -
# # 




