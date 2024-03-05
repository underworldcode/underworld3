# ## Surface integral / natural boundary conditions 
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
minY = -1.0
maxY = 1.0

resX = 4
resY = 4

cell_height = maxY / resY
cell_width = maxX / resX

meshQuad = uw.meshing.StructuredQuadBox(
        elementRes=(resX, resY), 
        minCoords=(minX, minY), 
        maxCoords=(maxX, maxY), qdegree=3)

meshTri = uw.meshing.UnstructuredSimplexBox(
        regular=True,
        cellSize=1/2, 
        minCoords=(minX, minY), 
        maxCoords=(maxX, maxY), qdegree=3)

mesh = meshTri
x,y = mesh.X

# -

uw.systems.Stokes.view()

# +

stokes0 = uw.systems.Stokes(mesh, solver_name="Stokes0")
stokes0.constitutive_model = uw.constitutive_models.ViscousFlowModel

v0 = stokes0.Unknowns.u
p0 = stokes0.Unknowns.p

stokes0.add_essential_bc( [0.,0.], "Bottom")  # no slip on the base
stokes0.add_essential_bc( [0.,sympy.oo], "Left")     # free slip Left/Right
stokes0.add_essential_bc( [0.,sympy.oo], "Right")     # no slip on the top
stokes0.bodyforce = sympy.Matrix([0, sympy.sin(x*sympy.pi)])

### see the SNES output
stokes0.petsc_options["snes_converged_reason"] = None
stokes0.petsc_options["snes_monitor_short"] = None
stokes0.tolerance = 1.0e-6

stokes0.solve()



# +
def JacViewer(stokes_solver):

    Jac, JacP, _ = stokes_solver.snes.getJacobian()
    ii,jj = Jac.getSize()

    ## Let's take a look (Sorry, Matt, using python for this)

    Jm = Jac.getValues(range(0,ii),range(0,jj))
    JPm = JacP.getValues(range(0,ii),range(0,jj))

    fig = plt.figure(figsize=(12,6), facecolor="none")
    ax  = plt.subplot(131)   # 2x1 array of plots, ax refers to the 1st of them
    ax2  = plt.subplot(132)   # 1x1 array of plots, ax refers to the 1st of them
    ax3  = plt.subplot(133)   # 1x1 array of plots, ax refers to the 1st of them
    
    ax.imshow(Jm, vmin=-1, vmax=1, cmap="coolwarm")
    ax2.imshow(JPm, vmin=-1, vmax=1, cmap="coolwarm")
    ax3.imshow(JPm-Jm, vmin=-0.01, vmax=0.01, cmap="coolwarm")
    
def JacDiffViewer(stokes_solver1, stokes_solver2):

    Jac1, JacP1, _ = stokes_solver1.snes.getJacobian()
    ii,jj = Jac1.getSize()

    Jac2, JacP2, _ = stokes_solver2.snes.getJacobian()

    
    ## Let's take a look (Sorry, Matt, using python for this)

    Jm1 = Jac1.getValues(range(0,ii),range(0,jj))
    JPm1 = JacP1.getValues(range(0,ii),range(0,jj))
    
    Jm2 = Jac2.getValues(range(0,ii),range(0,jj))
    JPm2 = JacP2.getValues(range(0,ii),range(0,jj))

    fig = plt.figure(figsize=(12,6), facecolor="none")
    ax  = plt.subplot(121)   # 2x1 array of plots, ax refers to the 1st of them
    ax2  = plt.subplot(122)   # 1x1 array of plots, ax refers to the 1st of them
    # ax3  = plt.subplot(133)   # 1x1 array of plots, ax refers to the 1st of them

    JmaxD = np.abs(Jm1-Jm2).max()
    JPmaxD = np.abs(JPm1-JPm2).max()

    print(f"Jac: min {(Jm1-Jm2).min()} / max {(Jm1-Jm2).max()}", flush=True) 
    print(f"JacP: min {(JPm1-JPm2).min()} / max {(JPm1-JPm2).max()}", flush=True) 
    
    ax.imshow(Jm1-Jm2,  vmin=-JmaxD, vmax=JmaxD, cmap="coolwarm")
    ax2.imshow(JPm1-JPm2, vmin=-JPmaxD, vmax=JPmaxD,  cmap="coolwarm")

JacViewer(stokes0)



# +
## Now try adding a null natural bc - Expect no changes to J, P though 
## the surface terms should be active if we check the debug output

stokes1 = uw.systems.Stokes(mesh, solver_name="Stokes1")
stokes1.constitutive_model = uw.constitutive_models.ViscousFlowModel

v1 = stokes1.Unknowns.u
p1 = stokes1.Unknowns.p

stokes1.petsc_options["snes_converged_reason"] = None
stokes1.petsc_options["snes_monitor_short"] = None
stokes1.tolerance = 1.0e-6

stokes1.add_essential_bc( [0.,0.], "Bottom")         # no slip on the base
stokes1.add_essential_bc( [0.,sympy.oo], "Left")     # free slip Left/Right
stokes1.add_essential_bc( [0.,sympy.oo], "Right")    # free slip Left/Right
stokes1.add_natural_bc( [0.,0.], "Top")              # Top is open (still)

stokes1.bodyforce = sympy.Matrix([0, sympy.sin(x*sympy.pi)])

# -


stokes1._setup_pointwise_functions()
stokes1._setup_discretisation()
stokes1._setup_problem_description()

# +

stokes1.solve(verbose=False, debug=False, zero_init_guess=True, picard=0, _force_setup=False)
# -


JacViewer(stokes1)
JacDiffViewer(stokes0, stokes1)
# +
## Now try adding a constant natural bc - Still expect no changes to J, P though 
## the surface terms should be active if we check the debug output

stokes2 = uw.systems.Stokes(mesh, solver_name="Stokes2")
stokes2.constitutive_model = uw.constitutive_models.ViscousFlowModel

v2 = stokes2.Unknowns.u
p2 = stokes2.Unknowns.p

stokes2.petsc_options["snes_converged_reason"] = None
stokes2.petsc_options["snes_monitor_short"] = None
stokes2.tolerance = 1.0e-6

stokes2.add_essential_bc( [0.,0.], "Bottom")         # no slip on the base
stokes2.add_essential_bc( [0.,sympy.oo], "Left")     # free slip Left/Right
stokes2.add_essential_bc( [0.,sympy.oo], "Right")    # free slip Left/Right
stokes2.add_natural_bc( [1.,0.], "Top")              # Top is driven (constant)

stokes2.bodyforce = sympy.Matrix([0, sympy.sin(x*sympy.pi)])

stokes2.solve(verbose=False, debug=False, zero_init_guess=True, picard=5, _force_setup=False)

# -

JacViewer(stokes2)
JacDiffViewer(stokes0, stokes2)

# +
## Now try adding a constant natural bc. J, P should be unchanged as this is 
## linear, but we should see a different RHS reflected in the SNES norms

stokes3 = uw.systems.Stokes(mesh, solver_name="Stokes3")
stokes3.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes3.petsc_options.setValue("snes_monitor", None)

v3 = stokes3.Unknowns.u
p3 = stokes3.Unknowns.p

stokes3.add_essential_bc( [0.,0.], "Bottom")         # no slip on the base
stokes3.add_essential_bc( [0.,sympy.oo], "Left")     # free slip Left/Right
stokes3.add_essential_bc( [0.,sympy.oo], "Right")    # free slip Left/Right
# stokes3.add_natural_bc(   [0.,1000000*v3.sym[1]], "Top")              # Top "free slip / penalty"

Gamma = mesh.Gamma # sympy.Piecewise((mesh.Gamma, x < 0.5), mesh.CoordinateSystem.unit_j
Gamma = sympy.Matrix([0,1])
stokes3.add_natural_bc( 1.0e8 *  Gamma.dot(v3.sym) * Gamma, "Top")              # Top "free slip / penalty"

stokes3.bodyforce = sympy.Matrix([0, sympy.sin(x*sympy.pi)])

# Set verbose / debug to True to see the functions being executed
stokes3.solve(verbose=False, debug=False, zero_init_guess=True, picard=2)

# -


JacViewer(stokes3)
JacDiffViewer(stokes3, stokes0)


# +
## Validation step - Did the penalty approach actually work

stokesN = uw.systems.Stokes(mesh, solver_name="StokesN")
stokesN.constitutive_model = uw.constitutive_models.ViscousFlowModel
vN = stokesN.Unknowns.u

stokesN.tolerance = 1.0e-6
stokesN.petsc_options.setValue("snes_monitor", None)
stokesN.petsc_options.setValue("ksp_monitor", None)

stokesN.add_essential_bc( [0.,0.], "Bottom")         # no slip on the base
stokesN.add_essential_bc( [0.,sympy.oo], "Left")     # free slip Left/Right
stokesN.add_essential_bc( [0.,sympy.oo], "Right")    # free slip Left/Right
stokesN.add_essential_bc( [sympy.oo, 0.], "Top")     # Top "free slip / penalty"


stokesN.bodyforce = sympy.Matrix([0, sympy.sin(x*sympy.pi)])

stokesN.solve(verbose=False, debug=False, zero_init_guess=True)


# +
# Visuals

# This creates a plot of the true free-surface solution, the penalized velocity solution, 
# and their difference


import underworld3 as uw
import pyvista as pv
import underworld3.visualisation

pl = pv.Plotter(window_size=(1000, 500))

pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)
pvmesh.point_data["V0"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, vN.sym)
pvmesh.point_data["V3"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, v3.sym)
pvmesh.point_data["P"] = uw.visualisation.scalar_fn_to_pv_points(pvmesh, p3.sym)
# pvmesh.point_data["Vmag"] = uw.visualisation.scalar_fn_to_pv_points(pvmesh, v.sym.dot(v.sym))

velocity_points = underworld3.visualisation.meshVariable_to_pv_cloud(v0)
velocity_points.point_data["V0"] = uw.visualisation.vector_fn_to_pv_points(velocity_points, vN.sym)
velocity_points.point_data["V3"] = uw.visualisation.vector_fn_to_pv_points(velocity_points, v3.sym)

pl.add_mesh(
    pvmesh,
    cmap="coolwarm",
    edge_color="Black",
    show_edges=True,
    scalars="P",
    use_transparency=False,
    opacity=1.0,
)


pl.add_arrows(velocity_points.points, velocity_points.point_data["V3"]-velocity_points.point_data["V0"], mag=100000, opacity=0.75)
pl.add_arrows(velocity_points.points+(0.01,0.0,0.0), velocity_points.point_data["V3"], mag=1.0, opacity=0.75)
pl.add_arrows(velocity_points.points+(0.00,0.0,0.0), velocity_points.point_data["V0"], mag=1.0, opacity=0.75)
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
