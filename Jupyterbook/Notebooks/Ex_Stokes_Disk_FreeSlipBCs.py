# # Cylindrical Stokes
#
# Let the mesh deform to create a free surface. If we iterate on this, then it is almost exactly the same as the free-slip boundary condition (though there are potentially instabilities here).
#
# The problem has a constant velocity nullspace in x,y. We eliminate this by fixing the central node in this example, but it does introduce a perturbation to the flow near the centre which is not always stagnant.

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

res=0.2

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None

import os
os.environ["SYMPY_USE_CACHE"]="no"
# -

meshball = uw.meshes.SphericalShell(dim=2, degree=1, radius_inner=0.0, 
                                    radius_outer=1.0, cell_size=res, 
                                    cell_size_upper=0.75*res)


v_soln  = uw.discretisation.MeshVariable('U', meshball, 2, degree=2 )
p_soln  = uw.discretisation.MeshVariable('P', meshball, 1, degree=1 )
t_soln  = uw.discretisation.MeshVariable("Delta T", meshball, 1, degree=3 )


t_soln.fn

# +
# check the mesh if in a notebook / serial

if uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = meshball.mesh2pyvista()
    pvmesh.plot(show_edges=True)


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre 
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10+radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff 

x = meshball.N.x
y = meshball.N.y
# z = meshball.N.z

r  = sympy.sqrt(x**2+y**2)  # cf radius_fn which is 0->1 
th = sympy.atan2(y+1.0e-5,x+1.0e-5)

Rayleigh = 1.0e5

hw = 1000.0 / res 
surface_fn = sympy.exp(-(r-1.0)**2 * hw)



# +
vtheta = r * sympy.sin(th)

vx = -vtheta*sympy.sin(th)
vy =  vtheta*sympy.cos(th)

# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                u_degree=2, p_degree=1, solver_name="stokes")

# Inexact Jacobian may be OK.
stokes.petsc_options["snes_type"]="newtonls"
stokes.petsc_options["snes_rtol"]=1.0e-4
stokes.petsc_options["snes_max_it"]=10
stokes.petsc_options["ksp_rtol"]=1.0e-2

# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"]  = 1.0e-2
# stokes.petsc_options["fieldsplit_pressure_ksp_rtol"]  = 1.0e-2

i_fn = 0.5 - 0.5 * sympy.tanh(1000.0*(r-0.5)) 
stokes.viscosity = 1.0 # + 10 * i_fn

# There is a null space with the unconstrained blob
# and it may be necessary to set at least one point 
# with a boundary condition. 

# stokes.add_dirichlet_bc( (0.0, 0.0), "Centre" , (0,1))

# -


t_init = 0.001 * sympy.exp(-5.0*(x**2+(y-0.5)**2))

# +
# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:,0] = uw.function.evaluate(t_init, t_soln.coords)
    print(t_soln.data.min(), t_soln.data.max())
    
t_mean = t_soln.mean()
print(t_soln.min(), t_soln.max())
# +

buoyancy_force = Rayleigh * gravity_fn * t_init 
buoyancy_force -= Rayleigh * 1000.0 *  v_soln.fn.dot(unit_rvec) * surface_fn 
stokes.bodyforce = unit_rvec * buoyancy_force 

# This may help the solvers - penalty in the preconditioner
stokes._Ppre_fn = 1.0 / (stokes.viscosity + 1000.0 * surface_fn)

# -

stokes.solve()

# +
# check the mesh if in a notebook / serial


if uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 600]
    
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pv.start_xvfb()
    
    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    with meshball.access():
        usol = stokes.u.data.copy()
       
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn, meshball.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    pl = pv.Plotter()
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                  use_transparency=False, opacity=0.5)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=1.0e-1)
    
    pl.show(cpos="xy")
    
# -




