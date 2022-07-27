# # Spherical Stokes
#
#

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None

import os
os.environ["SYMPY_USE_CACHE"]="no"

## 

Free_Slip = True
Rayleigh = 1.0e5
H_int = 1
res = 0.3
r_o = 1.0
r_i = 0.0
expt_name = "Sphere_Stokes_test_2"


# +
# meshball = uw.meshes.SphericalShell(dim=3, degree=1, 
#                                     radius_inner=r_i, 
#                                     centre_point=False,
#                                     radius_outer=r_o, 
#                                     cell_size=res, 
#                                     cell_size_upper=res,
#                                     verbose=True)
# -

meshball = uw.meshes.MeshFromGmshFile(dim=3, filename="Sample_Meshes_Gmsh/test_mesh_sphere_at_res_02.msh", 
                                      verbose=True, simplex=True)


v_soln  = uw.discretisation.MeshVariable('U', meshball, meshball.dim, degree=2 )
p_soln  = uw.discretisation.MeshVariable('P', meshball, 1, degree=1 )
t_soln  = uw.discretisation.MeshVariable("T", meshball, 1, degree=3 )


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
    
    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TETRA)
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
z = meshball.N.z

r  = sympy.sqrt(x**2+y**2+z**2)  # cf radius_fn which is 0->1 
th = sympy.atan2(y+1.0e-10,x+1.0e-10)
ph = sympy.acos(z / (r+1.0e-10))


# 1000 is the penalty value that enforces the free-slip condition

hw = 1000.0 / res 
surface_fn = sympy.exp(-((r - r_o) / r_o)**2 * hw)



# +
vtheta = r * sympy.sin(th)

vx = -vtheta*sympy.sin(th)
vy =  vtheta*sympy.cos(th)

# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                u_degree=2, p_degree=1, solver_name="stokes")

# Inexact Jacobian may be OK.
stokes.petsc_options["snes_rtol"]=1.0e-3
stokes.petsc_options["ksp_rtol"]=1.0e-3


# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"]  = 1.0e-2
# stokes.petsc_options["fieldsplit_pressure_ksp_rtol"]  = 1.0e-2

i_fn = 0.5 - 0.5 * sympy.tanh(1000.0*(r-0.5)) 
stokes.viscosity = 1.0 + 100 * i_fn

# -


t_init = 0.001 * sympy.exp(-5.0*(x**2+(y-0.5)**2))

# +
# Write temperature into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:,0] = uw.function.evaluate(t_init, t_soln.coords)
    
t_mean = t_soln.mean()

# +

buoyancy_force = Rayleigh * gravity_fn * t_init
buoyancy_force -= Rayleigh * 1000.0 *  v_soln.fn.dot(unit_rvec) * surface_fn
stokes.bodyforce = unit_rvec * buoyancy_force  

# This may help the solvers - penalty in the preconditioner
stokes._Ppre_fn = 1.0 / (stokes.viscosity + Rayleigh * 1000.0 * surface_fn)

# -

stokes.solve()

savefile = "output/{}.h5".format(expt_name)
meshball.save(savefile)
v_soln.save(savefile)
t_soln.save(savefile)
meshball.generate_xdmf(savefile)

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
    
    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TETRA)

    with meshball.access():
        usol = stokes.u.data.copy()
       
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn, meshball.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:3] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:3] = usol[...] 
    
    clipped = pvmesh.clip(normal="y")
            
    pl = pv.Plotter()
    
    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
    #               use_transparency=False, opacity=0.5)
    
    pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                  use_transparency=False, opacity=1.0)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=1.2e-1)
    
    pl.show(cpos="xy")
    
# -

