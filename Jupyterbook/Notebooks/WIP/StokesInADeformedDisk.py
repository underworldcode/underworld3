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

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None

# Is there some way to set the coordinate interpolation function space ?
# options["coord_dm_default_quadrature_order"] = 2

options["stokes_ksp_rtol"] =  1.0e-3
# options["stokes_ksp_monitor"] = None
options["stokes_snes_converged_reason"] = None
# options["stokes_snes_monitor_short"] = None

# options["stokes_snes_view"]=None
# options["stokes_snes_test_jacobian"] = None

options["stokes_snes_max_it"] = 15
options["stokes_snes_rtol"] = 1.0e-3
options["stokes_snes_atol"] = 1.0e-6
options["stokes_pc_type"] = "fieldsplit"
options["stokes_pc_fieldsplit_type"] = "schur"
options["stokes_pc_fieldsplit_schur_factorization_type"] ="full"
options["stokes_pc_fieldsplit_schur_precondition"] = "a11"
options["stokes_fieldsplit_velocity_pc_type"] = "lu"
options["stokes_fieldsplit_pressure_ksp_rtol"] = 1.e-4
options["stokes_fieldsplit_pressure_pc_type"] = "lu"

# Options directed at the poisson solver

# options["poisson_vr_pc_type"]  = "svd"
options["poisson_vr_ksp_rtol"] = 1.0e-2
#options["poisson_vr_ksp_monitor_short"] = None
# options["poisson_vr_snes_type"]  = "fas"
options["poisson_vr_snes_converged_reason"] = None
# options["poisson_vr_snes_monitor_short"] = None
# options["poisson_vr_snes_view"]=None
options["poisson_vr_snes_rtol"] = 1.0e-2
options["poisson_vr_snes_atol"] = 1.0e-4

import os
os.environ["SYMPY_USE_CACHE"]="no"
# -

meshball = uw.mesh.SphericalShell(dim=2, degree=2, radius_inner=0.0, 
                                  radius_outer=1.0, cell_size=0.1, cell_size_upper=0.075)


v_soln  = uw.mesh.MeshVariable('U', meshball, 2, degree=2 )
p_soln  = uw.mesh.MeshVariable('P', meshball, 1, degree=1 )
t_soln  = uw.mesh.MeshVariable("T", meshball, 1, degree=3 )
vr_soln = uw.mesh.MeshVariable('Vr',meshball, 1, degree=2 )


# this is how we deform the mesh
"""
coord_vec = meshball.dm.getCoordinates()

coords = coord_vec.array.reshape(-1,2)
coords *= (1.0 + 0.0005 * np.cos(2.0*np.arctan2(coords[:,1], coords[:,0]))).reshape(-1,1)
meshball.dm.setCoordinates(coord_vec)

meshball.meshio.points[:,0] = coords[:,0]
meshball.meshio.points[:,1] = coords[:,1]
"""


# +
# check the mesh if in a notebook / serial

import mpi4py
if mpi4py.MPI.COMM_WORLD.size==1:

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

# 
Rayleigh = 1.0e5

surface_exaggeration = 100
surface_fn = sympy.exp(-5.0/meshball.cell_size * (1.0-r)**2) / surface_exaggeration


# +
vtheta = r * sympy.sin(th)

vx = -vtheta*sympy.sin(th)
vy =  vtheta*sympy.cos(th)

# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                u_degree=2, p_degree=1, solver_name="stokes")
stokes.viscosity = 1.
stokes.add_dirichlet_bc( (0.0, 0.0), meshball.boundary.CENTRE , (0,1))


# +
# Diffusion solver for v_r pattern

# Create Poisson object

# Set some things
k = 1. 
h = 0.  

poisson_vr = uw.systems.Poisson(meshball, u_Field=vr_soln, 
                                solver_name="poisson_vr", 
                                degree=2, verbose=False)
poisson_vr.k = k
poisson_vr.f = h

# v_r 
vr_fn = v_soln.fn.dot(unit_rvec) 

# poisson_vr.add_dirichlet_bc( 0.0,  meshball.boundary.CENTRE )
poisson_vr.add_dirichlet_bc( vr_fn, meshball.boundary.TOP )

fs_residual = vr_fn * radius_fn
# -


t_init = 0.001 * sympy.exp(-5.0*(x**2+(y-0.5)**2))

# +
# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:,0] = uw.function.evaluate(t_init, t_soln.coords)
    print(t_soln.data.min(), t_soln.data.max())
    
t_mean = t_soln.mean()
print(t_soln.min(), t_soln.max())
# -
stokes.bodyforce = unit_rvec * (Rayleigh * t_init * gravity_fn - surface_fn * Rayleigh)

stokes.solve()

poisson_vr.solve(zero_init_guess=True)

# +
## With the mesh coords using 2nd order interpolation, there are some 
## issues with function evaluation.

kd = uw.algorithms.KDTree(meshball.data)
kd.build_index()

n,d,b = kd.find_closest_point(t_soln.coords)
t_mesh = np.zeros((meshball.data.shape[0]))
w = np.zeros((meshball.data.shape[0]))

with meshball.access():
    for i in range(0,n.shape[0]):
        t_mesh[n[i]] += t_soln.data[i] / (1.0e-10 + d[n[i]])
        w[n[i]] += 1.0 / (1.0e-10 + d[n[i]])
        
t_mesh /= w

        
n,d,b = kd.find_closest_point(vr_soln.coords)
vr_mesh = np.zeros((meshball.data.shape[0]))
err_mesh = np.zeros((meshball.data.shape[0]))

w = np.zeros((meshball.data.shape[0]))
kernel = uw.function.evaluate(surface_fn, meshball.data)

with meshball.access():
    for i in range(0,n.shape[0]):
        vr_mesh[n[i]] += vr_soln.data[i] / (1.0e-10 + d[n[i]])
        err_mesh[n[i]] += kernel[i] * vr_soln.data[i] / (1.0e-10 + d[n[i]])     
        w[n[i]] += 1.0 / (1.0e-10 + d[n[i]])

vr_mesh[np.where(w > 0.0)] /= w[np.where(w > 0.0)]
err_mesh[np.where(w > 0.0)] /= w[np.where(w > 0.0)]






# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = meshball.mesh2pyvista()
     
    with meshball.access():
        usol = stokes.u.data.copy()
        vrsol = vr_soln.data.copy()
        # fs_res = uw.function.evaluate(fs_residual, meshball.data)
        
        print("usol - magnitude {}".format(np.sqrt((usol**2).mean())))        
        print("vrsol - magnitude {}".format(np.sqrt((vrsol**2).mean())))
        # print("fs_residual - magnitude {}".format(np.sqrt((fs_res**2).mean())))
     
    pvmesh.point_data["T"]  = t_mesh
    pvmesh.point_data["Vr"] = vr_mesh

    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...]
    
    arrow_loc2 = np.zeros((vr_soln.coords.shape[0],3))
    arrow_loc2[:,0:2] = vr_soln.coords[...]
    
    arrow_length2 = np.zeros((vr_soln.coords.shape[0],3))
    arrow_length2[:,0:2] = vrsol * uw.function.evaluate(unit_rvec, vr_soln.coords)

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                  use_transparency=False, opacity=0.5)
    pl.add_arrows(arrow_loc, arrow_length, mag=5.0e-3)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=5.0e-3)

    pl.show()
# -


# ## So this gives us a smoothed mesh-deformation velocity everywhere
#
# We now need to move the mesh so the out-of-balance velocity vanishes. What we really need is 
# $\partial{v_r}/\partial{x_r}$ but we can also estimate the out-of-balance force term through an 
# isostatic balance.
#
# $F = \rho g \Delta x_r$
#
# Postglacial rebound timescale relates v, viscosity and amplitude
#
# $tau$ is the relaxation time, then this is the timescale to use for estimating the change in surface position that moves the system to equilibrium
#
# $$\dot{w} = -w \frac{\lambda g \rho }{4 \pi \eta} $$
#
# i.e.
#
# $$ w = -\dot{w} \frac{4\pi\eta}{\lambda \rho g} $$
#
# so the recipe should be to move the boundary by $-v_r \frac{4\pi\eta}{\lambda \rho g}$ where $\lambda$ is the scale of the deformation pattern (2 elements, perhaps) 
#
#

# +
vr0 = np.sqrt((vr_mesh**2).mean())
surface_residual_0 = np.sqrt((err_mesh**2).mean())

print("surface residual_0 - {} / {}".format(surface_residual_0, vr0))



# +

for it in range(0,10):
                  
    coord_vec = meshball.dm.getCoordinates()
    coords = coord_vec.array.reshape(-1,2)
    coords *= (1.0 + surface_exaggeration * vr_mesh / Rayleigh).reshape(-1,1)
       
    ## The following is crying out for some kind of context manager !    
        
    meshball.dm.setCoordinates(coord_vec)
    meshball.nuke_coords_and_rebuild()

    stokes._rebuild_after_mesh_update()
    poisson_vr._rebuild_after_mesh_update()
       
    stokes.solve()     
    poisson_vr.solve()
    
    with meshball.access():
        usol = stokes.u.data.copy()
        vrsol = vr_soln.data.copy()
        
        print("usol - magnitude {}".format(np.sqrt((usol**2).mean())))        
        print("vrsol - magnitude {}".format(np.sqrt((vrsol**2).mean())))

#     # re-map to nodes
        
    n,d,b = kd.find_closest_point(vr_soln.coords)
    vr_mesh = np.zeros((meshball.data.shape[0]))
    err_mesh = np.zeros((meshball.data.shape[0]))

    w = np.zeros((meshball.data.shape[0]))
    kernel = uw.function.evaluate(surface_fn, meshball.data)

    with meshball.access():
        for i in range(0,n.shape[0]):
            vr_mesh[n[i]] += vr_soln.data[i] / (1.0e-10 + d[n[i]])
            err_mesh[n[i]] += kernel[i] * vr_soln.data[i] / (1.0e-10 + d[n[i]])     
            w[n[i]] += 1.0 / (1.0e-10 + d[n[i]])

    vr_mesh[np.where(w > 0.0)] /= w[np.where(w > 0.0)]
    err_mesh[np.where(w > 0.0)] /= w[np.where(w > 0.0)]
    
    vr = np.sqrt((vr_mesh**2).mean())
    surface_residual = np.sqrt((err_mesh**2).mean())

    print("surface residual - {} / {}".format(surface_residual/surface_residual_0, vr/vr0))



# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = meshball.mesh2pyvista()
     
    with meshball.access():
        usol = stokes.u.data.copy()
        vrsol = vr_soln.data.copy()
        surfs = uw.function.evaluate(surface_fn, meshball.data)
        
        print("usol - magnitude {}".format(np.sqrt((usol**2).mean())))        
        print("vrsol - magnitude {}".format(np.sqrt((vrsol**2).mean())))
        # print("fs_residual - magnitude {}".format(np.sqrt((fs_res**2).mean())))
        
     
    pvmesh.point_data["T"]  = t_mesh
    pvmesh.point_data["Vr"] = vr_mesh
    pvmesh.point_data["Sfn"] = surfs


    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...]
    
    arrow_loc2 = np.zeros((vr_soln.coords.shape[0],3))
    arrow_loc2[:,0:2] = vr_soln.coords[...]
    
    arrow_length2 = np.zeros((vr_soln.coords.shape[0],3))
    arrow_length2[:,0:2] = vrsol * uw.function.evaluate(unit_rvec, vr_soln.coords)

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="T",
                  use_transparency=False, opacity=0.5)
    pl.add_arrows(arrow_loc, arrow_length, mag=7.0e-2)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=5.0e-2)

    pl.show()
# -


