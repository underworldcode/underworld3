# # Stokes plus surface deformation 
#
# Let the internal boundary of the mesh deform to create a "free surface". If we iterate on this, then it is almost exactly the same as the free-slip boundary condition (though there are potentially instabilities here).
#
# If we fix the outer mesh then we can eliminate the nullspace problem though we end up replacing it by a sticky-air approximation instead.

# +
visuals = 0
output_dir = "outputs_sticky01"
res = 0.1

import os

os.makedirs(output_dir, exist_ok=True)

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

## Use pyvista or not ?

visuals = 1

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

# options["stokes_snes_max_it"] = 10
# options["stokes_snes_rtol"] = 1.0e-4
# options["stokes_snes_atol"] = 1.0e-6
# options["stokes_pc_type"] = "fieldsplit"
# options["stokes_pc_fieldsplit_type"] = "schur"
# options["stokes_pc_fieldsplit_schur_factorization_type"] ="full"
# options["stokes_pc_fieldsplit_schur_precondition"] = "a11"
# options["stokes_fieldsplit_velocity_ksp_type"] = "fgmres"
# options["stokes_fieldsplit_velocity_pc_type"] = "gamg"
# options["stokes_fieldsplit_pressure_ksp_rtol"] = 1.e-5
# options["stokes_fieldsplit_pressure_pc_type"] = "gamg"

# Options directed at the poisson solver

# options["poisson_vr_pc_type"]  = "svd"
options["poisson_vr_ksp_rtol"] = 1.0e-2
# options["poisson_vr_ksp_monitor_short"] = None
# options["poisson_vr_snes_type"]  = "fas"
options["poisson_vr_snes_converged_reason"] = None
# options["poisson_vr_snes_monitor_short"] = None
# options["poisson_vr_snes_view"]=None
options["poisson_vr_snes_rtol"] = 1.0e-2
options["poisson_vr_snes_atol"] = 1.0e-4

import os
os.environ["SYMPY_USE_CACHE"]="no"

# +
# meshball = uw.discretisation.SphericalShell(dim=2, degree=2, radius_inner=0.0, 
#                                   radius_outer=1.0, cell_size=0.1, cell_size_upper=0.075)

# Build this one by hand 

csize_local = res
cell_size_lower = res*1.5
cell_size_upper = res
radius_outer = 1.0
radius_inner = 0.0

import pygmsh
import meshio

# Generate local mesh on rank 0


if uw.mpi.rank==0:

    with pygmsh.geo.Geometry() as geom:

        geom.characteristic_length_max = csize_local
        outer  = geom.add_circle((0.0,0.0,0.0),radius_outer, make_surface=False, mesh_size=cell_size_upper)

        if radius_inner > 0.0:   
            inner  = geom.add_circle((0.0,0.0,0.0),radius_inner, make_surface=False, mesh_size=cell_size_upper)
            domain = geom.add_circle((0.0,0.0,0.0), radius_outer*1.25, mesh_size=cell_size_upper, holes=[inner])
            geom.add_physical(inner.curve_loop.curves, label="Centre")       
            for l in inner.curve_loop.curves:
                geom.set_transfinite_curve(l, num_nodes=7, mesh_type="Progression", coeff=1.0)

        else:
            centre = geom.add_point((0.0,0.0,0.0), mesh_size=cell_size_lower)       
            domain = geom.add_circle((0.0,0.0,0.0), radius_outer*1.25, mesh_size=cell_size_upper)
            geom.in_surface(centre, domain.plane_surface)
            geom.add_physical(centre, label="Centre")

        for l in outer.curve_loop.curves:
            geom.in_surface(l, domain.plane_surface)
            geom.set_transfinite_curve(l, num_nodes=40, mesh_type="Progression", coeff=1.0)

        for l in domain.curve_loop.curves:
            geom.set_transfinite_curve(l, num_nodes=40, mesh_type="Progression", coeff=1.0)

        geom.add_physical(outer.curve_loop.curves, label="Upper")       
        geom.add_physical(domain.curve_loop.curves, label="Celestial_Sphere")

        # This is not really needed in the label list - it's everything else
        geom.add_physical(domain.plane_surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=True)
        geom.save_geometry("ignore_celestial.msh")
        geom.save_geometry("ignore_celestial.vtk")


meshball = uw.meshes.MeshFromGmshFile(dim=2, degree=1, filename="ignore_celestial.msh", label_groups=[], simplex=True)

# -


meshball.meshio.remove_lower_dimensional_cells()
meshball.meshio

# +
v_soln  = uw.discretisation.MeshVariable('U', meshball, 2, degree=2 )
p_soln  = uw.discretisation.MeshVariable('P', meshball, 1, degree=1 )
t_soln  = uw.discretisation.MeshVariable("T", meshball, 1, degree=3 )
vr_soln = uw.discretisation.MeshVariable('Vr',meshball, 1, degree=1 )

mask    = uw.discretisation.MeshVariable('M', meshball, 1, degree=1 )
surface = uw.discretisation.MeshVariable('S_f',meshball, 1, degree=1 )
r_mesh  = uw.discretisation.MeshVariable('R', meshball,  1, degree=1 )
r_mesh0 = uw.discretisation.MeshVariable('R0',meshball,  1, degree=1 ) 

# +
# Introduce a swarm so that we can introduce strain markers

swarm = uw.swarm.Swarm(meshball)
sv = swarm.add_variable("compo", num_components=1, proxy_degree=3, _nn_proxy=False )

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre 
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10+radius_fn)
gravity_fn = radius_fn

# This nukes everything outside the computational domain (but only the mesh_variable
# version is Lagrangian

mask_fn = 0.5 - 0.5 * sympy.tanh(1000.0*(r_mesh.fn-1.003)) 
i_mask_fn = 0.5 - 0.5 * sympy.tanh(1000.0*(r_mesh.fn-0.997)) 
sky_mask_fn = 1.0 - mask_fn

# Some useful coordinate stuff 

x = meshball.N.x
y = meshball.N.y
# z = meshball.N.z

r  = sympy.sqrt(x**2+y**2)  # cf radius_fn which is 0->1 
th = sympy.atan2(y+1.0e-5,x+1.0e-5)

# 
Rayleigh = 1.0e5

surface_exaggeration = 5.0
surface_fn = mask_fn - i_mask_fn
# sympy.exp(-25.0/csize_local * (1.0-r_mesh.fn)**2)

# Assign values for the mesh-based variables 

with meshball.access(r_mesh, r_mesh0):
    r_mesh.data[...]   = uw.function.evaluate(r, r_mesh.coords).reshape(-1,1) 
    r_mesh0.data[...]  = uw.function.evaluate(r, r_mesh0.coords).reshape(-1,1) 
    
with meshball.access(mask, surface):
    mask.data[...] = uw.function.evaluate(mask_fn, mask.coords).reshape(-1,1)
    surface.data[...] = uw.function.evaluate(mask_fn - i_mask_fn, surface.coords).reshape(-1,1)

t_init = 0.001 *  (sympy.exp(-5.0*(x**2+(y-0.9)**2)) + 
                            sympy.exp(-10.0*((x-0.8)**2+(y+0.25)**2)))
# -


swarm.populate(fill_param=8)

with swarm.access(sv):
    sv.data[...] =  uw.function.evaluate(t_init, swarm.particle_coordinates.data).reshape(-1,1)


# +
# Validate

from mpi4py import MPI

if visuals and MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = pv.read("ignore_celestial.vtk")

    pvmesh.point_data["T"]  = uw.function.evaluate(t_init**2, meshball.data)
    pvmesh.point_data["T2"] = uw.function.evaluate(sv.fn**2, meshball.data)
    pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"] 
    
    pl = pv.Plotter(off_screen=True)

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="DT",
                  use_transparency=False, opacity=0.5, clim=[-5.0e-9,5.0e-9])
    
    pl.camera_position="xy"
     
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")  
# -




# +
# this is how we deform the mesh

original_coords = meshball.data.copy()

# coord_vec = meshball.dm.getCoordinates()
# original_coord_vec = coord_vec.copy()

# coords = coord_vec.array.reshape(-1,2)
# deformation_fn = 1.0 + 0.00001 * sympy.cos(3.0*th)
# deformation_mesh_pts = uw.function.evaluate(deformation_fn, coords)
# coords *= deformation_mesh_pts.reshape(-1,1) # This is a view on the coord_vec 

# meshball.dm.setCoordinates(coord_vec)
# meshball.nuke_coords_and_rebuild() # should build all this into the same function !

# with meshball.access(r_mesh):
#     r_mesh.data[...]  = uw.function.evaluate(r, meshball.data).reshape(-1,1)

# deformation_mesh_pts = uw.function.evaluate(deformation_fn, meshball.meshio.points[:,0:2])
# meshball.meshio.points *= deformation_mesh_pts.reshape(-1,1)


# +
vtheta = r * sympy.sin(th)

vx = -vtheta*sympy.sin(th)
vy =  vtheta*sympy.cos(th)
# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                u_degree=2, p_degree=1, solver_name="stokes")

# just to test
stokes.petsc_options["snes_rtol"]=1.0e-4
stokes.petsc_options["ksp_rtol"]=1.0e-2
# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"]  = 1.0e-2
# stokes.petsc_options["fieldsplit_pressure_ksp_rtol"]  = 1.0e-2


# stokes.petsc_options.delValue("snes_monitor_short")
# stokes.petsc_options.delValue("ksp_monitor")

stokes.viscosity = 0.0 + 1.0 * (mask.fn * 0.9 + 0.1)
stokes.add_dirichlet_bc( (0.0, 0.0), "Celestial_Sphere" , (0,1))
# stokes.add_dirichlet_bc( (0.0, 0.0), "Centre" , (0,1))

# buoyancy_force = Rayleigh * gravity_fn * ( -(r_mesh.fn-r_mesh0.fn)*surface.fn/surface_exaggeration + t_init )  


buoyancy_force = Rayleigh * gravity_fn * t_init * mask_fn
buoyancy_force -= Rayleigh * 100.0 *  v_soln.fn.dot(unit_rvec) * surface_fn

stokes.bodyforce = unit_rvec * buoyancy_force  


# Define radial stress (Can't evaluate this with current function tools)
ur = sympy.Matrix([[unit_rvec.dot(meshball.N.i),unit_rvec.dot(meshball.N.j)]])
radial_stress = ur * stokes.stress * ur.T
rs = radial_stress[0,0]
# -




# +
# Diffusion solver for v_r pattern ... 
# 

# Create Poisson object

# Set some things
k = 1. 
h = 0.  

poisson_vr = uw.systems.Poisson(meshball, u_Field=vr_soln, 
                                solver_name="poisson_vr", 
                                degree=vr_soln.degree, verbose=False)
poisson_vr.k = k
poisson_vr.f = h

# v_r 
vr_fn = v_soln.fn.dot(unit_rvec) 

poisson_vr.add_dirichlet_bc( 0.0,  "Celestial_Sphere" )
poisson_vr.add_dirichlet_bc( 0.0,  "Centre" )
poisson_vr.add_dirichlet_bc( vr_fn, "Upper" )

fs_residual = vr_fn * radius_fn


# +
# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:,0] = uw.function.evaluate(t_init, t_soln.coords)
    
tsize, tmean, tmin, tmax, tsum, tnorm2, trms = t_soln.stats()
# -
stokes.solve()



poisson_vr.solve(zero_init_guess=True)

mask_mesh = uw.function.evaluate(mask.fn, meshball.data)
vr_mesh   = uw.function.evaluate(vr_soln.fn, meshball.data)
err_mesh  = uw.function.evaluate(surface_fn * vr_soln.fn, meshball.data)
norm_mesh = uw.function.evaluate(surface_fn, meshball.data)
t_mesh    = uw.function.evaluate(t_soln.fn, meshball.data)







# +
# check the mesh if in a notebook / serial


if visuals and uw.mpi.size==1:

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
        vrsol = vr_soln.data.copy()
        meshmask = uw.function.evaluate(mask.fn, meshball.data)
        
        # fs_res = uw.function.evaluate(fs_residual, meshball.data)
        print("usol - magnitude {}".format(np.sqrt((usol**2).mean())))        
        print("vrsol - magnitude {}".format(np.sqrt((vrsol**2).mean())))
        # print("fs_residual - magnitude {}".format(np.sqrt((fs_res**2).mean())))
        
  
    pvmesh.point_data["T"]  = t_mesh
    pvmesh.point_data["Vr"] = vr_mesh
    pvmesh.point_data["M"]  = mask_mesh
    pvmesh.point_data["E"]  = err_mesh
    pvmesh.point_data["S"]  = uw.function.evaluate(surface.fn, meshball.data)
    pvmesh.point_data["R"]  = uw.function.evaluate((r_mesh0.fn), meshball.data)

    mask_coords_3D = np.zeros((stokes.u.coords.shape[0],3))
    mask_coords_3D[:,0:2] = stokes.u.coords[...]

    pdata = pv.PolyData(mask_coords_3D)
    pdata['mask'] = uw.function.evaluate(surface.fn, stokes.u.coords)

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
    
    pl.add_arrows(arrow_loc, arrow_length, mag=1.0e-1)
    #pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)
    
    # pl.add_points(pdata)

    pl.show(cpos="xy")
    



# +
vsize, vmean, vmin, vmax, vsum, vnorm2, vrms = meshball.stats(vr_soln.fn)
vr0 = vrms

vsize, vmean, vmin, vmax, vsum, vnorm2, vrms = meshball.stats(surface.fn)
norm_0 = vrms

vsize, vmean, vmin, vmax, vsum, vnorm2, vrms = meshball.stats((surface.fn*vr_fn))
surface_residual_0 = vrms
surface_mean_flux_0 = vmean

if uw.mpi.rank==0:
    print("surface residual 0 - {} / {}".format(surface_residual_0, norm_0))
    print("surface mean flux 0 - {} / {}".format(surface_mean_flux_0, vr0))
# -

0/0

# +
# stokes.dt(), cell_size_upper

vr_mesh_1 = vr_mesh.copy()
with meshball.access():
    xy_mesh_1 = meshball.data.copy()

# +
for it in range(0,10):
     
    # Use v history to estimate dv/dt, v for half step forward. 
        
    delta_r = 0.5 * stokes.estimate_dt() * ( 1.5 * vr_mesh - 0.5 * vr_mesh_1)
    delta_x = delta_r * uw.function.evaluate(sympy.cos(th), xy_mesh_1)
    delta_y = delta_r * uw.function.evaluate(sympy.sin(th), xy_mesh_1)

    coords = xy_mesh_1.copy()
    coords[:,0] += delta_x
    coords[:,1] += delta_y
        
    meshball.deform_mesh(coords)

    # update the mesh-based r variable.
    with meshball.access(r_mesh):
        r_mesh.data[...]  = uw.function.evaluate(r, meshball.data).reshape(-1,1)

    stokes.solve(zero_init_guess=False)     
    poisson_vr.solve(zero_init_guess=False)
     
    # re-map to nodes

    vr_mesh_1 = vr_mesh
    vr_mesh   = uw.function.evaluate(vr_soln.fn, meshball.data)
    xy_mesh_1 = 1.0 * coords  + 0.0 * xy_mesh_1

    vsize, vmean, vmin, vmax, vsum, vnorm2, vrms = meshball.stats((surface.fn*vr_soln.fn))
    surface_residual = vrms
    surface_mean_flux = vmean
 
    # Save interim results
    
#     savefile = "{}/free_surface_disk_{}_iterati.h5".format(output_dir,it) 
#     meshball.save(savefile)
#     v_soln.save(savefile)
#     t_soln.save(savefile)
#     meshball.generate_xdmf(savefile)
    
    if uw.mpi.rank==0:
        
        print("surface residual - {} / {}".format(surface_residual/surface_residual_0, surface_residual/norm_0))
        print("surface mean flux - {} / {}".format(surface_mean_flux/vr0, vr0))
    
    if surface_residual/norm_0 < 0.001:
        break

        
mask_mesh = uw.function.evaluate(mask.fn, meshball.data)
err_mesh  = uw.function.evaluate(surface_fn * vr_soln.fn, meshball.data)
t_mesh    = uw.function.evaluate(t_soln.fn, meshball.data)



# +
# check the mesh if in a notebook / serial

## If we move the mesh coordinates, we break the 
## hash that identifies the variable locations and
## evaluations require interpolation (which is really quite broken)


if visuals and uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
     
    with meshball.access():
        usol = stokes.u.data.copy()
        vrsol = vr_soln.data.copy()
       
        print("usol - magnitude {}".format(np.sqrt((usol**2).mean())))        
        print("vrsol - magnitude {}".format(np.sqrt((vrsol**2).mean())))

    pvmesh.point_data["T"]  = t_mesh
    pvmesh.point_data["Vr"] = uw.function.evaluate(vr_soln.fn*surface_fn, meshball.data)
    pvmesh.point_data["M"] =  np.clip(mask_mesh, 0.0, 1.0)
    pvmesh.point_data["E"] =  err_mesh
    pvmesh.point_data["B"]  =  uw.function.evaluate(buoyancy_force, meshball.data)
    pvmesh.point_data["DR"] =  uw.function.evaluate(surface.fn*(r_mesh.fn - r_mesh0.fn), meshball.data)

    v_vectors = np.zeros((meshball.data.shape[0],3))
    v_vectors[:,0:2] = uw.function.evaluate(i_mask_fn * v_soln.fn, meshball.data)
    pvmesh.point_data["V"] = v_vectors 

    mesh_coords_3D = np.zeros((meshball.data.shape[0],3))
    mesh_coords_3D[:,0:2] = meshball.data[...]

    arrow_loc = np.zeros((v_soln.coords.shape[0],3))
    arrow_loc[:,0:2] = v_soln.coords[...]
    
    arrow_length = np.zeros((v_soln.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] * uw.function.evaluate(mask_fn, v_soln.coords).reshape(-1,1)
    
    arrow_loc2 = np.zeros((vr_soln.coords.shape[0],3))
    arrow_loc2[:,0:2] = vr_soln.coords[...]
    
    arrow_length2 = np.zeros((vr_soln.coords.shape[0],3))
    arrow_length2[:,0:2] = uw.function.evaluate(unit_rvec * vr_soln.fn, vr_soln.coords) 

    pl = pv.Plotter(notebook=True, off_screen=True)
    
    # Remove the sky 

    clipping_surface = pv.Circle(radius=1.05)
    clipped_mesh = pvmesh.clip_surface(clipping_surface, invert=False)
      
    # streams = clipped_mesh.streamlines_evenly_spaced_2D(vectors="V", integrator_type=4, separating_distance=0.25)              
    
    # cmcc = clipped_mesh.cell_centers()
    # cmcc_dataset = pv.PolyData(cmcc.points[::2])
    # streams = clipped_mesh.streamlines_from_source(cmcc_dataset, vectors="V", 
    #                                               surface_streamlines=True, max_steps=1000,
    #                                               max_step_length=0.01 )
    
        
 
    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(clipped_mesh, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="T",
                 use_transparency=False, opacity=0.5)
    
    # pl.add_mesh(streams, opacity=0.75)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=2.0e-1)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0)


    pl.show(cpos="xy")
    # pl.screenshot(filename="IC_with_IIC.png", window_size=(1000,1000))
# -

with meshball.access(v_soln):
    v_soln.data[...] *= uw.function.evaluate(mask.fn**2, v_soln.coords).reshape(-1,1)

# +
## Save data

savefile = "{}/free_surface_disk.h5".format(output_dir)
meshball.save(savefile)
v_soln.save(savefile)
t_soln.save(savefile)
meshball.generate_xdmf(savefile)
# -





