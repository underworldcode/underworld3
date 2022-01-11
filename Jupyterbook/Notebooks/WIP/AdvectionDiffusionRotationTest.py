# # Rigid body flow in a disc with adv_diff to solve T using back-in-time sampling with particles
#
# Here we perform a rotation test in a disc to check the time-scaling etc. The stokes solver is replaced with a rigid body rotation ... 

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

# Options directed at the Stokes solver

options["stokes_ksp_rtol"] =  1.0e-3
options["stokes_ksp_monitor"] = None
options["stokes_snes_converged_reason"] = None
options["stokes_snes_monitor_short"] = None
options["stokes_snes_max_it"] = 1
options["stokes_pc_type"] = "fieldsplit"
options["stokes_pc_fieldsplit_type"] = "schur"
options["stokes_pc_fieldsplit_schur_factorization_type"] ="full"
options["stokes_pc_fieldsplit_schur_precondition"] = "a11"
options["stokes_fieldsplit_velocity_pc_type"] = "lu"
options["stokes_fieldsplit_pressure_ksp_rtol"] = 1.e-3
options["stokes_fieldsplit_pressure_pc_type"] = "lu"

# Options directed at the adv_diff solver

# options["adv_diff_pc_type"]  = "svd"
options["adv_diff_snes_type"] = "qn"
options["adv_diff_ksp_rtol"] = 1.0e-3
options["adv_diff_ksp_monitor"] = None
options["adv_diff_ksp_type"] = "fgmres"
options["adv_diff_pre_type"] = "gamg"
options["adv_diff_snes_converged_reason"] = None
options["adv_diff_snes_monitor_short"] = None
# options["adv_diff_snes_view"]=None
options["adv_diff_snes_rtol"] = 1.0e-3

# import os
# os.environ["SYMPY_USE_CACHE"]="no"

# options.getAll()

# +
import meshio

meshball = uw.meshes.SphericalShell(dim=2, radius_inner=0.5,
                                    radius_outer=1.0, cell_size=0.1,
                                    degree=1, verbose=True)
# -


v_soln = uw.mesh.MeshVariable('U',    meshball, meshball.dim, degree=2 )
t_soln = uw.mesh.MeshVariable("T",    meshball, 1, degree=3)
t_0    = uw.mesh.MeshVariable("T0",   meshball, 1, degree=3)


swarm  = uw.swarm.Swarm(mesh=meshball)
T1 = uw.swarm.SwarmVariable("Tminus1", swarm, 1)
X1 = uw.swarm.SwarmVariable("Xminus1", swarm, 2)
swarm.populate(fill_param=3)


# +
# Advection scheme tracers ( a swarm based at the nodal points of t_sol)

# Create a swarm that is located at the mesh points for t_sol
# This needs to be called in a batman style before the 
# other dm field are finalised because the swarm uses variables ... 

nswarm = uw.swarm.Swarm(meshball)
nT1 = uw.swarm.SwarmVariable("nTminus1", nswarm, 1)
nX0 = uw.swarm.SwarmVariable("nX0", nswarm, nswarm.dim)

nswarm.dm.finalizeFieldRegister()
nswarm.dm.addNPoints(t_soln.coords.shape[0]+1) # why + 1 ? That's the number of spots actually allocated
cellid = nswarm.dm.getField("DMSwarm_cellid")
coords = nswarm.dm.getField("DMSwarmPIC_coor").reshape( (-1, nswarm.dim) )
coords[...] = t_soln.coords[...]
cellid[:] = meshball.get_closest_cells(coords)
nswarm.dm.restoreField("DMSwarmPIC_coor")
nswarm.dm.restoreField("DMSwarm_cellid")
nswarm.dm.migrate(remove_sent_points=True)

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre 
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10+radius_fn)

# Some useful coordinate stuff 

x = meshball.N.x
y = meshball.N.y

r  = sympy.sqrt(x**2+y**2)
th = sympy.atan2(y+1.0e-5,x+1.0e-5)

# Rigid body rotation v_theta = constant, v_r = 0.0

theta_dot = 2.0 * np.pi # i.e one revolution in time 1.0
v_x = - r * theta_dot * sympy.sin(th)
v_y =   r * theta_dot * sympy.cos(th)

with meshball.access(v_soln):
    v_soln.data[:,0] = uw.function.evaluate(v_x, v_soln.coords)    
    v_soln.data[:,1] = uw.function.evaluate(v_y, v_soln.coords)

# +
# Create adv_diff object

# Set some things
k = 1.0e-6
h = 0.0 
t_i = 2.
t_o = 1.
r_i = 0.5
r_o = 1.0
delta_t = 1.0

adv_diff = uw.systems.AdvDiffusion(meshball, u_Field=t_soln, 
                                   ustar_Field=nT1, 
                                   solver_name="adv_diff", 
                                   degree=3)
adv_diff.k = k
adv_diff.theta = 0.5

# +
# Define T boundary conditions via a sympy function

import sympy
abs_r  = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = 0.5 * sympy.sin(5.0*th) * sympy.sin(np.pi*(r-r_i)/(r_o-r_i)) + (r_o-r)/(r_o-r_i)

adv_diff.add_dirichlet_bc(  1.0,  "Lower" )
adv_diff.add_dirichlet_bc(  0.0,  "Upper" )

with nswarm.access(nT1):
    nT1.data[...] = uw.function.evaluate(init_t, nswarm.particle_coordinates.data).reshape(-1,1)
    
with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1,1)
    t_soln.data[...] = t_0.data[...]

# -

with nswarm.access(nT1):
    print(nT1.data.min(), nT1.data.max(), nT1.data.mean()) 


# +
# The position update should be built into the solve.
# However, it does mean that we can do an angular velocity 
# update of the positions

delta_t = 0.1

with nswarm.access(nswarm.particle_coordinates, nX0):
    nX0.data[...] = nswarm.data[...]

    n_x = uw.function.evaluate(r * sympy.cos(th+delta_t*theta_dot), nswarm.data)
    n_y = uw.function.evaluate(r * sympy.sin(th+delta_t*theta_dot), nswarm.data)

    nswarm.data[:,0] = n_x
    nswarm.data[:,1] = n_y


with nswarm.access(nT1):
    nT1.data[...] = uw.function.evaluate(t_soln.fn, nswarm.data).reshape(-1,1)

# restore coords 
with nswarm.access(nswarm.particle_coordinates):
    nswarm.data[...] = nX0.data[...]

# delta_t will be baked in when this is defined ... so re-define it 
adv_diff.solve(timestep=delta_t)

# -


print(meshball.stats((t_soln.fn)))


# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0] 
    pv.global_theme.camera['position'] = [0.0, 0.0, 1.0] 

    pv.start_xvfb()
    
    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    with meshball.access():
        usol = v_soln.data.copy()
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn, meshball.data)
 
    arrow_loc = np.zeros((v_soln.coords.shape[0],3))
    arrow_loc[:,0:2] = v_soln.coords[...]
    
    arrow_length = np.zeros((v_soln.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                  use_transparency=False, opacity=0.5)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=0.01)
    #pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)
    
    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -
def plot_T_mesh(filename):

    import mpi4py

    if mpi4py.MPI.COMM_WORLD.size==1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = 'white'
        pv.global_theme.window_size = [750, 750]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = 'pythreejs'
        pv.global_theme.smooth_shading = True
        pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0] 
        pv.global_theme.camera['position'] = [0.0, 0.0, 5.0] 

        pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

        with meshball.access():
            usol = v_soln.data.copy()

        pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn, meshball.data)

        arrow_loc = np.zeros((v_soln.coords.shape[0],3))
        arrow_loc[:,0:2] = v_soln.coords[...]

        arrow_length = np.zeros((v_soln.coords.shape[0],3))
        arrow_length[:,0:2] = usol[...] 

        pl = pv.Plotter()

        pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                      use_transparency=False, opacity=0.5)

        pl.add_arrows(arrow_loc, arrow_length, mag=0.005)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("mag")


        pl.screenshot(filename="{}.png".format(filename), window_size=(1000,1000), 
                      return_img=False)
        # pl.show()

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1,1)
    t_soln.data[...] = t_0.data[...]


# +
# Advection/diffusion model / update in time

delta_t = 0.05
adv_diff.k=0.0
expt_name="rotation_test_k_00"

plot_T_mesh(filename="{}_step_{}".format(expt_name,0))

for step in range(1,21):
    
## This first order update is a placeholder only !
    # with nswarm.access(nswarm.particle_coordinates, nX0):
    #     v_at_Tpts = uw.function.evaluate(v_soln.fn, nswarm.particle_coordinates.data)
    #     nX0.data[...] = nswarm.data[...]
    #     nswarm.data[...] -= delta_t * v_at_Tpts

## And this is a bit of a cheat really ... 
    with nswarm.access(nswarm.particle_coordinates, nX0):
        nX0.data[...] = nswarm.data[...]

        n_x = uw.function.evaluate(r * sympy.cos(th-delta_t*theta_dot), nswarm.data)
        n_y = uw.function.evaluate(r * sympy.sin(th-delta_t*theta_dot), nswarm.data)

        nswarm.data[:,0] = n_x
        nswarm.data[:,1] = n_y

    with nswarm.access(nT1):
        nT1.data[...] = uw.function.evaluate(t_soln.fn, nswarm.data).reshape(-1,1)

    # restore coords 
    with nswarm.access(nswarm.particle_coordinates):
        nswarm.data[...] = nX0.data[...]

    # delta_t will be baked in when this is defined ... so re-define it 
    adv_diff.solve(timestep=delta_t)
    
    # stats then loop
    
    tstats = t_soln.stats()
    dtstats = meshball.stats(t_soln.fn  - nT1.fn)
    
    
    if mpi4py.MPI.COMM_WORLD.rank==0:
        print("Timestep {}, dt {}".format(step, delta_t))
        print(tstats)
        print(dtstats)
        
    plot_T_mesh(filename="{}_step_{}".format(expt_name,step))

    # savefile = "output_conv/convection_cylinder_{}_iter.h5".format(step) 
    # meshball.save(savefile)
    # v_soln.save(savefile)
    # t_soln.save(savefile)
    # meshball.generate_xdmf(savefile)
 
# -


# savefile = "output_conv/convection_cylinder.h5".format(step) 
# meshball.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshball.generate_xdmf(savefile)



# +
import imageio as iio
from pathlib import Path

images = list()
for file in Path("output/RotationTestK001/").iterdir():
    im = iio.imread(file)
    images.append(im)
# -




