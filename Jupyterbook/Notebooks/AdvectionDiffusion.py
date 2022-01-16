# # Stokes in a disc with adv_diff to solve T and back-in-time sampling with particles
#
# This is a simple example in which we try to instantiate two solvers on the mesh and have them use a common set of variables.
#
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.
#
# The next step is to add particles at node points and sample back along the streamlines to find values of the T field at a previous time. 
#
# (Note, we keep all the pieces from previous increments of this problem to ensure that we don't break something along the way)

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

# options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None
# options.getAll()

# +
import meshio

meshball = uw.meshes.SphericalShell(dim=2, 
                                    radius_inner=0.5,
                                    radius_outer=1.0, cell_size=0.033,
                                    degree=1, verbose=False)


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
    pv.global_theme.camera['position'] = [0.0, 0.0, -5.0]     
    
    pvmesh = meshball.mesh2pyvista()
    
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, 
                  use_transparency=False, opacity=0.5)

    pl.show()
# -

v_soln = uw.mesh.MeshVariable('U',    meshball, meshball.dim, degree=2 )
p_soln = uw.mesh.MeshVariable('P',    meshball, 1, degree=1 )
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

# nswarm = uw.swarm.Swarm(meshball)
# nT1 = uw.swarm.SwarmVariable("nTminus1", nswarm, 1)
# nX0 = uw.swarm.SwarmVariable("nX0", nswarm, nswarm.dim)

# nswarm.dm.finalizeFieldRegister()
# nswarm.dm.addNPoints(t_soln.coords.shape[0]+1) # why + 1 ? That's the number of spots actually allocated
# cellid = nswarm.dm.getField("DMSwarm_cellid")
# coords = nswarm.dm.getField("DMSwarmPIC_coor").reshape( (-1, nswarm.dim) )
# coords[...] = t_soln.coords[...]
# cellid[:] = meshball.get_closest_cells(coords)
# nswarm.dm.restoreField("DMSwarmPIC_coor")
# nswarm.dm.restoreField("DMSwarm_cellid")
# nswarm.dm.migrate(remove_sent_points=True)

# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                u_degree=2, p_degree=1, solver_name="stokes")

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
stokes.petsc_options.delValue("ksp_monitor")

# Constant visc
stokes.viscosity = 1.

# Velocity boundary conditions
stokes.add_dirichlet_bc( (0.,0.), "Upper" , (0,1) )
stokes.add_dirichlet_bc( (0.,0.), "Lower" , (0,1) )

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

r  = sympy.sqrt(x**2+y**2)
th = sympy.atan2(y+1.0e-5,x+1.0e-5)

# +
# Create adv_diff object

# Set some things
k = 1.0
h = 0.0 
r_i = 0.5
r_o = 1.0

adv_diff = uw.systems.AdvDiffusion(meshball, 
                                   u_Field=t_soln, 
                                   V_Field=v_soln,
                                   solver_name="adv_diff", 
                                   degree=3)

adv_diff.k = k
adv_diff.theta = 0.5
# adv_diff.f = t_soln.fn / delta_t - t_star.fn / delta_t


# +
# Define T boundary conditions via a sympy function

import sympy
abs_r  = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = 0.01 * sympy.sin(15.0*th) * sympy.sin(np.pi*(r-r_i)/(r_o-r_i)) + (r_o-r)/(r_o-r_i)

adv_diff.add_dirichlet_bc(  1.0,  "Lower" )
adv_diff.add_dirichlet_bc(  0.0,  "Upper" )

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1,1)
    t_soln.data[...] = t_0.data[...]
# -



# +
buoyancy_force = 1.0e6 * t_soln.fn / (0.5)**3 
stokes.bodyforce = unit_rvec * buoyancy_force  

# check the stokes solve converges
stokes.solve()
# -

adv_diff

# Check the diffusion part of the solve converges 
adv_diff.solve(timestep=0.0001)


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

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                  use_transparency=False, opacity=0.5)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=0.0001)
    #pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)
    
    # pl.add_points(pdata)

    pl.show(cpos="xy")
# -


delta_t = 5.0*stokes.dt()

# +
# with meshball.access():

# mid pt scheme

# with nswarm.access(nswarm.particle_coordinates, nX0):
#     v_at_Tpts = uw.function.evaluate(v_soln.fn, nswarm.data)
#     nX0.data[...] = nswarm.data[...]
#     nswarm.data[...] -= 0.5 * delta_t * v_at_Tpts
    
# with nswarm.access(nswarm.particle_coordinates):
#     v_at_Tpts_half_dt = uw.function.evaluate(v_soln.fn, nswarm.data)
#     nswarm.data[...] = nX0.data[...] - delta_t * v_at_Tpts

# with nswarm.access(nT1):
#     nT1.data[...] = uw.function.evaluate(t_soln.fn, nswarm.data).reshape(-1,1)
    
# # restore coords 
# with nswarm.access(nswarm.particle_coordinates):
#     nswarm.data[...] = nX0.data[...]
# -

adv_diff.solve(timestep=delta_t)

# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

      
    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    with meshball.access():
        usol = stokes.u.data.copy()
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn-t_0.fn, meshball.data)
 
    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                  use_transparency=False, opacity=0.5)
    
    # pl.add_arrows(arrow_loc, arrow_length, mag=0.025)
    
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

        points = np.zeros((t_soln.coords.shape[0],3))
        points[:,0] = t_soln.coords[:,0]
        points[:,1] = t_soln.coords[:,1]

        point_cloud = pv.PolyData(points)

        with meshball.access():
            point_cloud.point_data["T"] = t_soln.data.copy()

        with meshball.access():
            usol = stokes.u.data.copy()

        pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn, meshball.data)

        arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
        arrow_loc[:,0:2] = stokes.u.coords[...]

        arrow_length = np.zeros((stokes.u.coords.shape[0],3))
        arrow_length[:,0:2] = usol[...] 

        pl = pv.Plotter()


        pl.add_arrows(arrow_loc, arrow_length, mag=0.00002, opacity=0.75)

        pl.add_points(point_cloud, cmap="coolwarm", 
                      render_points_as_spheres=False,
                      point_size=10, opacity=0.66
                    )
        

        pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("mag")

        pl.screenshot(filename="{}.png".format(filename), window_size=(1280,1280), 
                      return_img=False)
        # pl.show()

# +
# Convection model / update in time

expt_name="output/Ra1e6i"

for step in range(0,25):
    
    stokes.solve()
    delta_t = 5.0*stokes.dt() 
    adv_diff.solve(timestep=delta_t)
    
    # stats then loop
    tstats = t_soln.stats()
    
    
    if mpi4py.MPI.COMM_WORLD.rank==0:
        print("Timestep {}, dt {}".format(step, delta_t))
#         print(tstats)
        
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
    
    pv.start_xvfb()
    
    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
    
    
    points = np.zeros((t_soln.coords.shape[0],3))
    points[:,0] = t_soln.coords[:,0]
    points[:,1] = t_soln.coords[:,1]

    point_cloud = pv.PolyData(points)

    with meshball.access():
        point_cloud.point_data["T"] = t_soln.data.copy()


    with meshball.access():
        usol = stokes.u.data.copy()
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn, meshball.data)
 
    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    pl = pv.Plotter()
   
    pl.add_arrows(arrow_loc, arrow_length, mag=0.00002, opacity=0.75)
    #pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)
    
    
    pl.add_points(point_cloud, cmap="coolwarm", 
                  render_points_as_spheres=True,
                  point_size=7.5, opacity=0.25
                )
    
    
    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)


    pl.show(cpos="xy")
# -



0/0

# +
# Evaluate Tstar via 
# 1) node positions traced upstream
# 2) swarm positions traced upstream then projected back to nodes with inv. distance weights.

# T on the particles:

with swarm.access(T1):
    T1.data[:,0] = uw.function.evaluate(t_soln.fn, swarm.particle_coordinates.data)

# +
kd = uw.algorithms.KDTree(t_soln.coords)
kd.build_index()

with swarm.access():
    n,d,b = kd.find_closest_point(swarm.particle_coordinates.data)

tstar_mesh = np.zeros((t_soln.coords.shape[0]))
w = np.zeros((t_soln.coords.shape[0]))

with meshball.access():
    for i in range(0,n.shape[0]):
        with swarm.access():
            tstar_mesh[n[i]] += T1.data[i] / (1.0e-10 + d[n[i]])
        w[n[i]] += 1.0 / (1.0e-10 + d[n[i]])
        
tstar_mesh /= w
# -

np.where(w == 0.0)

# +
pl2 = pv.Plotter()

pvmesh2 = pvmesh.copy()
pvmesh2.scale([0.8]*3)


t_start = pv.PolyData(t_soln.coords)

with swarm.access():
    t_points = pv.PolyData(swarm.particle_coordinates.data[0:27])
    t_points["T"] = T1.data[0:27]


pl2.add_mesh(pvmesh,'Green', 'wireframe', opacity=0.5)
# pl2.add_mesh(pvmesh2,'White',  opacity=1.0)

# pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=False, 
#               use_transparency=False, opacity=0.5)
# pl.add_arrows(meshball.data, u_mesh, mag=25.0)

# pl2.add_points(t_start, color="Black", point_size=2.0)
pl2.add_points(t_points, cmap="coolwarm", scalars="T")

   
pl2.show()
# -

aa,bb =meshball.dm.getTransitiveClosure(895)

meshball.dm.getConeSize(895)

meshball.pygmesh.cells[0].data.shape

0/0

# +
pl = pv.Plotter()

pvmesh2 = pvmesh.copy()
pvmesh2.scale([0.95]*3)

pl.add_mesh(pvmesh,'Green', 'wireframe', opacity=0.5)
pl.add_mesh(pvmesh2,'White',  opacity=1.0)

# pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=False, 
#               use_transparency=False, opacity=0.5)
# pl.add_arrows(meshball.data, u_mesh, mag=25.0)
with s.access():
    pl.add_points(s.particle_coordinates.data, cmap="CoolWarm", color="Tstar", point_size=2.0 )
pl.show()
# -










