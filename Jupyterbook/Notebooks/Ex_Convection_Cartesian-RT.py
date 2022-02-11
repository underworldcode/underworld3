# # Thermochemical convection
#
# We have a thermal convection (advection-diffusion) problem and a material-swarm mediated density variation 
#
#

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
# -

meshbox = uw.meshes.Unstructured_Simplex_Box(dim=2, minCoords=(0.0,0.0,0.0), maxCoords=(1.0,1.0,1.0), cell_size=1.0/16.0, regular=True)
meshbox.dm.view()   

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
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True
    
    pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
    
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, edge_color="Black", show_edges=True)

    pl.show(cpos="xy")
# -

v_soln = uw.mesh.MeshVariable('U',    meshbox,  meshbox.dim, degree=2 )
p_soln = uw.mesh.MeshVariable('P',    meshbox, 1, degree=1 )
t_soln = uw.mesh.MeshVariable("T",    meshbox, 1, degree=3)
t_0    = uw.mesh.MeshVariable("T0",   meshbox, 1, degree=3)


swarm  = uw.swarm.Swarm(mesh=meshbox)
Mat = uw.swarm.SwarmVariable("Material", swarm, 1, proxy_degree=3)
X0 = uw.swarm.SwarmVariable("X0", swarm, meshbox.dim, _proxy=False)
swarm.populate(fill_param=3)


# +
# Create Stokes object

stokes = Stokes(meshbox, velocityField=v_soln, 
                pressureField=p_soln, 
                u_degree=v_soln.degree, 
                p_degree=p_soln.degree, 
                solver_name="stokes", 
                verbose=False)

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
stokes.petsc_options.delValue("ksp_monitor")

# Constant visc
stokes.viscosity = 1.

# Velocity boundary conditions
stokes.add_dirichlet_bc( (0.0,), "Left" ,   (0,) )
stokes.add_dirichlet_bc( (0.0,), "Right" ,  (0,) )
stokes.add_dirichlet_bc( (0.0,), "Top" ,    (1,) )
stokes.add_dirichlet_bc( (0.0,), "Bottom" , (1,) )



# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre 
# of the sphere to (say) 1 at the surface

import sympy

# Some useful coordinate stuff 

x = meshbox.N.x
y = meshbox.N.y


# +
# Create adv_diff object

# Set some things
k = 1.0
h = 0.0 
r_i = 0.5
r_o = 1.0

adv_diff = uw.systems.AdvDiffusion(meshbox, 
                                   u_Field=t_soln, 
                                   V_Field=v_soln,
                                   solver_name="adv_diff", 
                                   degree=3,
                                   verbose=False)

adv_diff.k = k
adv_diff.theta = 0.5



# +
# Define T boundary / initial conditions via a sympy function

import sympy
init_t = 0.01 * sympy.sin(5.0*x) * sympy.sin(np.pi*y) + (1.0-y)

adv_diff.add_dirichlet_bc(  1.0,  "Bottom" )
adv_diff.add_dirichlet_bc(  0.0,  "Top" )

with meshbox.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1,1)
    t_soln.data[...] = t_0.data[...]


# +
buoyancy_force = 1.0e6 * t_soln.fn 
stokes.bodyforce = meshbox.N.j * buoyancy_force  

# check the stokes solve is set up and that it converges
stokes.solve()
# -

# Check the diffusion part of the solve converges 
adv_diff.solve(timestep=0.01*stokes.estimate_dt())


# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pv.start_xvfb()
    
    pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    with meshbox.access():
        usol = stokes.u.data.copy()
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn, meshbox.data)
 
    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                  use_transparency=False, opacity=0.5)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=1.0e-4, opacity=0.5)
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

        pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

        points = np.zeros((t_soln.coords.shape[0],3))
        points[:,0] = t_soln.coords[:,0]
        points[:,1] = t_soln.coords[:,1]

        point_cloud = pv.PolyData(points)
        
        with swarm.access():
            points = np.zeros((swarm.data.shape[0],3))
            points[:,0] = swarm.data[:,0]
            points[:,1] = swarm.data[:,1]

        swarm_point_cloud = pv.PolyData(points)



        with meshbox.access():
            point_cloud.point_data["T"] = t_soln.data.copy()

        with meshbox.access():
            usol = stokes.u.data.copy()

        pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn, meshbox.data)

        arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
        arrow_loc[:,0:2] = stokes.u.coords[...]

        arrow_length = np.zeros((stokes.u.coords.shape[0],3))
        arrow_length[:,0:2] = usol[...] 

        pl = pv.Plotter()


        pl.add_arrows(arrow_loc, arrow_length, mag=0.00002, opacity=0.75)

        pl.add_points(point_cloud, cmap="coolwarm", 
                      render_points_as_spheres=False,
                      point_size=10, opacity=0.5
                    )
        
        pl.add_points(swarm_point_cloud, color="Black",
                  render_points_as_spheres=True,
                  point_size=2.0, opacity=1.0
                )

        pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("mag")

        pl.screenshot(filename="{}.png".format(filename), window_size=(1280,1280), 
                      return_img=False)
        # pl.show()



# +
# Convection model / update in time

expt_name="output/Ra1e6_TC"

for step in range(0,50):
    
    stokes.solve(zero_init_guess=False)
    delta_t = 5.0*stokes.estimate_dt() 
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)
    
    
    # update swarm locations. 
    
    # Ideally this would just be 
    # swarm.timestep(V, dt, order) ... but let's do this explicitly first
    
    # 1. Previous locations 
    with swarm.access(X0):
        X0.data[...] = swarm.data[...]
        
        
    with swarm.access(swarm.particle_coordinates):
        v_at_Vpts = uw.function.evaluate(v_soln.fn, swarm.data).reshape(-1,meshbox.dim)
        mid_pt_coords = swarm.data[...] + 0.5 * delta_t * v_at_Vpts

        # validate_coords to ensure they live within the domain (or there will be trouble)
        # mid_pt_coords = restore_points_to_domain_func(mid_pt_coords)

        swarm.data[...] = mid_pt_coords

        ## Let the swarm be updated, and then move the rest of the way

    with swarm.access(swarm.particle_coordinates):
        v_at_Vpts = uw.function.evaluate(v_soln.fn, swarm.data).reshape(-1,meshbox.dim)
        new_coords = X0.data[...] + delta_t * v_at_Vpts

        # validate_coords to ensure they live within the domain (or there will be trouble)
        # new_coords = self.restore_points_to_domain_func(new_coords)

        swarm.data[...] = new_coords
  
    
    # stats then loop
    tstats = t_soln.stats()
    
    
    if mpi4py.MPI.COMM_WORLD.rank==0:
        print("Timestep {}, dt {}".format(step, delta_t))
#         print(tstats)
        
    plot_T_mesh(filename="{}_step_{}".format(expt_name,step))

    # savefile = "{}_ts_{}.h5".format(expt_name,step) 
    # meshbox.save(savefile)
    # v_soln.save(savefile)
    # t_soln.save(savefile)
    # meshbox.generate_xdmf(savefile)



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
    
    pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
    
    
    points = np.zeros((t_soln.coords.shape[0],3))
    points[:,0] = t_soln.coords[:,0]
    points[:,1] = t_soln.coords[:,1]

    point_cloud = pv.PolyData(points)
    
    
    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]

    swarm_point_cloud = pv.PolyData(points)

    with meshbox.access():
        point_cloud.point_data["T"] = t_soln.data.copy()


    with meshbox.access():
        usol = stokes.u.data.copy()
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.fn, meshbox.data)
 
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
    
    pl.add_points(swarm_point_cloud, color="Black",
                  render_points_as_spheres=True,
                  point_size=3.0, opacity=1.0
                )
    
    
    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)


    pl.show(cpos="xy")
# -




