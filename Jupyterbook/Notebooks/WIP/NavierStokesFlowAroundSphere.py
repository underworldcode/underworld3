# # Navier Stokes test: flow around a circular inclusion (2D)
#
# Should be able to reproduce vortex shedding if free slip bc on the inner circle.

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3.systems import NavierStokesSwarm
from underworld3 import function

import numpy as np

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None
# options.getAll()

# +
import meshio, pygmsh

# Mesh a 2D pipe with a circular hole

csize = 0.1
csize_circle = 0.05
res = csize_circle

width = 3.0
height = 1.0
radius = 0.25


import mpi4py

if mpi4py.MPI.COMM_WORLD.rank==0:

    # Generate local mesh on boss process
    
    with pygmsh.geo.Geometry() as geom:

        geom.characteristic_length_max = csize
        
        p0 = geom.add_point((0.2,    csize,0.0), mesh_size=csize )
        p1 = geom.add_point((0.2,1.0-csize,0.0), mesh_size=csize )

        inclusion  = geom.add_circle((1.0,0.5,0.0), radius, make_surface=False, mesh_size=csize_circle)
        line     = geom.add_line(p0=p0, p1=p1)
        domain = geom.add_rectangle(xmin=0.0,ymin=0.0, xmax=width, ymax=height, z=0, holes=[inclusion], mesh_size=csize)
        
        geom.in_surface(line, domain.surface)
        
        geom.add_physical(domain.surface.curve_loop.curves[0], label="bottom")
        geom.add_physical(domain.surface.curve_loop.curves[1], label="right")
        geom.add_physical(domain.surface.curve_loop.curves[2], label="top")
        geom.add_physical(domain.surface.curve_loop.curves[3], label="left")
    
        geom.add_physical(inclusion.curve_loop.curves, label="inclusion")
        geom.add_physical(line, label="internal_boundary")

        geom.add_physical(domain.surface, label="Elements")    
        
        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry("ns_pipe_flow.msh")
        geom.save_geometry("ns_pipe_flow.vtk")

# -


pipemesh = uw.meshes.MeshFromGmshFile(dim=2, degree=1, filename="ns_pipe_flow.msh", label_groups=[], simplex=True)
pipemesh.dm.view()

# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:    
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0] 
    pv.global_theme.camera['position'] = [0.0, 0.0, 1.0]     
    
    pvmesh = pipemesh.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
    
    pl = pv.Plotter()

    points = np.zeros((pipemesh._centroids.shape[0],3))
    points[:,0] = pipemesh._centroids[:,0]
    points[:,1] = pipemesh._centroids[:,1]

    point_cloud = pv.PolyData(points)
  
    
    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, 
                  use_transparency=False, opacity=0.5)
    
    pl.add_points(point_cloud, color="Blue",
                  render_points_as_spheres=True,
                  point_size=2, opacity=1.0  )
    
    pl.show(cpos="xy")

# +
# Define some functions on the mesh

import sympy

# radius_fn = sympy.sqrt(pipemesh.rvec.dot(pipemesh.rvec)) # normalise by outer radius if not 1.0
# unit_rvec = pipemesh.rvec / (1.0e-10+radius_fn)

# Some useful coordinate stuff 

x = pipemesh.N.x
y = pipemesh.N.y

# relative to the centre of the inclusion
r  = sympy.sqrt((x-1.0)**2+(y-0.5)**2)
th = sympy.atan2(y-0.5,x-1.0)

# need a unit_r_vec equivalent

inclusion_rvec = pipemesh.rvec - 1.0 * pipemesh.N.i - 0.5 * pipemesh.N.j
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)

# -

v_soln = uw.mesh.MeshVariable('U',    pipemesh, pipemesh.dim, degree=2 )
p_soln = uw.mesh.MeshVariable('P',    pipemesh, 1, degree=1 )


swarm = uw.swarm.Swarm(mesh=pipemesh)
v_star = uw.swarm.SwarmVariable("Vs", swarm, pipemesh.dim, proxy_degree=3)
swarm.populate(fill_param=5)


def points_fell_out(coords): 
    coords[:,1] = coords[:,1] % width
    return coords


# +
# Create NS object

navier_stokes = NavierStokesSwarm(pipemesh, 
                velocityField=v_soln, 
                pressureField=p_soln, 
                velocityStar=v_star,
                u_degree=v_soln.degree, 
                p_degree=p_soln.degree, 
                rho=1.0,
                theta=1.0,
                verbose=False,
                projection=True,
                restore_points_func=points_fell_out,
                solver_name="navier_stokes")



# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

navier_stokes.rho=1.0
navier_stokes.theta=0.66
navier_stokes.penalty=0.0
navier_stokes.viscosity = 1.0
navier_stokes.bodyforce = 1.0e-16*pipemesh.N.i

Vb = 10.0
Free_Slip = False
expt_name = "pipe_flow_cylinder_R025_10_rho1"

if Free_Slip:
    hw = 1000.0 / res 
    surface_fn = sympy.exp(-((r - radius) / radius)**2 * hw)
    navier_stokes.bodyforce -= 1.0e5 * Vb * navier_stokes.rho * v_soln.fn.dot(inclusion_unit_rvec) * surface_fn * inclusion_unit_rvec
    navier_stokes._Ppre_fn = 1.0 / (navier_stokes.viscosity + navier_stokes.rho / navier_stokes.delta_t + 1.0e5 * Vb * surface_fn)
    # navier_stokes._Ppre_fn = 1.0 / (navier_stokes.viscosity )


else:
    surface_fn =  1.0e-32 * r
    navier_stokes.add_dirichlet_bc( (0.0,0.0), "inclusion" , (0,1) )  
    
    

# Velocity boundary conditions

navier_stokes.add_dirichlet_bc( (Vb,0.0),  "top" ,    (0,1) )
navier_stokes.add_dirichlet_bc( (Vb,0.0),  "bottom" , (0,1) )
navier_stokes.add_dirichlet_bc( (Vb,0.0),  "left" ,  (0,1) )
navier_stokes.add_dirichlet_bc( (Vb,0.0),  "right" ,  (0,1) )
# -


navier_stokes.solve(timestep=1.0)  # Stokes-like initial flow

with pipemesh.access(v_soln):
    v_soln.data[:,0] = 0.0


def plot_V_mesh(filename):

    import mpi4py

    if mpi4py.MPI.COMM_WORLD.size==1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = 'white'
        pv.global_theme.window_size = [1250, 1000]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = 'pythreejs'
        pv.global_theme.smooth_shading = True
        pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0] 
        pv.global_theme.camera['position'] = [0.0, 0.0, 2.0] 

        pvmesh = pipemesh.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
        
        
        with swarm.access():
            points = np.zeros((swarm.data.shape[0],3))
            points[:,0] = swarm.data[:,0]
            points[:,1] = swarm.data[:,1]

        point_cloud = pv.PolyData(points)
        
        
        # POINT CLOUD from centroids for streamlines


        with pipemesh.access():
             pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, pipemesh.data)

        with pipemesh.access():
            usol = v_soln.data.copy()
            

        v_vectors = np.zeros((pipemesh.data.shape[0],3))
        v_vectors[:,0:2] = uw.function.evaluate(v_soln.fn, pipemesh.data)
        pvmesh.point_data["V"] = v_vectors 

        arrow_loc = np.zeros((v_soln.coords.shape[0],3))
        arrow_loc[:,0:2] = v_soln.coords[...]

        arrow_length = np.zeros((v_soln.coords.shape[0],3))
        arrow_length[:,0:2] = usol[...] 
        
        pl = pv.Plotter()

        pl.add_arrows(arrow_loc, arrow_length, mag=0.1/Vb, opacity=0.75)

        # pvstream = pvmesh.streamlines_from_source(point_cloud, vectors="V", 
        #                                       integration_direction="both",
        #                                       max_steps=250
        #                                      )

        # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
        pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="P",
                  use_transparency=False, opacity=0.5)


        # pl.add_mesh(pvstream)
        
        
        pl.add_points(point_cloud, color="Black",
                      render_points_as_spheres=True,
                      point_size=2, opacity=0.66
                    )

        pl.remove_scalar_bar("P")
       # pl.remove_scalar_bar("mag")

        pl.screenshot(filename="{}.png".format(filename), window_size=(2560,1280), 
                      return_img=False)
        
        pl.close()

       # pl.show()

ts = 0

for step in range(0,50):
    delta_t = 2.0 * navier_stokes.estimate_dt()
    navier_stokes.solve(timestep=delta_t, zero_init_guess=False)
    
    with swarm.access(v_star):
        v_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data)
     
    # advect swarm
    print("Swarm advection")
    swarm.advection(v_soln.fn, delta_t)
    print("Swarm advection, complete")

    if mpi4py.MPI.COMM_WORLD.rank==0:
        print("Timestep {}, dt {}".format(step, delta_t))
                
    if ts%1 == 0:
        plot_V_mesh(filename="output/{}_step_{}".format(expt_name,step))

    ts += 1

    # savefile = "output/{}_ts_{}.h5".format(expt_name,step) 
    # pipemesh.save(savefile)
    # v_soln.save(savefile)
    # p_soln.save(savefile)
    # pipemesh.generate_xdmf(savefile)



# +
# display(navier_stokes._uu_g0)
# display(navier_stokes._uu_g3)
# display(navier_stokes._uu_g2)

# display(navier_stokes._up_g2)
# display(navier_stokes._up_g3)
# display(navier_stokes._pu_g1)


# +
# navier_stokes.petsc_options.getAll()

# +
# Solves a Stokes initial condition for this problem to avoid a crash-start 
# We could also just leave everything to V = (Vb,0)

# navier_stokes.petsc_options["snes_type"]="newtonls"
# navier_stokes.petsc_options["snes_rtol"]=1.0e-3
# navier_stokes.rho =0.01  # will trigger a rebuild and options are all re-read
# navier_stokes.solve(timestep=pipemesh.get_min_radius()/Vb)


# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [1250, 1250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0] 
    pv.global_theme.camera['position'] = [0.0, 0.0, 1.0] 

    pvmesh = pipemesh.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

#     points = np.zeros((t_soln.coords.shape[0],3))
#     points[:,0] = t_soln.coords[:,0]
#     points[:,1] = t_soln.coords[:,1]

#     point_cloud = pv.PolyData(points)

    with pipemesh.access():
        usol = v_soln.data.copy()
        
    with pipemesh.access():
        pvmesh.point_data["S"] = uw.function.evaluate(surface_fn, pipemesh.data)
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, pipemesh.data)


    v_vectors = np.zeros((pipemesh.data.shape[0],3))
    v_vectors[:,0:2] = uw.function.evaluate(v_soln.fn, pipemesh.data)
    pvmesh.point_data["V"] = v_vectors 
    
    arrow_loc = np.zeros((v_soln.coords.shape[0],3))
    arrow_loc[:,0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    # point sources at cell centres
    
    # points = np.zeros((pipemesh._centroids.shape[0],3))
    # points[:,0] = pipemesh._centroids[:,0]
    # points[:,1] = pipemesh._centroids[:,1]
    # point_cloud = pv.PolyData(points)
    
    pvstream = pvmesh.streamlines_from_source(point_cloud, vectors="V", 
                                              integration_direction="both",
                                              max_steps=100
                                             )
    
    
    pl = pv.Plotter()
 
    pl.add_arrows(arrow_loc, arrow_length, mag=0.01/Vb, opacity=0.75)

    # pl.add_points(point_cloud, cmap="coolwarm", 
    #               render_points_as_spheres=False,
    #               point_size=10, opacity=0.66
    #             )
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="P",
                  use_transparency=False, opacity=1.0)

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
    pl.add_mesh(pvstream)

    # pl.remove_scalar_bar("S")
    # pl.remove_scalar_bar("mag")

    pl.show()
# + active=""
#
# -

0/0



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

        pvmesh = pipemesh.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
        
        points = np.zeros((t_soln.coords.shape[0],3))
        points[:,0] = t_soln.coords[:,0]
        points[:,1] = t_soln.coords[:,1]

        point_cloud = pv.PolyData(points)

        with pipemesh.access():
            point_cloud.point_data["T"] = t_soln.data.copy()

        with pipemesh.access():
            usol = v_soln.data.copy()


        arrow_loc = np.zeros((v_soln.coords.shape[0],3))
        arrow_loc[:,0:2] = v_soln.coords[...]

        arrow_length = np.zeros((v_soln.coords.shape[0],3))
        arrow_length[:,0:2] = usol[...] 
        
        pl = pv.Plotter()

        pl.add_arrows(arrow_loc, arrow_length, mag=0.0001, opacity=0.75)

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
