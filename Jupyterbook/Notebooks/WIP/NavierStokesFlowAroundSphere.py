# # Navier Stokes test: boundary driven ring with step change in boundary conditions
#
# This should develop a boundary layer with sqrt(t) growth rate

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3.systems import NavierStokes
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

csize = 0.075
csize_circle = 0.033

width = 5.0
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
        
        geom.generate_mesh(dim=2, verbose=True)
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

radius_fn = sympy.sqrt(pipemesh.rvec.dot(pipemesh.rvec)) # normalise by outer radius if not 1.0
unit_rvec = pipemesh.rvec / (1.0e-10+radius_fn)

# Some useful coordinate stuff 

x = pipemesh.N.x
y = pipemesh.N.y

# relative to the centre of the inclusion
r  = sympy.sqrt((x-1.0)**2+(y-0.5)**2)
th = sympy.atan2(y-0.5,x-1.0)

# -

v_soln = uw.mesh.MeshVariable('U',    pipemesh, pipemesh.dim, degree=2 )
p_soln = uw.mesh.MeshVariable('P',    pipemesh, 1, degree=1 )


# +
# Create NS object

navier_stokes = NavierStokes(pipemesh, 
                velocityField=v_soln, 
                pressureField=p_soln, 
                u_degree=v_soln.degree, 
                p_degree=p_soln.degree, 
                rho=1.0,
                theta=0.5,
                verbose=True,
                solver_name="navier_stokes")



# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc
navier_stokes.viscosity = 1.0
navier_stokes.bodyforce = pipemesh.N.y * 1.0e-16

# Velocity boundary conditions

Vb = 25.0

navier_stokes.add_dirichlet_bc( (Vb,0.0),  "top" ,    (0,1) )
navier_stokes.add_dirichlet_bc( (Vb,0.0),  "bottom" , (0,1) )
navier_stokes.add_dirichlet_bc( (Vb,0.0),  "left" ,  (0,1) )
navier_stokes.add_dirichlet_bc( (Vb,0.0),  "right" ,  (0,1) )
navier_stokes.add_dirichlet_bc( (0.0,0.0), "inclusion" , (0,1) )

# +
# navier_stokes.petsc_options.getAll()
# -

with pipemesh.access(v_soln):
    v_soln.data[:,0] = Vb

# +
# Solves a Stokes initial condition for this problem to avoid a crash-start 
# We could also just leave everything to V = (Vb,0)

# navier_stokes.petsc_options["snes_type"]="newtonls"
# navier_stokes.petsc_options["snes_rtol"]=1.0e-3


# navier_stokes.rho =0.01  # will trigger a rebuild and options are all re-read
# navier_stokes.solve(timestep=pipemesh.get_min_radius()/Vb)


# +
# different options for NS solve cf to Stokes-like solve

navier_stokes.verbose=False
navier_stokes.petsc_options["snes_type"]="qn"
navier_stokes.petsc_options["snes_qn_type"]="lbfgs"
navier_stokes.petsc_options["snes_rtol"]=1.0e-4
navier_stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-5
navier_stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 1.0e-5
navier_stokes.petsc_options["snes_max_it"]=250
# navier_stokes.petsc_options["snes_qn_linesearch_type"]="basic"
navier_stokes.petsc_options["snes_qn_monitor"]=None
navier_stokes.petsc_options["snes_qn_scale_type"]="diagonal"
navier_stokes.petsc_options["snes_qn_restart_type"]="powell"
#navier_stokes.petsc_options["snes_qn_m"]=25

navier_stokes.rho=1.0
navier_stokes.theta=0.5

expt_name = "pipe_flow_cylinder_R025_V25"
# -

for step in range(0,200):
    navier_stokes.solve(timestep=2.0*navier_stokes.estimate_dt(), zero_init_guess=True)
    
    savefile = "output/{}_ts_{}.h5".format(expt_name,step) 
    pipemesh.save(savefile)
    v_soln.save(savefile)
    p_soln.save(savefile)
    pipemesh.generate_xdmf(savefile)



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
    pv.global_theme.camera['position'] = [0.0, 0.0, 10.0] 

    pvmesh = pipemesh.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

#     points = np.zeros((t_soln.coords.shape[0],3))
#     points[:,0] = t_soln.coords[:,0]
#     points[:,1] = t_soln.coords[:,1]

#     point_cloud = pv.PolyData(points)

    with pipemesh.access():
        usol = v_soln.data.copy()
        
    v_vectors = np.zeros((pipemesh.data.shape[0],3))
    v_vectors[:,0:2] = uw.function.evaluate(v_soln.fn, pipemesh.data)
    pvmesh.point_data["V"] = v_vectors 
    
    arrow_loc = np.zeros((v_soln.coords.shape[0],3))
    arrow_loc[:,0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    # point sources at cell centres
    
    points = np.zeros((pipemesh._centroids.shape[0],3))
    points[:,0] = pipemesh._centroids[:,0]
    points[:,1] = pipemesh._centroids[:,1]
    point_cloud = pv.PolyData(points)
    
    pvstream = pvmesh.streamlines_from_source(point_cloud, vectors="V", 
                                              integration_direction="both",
                                              max_steps=10
                                             )
    
    
    pl = pv.Plotter()

 
    pl.add_arrows(arrow_loc, arrow_length, mag=5.0e-2, opacity=0.75)

    # pl.add_points(point_cloud, cmap="coolwarm", 
    #               render_points_as_spheres=False,
    #               point_size=10, opacity=0.66
    #             )


    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
    pl.add_mesh(pvstream)

    # pl.remove_scalar_bar("T")
    pl.remove_scalar_bar("mag")

    pl.show()
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

        pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
        
        points = np.zeros((t_soln.coords.shape[0],3))
        points[:,0] = t_soln.coords[:,0]
        points[:,1] = t_soln.coords[:,1]

        point_cloud = pv.PolyData(points)

        with meshball.access():
            point_cloud.point_data["T"] = t_soln.data.copy()

        with meshball.access():
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
