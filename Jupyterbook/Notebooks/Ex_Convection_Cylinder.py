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
# -

meshball = uw.meshing.Annulus(radiusInner=0.5, 
                              radiusOuter=1.0,
                              cellSize=0.0333,
                                    degree=1)


# +
# check the mesh if in a notebook / serial


if uw.mpi.size==1:    
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0] 
    pv.global_theme.camera['position'] = [0.0, 0.0, -5.0]     
    
    meshball.vtk("ignore_meshball.vtk")
    pvmesh = pv.read("ignore_meshball.vtk")
    
    
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, 
                  use_transparency=False, opacity=0.5)

    pl.show()
# -

v_soln = uw.discretisation.MeshVariable('U',    meshball, meshball.dim, degree=2 )
p_soln = uw.discretisation.MeshVariable('P',    meshball, 1, degree=1 )
t_soln = uw.discretisation.MeshVariable("T",    meshball, 1, degree=3)
t_0    = uw.discretisation.MeshVariable("T0",   meshball, 1, degree=3)


swarm  = uw.swarm.Swarm(mesh=meshball)
T1 = uw.swarm.SwarmVariable("Tminus1", swarm, 1)
X1 = uw.swarm.SwarmVariable("Xminus1", swarm, 2)
swarm.populate(fill_param=3)


# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                solver_name="stokes")

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshball.dim)
stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(viscosity = 1)


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

radius_fn = sympy.sqrt(meshball.X.dot(meshball.X)) # normalise by outer radius if not 1.0
unit_rvec = meshball.X / (1.0e-10+radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff 

x = meshball.X[0]
y = meshball.X[1]

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
                                   degree=3,
                                   verbose=False)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshball.dim)
adv_diff.constitutive_model.material_properties = adv_diff.constitutive_model.Parameters(diffusivity = 1)

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
# +
buoyancy_force = 1.0e6 * t_soln.sym[0] / (0.5)**3 
stokes.bodyforce = unit_rvec * buoyancy_force  

# check the stokes solve converges
stokes.solve()

# +
# Check the diffusion part of the solve converges 
adv_diff.petsc_options["ksp_monitor"]=None
adv_diff.petsc_options["monitor"]=None

adv_diff.solve(timestep=0.00001*stokes.estimate_dt())


# +
diff = uw.systems.Poisson(meshball, 
                                   u_Field=t_soln, 
                                   solver_name="diff_only")

diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshball.dim)
diff.constitutive_model.material_properties = adv_diff.constitutive_model.Parameters(diffusivity = 1)


# -

diff.solve()

0/0

# +
# check the mesh if in a notebook / serial

if uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    # pv.start_xvfb()
    
    meshball.vtk("ignore_meshball.vtk")
    pvmesh = pv.read("ignore_meshball.vtk")
   
    with meshball.access():
        usol = stokes.u.data.copy()
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.sym[0], meshball.data)
    pvmesh.point_data["Ts"]  = uw.function.evaluate(adv_diff._u_star.sym[0], meshball.data)
    pvmesh.point_data["dT"]  = uw.function.evaluate(t_soln.sym[0] - adv_diff._u_star.sym[0], meshball.data)


 
    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="dT",
                  use_transparency=False, opacity=0.5)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=0.0001)
    #pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)
    
    # pl.add_points(pdata)

    pl.show(cpos="xy")
# -


pvmesh.point_data["dT"].min()

adv_diff.petsc_options["pc_gamg_agg_nsmooths"]= 1

# +
# check the mesh if in a notebook / serial


if uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk
   
    meshball.vtk("ignore_meshball.vtk")
    pvmesh = pv.read("ignore_meshball.vtk")
    
    with meshball.access():
        usol = stokes.u.data.copy()
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_soln.sym[0]-t_0.sym[0], meshball.data)
 
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


    if uw.mpi.size==1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = 'white'
        pv.global_theme.window_size = [750, 750]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = 'panel'
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

expt_name="output/Cylinder_Ra1e6i"

for step in range(0,25):
    
    stokes.solve()
    delta_t = 5.0*stokes.estimate_dt() 
    adv_diff.solve(timestep=delta_t)
    
    # stats then loop
    tstats = t_soln.stats()
    
    
    if uw.mpi.rank==0:
        print("Timestep {}, dt {}".format(step, delta_t))
#         print(tstats)
        
#     plot_T_mesh(filename="{}_step_{}".format(expt_name,step))

    savefile = "{}_{}_iter.h5".format(expt_name,step) 
    meshball.save(savefile)
    v_soln.save(savefile)
    t_soln.save(savefile)
    meshball.generate_xdmf(savefile)

# -


# savefile = "output_conv/convection_cylinder.h5".format(step) 
# meshball.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshball.generate_xdmf(savefile)


# +


if uw.mpi.size==1:

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




