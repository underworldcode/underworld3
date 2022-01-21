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
import meshio

meshball = uw.meshes.SphericalShell(dim=2, radius_inner=0.5,
                                    radius_outer=1.0, 
                                    cell_size=0.075,
                                    cell_size_lower=0.05,
                                    degree=1, verbose=False)


# +
# Define some functions on the mesh

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
v_x = -1.0 *  r * theta_dot * sympy.sin(th)
v_y =         r * theta_dot * sympy.cos(th)
# -

# coord_vec = meshball.dm.getCoordinates()
#
# coords = coord_vec.array.reshape(-1,2)
# mesh_th = np.arctan2(coords[:,1], coords[:,0]).reshape(-1,1)
# mesh_r  = np.hypot(coords[:,0], coords[:,1]).reshape(-1,1)
# coords *= 1.0 + 0.5 * (1.0-mesh_r) * np.cos(mesh_th*5.0)
# meshball.dm.setCoordinates(coord_vec)
#
# meshball.meshio.points[:,0] = coords[:,0]
# meshball.meshio.points[:,1] = coords[:,1]


v_soln = uw.mesh.MeshVariable('U',    meshball, meshball.dim, degree=2 )
p_soln = uw.mesh.MeshVariable('P',    meshball, 1, degree=1 )


# +
# Create Stokes object (switch out for NS in a minute)

navier_stokes = NavierStokes(meshball, 
                velocityField=v_soln, 
                pressureField=p_soln, 
                u_degree=2, 
                p_degree=1, 
                rho=1.0,
                theta=0.5,
                solver_name="navier_stokes")

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
navier_stokes.petsc_options.delValue("ksp_monitor")
navier_stokes.petsc_options["ksp_rtol"]=3.0e-4
navier_stokes.petsc_options["snes_type"]="newtonls"
navier_stokes.petsc_options["snes_max_it"]=150
navier_stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fgmres"
navier_stokes.petsc_options["fieldsplit_velocity_pc_type"] = "lu"
# navier_stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
# navier_stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None



# Constant visc
navier_stokes.viscosity = 1.0
navier_stokes.penalty=1.0

navier_stokes.bodyforce = unit_rvec * 1.0e-16

# Velocity boundary conditions
navier_stokes.add_dirichlet_bc( (v_x,v_y), "Upper" , (0,1) )
navier_stokes.add_dirichlet_bc( (0.0,0.0), "Lower" , (0,1) )

# +
# navier_stokes.petsc_options.getAll()
# -

with meshball.access(v_soln):
    v_soln.data[...] = 0.0

navier_stokes.estimate_dt()

navier_stokes.solve(timestep=0.01)
navier_stokes.estimate_dt()

navier_stokes._uu_g0
# navier_stokes._pp_g0

navier_stokes.penalty



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

    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

#     points = np.zeros((t_soln.coords.shape[0],3))
#     points[:,0] = t_soln.coords[:,0]
#     points[:,1] = t_soln.coords[:,1]

#     point_cloud = pv.PolyData(points)

    with meshball.access():
        usol = v_soln.data.copy()

    arrow_loc = np.zeros((v_soln.coords.shape[0],3))
    arrow_loc[:,0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 

    pl = pv.Plotter()

 
    pl.add_arrows(arrow_loc, arrow_length, mag=5.0e-2, opacity=0.75)

    # pl.add_points(point_cloud, cmap="coolwarm", 
    #               render_points_as_spheres=False,
    #               point_size=10, opacity=0.66
    #             )


    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

    # pl.remove_scalar_bar("T")
    pl.remove_scalar_bar("mag")

    pl.show()


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





0/0

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1,1)
    t_soln.data[...] = t_0.data[...]


# +
# Advection/diffusion model / update in time

delta_t = 0.05
adv_diff.k=0.01
expt_name="output/rotation_test_k_001"

plot_T_mesh(filename="{}_step_{}".format(expt_name,0))

for step in range(1,21):
    
    # This shows how we over-rule the mid-point scheme that is provided
    # by the adv_diff solver
    
    with adv_diff._nswarm.access():
        print(adv_diff._nswarm.data.shape)
        coords0 = adv_diff._nswarm.data.copy()

        n_x = uw.function.evaluate(r * sympy.cos(th-delta_t*theta_dot), coords0)
        n_y = uw.function.evaluate(r * sympy.sin(th-delta_t*theta_dot), coords0)

        coords = np.empty_like(coords0)
        coords[:,0] = n_x
        coords[:,1] = n_y

   
    # delta_t will be baked in when this is defined ... so re-define it 
    adv_diff.solve(timestep=delta_t, coords=coords)
    
    # stats then loop
    
    tstats = t_soln.stats()    
    
    if mpi4py.MPI.COMM_WORLD.rank==0:
        print("Timestep {}, dt {}".format(step, delta_t))
        print(tstats)
        
    plot_T_mesh(filename="{}_step_{}".format(expt_name,step))

    # savefile = "output_conv/convection_cylinder_{}_iter.h5".format(step) 
    # meshball.save(savefile)
    # v_soln.save(savefile)
    # t_soln.save(savefile)
    # meshball.generate_xdmf(savefile)
 


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
    pv.global_theme.camera['position'] = [0.0, 0.0, 5.0] 

    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    points = np.zeros((t_soln.coords.shape[0],3))
    points[:,0] = t_soln.coords[:,0]
    points[:,1] = t_soln.coords[:,1]

    point_cloud = pv.PolyData(points)

    with meshball.access():
        point_cloud.point_data["T"] = t_soln.data-t_0.data

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

    pl.show()
# -

# savefile = "output_conv/convection_cylinder.h5".format(step) 
# meshball.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshball.generate_xdmf(savefile)


