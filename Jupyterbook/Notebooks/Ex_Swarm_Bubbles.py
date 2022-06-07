# # Multiple materials 
#
# We introduce the notion of an `IndexSwarmVariable` which automatically generates masks for a swarm
# variable that consists of discrete level values (integers).
#
# For a variable $M$, the mask variables are $\left\{ M^0, M^1, M^2 \ldots M^{N-1} \right\}$ where $N$ is the number of indices (e.g. material types) on the variable. This value *must be defined in advance*.
#
# The masks are orthogonal in the sense that $M^i * M^j = 0$ if $i \ne j$, and they are complete in the sense that $\sum_i M^i = 1$ at all points.
#
# The masks are implemented as continuous mesh variables (the user can specify the interpolation order) and so they are also differentiable (once). 
#

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

render = True
# -

meshbox = uw.util_mesh.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
                                              maxCoords=(1.0,1.0), 
                                              cellSize=1.0/48.0, 
                                              regular=True)


# +
import sympy

# Some useful coordinate stuff 

x = meshbox.N.x
y = meshbox.N.y
# -

v_soln = uw.mesh.MeshVariable('U',    meshbox,  meshbox.dim, degree=2 )
p_soln = uw.mesh.MeshVariable('P',    meshbox,  1, degree=1 )


swarm     = uw.swarm.Swarm(mesh=meshbox)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=4)
swarm.populate(fill_param=4)


# +
blobs = np.array(
        [[ 0.25, 0.75, 0.1,  1], 
         [ 0.45, 0.70, 0.05, 2], 
         [ 0.65, 0.60, 0.06, 3], 
         [ 0.85, 0.40, 0.06, 1], 
         [ 0.65, 0.20, 0.06, 2], 
         [ 0.45, 0.20, 0.12, 3] ])


with swarm.access(material):
    material.data[...] = 0
    
    for i in range(blobs.shape[0]):
        cx, cy, r, m = blobs[i,:]              
        inside = (swarm.data[:,0] - cx)**2 + (swarm.data[:,1] - cy)**2 < r**2
        material.data[inside] = m
        
# -


material.f

meshbox.X

# +
# The material fields are differentiable 

sympy.derive_by_array(material.f, meshbox.X).reshape(2,4).tomatrix()
# -

mat_density = np.array([1,0.1,0.1,2])
density = mat_density[0] * material.f[0] + mat_density[1] * material.f[1] + \
          material.f[2] * mat_density[2] + mat_density[3] * material.f[3] 

mat_viscosity = np.array([1,0.1,10.0,10.0])
viscosity = mat_viscosity[0] * material.f[0] + \
            mat_viscosity[1] * material.f[1] + \
            mat_viscosity[2] * material.f[2] + \
            mat_viscosity[3] * material.f[3] 

if render:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True


    meshbox.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")    

    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]
        points[:,2] = 0.0

    point_cloud = pv.PolyData(points)

    with meshbox.access():
        pvmesh.point_data["M0"] = uw.function.evaluate(material.f[0], meshbox.data)
        pvmesh.point_data["M1"] = uw.function.evaluate(material.f[1], meshbox.data)
        pvmesh.point_data["M2"] = uw.function.evaluate(material.f[2], meshbox.data)
        pvmesh.point_data["M3"] = uw.function.evaluate(material.f[3], meshbox.data)
        pvmesh.point_data["M"] = 1.0 * pvmesh.point_data["M1"] +  2.0 * pvmesh.point_data["M2"] + 3.0 * pvmesh.point_data["M3"]

        pvmesh.point_data["rho"]  = uw.function.evaluate(density,   meshbox.data)
        pvmesh.point_data["visc"] = uw.function.evaluate(sympy.log(viscosity), meshbox.data)



    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter(notebook=True)

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)         


    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="rho",
                        use_transparency=False, opacity=0.95)


    pl.show(cpos="xy")
# +
# Create Stokes object

stokes = uw.systems.Stokes(meshbox, 
                velocityField=v_soln, 
                pressureField=p_soln, 
                u_degree=v_soln.degree, 
                p_degree=p_soln.degree, 
                solver_name="stokes", 
                verbose=False)

# Set some things
import sympy
from sympy import Piecewise

stokes.viscosity = viscosity
stokes.penalty = 1.0
stokes.bodyforce = - meshbox.N.j * density
stokes._Ppre_fn = 1.0 / (stokes.viscosity + stokes.penalty)

# free slip.  
# note with petsc we always need to provide a vector of correct cardinality. 
stokes.add_dirichlet_bc( (0.,0.), ["Bottom",  "Top"], 1 )  # top/bottom: components, function, markers 
stokes.add_dirichlet_bc( (0.,0.), ["Left", "Right"],  0 )  # left/right: components, function, markers
# -


stokes.solve()

# +
# check the solution


if uw.mpi.size==1 and render:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    # pv.start_xvfb()
    
    meshbox.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")    


    with meshbox.access():
        usol = stokes.u.data.copy()
  
    pvmesh.point_data["rho"]  = uw.function.evaluate(density,   meshbox.data)
    pvmesh.point_data["visc"] = uw.function.evaluate(sympy.log(viscosity), meshbox.data)


    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    pl = pv.Plotter()

    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="rho",
                  opacity=0.5)
    

    pl.add_arrows(arrow_loc, arrow_length, mag=1.0e1, opacity=0.5)
    #pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)
    
    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -


def plot_mesh(filename):


    if uw.mpi.size==1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = 'white'
        pv.global_theme.window_size = [750, 750]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = 'pythreejs'
        pv.global_theme.smooth_shading = False
        pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0] 
        pv.global_theme.camera['position'] = [0.0, 0.0, 5.0] 

        meshbox.vtk("tmp_box.vtk")
        pvmesh = pv.read("tmp_box.vtk")    

        with meshbox.access():
            usol = stokes.u.data.copy()
            
        with swarm.access():
            points = np.zeros((swarm.data.shape[0],3))
            points[:,0] = swarm.data[:,0]
            points[:,1] = swarm.data[:,1]
            points[:,2] = 0.0

        point_cloud = pv.PolyData(points)
        
        with swarm.access():
            point_cloud.point_data["M"] = material.data.astype(float)


        pvmesh.point_data["rho"]  = uw.function.evaluate(density,   meshbox.data)
        pvmesh.point_data["visc"] = uw.function.evaluate(sympy.log(viscosity), meshbox.data)

        arrow_loc = np.zeros((meshbox.data.shape[0],3))
        arrow_loc[:,0:2] = meshbox.data[...]

        arrow_length = np.zeros((meshbox.data.shape[0],3))
        arrow_length[:,0:2] = uw.function.evaluate(v_soln.fn, meshbox.data)

        pl = pv.Plotter(off_screen=True)


        pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", 
                    show_edges=True, scalars="rho",
                     opacity=1.0)
        
        pl.add_points(point_cloud, cmap="gray_r", 
                      render_points_as_spheres=True,
                      point_size=5, opacity=0.5
                    )
        pl.add_arrows(arrow_loc, arrow_length, mag=50, opacity=0.8)

        pl.remove_scalar_bar("M")
        pl.remove_scalar_bar("mag")
        pl.remove_scalar_bar("rho")


        pl.screenshot(filename="{}.png".format(filename), window_size=(1250,1250), 
                      return_img=False)
        

        
        # pl.show()
        pl.close()

t_step = 0

# +
# Update in time

expt_name="output/blobs"

for step in range(0,150):
    
    stokes.solve(zero_init_guess=False)
    delta_t = min(10.0,2.5*stokes.estimate_dt())
        
    # update swarm / swarm variables
    
    if uw.mpi.rank==0:
        print("Timestep {}, dt {}".format(t_step, delta_t))
 
    # advect swarm
    swarm.advection(v_soln.fn, delta_t)
    
    visc = uw.function.evaluate(viscosity, material._meshLevelSetVars[0].coords)
    mat  = uw.function.evaluate(material.f[0]+material.f[1]+material.f[2]+material.f[3],
                                material._meshLevelSetVars[0].coords)
    with meshbox.access():
        mask = material._meshLevelSetVars[0].data + \
               material._meshLevelSetVars[1].data + \
               material._meshLevelSetVars[2].data + \
               material._meshLevelSetVars[3].data   

    print("Viscosity min/max: {} / {} ".format(visc.min(), visc.max()))
    print("Mat field min/max: {} / {} ".format(mat.min(), mat.max()))
    print("Mask field min/max: {} / {} ".format(mask.min(), mask.max()))


    if (t_step%1==0):
        plot_mesh(filename="{}_step_{}".format(expt_name,t_step))
        
    t_step += 1



# +
# savefile = "output/bubbles.h5".format(step) 
# meshbox.save(savefile)
# v_soln.save(savefile)
# meshbox.generate_xdmf(savefile)

