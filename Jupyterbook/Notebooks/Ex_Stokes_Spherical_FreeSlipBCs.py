# Stokes flow in a Spherical Domain
#
#
# ## Mathematical formulation
#
# The Navier-Stokes equation describes the time-dependent flow of a viscous fluid in response to buoyancy forces and pressure gradients:
#
# \\[
# \rho \frac{\partial \mathbf{u}}{\partial t} + \eta\nabla^2 \mathbf{u} -\nabla p = \rho \mathbf{g}
# \\]
#
# Where $\rho$ is the density, $\eta$ is dynamic viscosity and $\mathbf{g}$ is the gravitational acceleration vector. We here assume that density changes are due to temperature and are small enough to be consistent with an assumption of incompressibility (the Boussinesq approximation). We can rescale this equation of motion using units for length, time, temperature and mass that are specific to the problem and, in this way, obtain a scale-independent form:
#
# \\[
# \frac{1}{\mathrm{Pr}} \frac{\partial \mathbf{u}}{\partial t} + \nabla^2 \mathbf{u} -\nabla p = \mathrm{Ra} T' \hat{\mathbf{g}}
# \\]
#

# where we have assumed that buoyancy forces on the right hand side are due to temperature variations, and the two dimensionless numbers, $\mathrm{Ra}$ and $\mathrm{Pr}$ are measures of the importance of buoyancy forcing and intertial terms, respectively.
#
# \\[
# \mathrm{Ra} = \frac{g\rho_0 \alpha \Delta T d^3}{\kappa \eta} 
# \quad \textrm{and} \quad
# \mathrm{Pr} = \frac{\eta}{\rho \kappa}
# \\]
#
# Here $\alpha$ is the thermal expansivity, $\Delta T$ is the range of the temperature variation, $d$ is the typical length scale over which temperature varies, $\kappa$ is the thermal diffusivity ( $\kappa = k / \rho_0 C_p$; $k$ is thermal conductivity, and $C_p$ is heat capacity). 

# If we assume that the Prandtl number is large, then the inertial terms will not contribute significantly to the balance of forces in the equation of motion because we have rescaled the equations so that the velocity and pressure gradient terms are of order 1. This assumption eliminates the time dependent terms in the equations and tells us that the flow velocity and pressure field are always in equilibrium with the pattern of density variations and this also tells us that we can evaluate the flow without needing to know the history or origin of the buoyancy forces. When the viscosity is independent of velocity and dynamic pressure, the velocity and pressure scale proportionally with $\mathrm{Ra}$ but the flow pattern itself is unchanged. 
#
# The scaling that we use for the non-dimensionalisation is as follows:
#  
# \\[
#     x = d x', \quad t = \frac{d^2}{\kappa} t', \quad T=\Delta T T', \quad 
#     p = p_0 + \frac{\eta \kappa}{d^2} p'
# \\]
#
# where the stress (pressure) scaling using viscosity ($\eta$) determines how the mass scales. In the above, $d$ is the radius of the inner core, a typical length scale for the problem, $\Delta T$ is the order-of-magnitude range of the temperature variation from our observations, and $\kappa$ is thermal diffusivity. The scaled velocity is obtained as $v = \kappa / d v'$.

# ## Formulation & model
#
#
# The model consists of a spherical ball divided into an unstructured tetrahedral mesh of quadratic velocity, linear pressure elements with a free slip upper boundary and with a buoyancy force pre-defined :
#
# \\[
# T(r,\theta,\phi) =  T_\textrm{TM}(\theta, \phi) \cdot r  \sin(\pi r) 
# \\] 

#

# ## Computational script in python

# +
visuals = 0
output_dir = "output"
expt_name="Stokes_Sphere_i"

# Some gmsh issues, so we'll use a pre-built one
mesh_file = "Sample_Meshes_Gmsh/test_mesh_sphere_at_res_005_c.msh"
res = 0.2
r_o = 1.0
r_i = 0.0

Rayleigh = 1.0  # Doesn't actually matter to the solution pattern, 
                # choose 1 to make re-scaling simple

iic_radius = 0.1
iic_delta_eta = 100.0
import os

os.makedirs(output_dir, exist_ok=True)

# +
# Imports here seem to be order dependent again (pygmsh / gmsh v. petsc

import pygmsh
import meshio

import petsc4py
from petsc4py import PETSc
import mpi4py

import underworld3 as uw
import numpy as np
import sympy
# -


meshball = uw.meshing.SphericalShell( radiusInner=r_i, 
                                        radiusOuter=r_o, 
                                        cellSize=res)

v_soln   = uw.discretisation.MeshVariable('U', meshball, meshball.dim, degree=2 )
p_soln   = uw.discretisation.MeshVariable('P', meshball, 1, degree=1 )
t_soln   = uw.discretisation.MeshVariable("T", meshball, 1, degree=2 )
r        = uw.discretisation.MeshVariable('R', meshball, 1, degree=1 )


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre 
# of the sphere to (say) 1 at the surface

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10+radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff 

x = meshball.N.x
y = meshball.N.y
z = meshball.N.z

# r  = sympy.sqrt(x**2+y**2+z**2)  # cf radius_fn which is 0->1 
th = sympy.atan2(y+1.0e-10,x+1.0e-10)
ph = sympy.acos(z / (r.fn+1.0e-10))

hw = 1000.0 / res 
surface_fn = sympy.exp(-((r.fn - r_o) / r_o)**2 * hw)

hw2 = 100.0 / res 
body_fn = 1.0 - sympy.exp(-((r.fn - r_o) / r_o)**2 * hw2)

## Buoyancy (T) field

t_forcing_fn = 1.0 * ( sympy.exp(-10.0*(x**2+(y-0.8)**2+z**2)) + 
                       sympy.exp(-10.0*((x-0.8)**2+y**2+z**2)) +
                       sympy.exp(-10.0*(x**2+y**2+(z-0.8)**2))        
                     )


# +
# Rigid body rotations that are null-spaces for this set of bc's 

# We can remove these after the fact, but also useful to double check
# that we are not adding anything to excite these modes in the forcing terms. 

orientation_wrt_z = sympy.atan2(y+1.0e-10,x+1.0e-10)
v_rbm_z_x = -r.fn * sympy.sin(orientation_wrt_z) * meshball.N.i
v_rbm_z_y =  r.fn * sympy.cos(orientation_wrt_z) * meshball.N.j
v_rbm_z   =  v_rbm_z_x + v_rbm_z_y

orientation_wrt_x = sympy.atan2(z+1.0e-10,y+1.0e-10)
v_rbm_x_y = -r.fn * sympy.sin(orientation_wrt_x) * meshball.N.j
v_rbm_x_z =  r.fn * sympy.cos(orientation_wrt_x) * meshball.N.k
v_rbm_x   =  v_rbm_x_z + v_rbm_x_y 

orientation_wrt_y = sympy.atan2(z+1.0e-10,x+1.0e-10)
v_rbm_y_x = -r.fn * sympy.sin(orientation_wrt_y) * meshball.N.i
v_rbm_y_z =  r.fn * sympy.cos(orientation_wrt_y) * meshball.N.k
v_rbm_y   =  v_rbm_y_z + v_rbm_y_x 

# +
# Create NS object

stokes = uw.systems.Stokes(meshball, 
                velocityField=v_soln, 
                pressureField=p_soln, 
                u_degree=v_soln.degree, 
                p_degree=p_soln.degree, 
                verbose=False,
                solver_name="stokes")

# stokes.petsc_options.delValue("ksp_monitor") # We can flip the default behaviour at some point
stokes.petsc_options["snes_rtol"]=3.0e-3 
stokes.petsc_options["snes_max_it"]=10

stokes.viscosity = 1.0

# thermal buoyancy force
buoyancy_force = Rayleigh * gravity_fn * t_forcing_fn * body_fn

# Free slip condition by penalizing radial velocity at the surface (non-linear term)
free_slip_penalty  = 1.0e6 *  v_soln.fn.dot(unit_rvec) * unit_rvec * surface_fn

# Stagnation in the centre of the sphere (null space suppression) 
centre_stagnation_penalty = 1.0e3 * v_soln.fn * sympy.Piecewise( ((0.5 - r.fn)**0.25, r.fn < 0.5), (0.0, True) )

stokes.bodyforce  = unit_rvec * buoyancy_force 
stokes.bodyforce -= free_slip_penalty 
stokes.bodyforce -= centre_stagnation_penalty

stokes._Ppre_fn = 1.0 / (stokes.viscosity + 
                         1.0 * sympy.Piecewise( ((0.5 - r.fn)**2, r.fn < 0.5), (0.0, True) ) +
                         10.0 * surface_fn)

# Velocity boundary conditions
# stokes.add_dirichlet_bc( (0.0, 0.0, 0.0), "Upper", (0,1,2))
# stokes.add_dirichlet_bc( (0.0, 0.0, 0.0), "Centre", (0,1,2))

# +
with meshball.access(r):
    r.data[:,0]   = uw.function.evaluate(sympy.sqrt(x**2+y**2+z**2), meshball.data)  # cf radius_fn which is 0->1 
    
with meshball.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(t_forcing_fn, t_soln.coords).reshape(-1,1)
# -


stokes.solve()

# +
  
# Note: we should remove the rigid body rotation nullspace
# This should be done during the solve, but it is also reasonable to
# remove it from the force terms and solution to prevent it growing if present


I0 = uw.maths.Integral(meshball, v_rbm_y.dot(v_rbm_y))
norm = I0.evaluate()

for i in range(10):
    
    I0.fn = v_soln.fn.dot(v_rbm_x)
    x_ns = I0.evaluate() / norm
    I0.fn = v_soln.fn.dot(v_rbm_y)
    y_ns = I0.evaluate() / norm
    I0.fn = v_soln.fn.dot(v_rbm_z)
    z_ns = I0.evaluate() / norm

    with meshball.access(v_soln):
        ## Note, we have to add in something in the missing component (and it has to be spatially variable ??)
        v_soln.data[...] -= x_ns * uw.function.evaluate(v_rbm_x + 1.0e-16 * r.fn * meshball.N.i , v_soln.coords)
        v_soln.data[...] -= y_ns * uw.function.evaluate(v_rbm_y + 1.0e-16 * r.fn * meshball.N.j , v_soln.coords)
        v_soln.data[...] -= z_ns * uw.function.evaluate(v_rbm_z + 1.0e-16 * r.fn * meshball.N.k , v_soln.coords)

    null_space_err = np.sqrt(x_ns**2 + y_ns**2 + z_ns**2)
   
    if null_space_err < 1.0e-12:
        if uw.mpi.rank==0:
            print("{}: Rigid body: {}, {}, {} / {}  (x,y,z axis / total)".format(i, x_ns, y_ns, z_ns, null_space_err))
        break


# +
# savefile = "output/{}_ts_{}.h5".format(expt_name,0) 
# meshball.save(savefile)
# v_soln.save(savefile)
# p_soln.save(savefile)
# om.save(savefile)
# # vorticity.save(savefile)
# meshball.generate_xdmf(savefile)
# -




# +
# OR

# # +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:
    

    import numpy as np
    import pyvista as pv
    import vtk

    
    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    meshball.vtk("tmp_meshball.vtk")
    pvmesh = pv.read("tmp_meshball.vtk")
    
    pvmesh.point_data["T"]  = uw.function.evaluate(t_forcing_fn, meshball.data)
    pvmesh.point_data["P"]  = uw.function.evaluate(p_soln.fn, meshball.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[...] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[...] = uw.function.evaluate(stokes.u.fn, stokes.u.coords)
    
    clipped = pvmesh.clip(origin=(0.001,0.0,0.0), normal=(0, 0, 1), invert=True)

    pl = pv.Plotter(window_size=[1000,1000])
    pl.add_axes()
    

    pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                  use_transparency=False, opacity=1.0)

    
    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
    #               use_transparency=False, opacity=1.0)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=50)
    
    pl.show(cpos="xy")

# -
meshball.stats(v_soln.fn.dot(v_rbm_y)), meshball.stats(v_star.fn.dot(v_rbm_y))
meshball.stats(v_soln.fn.dot(v_rbm_z)), meshball.stats(v_star.fn.dot(v_rbm_z))
meshball.stats(v_soln.fn.dot(v_rbm_x)), meshball.stats(v_star.fn.dot(v_rbm_x))

meshball.stats(sympy.vector.cross(Omega, v_soln.fn).dot(v_rbm_z))


