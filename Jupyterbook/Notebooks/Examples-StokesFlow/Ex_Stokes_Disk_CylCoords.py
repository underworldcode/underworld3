# # Cylindrical Stokes

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy

import os

os.environ["SYMPY_USE_CACHE"] = "no"

res = 0.1
r_o = 2.0
r_i = 1.0
free_slip_upper = False

# -


meshball_xyz = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=res)


xy_vec = meshball_xyz.dm.getCoordinates()
xy = xy_vec.array.reshape(-1, 2)
dmplex = meshball_xyz.dm.clone()
rtheta = np.empty_like(xy)
rtheta[:, 0] = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
rtheta[:, 1] = np.arctan2(xy[:, 1] + 1.0e-16, xy[:, 0] + 1.0e-16) 
rtheta_vec = xy_vec.copy()
rtheta_vec.array[...] = rtheta.reshape(-1)[...]
dmplex.setCoordinates(rtheta_vec)

meshball = uw.meshing.Mesh(dmplex, coordinate_system_type=uw.coordinates.CoordinateSystemType.CYLINDRICAL2D_NATIVE)
uw.cython.petsc_discretisation.petsc_dm_set_periodicity(meshball.dm, 
                                                        [0.0,1.0], [0.0,0.0], [0.0,2.0*np.pi])
meshball.dm.view()



display(meshball_xyz.CoordinateSystem.type)
display(meshball_xyz.CoordinateSystem.N)
display(meshball_xyz.CoordinateSystem.R)
display(meshball_xyz.CoordinateSystem.r)
display(meshball_xyz.CoordinateSystem.X)
display(meshball_xyz.CoordinateSystem.x)

display(meshball.CoordinateSystem.type)
display(meshball.CoordinateSystem.N)
display(meshball.CoordinateSystem.R)
display(meshball.CoordinateSystem.r)
display(meshball.CoordinateSystem.X)
display(meshball.CoordinateSystem.x)

x,y = meshball.CoordinateSystem.X
r,t = meshball.CoordinateSystem.R

# +
# uw.function.evaluate(meshball.CoordinateSystem.R[0], meshball.data)
# -

v_soln = uw.discretisation.MeshVariable("U", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1, continuous=False)
p_cont = uw.discretisation.MeshVariable("Pc", meshball, 1, degree=2)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    meshball_xyz.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")
    pvmesh.points[:, 0:2] = xy[:, 0:2]

    pvmesh.plot(show_edges=True, cpos="xy")
# -


# #### Strain rate in Cylindrical (2D) geometry is this:
#
# $$ \dot\epsilon_{rr} = \frac{\partial u_r}{\partial r}$$
#
# $$ \dot\epsilon_{\theta\theta} = \frac{1}{r} \frac{\partial u_\theta}{\partial \theta} + \frac{u_r}{r} $$
#
# $$ 2\dot\epsilon_{r\theta} = \frac{1}{r} \frac{\partial u_r}{\partial \theta} + \frac{\partial u_\theta}{\partial r} - \frac{u_\theta}{r} $$

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

# Some useful coordinate stuff

r, th = meshball.CoordinateSystem.R
x, y = meshball.CoordinateSystem.X

gravity_fn = r / r_o  

e = 0  # sympy.sympify(10)**sympy.sympify(-10)

#
Rayleigh = 1.0e5

# +
# Create Stokes object

stokes = uw.systems.Stokes(meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshball.dim)
stokes.constitutive_model.Parameters.viscosity=1
stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity

# Velocity boundary conditions

if not free_slip_upper:
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
else:
    stokes.add_dirichlet_bc((0.0), "Upper", (0,))
    
stokes.add_dirichlet_bc((0.0,0.0), "Lower", (0,1))
# -


pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-3

# +
t_init = 10.0 * sympy.exp(-5.0 * (x**2 + (y - 0.5) ** 2))
t_init = sympy.cos(3 * th)

with meshball.access(t_soln):
    t_soln.data[:, 0] = uw.function.evaluate(t_init, t_soln.coords)

# -
stokes.bodyforce = sympy.Matrix([Rayleigh * t_init * gravity_fn, 0])

# +
# stokes.petsc_options["snes_test_jacobian"] = None
stokes.petsc_options["snes_rtol"] = 1.0e-3

# stokes.petsc_options["ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None

# stokes.snes.view()
# -


stokes._setup_terms()

stokes.solve(zero_init_guess=True)
pressure_solver.solve()

U_xy = meshball.CoordinateSystem.xRotN * v_soln.sym.T

# +
# An alternative is to use the swarm project_from method using these points to make a swarm

# +
## Periodic in theta - the nodes which have been "moved" to a
## different coordinate sheet are plotted incorrectly and there is
## not much to be done about that ... we could define a v_soln/p_soln on
## the xyz mesh for Uxy, and use it for plotting. 

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1000, 1000]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    meshball_xyz.vtk("tmp.vtk")
    pvmesh = pv.read("tmp.vtk")

    with meshball.access():
        pvmesh.point_data["V"] = uw.function.evaluate(v_soln.sym.dot(v_soln.sym), meshball.data)
        pvmesh.point_data["P"] = uw.function.evaluate(p_cont.sym[0], meshball.data)
        pvmesh.point_data["T"] = uw.function.evaluate(t_init, meshball.data)


    usol = np.empty_like(v_soln.coords)
    usol[:, 0] = uw.function.evaluate(U_xy[0], v_soln.coords)
    usol[:, 1] = uw.function.evaluate(U_xy[1], v_soln.coords)
    
    xy = np.empty_like(v_soln.coords)
    xy[:,0] = uw.function.evaluate(meshball.CoordinateSystem.X[0], v_soln.coords)
    xy[:,1] = uw.function.evaluate(meshball.CoordinateSystem.X[1], v_soln.coords)


    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = xy[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Grey", scalars="P", show_edges=True, use_transparency=False, opacity=0.75
    )
    pl.add_arrows(arrow_loc, arrow_length, mag=0.0001)
    pl.show(cpos="xy")
# -
usol_rms = np.sqrt(usol[:,0]**2 + usol[:,1]**2).mean()
usol_rms

# +
with meshball.access():
    usolrt = stokes.u.data
    
usolrt_rms = np.sqrt(usolrt[:,0]**2 + usolrt[:,1]**2).mean()
usolrt_rms
# -

meshball.vector.gradient(p_cont.sym)

meshball.vector.divergence(v_soln.sym)

stokes.strainrate.trace()


