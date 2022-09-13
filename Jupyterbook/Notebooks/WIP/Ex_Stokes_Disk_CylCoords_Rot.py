# # Cylindrical Stokes
#
# Use rotation matrix for coordinate transformations

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy

# -


meshball = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, centre=True)

v_soln = uw.discretisation.MeshVariable("U", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
theta = uw.discretisation.MeshVariable(r"\theta", meshball, 1, degree=1)


v_soln.sym[0]

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

    meshball.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")
    # pvmesh.points[:,0:2] = xy[:,0:2]

    pvmesh.plot(show_edges=True, cpos="xy")


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec))  # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10 + radius_fn)
gravity_fn = radius_fn

e = 0  # sympy.sympify(10)**sympy.sympify(-10)

# Some useful coordinate stuff

x, y = meshball.X

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y, x)

with meshball.access(theta):
    theta.data[:, 0] = uw.function.evaluate(th, theta.coords)

#
Rayleigh = 1.0e2

# +
symtheta = sympy.symbols(r"\vartheta")

Rotate = sympy.Matrix([[sympy.cos(symtheta), -sympy.sin(symtheta)], [sympy.sin(symtheta), sympy.cos(symtheta)]])

meshball.Rot = Rotate
meshball.theta = theta

# +
gradU = v_soln.sym.jacobian(meshball.X)
E = gradU + gradU.T
Eprime = Rotate * E * Rotate.T

# Eprime.subs(symtheta,sympy.pi/2)

G3 = Eprime.diff(stokes._L)
dim = meshball.dim
G3mat = sympy.Matrix(sympy.permutedims(G3, (1, 3, 0, 2)).reshape(dim * dim, dim * dim))
G3mat.subs(symtheta, sympy.pi)

# +
## Test the gradient ...

rotvar = Rotate.subs(symtheta, theta.sym[0])
rotvar
# -
grad = (rotvar * v_soln.sym.T).jacobian(meshball.X)
grad
grad2 = (meshball.Rot * v_soln.sym.T).jacobian(meshball.X)


(grad2 + grad2.T).subs(symtheta, sympy.pi / 2)

# +
strainrate = grad2 + grad2.T
strainrate2 = strainrate.subs(symtheta, theta)

L = v_soln.sym.jacobian(meshball.X)

strainrate2.diff(L)

# +
## gradient in R,theta form

# Vx = v_soln.sym[0]
# Vy = v_soln.sym[1]

# ## Take R/theta velocity,

# vr =  sympy.cos(theta) * Vx + sympy.sin(theta) * Vy
# vt = -sympy.sin(theta) * Vx + sympy.cos(theta) * Vy

# Vrt = sympy.Matrix([vr, vt]).T
# +
# Ur = sympy.symbols(r'U_r')
# Ut = sympy.symbols(r'U_\theta')
# Urt = sympy.Matrix([Ur, Ut]).T

# +
# meshball.r_vec

# +
# RR = meshball.Rot*Urt.T  # For checking, this is how we convert  [vx vy].T
# display(RR)
# # Now substitute for U_theta etc and show this is just vx, vy
# sympy.simplify(RR.subs(Ut, vt).subs(Ur, vr))
# -


# +
# 1. Rotate * stress_C * Rotate.T # standard
# 2. Rotate * visc * strain_rate_C * Rotate.T
# 3. Rotate * visc_C_ijkl * grad_C( Rotate * Vrt ) * Rotate.T

# (Try that !)
# -


theta.sym[0]

# +
# Create Stokes object

stokes = uw.systems.Stokes(meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshball.dim)
stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(viscosity=1)

# Velocity boundary conditions

stokes.add_dirichlet_bc((0.0), "Upper", (0))
stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))
# +
# Write density into a variable for saving
t_init = sympy.cos(4 * th)

with meshball.access(t_soln):
    t_soln.data[:, 0] = uw.function.evaluate(t_init, t_soln.coords)
    print(t_soln.data.min(), t_soln.data.max())
# +
t = sympy.pi / 2

stokes.constitutive_model.flux(stokes.strainrate).subs(theta.sym[0], sympy.pi / 2)
# -

flux = stokes.constitutive_model.flux(stokes.strainrate)
G3 = flux.diff(stokes._L)
dim = meshball.dim
sympy.Matrix(sympy.permutedims(G3, (1, 3, 0, 2)).reshape(dim * dim, dim * dim)).subs(theta.sym[0], sympy.pi / 2)


stokes.bodyforce = sympy.Matrix([t_init * Rayleigh, 0.0])

stokes._setup_terms()


# +
stokes.petsc_options["snes_test_jacobian"] = None
stokes.petsc_options["snes_rtol"] = 1.0e-3

stokes.petsc_options["snes_type"] = "qn"

stokes.petsc_options["pc_type"] = "fieldsplit"
stokes.petsc_options["pc_fieldsplit_type"] = "schur"
stokes.petsc_options["pc_fieldsplit_schur_fact_type"] = "diag"
stokes.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"
stokes.petsc_options["pc_fieldsplit_detect_saddle_point"] = None
stokes.petsc_options["pc_fieldsplit_off_diag_use_amat"] = None  # These two seem to be needed in petsc 3.17
stokes.petsc_options["pc_use_amat"] = None  # These two seem to be needed in petsc 3.17
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fgmres"
stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-4
stokes.petsc_options["fieldsplit_velocity_pc_type"] = "gamg"
stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 3.0e-4
stokes.petsc_options["fieldsplit_pressure_pc_type"] = "gamg"

# stokes.petsc_options.delValue("pc_fieldsplit_off_diag_use_amat")
# stokes.petsc_options.delValue("pc_use_amat")

stokes.petsc_options["ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None


# stokes.snes.view()
# -


stokes.solve()

U_xy = meshball.Rot * v_soln.sym.T
# U_xy = meshball.Rot * stokes.bodyforce.T

U_xy

# +
# An alternative is to use the swarm project_from method using these points to make a swarm

# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1000, 1000]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    meshball.vtk("tmp.vtk")
    pvmesh = pv.read("tmp.vtk")

    with meshball.access():
        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshball.data)

    usol = np.empty_like(v_soln.coords)
    usol[:, 0] = uw.function.evaluate(U_xy[0], v_soln.coords)
    usol[:, 1] = uw.function.evaluate(U_xy[1], v_soln.coords)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]
    # -

    pl = pv.Plotter(window_size=[750, 750])

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)
    pl.add_arrows(arrow_loc, arrow_length, mag=0.1)
    pl.show(cpos="xy")

sympy.simplify((meshball.Rot * stokes._E * meshball.Rot.T).subs(x, 0))

sympy.simplify((meshball.Rot * stokes._E * meshball.Rot.T).subs(y, 0))

R
