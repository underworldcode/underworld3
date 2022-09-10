# # Cylindrical Stokes
#
# Let the mesh deform to create a free surface. If we iterate on this, then it is almost exactly the same as the free-slip boundary condition (though there are potentially instabilities here).
#
# The problem has a constant velocity nullspace in x,y. We eliminate this by fixing the central node in this example, but it does introduce a perturbation to the flow near the centre which is not always stagnant.

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

res = 0.2

free_slip_upper = True

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None

import os

os.environ["SYMPY_USE_CACHE"] = "no"
# -

meshball = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)


# +
# Test that the second one is skipped
v_soln = uw.discretisation._MeshVariable(r"u", meshball, 2, degree=2)

v_soln = uw.discretisation.MeshVariable(r"u", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable(r"p", meshball, 1, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=3)
maskr = uw.discretisation.MeshVariable("r", meshball, 1, degree=1)


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

# radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
# unit_rvec = meshball.rvec / (1.0e-10+radius_fn)

radius_fn = meshball.CoordinateSystem.xR[0]
unit_rvec = meshball.CoordinateSystem.unit_e_0
gravity_fn = radius_fn

# Some useful coordinate stuff

x, y = meshball.CoordinateSystem.X
r, th = meshball.CoordinateSystem.xR

Rayleigh = 1.0e5

hw = 1000.0
surface_fn = sympy.exp(-((maskr.sym[0] - 1.0) ** 2) * hw)


# +
vtheta = 1

vx = -vtheta * sympy.sin(th)
vy = vtheta * sympy.cos(th)

# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshball.dim)
stokes.constitutive_model.material_properties=stokes.constitutive_model.Parameters(viscosity=1)

# There is a null space if there are no fixed bcs, so we'll do this:

if not free_slip_upper:
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))

stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

# -


t_init = 0.001 * sympy.exp(-5.0 * (x**2 + (y - 0.5) ** 2))
t_init = sympy.cos(3 * th)

# +
# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:, 0] = uw.function.evaluate(t_init, t_soln.coords)
    print(t_soln.data.min(), t_soln.data.max())

with meshball.access(maskr):
    maskr.data[:, 0] = uw.function.evaluate(r, maskr.coords)

t_mean = t_soln.mean()
print(t_soln.min(), t_soln.max())
# -
I = uw.maths.Integral(meshball, surface_fn)
s_norm = I.evaluate()
s_norm

# +

buoyancy_force = Rayleigh * gravity_fn * t_init
buoyancy_force -= 100000 * v_soln.sym.dot(unit_rvec) * surface_fn / s_norm

stokes.bodyforce = unit_rvec * buoyancy_force

# This may help the solvers - penalty in the preconditioner
stokes._Ppre_fn = 1.0

# -

stokes.solve()

# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    meshball.vtk("tmp_ball.vtk")
    pvmesh = pv.read("tmp_ball.vtk")

    with meshball.access():
        pvmesh.point_data["T"] = uw.function.evaluate(maskr.sym[0] * v_soln.sym.dot(unit_rvec), meshball.data)

    with meshball.access():
        usol = stokes.u.data

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]
    # -
    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)
    pl.add_arrows(arrow_loc, arrow_length, mag=0.0003)
    pl.show(cpos="xy")
