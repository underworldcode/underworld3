# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Cylindrical Stokes
# (In cylindrical coordinates)

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3 import timing
import numpy as np
import sympy

from IPython.display import display
import os

os.environ["UW_TIMING_ENABLE"] = "1"

# +
# Define the problem size
#      1 - ultra low res for automatic checking
#      2 - low res problem to play with this notebook
#      3 - medium resolution (be prepared to wait)
#      4 - highest resolution (benchmark case from Spiegelman et al)

problem_size = 2

# For testing and automatic generation of notebook output,
# over-ride the problem size if the UW_TESTING_LEVEL is set

uw_testing_level = os.environ.get("UW_TESTING_LEVEL")
if uw_testing_level:
    try:
        problem_size = int(uw_testing_level)
    except ValueError:
        # Accept the default value
        pass

r_o = 1.0
r_i = 0.5
free_slip_upper = True

if problem_size <= 1:
    res = 0.1
elif problem_size == 2:
    res = 0.075
elif problem_size == 3:
    res = 0.05
elif problem_size >= 4:
    res = 0.01

# -


meshball_xyz_tmp = uw.meshing.Annulus(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=res,
    refinement=0,
    filename="tmp_meshball.msh",
)


# +
## We don't have the native coordinates built in to this mesh

xy_vec = meshball_xyz_tmp.dm.getCoordinates()
xy = xy_vec.array.reshape(-1, 2)

dmplex = meshball_xyz_tmp.dm.clone()

rtheta = np.empty_like(xy)
rtheta[:, 0] = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
rtheta[:, 1] = np.arctan2(xy[:, 1] + 1.0e-16, xy[:, 0] + 1.0e-16)
rtheta_vec = xy_vec.copy()
rtheta_vec.array[...] = rtheta.reshape(-1)[...]
dmplex.setCoordinates(rtheta_vec)
# del meshball_xyz_tmp

from enum import Enum
class boundaries(Enum):
    Lower = 1
    Upper = 2
    Centre = 10

meshball = uw.meshing.Mesh(
    dmplex,
    boundaries = boundaries,
    coordinate_system_type=uw.coordinates.CoordinateSystemType.CYLINDRICAL2D_NATIVE,
    
    qdegree=3,
)
uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
    meshball.dm, [0.0, 1.0], [0.0, 0.0], [0.0, 2 * np.pi]
)
meshball.dm.view()

meshball_xyz = uw.meshing.Annulus(
    radiusOuter=r_o, radiusInner=r_i, cellSize=res, qdegree=3
)

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

x, y = meshball.CoordinateSystem.X
r, t = meshball.CoordinateSystem.R

v_soln = uw.discretisation.MeshVariable("U", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("Pc", meshball, 1, degree=2)

v_soln_xy = uw.discretisation.MeshVariable("Uxy", meshball_xyz, 2, degree=2)
p_soln_xy = uw.discretisation.MeshVariable(
    "Pxy", meshball_xyz, 1, degree=1, continuous=True
)
r_xy = uw.discretisation.MeshVariable("Rxy", meshball_xyz, 1, degree=1, continuous=True)

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

# Some useful coordinate stuff

r, th = meshball.CoordinateSystem.R
x, y = meshball.CoordinateSystem.X

unit_rvec = meshball.CoordinateSystem.unit_e_0
gravity_fn = r / r_o

#
Rayleigh = 1.0e5

# +
# Create Stokes object (r, theta)

stokes = uw.systems.Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

stokes.tolerance = 1.0e-6
stokes.petsc_options["snes_monitor"] = None

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

# Velocity boundary conditions

if not free_slip_upper:
    stokes.add_dirichlet_bc(0.0, "Upper", 0)
    stokes.add_dirichlet_bc(0.0, "Upper", 1)

else:
    stokes.add_dirichlet_bc(0.0, "Upper", 0)

stokes.add_dirichlet_bc(0.0, "Lower", 0)
# stokes.add_dirichlet_bc(0.0, "Lower", 1)
# -


stokes.view()

# #### Strain rate in Cylindrical (2D) geometry is this:
#
# $$ \dot\epsilon_{rr} = \frac{\partial u_r}{\partial r}$$
#
# $$ \dot\epsilon_{\theta\theta} = \frac{1}{r} \frac{\partial u_\theta}{\partial \theta} + \frac{u_r}{r} $$
#
# $$ 2\dot\epsilon_{r\theta} = \frac{1}{r} \frac{\partial u_r}{\partial \theta} + \frac{\partial u_\theta}{\partial r} - \frac{u_\theta}{r} $$

meshball.vector.strain_tensor(stokes.Unknowns.u.sym)

# +
# Create Stokes object (x,y)

radius_fn = meshball_xyz.CoordinateSystem.xR[0]
radius_fn = r_xy.sym[0]

unit_rvec_xy = meshball_xyz.CoordinateSystem.unit_e_0

stokes_xy = uw.systems.Stokes(
    meshball_xyz,
    velocityField=v_soln_xy,
    pressureField=p_soln_xy,
    solver_name="stokes_xy",
)

stokes_xy.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_xy.constitutive_model.Parameters.shar_viscosity_0 = 1
stokes_xy.petsc_options["snes_rtol"] = 1.0e-6
stokes_xy.petsc_options["snes_monitor"] = None

# Velocity boundary conditions

if not free_slip_upper:
    stokes_xy.add_dirichlet_bc(0.0, "Upper", 0)
    stokes_xy.add_dirichlet_bc(0.0, "Upper", 1)
else:
    print("Free slip !")
    penalty = 100000
    stokes_xy.add_natural_bc(
            penalty * unit_rvec_xy.dot(v_soln_xy.sym) * unit_rvec_xy, "Upper"
        )
    stokes_xy.add_natural_bc(
            penalty * unit_rvec_xy.dot(v_soln_xy.sym) * unit_rvec_xy, "Lower"
        )

# stokes_xy.add_dirichlet_bc([0.0, 0.0], "Lower")
# -

unit_rvec_xy

penalty * unit_rvec_xy.dot(v_soln_xy.sym) * unit_rvec_xy

# +
# t_init = 10.0 * sympy.exp(-5.0 * (x**2 + (y - 0.5) ** 2))
t_init = sympy.cos(4 * th)
stokes.bodyforce = sympy.Matrix([Rayleigh * t_init, 0])

# ----
t_init_xy = sympy.cos(4 * meshball_xyz.CoordinateSystem.xR[1])
unit_rvec = meshball_xyz.CoordinateSystem.unit_e_0
stokes_xy.bodyforce = Rayleigh * t_init_xy * unit_rvec

# -

stokes_xy.bodyforce



with meshball_xyz.access(r_xy):
    r_xy.data[:, 0] = uw.function.evaluate(
        meshball_xyz.CoordinateSystem.xR[0],
        coords=r_xy.coords,
        coord_sys=meshball_xyz.N,
    )

timing.start()
stokes.solve(zero_init_guess=True)
timing.print_table()

timing.start()

stokes_xy.solve(zero_init_guess=True)

timing.print_table()

U_xy = meshball.CoordinateSystem.xRotN * v_soln.sym.T


# +
# Visuals

if uw.mpi.size == 1:
    import underworld3.visualisation as vis  # use this module for plotting
    import pyvista as pv
    import vtk

pl = pv.Plotter(window_size=(1000, 1000))

pvmesh = uw.visualisation.mesh_to_pv_mesh(meshball_xyz)
pvmesh.point_data["T"] = uw.visualisation.scalar_fn_to_pv_points(pvmesh, t_init_xy)

velocity_points = uw.visualisation.meshVariable_to_pv_cloud(v_soln_xy)
velocity_points_rt = uw.visualisation.meshVariable_to_pv_cloud(v_soln)

velocity_points.point_data["Vxy"] = uw.visualisation.vector_fn_to_pv_points(
    velocity_points, v_soln_xy.sym
)
velocity_points.point_data["Vrt"] = uw.visualisation.vector_fn_to_pv_points(
    velocity_points_rt, U_xy.T
)

pl.add_mesh(
    pvmesh,
    cmap="coolwarm",
    edge_color="Black",
    show_edges=True,
    scalars="T",
    use_transparency=False,
    opacity=1.0,
)

pl.add_arrows(
    velocity_points.points,
    velocity_points.point_data["Vxy"],
    mag=1.0e-4,
    opacity=0.75,
    color="Black",
)
pl.add_arrows(
    velocity_points.points,
    velocity_points.point_data["Vrt"],
    mag=1.0e-4,
    opacity=0.75,
    color="Green",
)

pl.camera.SetPosition(0.75, 0.2, 1.5)
pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
pl.camera.SetClippingRange(1.0, 8.0)

pl.show()


