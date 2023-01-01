# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Cylindrical Stokes
# (In cylindrical coordinates)

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy

import os

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"


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
    radiusOuter=r_o, radiusInner=r_i, cellSize=res, filename="./tmp_meshball.msh"
)


xy_vec = meshball_xyz_tmp.dm.getCoordinates()
xy = xy_vec.array.reshape(-1, 2)
dmplex = meshball_xyz_tmp.dm.clone()
rtheta = np.empty_like(xy)
rtheta[:, 0] = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
rtheta[:, 1] = np.arctan2(xy[:, 1] + 1.0e-16, xy[:, 0] + 1.0e-16)
rtheta_vec = xy_vec.copy()
rtheta_vec.array[...] = rtheta.reshape(-1)[...]
dmplex.setCoordinates(rtheta_vec)
del meshball_xyz_tmp


meshball = uw.meshing.Mesh(
    dmplex,
    coordinate_system_type=uw.coordinates.CoordinateSystemType.CYLINDRICAL2D_NATIVE,
)
uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
    meshball.dm, [0.0, 1.0], [0.0, 0.0], [0.0, 2 * np.pi]
)
meshball.dm.view()

meshball_xyz = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=res)

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

# +
# uw.function.evaluate(meshball.CoordinateSystem.R[0], meshball.data)
# -

v_soln = uw.discretisation.MeshVariable("U", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1, continuous=False)
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

options = stokes.petsc_options
options.setValue("snes_rtol", 1.0e-4)
options.setValue("pc_gamg_type", "agg")
options.setValue("pc_gamg_agg_nsmooths", 3)
options.setValue("pc_gamg_threshold", 0.5)

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(
    meshball.dim
)
stokes.constitutive_model.Parameters.viscosity = 1
stokes.penalty = 0.0
stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity
stokes.petsc_options["snes_rtol"] = 1.0e-4


# Velocity boundary conditions

if not free_slip_upper:
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
else:
    stokes.add_dirichlet_bc((0.0), "Upper", (0,))

stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))


# stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None

stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "gmres"
stokes.petsc_options["fieldsplit_pressure_pc_type"] = "gamg"

stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "gmres"
stokes.petsc_options["fieldsplit_velocity_pc_type"] = "gamg"


stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 3
stokes.petsc_options["fieldsplit_pressure_mg_levels_ksp_max_it"] = 3

# -


# #### Strain rate in Cylindrical (2D) geometry is this:
#
# $$ \dot\epsilon_{rr} = \frac{\partial u_r}{\partial r}$$
#
# $$ \dot\epsilon_{\theta\theta} = \frac{1}{r} \frac{\partial u_\theta}{\partial \theta} + \frac{u_r}{r} $$
#
# $$ 2\dot\epsilon_{r\theta} = \frac{1}{r} \frac{\partial u_r}{\partial \theta} + \frac{\partial u_\theta}{\partial r} - \frac{u_\theta}{r} $$

stokes.strainrate

stokes.stress

# +
# Create Stokes object (x,y)

radius_fn = meshball_xyz.CoordinateSystem.xR[0]
radius_fn = r_xy.sym[0]

hw = 1000.0 / res
surface_fn = sympy.exp(-((radius_fn - r_o) ** 2) * hw)
base_fn = sympy.exp(-((radius_fn - r_i) ** 2) * hw)

stokes_xy = uw.systems.Stokes(
    meshball_xyz,
    velocityField=v_soln_xy,
    pressureField=p_soln_xy,
    solver_name="stokes_xy",
)
stokes_xy.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(
    meshball_xyz.dim
)
stokes_xy.constitutive_model.Parameters.viscosity = 1
stokes_xy.saddle_preconditioner = 1 / stokes_xy.constitutive_model.Parameters.viscosity
stokes_xy.petsc_options["snes_rtol"] = 1.0e-8

# Velocity boundary conditions

if not free_slip_upper:
    stokes_xy.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))

stokes_xy.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))
# -


pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-3

# +
# t_init = 10.0 * sympy.exp(-5.0 * (x**2 + (y - 0.5) ** 2))
t_init = sympy.cos(4 * th)
stokes.bodyforce = sympy.Matrix([Rayleigh * t_init, 0])

# ----

t_init_xy = sympy.cos(4 * meshball_xyz.CoordinateSystem.xR[1])
unit_rvec = meshball_xyz.CoordinateSystem.unit_e_0
stokes_xy.bodyforce = Rayleigh * t_init_xy * unit_rvec
stokes_xy.bodyforce -= 1.0e6 * v_soln_xy.sym.dot(unit_rvec) * surface_fn * unit_rvec
# -

with meshball_xyz.access(r_xy):
    r_xy.data[:, 0] = uw.function.evaluate(
        r, coords=r_xy.coords, coord_sys=meshball_xyz.N
    )

# +
from underworld3 import timing

timing.start()
stokes.solve(zero_init_guess=True)
timing.print_table()

pressure_solver.solve()
# -
stokes_xy.tolerance = 1.0e-8


from underworld3 import timing

timing.start()
stokes_xy.solve(zero_init_guess=True)
timing.print_table()

U_xy = meshball.CoordinateSystem.xRotN * v_soln.sym.T

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

    pvmesh = pv.read("./tmp_meshball.msh")

    with meshball.access():
        pvmesh.point_data["V"] = uw.function.evaluate(
            v_soln.sym.dot(v_soln.sym), meshball.data
        )
        pvmesh.point_data["P"] = uw.function.evaluate(p_cont.sym[0], meshball.data)
        pvmesh.point_data["T"] = uw.function.evaluate(
            t_init_xy, meshball_xyz.data, coord_sys=meshball_xyz.N
        )

    usol = np.empty_like(v_soln.coords)
    usol[:, 0] = uw.function.evaluate(U_xy[0], v_soln.coords)
    usol[:, 1] = uw.function.evaluate(U_xy[1], v_soln.coords)

    usol_xy = np.empty_like(v_soln_xy.coords)
    usol_xy[:, 0] = uw.function.evaluate(v_soln_xy.sym[0], v_soln_xy.coords)
    usol_xy[:, 1] = uw.function.evaluate(v_soln_xy.sym[1], v_soln_xy.coords)

    xy = np.empty_like(v_soln.coords)
    xy[:, 0] = uw.function.evaluate(
        meshball.CoordinateSystem.X[0], v_soln.coords, coord_sys=meshball.N
    )
    xy[:, 1] = uw.function.evaluate(
        meshball.CoordinateSystem.X[1], v_soln.coords, coord_sys=meshball.N
    )

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = xy[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    arrow_length_xy = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length_xy[:, 0:2] = usol_xy[...]

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        scalars="P",
        show_edges=True,
        use_transparency=False,
        opacity=0.75,
    )

    pl.add_arrows(arrow_loc, arrow_length_xy, mag=0.00005, color="Blue")
    pl.add_arrows(arrow_loc + (0.0, 0.0, 0.0), arrow_length, mag=0.00005, color="Red")

    pl.show(cpos="xy")
# +
usol_rms = np.sqrt(usol[:, 0] ** 2 + usol[:, 1] ** 2).mean()
usol_xy_rms = np.sqrt(usol_xy[:, 0] ** 2 + usol_xy[:, 1] ** 2).mean()

print(f"MEAN: {usol_rms / usol_xy_rms}")

usol_rms = np.sqrt(usol[:, 0] ** 2 + usol[:, 1] ** 2).max()
usol_xy_rms = np.sqrt(usol_xy[:, 0] ** 2 + usol_xy[:, 1] ** 2).max()

print(f"MAX:  {usol_rms / usol_xy_rms}")


# +
# 0.2
# MEAN: 0.8721957519400886
# MAX:  1.0823938969017228
# 0.1
# MEAN: 0.8601596694865591
# MAX:  1.0587809789060159
# -

stokes
