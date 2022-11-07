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


# # Spherical Stokes
# (In spherical coordinates)

# + tags=[]
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy

import os

os.environ["SYMPY_USE_CACHE"] = "no"

r_o = 1.0
r_i = 0.5

num_els = 4
res = np.pi * (r_o + r_i) / (4 * num_els)

free_slip_upper = True



# +
# meshball_xyz_tmp = uw.meshing.Annulus(
#     radiusOuter=r_o, radiusInner=r_i, cellSize=res, filename="./tmp_meshball.msh"
# )

meshball_xyz_tmp = uw.meshing.CubedSphere(
    radiusOuter=r_o, 
    radiusInner=r_i, 
    numElements=num_els, 
    filename="./tmp_meshball3.msh", 
    qdegree=3, simplex=True,
)



# +
xyz_vec = meshball_xyz_tmp.dm.getCoordinates()
xyz = xyz_vec.array.reshape(-1, 3)
dmplex = meshball_xyz_tmp.dm.clone()

rl1l2 = np.empty_like(xyz)


## Fix this one ... 

rl1l2[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
rl1l2[:, 1] = np.arctan2(xyz[:, 1] + 1.0e-16, xyz[:, 0] + 1.0e-16)
rl1l2[:, 2] = np.arcsin( xyz[:, 2] + 1.0e-16 / rl1l2[:, 0] + 1.0e-16)

rl1l2_vec = xyz_vec.copy()
rl1l2_vec.array[...] = rl1l2.reshape(-1)[...]
dmplex.setCoordinates(rl1l2_vec)
del meshball_xyz_tmp
# -


meshball = uw.meshing.Mesh(
    dmplex,
    coordinate_system_type=uw.coordinates.CoordinateSystemType.SPHERICAL_NATIVE,
)
uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
    meshball.dm, [0.0, 6.28, 0.0], [0.0, -np.pi, 0.0], [0.0, 2*np.pi, 0.0]
)
meshball.dm.view()

meshball.dm.localizeCoordinates()
meshball.dm.view()

meshball_xyz = uw.meshing.CubedSphere(
    radiusOuter=r_o, 
    radiusInner=r_i, 
    numElements=num_els, 
    qdegree=3, simplex=True
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

x, y, z = meshball.CoordinateSystem.X
r, l1, l2  = meshball.CoordinateSystem.R

# +
# uw.function.evaluate(meshball.CoordinateSystem.R[0], meshball.data)
# -

v_soln = uw.discretisation.MeshVariable("U", meshball, 3, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1, continuous=False)
p_cont = uw.discretisation.MeshVariable("Pc", meshball, 1, degree=2)


v_soln_xyz = uw.discretisation.MeshVariable("Uxyz", meshball_xyz, 3, degree=2)
p_soln_xyz = uw.discretisation.MeshVariable(
    "Pxy", meshball_xyz, 1, degree=1, continuous=True
)

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

# Some useful coordinate stuff

unit_rvec = meshball.CoordinateSystem.unit_e_0
gravity_fn = r / r_o

#
Rayleigh = 1.0e5

# +
# Create Stokes object (r, theta)

stokes = uw.systems.Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)
stokes.petsc_options["snes_rtol"] = 1.0e-5

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(
    meshball.dim
)
stokes.constitutive_model.Parameters.viscosity = 1
stokes.penalty = 1.0
stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity

# Velocity boundary conditions

if not free_slip_upper:
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
else:
    stokes.add_dirichlet_bc((0.0), "Upper", (0,))

stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))
# -


# ### Strain rate in Spherical geometry 
#
# Note: the standard formulation for this is usually in $r,\theta, \phi$ coordinates and there are different conventions for which angle is listed first. In UW3, we are using latitute ($\lambda_1$) and latitude ($\lambda_2$) which is not a standard form of spherical coordinates even if it is quite a logical choice when using geographical data.
#
# $$ \dot\epsilon_{rr} \equiv \dot\epsilon_{00} = U_{ 0,0}(\mathbf{r}) $$
#
# $$ \dot\epsilon_{r\lambda_1} \equiv \dot\epsilon_{01} = \frac{U_{ 1,0}(\mathbf{r})}{2} + \frac{U_{ 0,1}(\mathbf{r})}{2 r \cos{\left(\lambda_2 \right)}} - \frac{U_{ 1 }(\mathbf{r})}{2 r} $$
#
# $$ \dot\epsilon_{r\lambda_2} \equiv \dot\epsilon_{02} = \frac{U_{ 2,0}(\mathbf{r})}{2} + \frac{U_{ 0,2}(\mathbf{r})}{2 r} - \frac{U_{ 2 }(\mathbf{r})}{2 r} $$
#

stokes.strainrate

stokes.stress

# +
# Create Stokes object (x,y)

radius_fn = meshball_xyz.CoordinateSystem.xR[0]

hw = 1000.0 / res
surface_fn = sympy.exp(-((radius_fn - r_o) ** 2) * hw)
base_fn = sympy.exp(-((radius_fn - r_i) ** 2) * hw)

stokes_xyz = uw.systems.Stokes(
    meshball_xyz,
    velocityField=v_soln_xyz,
    pressureField=p_soln_xyz,
    solver_name="stokes_xyz",
)
stokes_xyz.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(
    meshball_xyz.dim
)
stokes_xyz.constitutive_model.Parameters.viscosity = 1
stokes_xyz.saddle_preconditioner = 1 / stokes_xyz.constitutive_model.Parameters.viscosity
stokes_xyz.petsc_options["snes_rtol"] = 1.0e-5

# Velocity boundary conditions

if not free_slip_upper:
    stokes_xyz.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))

stokes_xyz.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))
# -


pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-3

# +
# t_init = 10.0 * sympy.exp(-5.0 * (x**2 + (y - 0.5) ** 2))
t_init = sympy.cos(4 * l1)
stokes.bodyforce = sympy.Matrix([Rayleigh * t_init, 0, 0])

# ----

t_init_xyz = sympy.cos(4 * meshball_xyz.CoordinateSystem.xR[1])
unit_rvec = meshball_xyz.CoordinateSystem.unit_e_0
stokes_xyz.bodyforce = Rayleigh * t_init_xyz * unit_rvec
stokes_xyz.bodyforce -= 1.0e6 * v_soln_xyz.sym.dot(unit_rvec) * surface_fn * unit_rvec
# -

meshball.CoordinateSystem.xRotN

stokes_xyz._setup_terms()
stokes_xyz.solve(zero_init_guess=True)

0/0

stokes._setup_terms()
stokes.solve(zero_init_guess=False)
pressure_solver.solve()

U_xyz = meshball.CoordinateSystem.xRotN * v_soln.sym.T

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
    pl.add_arrows(
        arrow_loc + (0.005, 0.005, 0.0), arrow_length, mag=0.00005, color="Red"
    )

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
