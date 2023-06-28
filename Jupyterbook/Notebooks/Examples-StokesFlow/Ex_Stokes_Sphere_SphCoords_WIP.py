# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Spherical Stokes
# (In a spherical coordinate system)

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
meshball = uw.meshing.SegmentedSphere(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=0.25,
    numSegments=7, 
    qdegree=3, 
    coordinatesNative=True,
)

uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
    meshball.dm, [0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 2 * np.pi, 0.0]
)

meshball_xyz = uw.meshing.SegmentedSphere(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=0.25,
    numSegments=6, 
    qdegree=3, 
    filename="tmpWedgeX.msh",
    coordinatesNative=False,
)

# -

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]
 
    pvmesh = pv.read("tmpWedgeX.msh")
   
    pl = pv.Plotter()
    
    clipped = pvmesh.clip(normal='x', crinkle=True)
  
    # pl.add_mesh(
    #     pvmesh,
    #     show_edges=True,
    #     opacity=0.1,  
    #     # clim=[0,1]
    # )

    pl.add_mesh(
        pvmesh,
        show_edges=True,
        opacity=1.0,  
        # clim=[0,1]
    )
    
    pl.add_axes(labels_off=False)


    pl.show(cpos="xy")

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

x, y, z = meshball.CoordinateSystem.X
r, t, p  = meshball.CoordinateSystem.R

v_soln = uw.discretisation.MeshVariable("U", meshball, 3, degree=2)
vector = uw.discretisation.MeshVariable("V", meshball, 3, degree=1)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1, continuous=True)
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
    v_soln
)
stokes.constitutive_model.Parameters.viscosity = 1
stokes.penalty = 0.0

# Velocity boundary conditions

if not free_slip_upper:
    stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Upper", (0, 1, 2))
else:
    stokes.add_dirichlet_bc((0.0), "Upper", (0,))

stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Lower", (0, 1, 2))
# -


# ### Strain rate in Spherical geometry 
#
# Note: the standard formulation for this is usually in $r,\theta, \phi$ coordinates and there are different conventions for which angle is listed first but we stick with the standard one (radius, colatitude, longitude) which is the right handed coordinate system with colatitude increasing from the N pole.

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
    v_soln_xyz
)
stokes_xyz.constitutive_model.Parameters.viscosity = 1
stokes_xyz.petsc_options["snes_rtol"] = 1.0e-5

# Velocity boundary conditions

if not free_slip_upper:
    stokes_xyz.add_dirichlet_bc((0.0, 0.0, 0.0), "Upper", (0, 1, 2))


stokes_xyz.add_dirichlet_bc((0.0, 0.0, 0.0), "Lower", (0, 1, 2))
# -


pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-3

# +
# t_init = 10.0 * sympy.exp(-5.0 * (x**2 + (y - 0.5) ** 2))
t_init = sympy.cos(4 * p) * sympy.sin(t)
stokes.bodyforce = sympy.Matrix([Rayleigh*t_init, 0, 0])
stokes.add_dirichlet_bc(0.0, ["PoleAxisN", "PolePtNo", "PolePtNi"], 2)
stokes.add_dirichlet_bc(0.0, ["PoleAxisS", "PolePtSo", "PolePtSi"], 2)

# ----

t_init_xyz = sympy.cos(4 * meshball_xyz.CoordinateSystem.xR[1])
unit_rvec = meshball_xyz.CoordinateSystem.unit_e_0
stokes_xyz.bodyforce = Rayleigh * t_init_xyz * unit_rvec
stokes_xyz.bodyforce -= 1.0e6 * v_soln_xyz.sym.dot(unit_rvec) * surface_fn * unit_rvec
# -

stokes_xyz._setup_terms()


# + tags=[]
stokes._setup_terms()
stokes.solve(zero_init_guess=True)
# -

stokes._u_f0

pressure_solver.solve()

# +
## Projection operator - see if this works

projector = uw.systems.Vector_Projection(meshball, vector)
projector.uw_function = stokes._u_f0
projector.smoothing = 1.0e-6
projector.add_dirichlet_bc(0.0, ["PoleAxisN", "PolePtNo", "PolePtNi"], 2)
projector.add_dirichlet_bc(0.0, ["PoleAxisS", "PolePtSo", "PolePtSi"], 2)

options = projector.petsc_options
options.setValue("snes_rtol",1.0e-4)

projector.solve()

# -

0/0

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

    pvmesh = pv.read("./tmpWedgeX.msh")

    with meshball.access():
        pvmesh.point_data["V"] = uw.function.evaluate(
            v_soln.sym.dot(v_soln.sym), meshball.data
        )
        pvmesh.point_data["P"] = uw.function.evaluate(p_cont.sym[0], meshball.data)
        pvmesh.point_data["T"] = uw.function.evaluate(
            t_init_xyz, meshball_xyz.data, coord_sys=meshball_xyz.N
        )

    usol = np.empty_like(v_soln.coords)
    usol[:, 0] = uw.function.evaluate(U_xyz[0], v_soln.coords)
    usol[:, 1] = uw.function.evaluate(U_xyz[1], v_soln.coords)
    usol[:, 2] = uw.function.evaluate(U_xyz[2], v_soln.coords)

    usol_xyz = np.empty_like(v_soln_xyz.coords)
    usol_xyz[:, 0] = uw.function.evaluate(v_soln_xyz.sym[0], v_soln_xyz.coords)
    usol_xyz[:, 1] = uw.function.evaluate(v_soln_xyz.sym[1], v_soln_xyz.coords)

    xyz = np.empty_like(v_soln.coords)
    xyz[:, 0] = uw.function.evaluate(
        meshball.CoordinateSystem.X[0], v_soln.coords, coord_sys=meshball.N
    )
    xyz[:, 1] = uw.function.evaluate(
        meshball.CoordinateSystem.X[1], v_soln.coords, coord_sys=meshball.N
    )
    xyz[:, 2] = uw.function.evaluate(
        meshball.CoordinateSystem.X[2], v_soln.coords, coord_sys=meshball.N
    )

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, :] = xyz[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:,:] = usol[...] * 0.0001

    arrow_length_xy = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length_xy[:, :] = usol_xyz[...]

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh,'Black', 'wireframe')
    # pl.add_mesh(
    #     pvmesh,
    #     cmap="coolwarm",
    #     edge_color="Grey",
    #     scalars="P",
    #     show_edges=True,
    #     use_transparency=False,
    #     opacity=0.75,
    # )

    # pl.add_arrows(arrow_loc, arrow_length_xy, mag=0.0001, color="Blue")
    pl.add_arrows(
        arrow_loc + (0.005, 0.005, 0.0), arrow_length, mag=0.001, color="Red"
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
