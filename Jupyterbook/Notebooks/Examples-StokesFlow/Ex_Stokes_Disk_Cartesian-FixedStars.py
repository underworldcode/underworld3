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


# # Cylindrical Stokes (Cartesian formulation)
#
# Let the embedded mesh deform to mimic a free surface. If we iterate on this, then it is almost exactly the same as the free-slip boundary condition (though there are potentially instabilities here).
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

res = 0.05
r_s = 1.25
r_o = 1.0
r_i = 0.5

free_slip_upper = True

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None

import os

os.environ["SYMPY_USE_CACHE"] = "no"
# -

meshball = uw.meshing.AnnulusFixedStars(
    radiusFixedStars=r_s,
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=res,
    cellSize_FS=res * 3,
    filename="tmp_fixedstarsMesh.msh",
)


v_soln = uw.discretisation.MeshVariable(r"\mathbf{u}", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable(r"p", meshball, 1, degree=1, continuous=False)
p_cont = uw.discretisation.MeshVariable(r"p_c", meshball, 1, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=3)
phi_g = uw.discretisation.MeshVariable(r"\phi", meshball, 1, degree=3)


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = meshball.CoordinateSystem.xR[0]
unit_rvec = meshball.CoordinateSystem.unit_e_0
gravity_fn = 1  # radius_fn / r_o

# Some useful coordinate stuff

x, y = meshball.CoordinateSystem.X
r, th = meshball.CoordinateSystem.xR

Rayleigh = 1.0e5

hw = 1000.0 / res
celestial_fn = sympy.exp(-((radius_fn - r_s) ** 2) * hw)
surface_fn = sympy.exp(-((radius_fn - r_o) ** 2) * hw)
base_fn = sympy.exp(-((radius_fn - r_i) ** 2) * hw)


# +
## Define some domain masks:

swarm = uw.swarm.Swarm(mesh=meshball)
material = uw.swarm.SwarmVariable(
    "M", swarm, num_components=1, proxy_continuous=False, proxy_degree=0
)
swarm.populate(fill_param=1)

with swarm.access(material):
    material.data[:, 0] = uw.function.evaluate(r < r_o, swarm.data, meshball.N)
# -

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 20.0]

    pvmesh = pv.read("tmp_fixedstarsMesh.msh")

    with swarm.access():
        points = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
        points[:, 0] = swarm.particle_coordinates.data[:, 0]
        points[:, 1] = swarm.particle_coordinates.data[:, 1]

    pvmesh.point_data["M"] = uw.function.evaluate(
        material.sym[0], meshball.data, meshball.N
    )
    pvmesh.point_data["C"] = uw.function.evaluate(
        celestial_fn, meshball.data, meshball.N
    )
    pvmesh.point_data["S"] = uw.function.evaluate(
        surface_fn - base_fn, meshball.data, meshball.N
    )

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter(window_size=(750, 750))
    # pl.camera_position = "xy"

    pl.add_mesh(pvmesh, "Grey", "wireframe")

    pl.add_mesh(
        pvmesh,
        cmap="Greens",
        edge_color="Grey",
        scalars="C",
        show_edges=True,
        use_transparency=False,
        clim=[0.66, 1],
        opacity=0.75,
    )

    pl.add_mesh(
        pvmesh,
        cmap="RdBu",
        edge_color="Grey",
        scalars="S",
        show_edges=True,
        use_transparency=False,
        clim=[-1, 1],
        opacity=0.5,
    )

    pl.add_points(
        point_cloud,
        cmap="Greys",
        render_points_as_spheres=True,
        clim=[-0.5, 1.0],
        point_size=5,
        opacity=0.66,
    )

    pl.screenshot(filename="Surface.png", window_size=(1000, 1000), return_img=False)
    pl.show(cpos="xy")

# +
# Create Stokes object

stokes = Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(meshball.dim)


stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# There is a null space if there are no fixed bcs, so we'll do this:

stokes.add_dirichlet_bc((0.0, 0.0), "FixedStars", (0, 1))
# stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

# -


pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-3

t_init = 10.0 * sympy.exp(-5.0 * (x**2 + (y - 0.5) ** 2)) / 3.5


# +
# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:, 0] = uw.function.evaluate(
        t_init, coords=t_soln.coords, coord_sys=meshball.N
    )
    print(t_soln.data.min(), t_soln.data.max())


# +
I = uw.maths.Integral(meshball, surface_fn)
s_norm = I.evaluate()
display(s_norm)

I.fn = base_fn
b_norm = I.evaluate()
display(b_norm)
# +

buoyancy_force = Rayleigh * gravity_fn * t_init
buoyancy_force -= 1.0e6 * v_soln.sym.dot(unit_rvec) * surface_fn / s_norm
buoyancy_force -= 1.0e6 * v_soln.sym.dot(unit_rvec) * base_fn / b_norm

stokes.bodyforce = unit_rvec * buoyancy_force * material.sym[0]

# This may help the solvers - penalty in the preconditioner
# stokes.saddle_preconditioner = 1.0

# -

stokes.solve()

# +
stokes.constitutive_model.Parameters.viscosity = material.sym[0] + 0.1
stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.viscosity

stokes.solve(zero_init_guess=False)

# +
stokes.constitutive_model.Parameters.viscosity = material.sym[0] + 0.01
stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.viscosity

stokes.solve(zero_init_guess=False)
# -

# Pressure at mesh nodes
pressure_solver.solve()

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
        pvmesh.point_data["V"] = uw.function.evaluate(
            v_soln.sym.dot(v_soln.sym), meshball.data
        )
        pvmesh.point_data["P"] = uw.function.evaluate(p_cont.sym[0], meshball.data)
        pvmesh.point_data["T"] = uw.function.evaluate(
            t_init, meshball.data, coord_sys=meshball.N
        )

    with swarm.access():
        pvmesh.cell_data["M"] = material.data[:, 0]

    with meshball.access():
        usol = stokes.u.data

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]
    # -
    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        scalars="T",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=0.0002)

    pl.add_mesh(
        pvmesh,
        cmap="Greys_r",
        edge_color="Grey",
        scalars="M",
        show_edges=True,
        use_transparency=False,
        opacity=0.3,
    )

    pl.show(cpos="xy")
# -
pvmesh.n_cells


usol_rms = np.sqrt(usol[:, 0] ** 2 + usol[:, 1] ** 2).mean()
usol_rms
