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


# # Rayleigh-Taylor (Level-set based) in the sphere
#
# If there are just two materials, then an efficient way to manage the interface tracking is through a "level-set" which tracks not just the material type, but the distance to the interface. The distance is a continuous quantity that is not degraded quickly by classical advection schemes. A particle-based level set also has advantages because the smooth signed-distance quantity can be projected to the mesh more accurately than a sharp condition function.

# +
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
from underworld3 import timing

import numpy as np
import sympy

render = True


# +
lightIndex = 0
denseIndex = 1

viscosityRatio = 1.0

r_layer = 0.7
r_o = 1.0
r_i = 0.5

res = 0.25

Rayleigh = 1.0e6 / (r_o - r_i) ** 3

offset = 0.5 * res


# +
cell_size = uw.options.getReal("mesh_cell_size", default=0.1)
particle_fill = uw.options.getInt("particle_fill", default=5)
viscosity_ratio = uw.options.getReal("rt_viscosity_ratio", default=1.0)


mesh = uw.meshing.SphericalShell(
    radiusInner=r_i, radiusOuter=r_o, cellSize=res, qdegree=2
)

# -


v_soln = uw.discretisation.MeshVariable(r"U", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable(r"P", mesh, 1, degree=1)
meshr = uw.discretisation.MeshVariable(r"r", mesh, 1, degree=1)


swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.SwarmVariable(r"\cal{L}", swarm, proxy_degree=1, num_components=1)
swarm.populate(fill_param=2)


with swarm.access(material):
    r = np.sqrt(
        swarm.particle_coordinates.data[:, 0] ** 2
        + swarm.particle_coordinates.data[:, 1] ** 2
        + (swarm.particle_coordinates.data[:, 2] - offset) ** 2
    )

    material.data[:, 0] = r - r_layer

# +

# Some useful coordinate stuff

x, y, z = mesh.CoordinateSystem.X
ra, l1, l2 = mesh.CoordinateSystem.xR

hw = 1000.0 / res
surface_fn_a = sympy.exp(-(((ra - r_o) / r_o) ** 2) * hw)
surface_fn = sympy.exp(-(((meshr.sym[0] - r_o) / r_o) ** 2) * hw)

base_fn_a = sympy.exp(-(((ra - r_i) / r_o) ** 2) * hw)
base_fn = sympy.exp(-(((meshr.sym[0] - r_i) / r_o) ** 2) * hw)


# +

density = sympy.Piecewise((0.0, material.sym[0] < 0.0), (1.0, True))
display(density)

viscosity = sympy.Piecewise((1.0, material.sym[0] < 0.0), (1.0, True))
display(viscosity)

# -

with swarm.access():
    print(material.data.max(), material.data.min())

if False:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]
        points[:, 2] = swarm.data[:, 2]

    point_cloud = pv.PolyData(points)

    with mesh.access():
        pvmesh.point_data["M"] = uw.function.evaluate(material.sym[0], mesh.data)
        pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)
        pvmesh.point_data["visc"] = uw.function.evaluate(
            sympy.log(viscosity), mesh.data
        )

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, "Black", "wireframe")

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        scalars="M",
        render_points_as_spheres=True,
        point_size=2,
        opacity=0.5,
    )

    # pl.add_mesh(
    #     pvmesh,
    #     cmap="coolwarm",
    #     edge_color="Black",
    #     show_edges=True,
    #     scalars="M1",
    #     use_transparency=False,
    #     opacity=0.25,
    # )

    pl.show(cpos="xy")


# +
stokes = uw.systems.Stokes(
    mesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
    solver_name="stokes",
)

# stokes.petsc_options.delValue("ksp_monitor") # We can flip the default behaviour at some point
stokes.petsc_options["snes_rtol"] = 1.0e-4
stokes.petsc_options["snes_rtol"] = 1.0e-3
stokes.petsc_options["ksp_monitor"] = None

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(mesh.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity

# buoyancy (magnitude)
buoyancy = Rayleigh * density  # * (1 - surface_fn) * (1 - base_fn)

unit_vec_r = mesh.CoordinateSystem.X / mesh.CoordinateSystem.xR[0]

# Free slip condition by penalizing radial velocity at the surface (non-linear term)
free_slip_penalty_upper = v_soln.sym.dot(unit_vec_r) * unit_vec_r * surface_fn
free_slip_penalty_lower = v_soln.sym.dot(unit_vec_r) * unit_vec_r * base_fn

stokes.bodyforce = -unit_vec_r * buoyancy
stokes.bodyforce -= 1000000 * (free_slip_penalty_upper + free_slip_penalty_lower)

stokes.saddle_preconditioner = 1 / viscosity

# -

mesh.CoordinateSystem.unit_e_0.shape
(mesh.CoordinateSystem.X / mesh.CoordinateSystem.xR[0]).shape

with mesh.access(meshr):
    meshr.data[:, 0] = uw.function.evaluate(
        sympy.sqrt(x**2 + y**2 + z**2), mesh.data, mesh.N
    )  # cf radius_fn which is 0->1


# +
timing.reset()
timing.start()

stokes.solve(zero_init_guess=True)

timing.print_table()

# +
# check the solution

if uw.mpi.size == 1 and render:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [1.0, 1.0, 1.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

    # pv.start_xvfb()

    mesh.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")

    pvmesh.point_data["M"] = uw.function.evaluate(material.sym[0], mesh.data)
    pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)
    pvmesh.point_data["visc"] = uw.function.evaluate(sympy.log(viscosity), mesh.data)

    velocity = np.zeros((mesh.data.shape[0], 3))
    velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], mesh.data)
    velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], mesh.data)
    velocity[:, 2] = uw.function.evaluate(v_soln.sym[2], mesh.data)

    pvmesh.point_data["V"] = 10.0 * velocity / velocity.max()

    # point sources at cell centres

    subsample = 2

    cpoints = np.zeros((mesh._centroids[::subsample, 0].shape[0], 3))
    cpoints[:, 0] = mesh._centroids[::subsample, 0]
    cpoints[:, 1] = mesh._centroids[::subsample, 1]
    cpoints[:, 2] = mesh._centroids[::subsample, 2]

    cpoint_cloud = pv.PolyData(cpoints)

    pvstream = pvmesh.streamlines_from_source(
        cpoint_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="both",
        compute_vorticity=False,
        surface_streamlines=False,
    )

    with swarm.access():
        spoints = np.zeros((swarm.data.shape[0], 3))
        spoints[:, 0] = swarm.data[:, 0]
        spoints[:, 1] = swarm.data[:, 1]
        spoints[:, 2] = swarm.data[:, 2]

    spoint_cloud = pv.PolyData(spoints)

    with swarm.access():
        spoint_cloud.point_data["M"] = material.data[...]

    contours = pvmesh.contour(isosurfaces=[0.0], scalars="M")

    pl = pv.Plotter(window_size=(1000, 1000))

    pl.add_mesh(pvmesh, "Gray", "wireframe")
    # pl.add_arrows(arrow_loc, velocity_field, mag=0.2/vmag, opacity=0.5)

    pl.add_mesh(pvstream, opacity=1.0)
    # pl.add_mesh(pvmesh, cmap="Blues_r", edge_color="Gray", show_edges=True, scalars="rho", opacity=0.25)

    pl.add_mesh(contours, opacity=0.75, color="Yellow")

    # pl.add_points(spoint_cloud, cmap="Reds_r", scalars="M", render_points_as_spheres=True, point_size=2, opacity=0.3)
    # pl.add_points(pdata)

    pl.show(cpos="xz")


# +
pv.global_theme.background = "white"
pv.global_theme.window_size = [750, 750]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = "panel"
pv.global_theme.smooth_shading = False
pv.global_theme.camera["viewup"] = [1.0, 1.0, 1.0]
pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

pl = pv.Plotter()


def plot_mesh(filename):
    if uw.mpi.size != 1:
        return

    import numpy as np
    import pyvista as pv
    import vtk

    mesh.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")

    pvmesh.point_data["Mat"] = uw.function.evaluate(material.sym[0], mesh.data)
    pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)
    pvmesh.point_data["visc"] = uw.function.evaluate(sympy.log(viscosity), mesh.data)

    velocity = np.zeros((mesh.data.shape[0], 3))
    velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], mesh.data)
    velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], mesh.data)

    pvmesh.point_data["V"] = 10.0 * velocity / velocity.max()
    print(f"Vscale {velocity.max()}")

    # point sources at cell centres

    cpoints = np.zeros((mesh._centroids[::2].shape[0], 3))
    cpoints[:, 0] = mesh._centroids[::2, 0]
    cpoints[:, 1] = mesh._centroids[::2, 1]
    cpoint_cloud = pv.PolyData(cpoints)

    pvstream = pvmesh.streamlines_from_source(
        cpoint_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="both",
        compute_vorticity=False,
        surface_streamlines=False,
    )

    with swarm.access():
        spoints = np.zeros((swarm.data.shape[0], 3))
        spoints[:, 0] = swarm.data[:, 0]
        spoints[:, 1] = swarm.data[:, 1]
        spoints[:, 2] = 0.0

    spoint_cloud = pv.PolyData(spoints)

    with swarm.access():
        spoint_cloud.point_data["M"] = material.data[...]

    contours = pvmesh.contour(isosurfaces=[0.0], scalars="Mat")

    ## Plotting into existing pl (memory leak in pyvista)
    pl.clear()

    pl.add_mesh(pvmesh, "Gray", "wireframe")
    # pl.add_arrows(arrow_loc, velocity_field, mag=0.2/vmag, opacity=0.5)

    pl.add_mesh(pvstream, opacity=0.33)
    # pl.add_mesh(pvmesh, cmap="Blues_r", edge_color="Gray", show_edges=True, scalars="rho", opacity=0.25)

    # pl.add_points(
    #     spoint_cloud, cmap="Reds_r", scalars="M", render_points_as_spheres=True, point_size=2, opacity=0.3
    # )

    pl.add_mesh(contours, opacity=0.75, color="Yellow")

    # pl.remove_scalar_bar("Mat")
    pl.remove_scalar_bar("V")
    # pl.remove_scalar_bar("rho")

    pl.camera_position = "xz"
    pl.screenshot(
        filename="{}.png".format(filename),
        window_size=(1000, 1000),
        return_img=False,
    )

    return


# -

t_step = 0

# +
# Update in time

expt_name = "output/swarm_rt_sph"

for step in range(0, 200):
    stokes.solve(zero_init_guess=False)
    delta_t = 2.0 * stokes.estimate_dt()

    # update swarm / swarm variables

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(t_step, delta_t))

    # advect swarm
    swarm.advection(v_soln.sym, delta_t)

    if t_step < 10 or t_step % 5 == 0:
        plot_mesh(filename="{}_step_{}".format(expt_name, t_step))

        savefile = "output/swarm_rt_{}.h5".format(t_step)
        mesh.save(savefile)
        v_soln.save(savefile)
        mesh.generate_xdmf(savefile)

    t_step += 1

# -


savefile = "output/swarm_rt.h5".format(step)
mesh.save(savefile)
v_soln.save(savefile)
mesh.generate_xdmf(savefile)

material
