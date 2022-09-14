# # Multiple materials
#
# We introduce the notion of an `IndexSwarmVariable` which automatically generates masks for a swarm
# variable that consists of discrete level values (integers).
#
# For a variable $M$, the mask variables are $\left\{ M^0, M^1, M^2 \ldots M^{N-1} \right\}$ where $N$ is the number of indices (e.g. material types) on the variable. This value *must be defined in advance*.
#
# The masks are orthogonal in the sense that $M^i * M^j = 0$ if $i \ne j$, and they are complete in the sense that $\sum_i M^i = 1$ at all points.
#
# The masks are implemented as continuous mesh variables (the user can specify the interpolation order) and so they are also differentiable (once).
#

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

render = True
# -


# +
lightIndex = 0
denseIndex = 1

viscosityRatio = 1.0

r_layer = 0.7
offset = 0.025
r_o = 1.0
r_i = 0.5

elements = 7
res = 1.0 / elements

# +
# mesh = uw.meshing.CubedSphere(
#     radiusInner=r_i,
#     radiusOuter=r_o,
#     numElements=elements,
#     simplex=True,
#     qdegree=2,
# )

# or

mesh = uw.meshing.SphericalShell(radiusInner=r_i, radiusOuter=r_o, cellSize=res, qdegree=2)

# -


v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
meshr = uw.discretisation.MeshVariable(r"r", mesh, 1, degree=1)


swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_degree=1)
swarm.populate(fill_param=1)


# +
with swarm.access(material):
    material.data[...] = 0

with swarm.access(material):
    r = np.sqrt(
        swarm.particle_coordinates.data[:, 0] ** 2
        + swarm.particle_coordinates.data[:, 1] ** 2
        + (swarm.particle_coordinates.data[:, 2] - offset) ** 2
    )

    material.data[:, 0] = np.where(r < r_layer, lightIndex, denseIndex)

# -


swarm.dm.migrate()

with swarm.access():
    print(swarm.particle_coordinates.data.shape)
    print(material.data.shape)

# +

# Some useful coordinate stuff

x, y, z = mesh.CoordinateSystem.X
ra, l1, l2 = mesh.CoordinateSystem.xR

hw = 1000.0 / res
surface_fn_a = sympy.exp(-(((ra - r_o) / r_o) ** 2) * hw)
surface_fn = sympy.exp(-(((meshr.sym[0] - r_o) / r_o) ** 2) * hw)

base_fn_a = sympy.exp(-(((ra - r_i) / r_o) ** 2) * hw)
base_fn = sympy.exp(-(((meshr.sym[0] - r_i) / r_o) ** 2) * hw)

# -

mesh.CoordinateSystem.Rot

rl1 = mesh.CoordinateSystem.R[1]
rl2 = mesh.CoordinateSystem.R[2]
l1 = mesh.CoordinateSystem.xR[1]
l2 = mesh.CoordinateSystem.xR[2]


mesh.CoordinateSystem.Rot.subs([(rl1, l1), (rl2, l2)])


mat_density = np.array([0, 1])  # lightIndex, denseIndex
density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]

mat_viscosity = np.array([viscosityRatio, 1])
viscosity = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]

if render:

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
        pvmesh.point_data["M0"] = uw.function.evaluate(material.sym[0], mesh.data)
        pvmesh.point_data["M1"] = uw.function.evaluate(material.sym[1], mesh.data)
        pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)
        pvmesh.point_data["visc"] = uw.function.evaluate(sympy.log(viscosity), mesh.data)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh, "Black", "wireframe")

    pl.add_points(point_cloud, cmap="coolwarm", scalars="M", render_points_as_spheres=True, point_size=2, opacity=0.5)

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
stokes.petsc_options["snes_rtol"] = 1.0e-3
stokes.petsc_options["ksp_rtol"] = 1.0e-3
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(viscosity=viscosity)

# buoyancy (magnitude)
buoyancy = density * (1 - surface_fn) * (1 - base_fn)

unit_vec_r = mesh.CoordinateSystem.unit_e_0

# Free slip condition by penalizing radial velocity at the surface (non-linear term)
free_slip_penalty_upper = v_soln.sym.dot(unit_vec_r) * unit_vec_r * surface_fn
free_slip_penalty_lower = v_soln.sym.dot(unit_vec_r) * unit_vec_r * base_fn

stokes.bodyforce = unit_vec_r * buoyancy
stokes.bodyforce -= 100000 * (free_slip_penalty_upper + free_slip_penalty_lower)

stokes.saddle_preconditioner = 1 / viscosity

# -

with mesh.access(meshr):
    meshr.data[:, 0] = uw.function.evaluate(
        sympy.sqrt(x**2 + y**2 + z**2), mesh.data
    )  # cf radius_fn which is 0->1


stokes._setup_terms(verbose=False)

stokes.solve(zero_init_guess=True)

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

    # pv.start_xvfb()

    mesh.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")

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

    # pv.start_xvfb()

    mesh.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")

    pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)
    pvmesh.point_data["visc"] = uw.function.evaluate(sympy.log(viscosity), mesh.data)

    velocity = np.zeros((mesh.data.shape[0], 3))
    velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], mesh.data)
    velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], mesh.data)
    velocity[:, 2] = uw.function.evaluate(v_soln.sym[2], mesh.data)

    pvmesh.point_data["V"] = velocity * 100

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
        integration_direction="forward",
        compute_vorticity=False,
        max_steps=250,
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

    pl = pv.Plotter(window_size=(1000, 1000))

    pl.add_mesh(pvmesh, "Gray", "wireframe")
    # pl.add_arrows(arrow_loc, velocity_field, mag=0.2/vmag, opacity=0.5)

    pl.add_mesh(pvstream, opacity=1.0)
    # pl.add_mesh(pvmesh, cmap="Blues_r", edge_color="Gray", show_edges=True, scalars="rho", opacity=0.25)

    pl.add_points(spoint_cloud, cmap="Reds_r", scalars="M", render_points_as_spheres=True, point_size=2, opacity=0.3)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# +
# OR

# # +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, mesh.data)
    pvmesh.point_data["S"] = uw.function.evaluate(v_soln.sym.dot(unit_vec_r) * (base_fn + surface_fn), mesh.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[...] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[...] = uw.function.evaluate(stokes.u.fn, stokes.u.coords)

    clipped = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=(0.1, 0, 1), invert=True)

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        clipped, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S", use_transparency=False, opacity=1.0
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=100)

    pl.show(cpos="xy")

# -


def plot_mesh(filename):

    if uw.mpi.size == 1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = "white"
        pv.global_theme.window_size = [750, 750]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = "pythreejs"
        pv.global_theme.smooth_shading = False
        pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
        pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

        mesh.vtk("tmp_box.vtk")
        pvmesh = pv.read("tmp_box.vtk")

        pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)
        pvmesh.point_data["visc"] = uw.function.evaluate(sympy.log(viscosity), mesh.data)

        velocity = np.zeros((mesh.data.shape[0], 3))
        velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], mesh.data)
        velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], mesh.data)

        pvmesh.point_data["V"] = velocity

        # point sources at cell centres

        cpoints = np.zeros((mesh._centroids.shape[0] // 4, 3))
        cpoints[:, 0] = mesh._centroids[::4, 0]
        cpoints[:, 1] = mesh._centroids[::4, 1]
        cpoint_cloud = pv.PolyData(cpoints)

        pvstream = pvmesh.streamlines_from_source(
            cpoint_cloud,
            vectors="V",
            integrator_type=45,
            integration_direction="forward",
            compute_vorticity=False,
            max_steps=25,
            surface_streamlines=True,
        )

        with swarm.access():
            spoints = np.zeros((swarm.data.shape[0], 3))
            spoints[:, 0] = swarm.data[:, 0]
            spoints[:, 1] = swarm.data[:, 1]
            spoints[:, 2] = 0.0

        spoint_cloud = pv.PolyData(spoints)

        with swarm.access():
            spoint_cloud.point_data["M"] = material.data[...]

        pl = pv.Plotter()

        # pl.add_mesh(pvmesh, "Gray",  "wireframe")
        # pl.add_arrows(arrow_loc, velocity_field, mag=0.2/vmag, opacity=0.5)

        pl.add_mesh(pvstream, opacity=0.33)
        pl.add_mesh(pvmesh, cmap="Blues_r", edge_color="Gray", show_edges=True, scalars="rho", opacity=0.25)

        pl.add_points(
            spoint_cloud, cmap="Reds_r", scalars="M", render_points_as_spheres=True, point_size=5, opacity=0.3
        )

        pl.remove_scalar_bar("M")
        pl.remove_scalar_bar("V")
        pl.remove_scalar_bar("rho")

        pl.screenshot(filename="{}.png".format(filename), window_size=(1250, 1250), return_img=False)

        pl.close()
        pv.close_all()

        return


0 / 0

t_step = 0

# +
# Update in time

expt_name = "output/swarm_rt"

for step in range(0, 200):

    stokes.solve(zero_init_guess=True)
    delta_t = min(10.0, stokes.estimate_dt())

    # update swarm / swarm variables

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(t_step, delta_t))

    # advect swarm
    swarm.advection(v_soln.fn, delta_t)

    if t_step % 5 == 0:
        plot_mesh(filename="{}_step_{}".format(expt_name, t_step))

    t_step += 1

# -


savefile = "output/swarm_rt.h5".format(step)
mesh.save(savefile)
v_soln.save(savefile)
mesh.generate_xdmf(savefile)
