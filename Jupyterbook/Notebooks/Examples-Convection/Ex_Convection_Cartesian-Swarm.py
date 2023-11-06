# # Constant viscosity convection, Cartesian domain (benchmark)
#
# This is a simple example in which we try to instantiate two solvers on the mesh and have them use a common set of variables.
#
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.
#
# The next step is to add particles at node points and sample back along the streamlines to find values of the T field at a previous time.
#
# (Note, we keep all the pieces from previous increments of this problem to ensure that we don't break something along the way)

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

# -

meshbox = uw.meshes.Unstructured_Simplex_Box(
    dim=2,
    minCoords=(0.0, 0.0, 0.0),
    maxCoords=(1.0, 1.0, 1.0),
    cell_size=1.0 / 32.0,
    regular=True,
)
meshbox.dm.view()

# +

import sympy

# Some useful coordinate stuff

x = meshbox.N.x
y = meshbox.N.y

# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "pythreejs"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, -5.0]
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, edge_color="Black", show_edges=True)

    pl.show(cpos="xy")
# -

v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=3)


swarm = uw.swarm.Swarm(mesh=meshbox)
T1 = uw.swarm.SwarmVariable("Tminus1", swarm, 1, proxy_degree=3)
swarm.populate(fill_param=5)


# +
ad = uw.systems.AdvDiffusionSwarm(meshbox, t_soln, T1.fn, degree=3, projection=True)

ad._u_star_projector.smoothing = 0.0

ad.add_dirichlet_bc(1.0, "Bottom")
ad.add_dirichlet_bc(0.0, "Top")

init_t = 0.01 * sympy.sin(5.0 * x) * sympy.sin(np.pi * y) + (1.0 - y)

with meshbox.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]

with swarm.access(T1):
    T1.data[...] = uw.function.evaluate(
        init_t, swarm.particle_coordinates.data
    ).reshape(-1, 1)

# +
# Create Stokes object

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    u_degree=v_soln.degree,
    p_degree=p_soln.degree,
    solver_name="stokes",
    verbose=False,
)

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
stokes.petsc_options.delValue("ksp_monitor")

# Constant visc
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(meshbox.dim)
stokes.constitutive_model.Parameters.viscosity = 1

# Velocity boundary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))
stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))


# +
buoyancy_force = 1.0e6 * t_soln.fn
stokes.bodyforce = meshbox.N.j * buoyancy_force

# check the stokes solve is set up and that it converges
stokes.solve()


# +
# check the projection


if uw.mpi.size == 1 and ad.projection:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "pythreejs"
    pv.global_theme.smooth_shading = True

    pv.start_xvfb()

    pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    with meshbox.access():
        usol = stokes.u.data.copy()

    pvmesh.point_data["mT1"] = uw.function.evaluate(
        ad._u_star_projected.fn, meshbox.data
    )
    pvmesh.point_data["T1"] = uw.function.evaluate(T1.fn, meshbox.data)
    pvmesh.point_data["dT1"] = uw.function.evaluate(
        T1.fn - ad._u_star_projected.fn, meshbox.data
    )

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="dT1",
        use_transparency=False,
        opacity=0.5,
    )

    # pl.add_arrows(arrow_loc, arrow_length, mag=1.0e-4, opacity=0.5)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -


def plot_T_mesh(filename):
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

        pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

        points = np.zeros((t_soln.coords.shape[0], 3))
        points[:, 0] = t_soln.coords[:, 0]
        points[:, 1] = t_soln.coords[:, 1]

        point_cloud = pv.PolyData(points)

        with meshbox.access():
            point_cloud.point_data["T"] = t_soln.data.copy()

        with swarm.access():
            points = np.zeros((swarm.data.shape[0], 3))
            points[:, 0] = swarm.data[:, 0]
            points[:, 1] = swarm.data[:, 1]

        swarm_point_cloud = pv.PolyData(points)

        with swarm.access():
            swarm_point_cloud.point_data["T1"] = T1.data.copy()

        with meshbox.access():
            usol = stokes.u.data.copy()

        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshbox.data)

        arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
        arrow_loc[:, 0:2] = stokes.u.coords[...]

        arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
        arrow_length[:, 0:2] = usol[...]

        pl = pv.Plotter()

        pl.add_arrows(arrow_loc, arrow_length, mag=0.00001, opacity=0.75)

        pl.add_points(
            swarm_point_cloud,  # cmap="RdYlBu_r", scalars="T1",
            color="Black",
            render_points_as_spheres=True,
            clim=[0.0, 1.0],
            point_size=1.0,
            opacity=0.5,
        )

        pl.add_points(
            point_cloud,
            cmap="coolwarm",
            scalars="T",
            render_points_as_spheres=False,
            clim=[0.0, 1.0],
            point_size=10.0,
            opacity=0.66,
        )

        # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black",
        #             show_edges=True, scalars="T",clim=[0.0,1.0],
        #               use_transparency=False, opacity=0.5)

        pl.remove_scalar_bar("T")
        # pl.remove_scalar_bar("T1")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1250, 1250),
            return_img=False,
        )
        # pl.show()
        pl.close()


# +
# Convection model / update in time

expt_name = "output/Ra1e6_swarm_pnots"

ad_delta_t = 0.000033  # target

for step in range(0, 250):
    stokes.solve(zero_init_guess=False)
    stokes_delta_t = 5.0 * stokes.estimate_dt()
    delta_t = stokes_delta_t

    ad.solve(timestep=delta_t, zero_init_guess=True)

    # update swarm / swarm variables

    with swarm.access(T1):
        T1.data[:, 0] = uw.function.evaluate(t_soln.fn, swarm.particle_coordinates.data)

    # advect swarm
    swarm.advection(v_soln.fn, delta_t)

    tstats = t_soln.stats()
    tstarstats = T1._meshVar.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
        print(tstats[2], tstats[3])
        print(tstarstats[2], tstarstats[3])

    plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

# savefile = "{}_ts_{}.h5".format(expt_name,step)
# meshbox.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshbox.generate_xdmf(savefile)

# -


# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshbox.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshbox.generate_xdmf(savefile)


# +


if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "pythreejs"
    pv.global_theme.smooth_shading = True

    pv.start_xvfb()

    pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    points = np.zeros((t_soln.coords.shape[0], 3))
    points[:, 0] = t_soln.coords[:, 0]
    points[:, 1] = t_soln.coords[:, 1]

    point_cloud = pv.PolyData(points)

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]

    swarm_point_cloud = pv.PolyData(points)

    with swarm.access():
        swarm_point_cloud.point_data["T1"] = T1.data.copy()

    with meshbox.access():
        point_cloud.point_data["T"] = t_soln.data.copy()

    with meshbox.access():
        usol = stokes.u.data.copy()

    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshbox.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter()

    pl.add_arrows(arrow_loc, arrow_length, mag=0.00002, opacity=0.75)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(point_cloud, cmap="coolwarm",
    #               render_points_as_spheres=True,
    #               point_size=7.5, opacity=0.25
    #             )

    pl.add_points(
        swarm_point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=2.5,
        opacity=0.5,
        clim=[0.0, 1.0],
    )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=0.5,
        clim=[0.0, 1.0],
    )

    pl.show(cpos="xy")
