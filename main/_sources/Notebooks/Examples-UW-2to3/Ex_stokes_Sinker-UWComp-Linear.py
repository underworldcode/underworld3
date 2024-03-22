# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] magic_args="[markdown]"
# ''Stokes Sinker''
# ======
#
# Testing the Stokes sinker problem between UW2 and UW3. This system consists of a dense, high viscosity sphere falling through a background lower density and a viscous fluid.
#
#
#
#
# -

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# %%
import underworld as uw2
from underworld import function as fn
import underworld.visualisation as vis
import numpy as np
import math

rank = uw2.mpi.rank

# + [markdown] magic_args="[markdown]"
# Setup parameters
# -----
#
# Set simulation parameters for the test and position of the spherical sinker.
# -

# %%
# Set the resolution.
res = 32

# Set size and position of dense sphere.
sphereRadius = 0.1
sphereCentre = (0.0, 0.7)

# define some names for our index
materialLightIndex = 0
materialHeavyIndex = 1


# Set constants for the viscosity and density of the sinker.
viscBG = 1.0
viscSphere = 10.0

densityBG = 1.0
densitySphere = 10.0


# location of tracer at bottom of sinker
x_pos = sphereCentre[0]
y_pos = sphereCentre[1] - sphereRadius


nsteps = 10


swarmGPC = 4


# + [markdown] magic_args="[markdown]"
# Create UW2 version
# ------
# -

# %%
def uw2_stokesSinker():

    mesh = uw2.mesh.FeMesh_Cartesian(
        elementType=("Q1/dQ0"),
        elementRes=(int(res * 2), int(res)),
        minCoord=(-1.0, 0.0),
        maxCoord=(1.0, 1.0),
    )

    velocityField = mesh.add_variable(nodeDofCount=2)
    pressureField = mesh.subMesh.add_variable(nodeDofCount=1)

    velocityField.data[:] = [0.0, 0.0]
    pressureField.data[:] = 0.0

    # Create the swarm and an advector associated with it
    swarm = uw2.swarm.Swarm(mesh=mesh)
    advector = uw2.systems.SwarmAdvector(
        swarm=swarm, velocityField=velocityField, order=2
    )

    # Add a data variable which will store an index to determine material.
    materialIndex = swarm.add_variable(dataType="int", count=1)

    # Create a layout object that will populate the swarm across the whole domain.
    swarmLayout = uw2.swarm.layouts.PerCellGaussLayout(
        swarm=swarm, gaussPointCount=swarmGPC
    )
    # swarmLayout = uw2.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=swarmFill )

    # Go ahead and populate the swarm.
    swarm.populate_using_layout(layout=swarmLayout)

    # create a function for a sphere. returns `True` if query is inside sphere, `False` otherwise.
    coord = fn.input() - sphereCentre
    fn_sphere = fn.math.dot(coord, coord) < sphereRadius**2

    # set up the condition for being in a sphere. If not in sphere then will return light index.
    conditions = [(fn_sphere, materialHeavyIndex), (True, materialLightIndex)]

    # Execute the branching conditional function, evaluating at the location of each particle in the swarm.
    # The results are copied into the materialIndex swarm variable.
    materialIndex.data[:] = fn.branching.conditional(conditions).evaluate(swarm)

    # build a tracer swarm with one particle
    tracerSwarm = uw2.swarm.Swarm(mesh)
    advector_tracer = uw2.systems.SwarmAdvector(
        swarm=tracerSwarm, velocityField=velocityField, order=2
    )

    # build a numpy array with one particle, specifying it's exact location
    coord_array = np.array(object=(x_pos, y_pos), ndmin=2)
    tracerSwarm.add_particles_with_coordinates(coord_array)

    tracer = numpy.zeros(shape=(1, 2))
    tracer[:, 0], tracer[:, 1] = x_pos, y_pos

    fig1 = vis.Figure(figsize=(800, 400))
    fig1.Points(swarm, materialIndex, colourBar=False, pointSize=2.0)
    fig1.VectorArrows(mesh, velocityField)
    fig1.show()

    # Here we set a viscosity value of '1.' for both materials
    mappingDictViscosity = {materialLightIndex: viscBG, materialHeavyIndex: viscSphere}
    # Create the viscosity map function.
    viscosityMapFn = fn.branching.map(
        fn_key=materialIndex, mapping=mappingDictViscosity
    )
    # Here we set a density of '0.' for the lightMaterial, and '1.' for the heavymaterial.
    mappingDictDensity = {
        materialLightIndex: densityBG,
        materialHeavyIndex: densitySphere,
    }
    # Create the density map function.
    densityFn = fn.branching.map(fn_key=materialIndex, mapping=mappingDictDensity)

    # And the final buoyancy force function.
    z_hat = (0.0, 1.0)
    buoyancyFn = -densityFn * z_hat

    iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
    jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]

    freeslipBC = uw2.conditions.DirichletCondition(
        variable=velocityField, indexSetsPerDof=(iWalls, jWalls)
    )

    noslipBC = uw2.conditions.DirichletCondition(
        variable=velocityField, indexSetsPerDof=(iWalls + jWalls, iWalls + jWalls)
    )

    stokes = uw2.systems.Stokes(
        velocityField=velocityField,
        pressureField=pressureField,
        voronoi_swarm=swarm,
        conditions=noslipBC,
        fn_viscosity=viscosityMapFn,
        fn_bodyforce=buoyancyFn,
    )

    solver = uw2.systems.Solver(stokes)

    top = mesh.specialSets["MaxJ_VertexSet"]
    surfaceArea = uw2.utils.Integral(
        fn=1.0, mesh=mesh, integrationType="surface", surfaceIndexSet=top
    )
    surfacePressureIntegral = uw2.utils.Integral(
        fn=pressureField, mesh=mesh, integrationType="surface", surfaceIndexSet=top
    )

    # a callback function to calibrate the pressure - will pass to solver later
    def pressure_calibrate():
        (area,) = surfaceArea.evaluate()
        (p0,) = surfacePressureIntegral.evaluate()
        offset = p0 / area
        if rank == 0:
            print(
                "Zeroing pressure using mean upper surface pressure {}".format(offset)
            )
        pressureField.data[:] -= offset

    vdotv = fn.math.dot(velocityField, velocityField)

    # Stepping. Initialise time and timestep.
    time = 0.0
    step = 0

    tSinker = np.zeros(nsteps)
    ySinker0 = np.zeros(nsteps)

    ySinker1 = np.zeros(nsteps)

    # Perform 10 steps
    while step < nsteps:
        # Get velocity solution - using callback
        solver.solve()
        # Calculate the RMS velocity
        vrms = math.sqrt(mesh.integrate(vdotv)[0] / mesh.integrate(1.0)[0])

        if rank == 0:
            ymin0 = np.copy(tracerSwarm.data[:, 1].min())
            ymin1 = np.copy(tracer[:, 1].min())

            ySinker0[step] = ymin0
            ySinker1[step] = ymin1

            tSinker[step] = time

            print(
                "step = {0:6d}; time = {1:.3e}; v_rms = {2:.3e}; height0 = {3:.3e}; height1 = {3:.3e}".format(
                    step, time, vrms, ymin0, ymin1
                )
            )

        # Retrieve the maximum possible timestep for the advection system.
        dt = advector.get_max_dt()
        # Advect using this timestep size.
        advector.integrate(dt)
        advector_tracer.integrate(dt)

        vel_on_tracer = velocityField.evaluate(tracer)
        tracer += dt * vel_on_tracer

        step += 1
        time += dt

    if rank == 0:
        print(
            "Initial position: t = {0:.3f}, y = {1:.3f}".format(tSinker[0], ySinker0[0])
        )
        print(
            "Final position:   t = {0:.3f}, y = {1:.3f}".format(
                tSinker[nsteps - 1], ySinker0[nsteps - 1]
            )
        )

        uw2.utils.matplotlib_inline()
        import matplotlib.pyplot as pyplot

        fig = pyplot.figure()
        fig.set_size_inches(12, 6)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(tSinker, ySinker0)
        ax.plot(tSinker, ySinker1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Sinker position")

        fig1.show()

        return tSinker, ySinker0, ySinker1


# + [markdown] magic_args="[markdown]"
# Create UW3 version
# ------
# -

# %%
from petsc4py import PETSc
import underworld3 as uw3
from underworld3.systems import Stokes
import numpy
import sympy
from mpi4py import MPI

options = PETSc.Options()

# %%
def uw3_stokesSinker(render=True):

    sys = PETSc.Sys()
    sys.pushErrorHandler("traceback")

    options = PETSc.Options()
    options["snes_converged_reason"] = None
    options["snes_monitor_short"] = None

    mesh = uw3.meshing.StructuredQuadBox(
        elementRes=(int(res), int(res)), minCoords=(-1.0, 0.0), maxCoords=(1.0, 1.0)
    )

    v = uw3.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw3.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw3.systems.Stokes(mesh, velocityField=v, pressureField=p)

    ### free slip.
    ### note with petsc we always need to provide a vector of correct cardinality.
    # stokes.add_dirichlet_bc( (0.,0.), ["Bottom",  "Top"], 1 )  # top/bottom: components, function, markers
    # stokes.add_dirichlet_bc( (0.,0.), ["Left", "Right"],  0 )  # left/right: components, function, markers

    ### No slip (?)
    sol_vel = sympy.Matrix([0, 0])

    stokes.add_dirichlet_bc(
        sol_vel, ["Top", "Bottom"], [0, 1]
    )  # top/bottom: components, function, markers
    stokes.add_dirichlet_bc(
        sol_vel, ["Left", "Right"], [0, 1]
    )  # left/right: components, function, markers

    swarm = uw3.swarm.Swarm(mesh=mesh)
    material = uw3.swarm.IndexSwarmVariable("M", swarm, indices=4)
    swarm.populate(fill_param=swarmGPC)

    blob = numpy.array(
        # [[ 0.25, 0.75, 0.1,  1],
        #  [ 0.45, 0.70, 0.05, 2],
        #  [ 0.65, 0.60, 0.06, 3],
        [[sphereCentre[0], sphereCentre[1], sphereRadius, 1]]
    )
    # [ 0.65, 0.20, 0.06, 2],
    # [ 0.45, 0.20, 0.12, 3] ])

    with swarm.access(material):
        material.data[...] = materialLightIndex

        for i in range(blob.shape[0]):
            cx, cy, r, m = blob[i, :]
            inside = (swarm.data[:, 0] - cx) ** 2 + (
                swarm.data[:, 1] - cy
            ) ** 2 < r**2
            material.data[inside] = materialHeavyIndex

    tracer = numpy.zeros(shape=(1, 2))
    tracer[:, 0], tracer[:, 1] = x_pos, y_pos

    mat_density = numpy.array([densityBG, densitySphere])

    density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]

    mat_viscosity = numpy.array([viscBG, viscSphere])

    viscosity = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]

    def plot_fig():

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = "white"
        pv.global_theme.window_size = [750, 750]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = "trame"
        pv.global_theme.smooth_shading = True

        mesh.vtk("tempMsh.vtk")
        pvmesh = pv.read("tempMsh.vtk")

        with swarm.access():
            points = numpy.zeros((swarm.data.shape[0], 3))
            points[:, 0] = swarm.data[:, 0]
            points[:, 1] = swarm.data[:, 1]
            points[:, 2] = 0.0

        point_cloud = pv.PolyData(points)

        with swarm.access():
            point_cloud.point_data["M"] = material.data.copy()

        pl = pv.Plotter(notebook=True)

        pl.add_mesh(pvmesh, "Black", "wireframe")

        # pl.add_points(point_cloud, color="Black",
        #                   render_points_as_spheres=False,
        #                   point_size=2.5, opacity=0.75)

        pl.add_mesh(
            point_cloud,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=False,
            scalars="M",
            use_transparency=False,
            opacity=0.95,
        )

        pl.show(cpos="xy")

    if render:
        plot_fig()

    stokes.constitutive_model = uw3.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.material_properties = (
        stokes.constitutive_model.Parameters(viscosity=viscosity)
    )

    # stokes.viscosity = viscosity

    stokes.bodyforce = -1.0 * density * mesh.N.j

    step = 0
    time = 0.0

    tSinker = np.zeros(nsteps)
    ySinker = np.zeros(nsteps)

    while step < nsteps:

        ### solve stokes
        stokes.solve()

        if MPI.COMM_WORLD.rank == 0:
            ymin = tracer[:, 1].min()
            ySinker[step] = ymin
            tSinker[step] = time
            print(
                f"Step: {str(step).rjust(3)}, time: {time:6.2f}, tracer:  {ymin:6.2f}"
            )  # , vrms {vrms_val:.3e}")

        ### estimate dt
        dt = stokes.estimate_dt()

        with swarm.access():
            vel_on_particles = uw3.function.evaluate(
                stokes.u.fn, swarm.particle_coordinates.data
            )

        ### advect swarm
        with swarm.access(swarm.particle_coordinates):
            swarm.particle_coordinates.data[:] += dt * vel_on_particles

        vel_on_tracer = uw3.function.evaluate(stokes.u.fn, tracer)
        tracer += dt * vel_on_tracer

        # if MPI.COMM_WORLD.rank==0:
        # print('step = {0:6d}; time = {1:.3e}; v_rms = {2:.3e}; height = {3:.3e}'
        #       .format(step,time,vrms,ymin))

        step += 1
        time += dt

    if rank == 0:
        print(
            "Initial position: t = {0:.3f}, y = {1:.3f}".format(tSinker[0], ySinker[0])
        )
        print(
            "Final position:   t = {0:.3f}, y = {1:.3f}".format(
                tSinker[nsteps - 1], ySinker[nsteps - 1]
            )
        )

        uw2.utils.matplotlib_inline()
        import matplotlib.pyplot as pyplot

        fig = pyplot.figure()
        fig.set_size_inches(12, 6)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(tSinker, ySinker)
        ax.set_xlabel("Time")
        ax.set_ylabel("Sinker position")

        # fig1.show()

        if render:
            plot_fig()

        return tSinker, ySinker


# %%
tSinker_UW2, ySinker0_UW2, ySinker1_UW2 = uw2_stokesSinker()

# %%
tSinker_UW3, ySinker_UW3 = uw3_stokesSinker()

# + [markdown] magic_args="[markdown]"
# Compare velocity of tracer in both models to terminal velocity estimation
# ------
# -

# ### Terminal velocity estimation is closest to model with no slip boundaries, free slip results in a higher velocity


# ## UW2 and UW3 velocities match when the viscosities in UW3 are doubled or densities are halved


stokes_vel0 = (-1 * (2 * sphereRadius) ** 2 * (densitySphere - densityBG)) / (
    18 * viscBG
)
stokes_vel1 = -1 * (2 / 9) * ((densitySphere - densityBG) / viscBG) * sphereRadius**2

# print(f'stokes vel0: {stokes_vel0}, stokes vel1: {stokes_vel1}')

time = np.arange(0, 7.5, 0.1)
terminalStokes = stokes_vel0 * time

UW2_vel = (ySinker0_UW2[0] - ySinker0_UW2[-1]) / (tSinker_UW2[0] - tSinker_UW2[-1])
UW3_vel = (ySinker_UW3[0] - ySinker_UW3[-1]) / (tSinker_UW3[0] - tSinker_UW3[-1])


# %%
print(
    f"\n\n\n alaytical velocity: {stokes_vel0}, UW2 velocity: {UW2_vel}, UW3 velocity: {UW3_vel/2.} \n\n\n"
)

# %%
if rank == 0:
    import matplotlib.pyplot as pyplot

    fig = pyplot.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tSinker_UW2, ySinker0_UW2, c="blue", ls="--", label="UW2")

    ax.plot(tSinker_UW2, ySinker1_UW2, c="orange", ls="-.", label="UW2")

    ax.plot(tSinker_UW3, ySinker_UW3, c="red", ls=":", label="UW3")

    ax.plot(
        time,
        (sphereCentre[1] - sphereRadius) + terminalStokes,
        c="k",
        ls=":",
        label="Terminal velocity",
    )

    ax.legend()

    ax.set_xlabel("Time")
    ax.set_ylabel("Sinker position")

    # fig1.show()

