# %% [markdown]
# # Stokes sinker - sinking block
#
# Sinking block benchmark as outlined in:
#
# - [Gerya, T.V. and Yuen, D.A., 2003. Characteristics-based marker-in-cell method with conservative finite-differences schemes for modeling geological flows with strongly variable transport properties. Physics of the Earth and Planetary Interiors, 140(4), pp.293-318.](http://jupiter.ethz.ch/~tgerya/reprints/2003_PEPI_method.pdf)
#
# - Only value to change is: **viscBlock**
#
# - utilises the UW scaling module to convert from dimensional to non-dimensional values
#
# - Includes a passive tracer that is handled by UW swarm routines (and should be parallel safe).

# %%
# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# %%
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

options = PETSc.Options()


# %%
sys = PETSc.Sys()
sys.pushErrorHandler("traceback")

options["snes_converged_reason"] = None
options["snes_monitor_short"] = None


# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim = uw.scaling.non_dimensionalise
dim = uw.scaling.dimensionalise

# %%
# Set the resolution, a structured quad box of 51x51 is used in the paper
# res = 51
res = 21  # use lower res for testing

nsteps = 1  # number of time steps
swarmGPC = 2  # swarm fill parameter
render = True  # plot images

# %%
refLength = 500e3
refDensity = 3.3e3
refGravity = 9.81
refVelocity = (1 * u.centimeter / u.year).to(u.meter / u.second).m  ### 1 cm/yr in m/s
refViscosity = 1e21
refPressure = refDensity * refGravity * refLength
refTime = refViscosity / refPressure

bodyforce = (
    refDensity * u.kilogram / u.metre**3 * refGravity * u.meter / u.second**2
)

# %%
KL = refLength * u.meter
Kt = refTime * u.second
KM = bodyforce * KL**2 * Kt**2

scaling_coefficients = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"] = KM
scaling_coefficients

# %%
### fundamental values
ref_length = uw.scaling.dimensionalise(1.0, u.meter).magnitude
ref_length_km = uw.scaling.dimensionalise(1.0, u.kilometer).magnitude
ref_density = uw.scaling.dimensionalise(1.0, u.kilogram / u.meter**3).magnitude
ref_gravity = uw.scaling.dimensionalise(1.0, u.meter / u.second**2).magnitude
ref_temp = uw.scaling.dimensionalise(1.0, u.kelvin).magnitude
ref_velocity = uw.scaling.dimensionalise(1.0, u.meter / u.second).magnitude

### derived values
ref_time = ref_length / ref_velocity
ref_time_Myr = dim(1, u.megayear).m
ref_pressure = ref_density * ref_gravity * ref_length
ref_stress = ref_pressure
ref_viscosity = ref_pressure * ref_time

### Key ND values
ND_gravity = 9.81 / ref_gravity

# %%
# define some names for our index
materialLightIndex = 0
materialHeavyIndex = 1

## Set constants for the viscosity and density of the sinker.
viscBG = 1e21 / ref_viscosity
viscBlock = 1e21 / ref_viscosity

## set density of blocks
densityBG = 3.2e3 / ref_density
densityBlock = 3.3e3 / ref_density

# %%
xmin, xmax = 0.0, ndim(500 * u.kilometer)
ymin, ymax = 0.0, ndim(500 * u.kilometer)

# %%
xmin, xmax

# %%
# Set the box min and max coords
boxCentre_x, boxCentre_y = ndim(250.0 * u.kilometer), ndim(375.0 * u.kilometer)

box_xmin, box_xmax = boxCentre_x - ndim(50 * u.kilometer), boxCentre_x + ndim(
    50 * u.kilometer
)
box_ymin, box_ymax = boxCentre_y - ndim(50 * u.kilometer), boxCentre_y + ndim(
    50 * u.kilometer
)

# location of tracer at bottom of sinker
x_pos = box_xmax - ((box_xmax - box_xmin) / 2.0)
y_pos = box_ymin

# %%
# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0),
#                                               maxCoords=(1.0,1.0),
#                                               cellSize=1.0/res,
#                                               regular=True)

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / res, regular=False)

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax)
)


# %%
### Create Stokes object

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# %%
#### No slip
sol_vel = sympy.Matrix([0.0, 0.0])

### free slip
stokes.add_dirichlet_bc(sol_vel, "Left", 0)  # left/right: components, function, markers
stokes.add_dirichlet_bc(sol_vel, "Right", 0)  # left/right: components, function, markers
stokes.add_dirichlet_bc(sol_vel, "Top", 1)  # left/right: components, function, markers
stokes.add_dirichlet_bc(sol_vel, "Bottom", 1)  # left/right: components, function, markers


# %%
## Solver

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# gasm is super-fast ... but mg seems to be bulletproof
# gamg is toughest wrt viscosity

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")


# %%
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2)
swarm.populate(fill_param=swarmGPC)

# %%
with swarm.access(material):
    material.data[...] = materialLightIndex
    material.data[
        (swarm.data[:, 0] >= box_xmin)
        & (swarm.data[:, 0] <= box_xmax)
        & (swarm.data[:, 1] >= box_ymin)
        & (swarm.data[:, 1] <= box_ymax)
    ] = materialHeavyIndex


# %%
### add tracer for sinker velocity
tracer = np.zeros(shape=(1, 2))
tracer[:, 0], tracer[:, 1] = x_pos, y_pos

# %%
passiveSwarm = uw.swarm.Swarm(mesh)
passiveSwarm.dm.finalizeFieldRegister()
passiveSwarm.dm.addNPoints(npoints=len(tracer))
passiveSwarm.dm.setPointCoordinates(tracer)

# %%
mat_density = np.array([densityBG, densityBlock])

density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]

# %%
mat_viscosity = np.array([viscBG, viscBlock])

viscosityMat = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]


# %%
def plot_mat():
    
    import pyvista as pv
    import underworld3.visualisation as vis
    
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    points = vis.swarm_to_pv_cloud(swarm)

    point_cloud = pv.PolyData(points)
    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh, "Black", "wireframe")

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


if render and uw.mpi.size == 1:
    plot_mat()

# %%
### linear solve
stokes.constitutive_model.Parameters.shear_viscosity_0 = ndim(
    ref_viscosity * u.pascal * u.second
)
stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density])


# %%
stokes.solve()

# %%
### add in material-based viscosity
stokes.constitutive_model.Parameters.viscosity = viscosityMat

# %%
# stokes.petsc_options.view()
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
options["snes_test_jacobian"] = None
options["snes_test_jacobian_view"] = None
# stokes.petsc_options['snes_test_jacobian'] = None
# stokes.petsc_options['snes_test_jacobian_view'] = None

# %%
stokes.bodyforce

# %%
stokes.constitutive_model.Parameters.viscosity

# %%
step = 0
time = 0.0
tSinker = np.zeros(nsteps + 1) * np.nan
ySinker = np.zeros(nsteps + 1) * np.nan


# %%
def record_tracer(step, time):
    ### Get the position of the sinking ball
    with passiveSwarm.access(passiveSwarm):
        if passiveSwarm.dm.getLocalSize() > 0:
            ymin = passiveSwarm.data[:, 1].min()
    ySinker[step] = ymin
    tSinker[step] = time

    ### print some stuff
    if passiveSwarm.dm.getLocalSize() > 0:
        print(
            f"Step: {str(step).rjust(3)}, time: {dim(time, u.megayear).m:6.2f} [Myr], tracer:  {dim(ymin, u.kilometer):6.2f} [km]"
        )


record_tracer(step, time)

while step < nsteps:
    ### solve stokes
    stokes.solve()
    ### estimate dt
    dt = 0.5 * stokes.estimate_dt()

    ### advect the swarm
    swarm.advection(stokes.u.sym, dt, corrector=False)
    passiveSwarm.advection(stokes.u.sym, dt, corrector=False)

    ### advect tracer
    # vel_on_tracer = uw.function.evaluate(stokes.u.fn,tracer)
    # tracer += dt*vel_on_tracer
    step += 1
    time += dt

    record_tracer(step, time)

# %%
if passiveSwarm.dm.getLocalSize() > 0:
    import matplotlib.pyplot as plt

    ### remove nan values, if any. Convert to km and Myr
    ySinker = dim(ySinker[~np.isnan(ySinker)], u.kilometer)
    tSinker = dim(tSinker[~np.isnan(tSinker)], u.megayear)

    print("Initial position: t = {0:.3f}, y = {1:.3f}".format(tSinker[0], ySinker[0]))
    print("Final position:   t = {0:.3f}, y = {1:.3f}".format(tSinker[-1], ySinker[-1]))

    UWvelocity = (
        ((ySinker[0] - ySinker[-1]) / (tSinker[-1] - tSinker[0]))
        .to(u.meter / u.second)
        .m
    )
    print(f"Velocity:         v = {UWvelocity} m/s")

    if uw.mpi.size == 0:
        fig = plt.figure()
        fig.set_size_inches(12, 6)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(tSinker.m, ySinker.m)
        ax.set_xlabel("Time [Myr]")
        ax.set_ylabel("Sinker position [km]")

# %% [markdown]
# ##### compare values against published results
#
#
# - The marker, representing the velocity calculated from the UW model, should fit along the curved line.
# - These velocities are taken from _Gerya (2010), Introduction to numerical modelling (2nd Ed), page 345_, but show the same model referenced in the paper above

# %%
from scipy.interpolate import interp1d


#### col 0 is log10( visc_block / visc_BG ), col 1 is block velocity, m/s

paperData = np.array(
    [
        (-6.01810758939326, 1.3776991077026654e-9),
        (-5.014458950015076, 1.3792676876049961e-9),
        (-4.018123543216514, 1.3794412652019993e-9),
        (-3.021737084183539, 1.3740399388011341e-9),
        (-2.0104944249364634, 1.346341549020515e-9),
        (-1.0053652707603105, 1.1862379129846573e-9),
        (-0.005609364256097038, 8.128929227244664e-10),
        (0.993865754958847, 4.702099044525527e-10),
        (2.005950776073732, 3.505255987071023e-10),
        (3.0024521026341358, 3.3258073831103253e-10),
        (4.006139031188129, 3.2996814021496194e-10),
        (5.00247443798669, 3.301417178119651e-10),
        (6.013474599120308, 3.289241220212219e-10),
    ]
)

### some errors from sampling, so rounding are used
visc_ratio = np.round(paperData[:, 0])
paperVelocity = np.round(paperData[:, 1], 11)

f = interp1d(visc_ratio, paperVelocity, kind="cubic")
x = np.arange(-6, 6, 0.01)

if uw.mpi.size == 1 and render:
    import matplotlib.pyplot as plt

    plt.title("check benchmark")
    plt.plot(x, f(x), label="benchmark velocity curve", c="k")
    plt.scatter(
        np.log10(viscBlock / viscBG),
        UWvelocity,
        label="model velocity",
        c="red",
        marker="x",
    )
    plt.legend()


# %%
### for uw testing system
def test_sinkBlock():
    if uw.mpi.size == 1:
        assert np.isclose(f(0.0), UWvelocity)


# %%
if render and uw.mpi.size == 1:
    plot_mat()

# %%
