# # Stokes flow in a Spherical Domain (Inner Core)
#
# This notebook models slow flow in the viscous, solid Inner Core of the Earth.
#
# ## Problem description
#
# We expect the Earth's Inner core to be in a creeping flow (Stokes) regime, potentially convecting if heat sources (surface cooling) are sufficiently strong. Observations of seismic velocity variations suggest large-amplitude, long-wavelength density (temperature) variations are present near the surface of the inner core that are expected to induce an instantaneous flow pattern that we can model. We note that it is more difficult to determine whether this pattern is the result of self-sustaining thermal / thermo-chemical convection.
#
# ## Mathematical formulation
#
# The Navier-Stokes equation describes the flow of a viscous fluid in response to buoyancy forces and pressure gradients:
#
# \\[
# \rho \frac{\partial \mathbf{u}}{\partial t} + \eta\nabla^2 \mathbf{u} -\nabla p = \rho \mathbf{g}
# \\]
#
# Where $\rho$ is the density, $\eta$ is dynamic viscosity and $\mathbf{g}$ is the gravitational acceleration vector. We here assume that density changes are due to temperature and are small enough to be consistent with an assumption of incompressibility (the Boussinesq approximation). We can rescale this equation of motion using units for length, time, temperature and mass that are specific to the problem and, in this way, obtain a scale-independent form:
#
# \\[
# \frac{1}{\mathrm{Pr}} \frac{\partial \mathbf{u}}{\partial t} + \nabla^2 \mathbf{u} -\nabla p = \mathrm{Ra} T' \hat{\mathbf{g}}
# \\]

# where we have assumed that buoyancy forces on the right hand side are due to temperature variations, and the two dimensionless numbers, $\mathrm{Ra}$ and $\mathrm{Pr}$ are measures of the importance of buoyancy forcing and intertial terms, respectively.
#
# \\[
# \mathrm{Ra} = \frac{g\rho_0 \alpha \Delta T d^3}{\kappa \eta}
# \quad \textrm{and} \quad
# \mathrm{Pr} = \frac{\eta}{\rho \kappa}
# \\]
#
# Here $\alpha$ is the thermal expansivity, $\Delta T$ is the range of the temperature variation, $d$ is the typical length scale over which temperature varies, $\kappa$ is the thermal diffusivity ( $\kappa = k / \rho_0 C_p$; $k$ is thermal conductivity, and $C_p$ is heat capacity).

# If we assume that the Prandtl number is large, then the inertial terms will not contribute significantly to the balance of forces in the equation of motion because we have rescaled the equations so that the velocity and pressure gradient terms are of order 1. This assumption eliminates the time dependent terms in the equations and tells us that the flow velocity and pressure field are always in equilibrium with the pattern of density variations and this also tells us that we can evaluate the flow without needing to know the history or origin of the buoyancy forces. When the viscosity is independent of velocity and dynamic pressure, the velocity and pressure scale proportionally with $\mathrm{Ra}$ but the flow pattern itself is unchanged.
#
# The scaling that we use for the non-dimensionalisation is as follows:
#
# \\[
#     x = d x', \quad t = \frac{d^2}{\kappa} t', \quad T=\Delta T T', \quad
#     p = p_0 + \frac{\eta \kappa}{d^2} p'
# \\]
#
# where the stress (pressure) scaling using viscosity ($\eta$) determines how the mass scales. In the above, $d$ is the radius of the inner core, a typical length scale for the problem, $\Delta T$ is the order-of-magnitude range of the temperature variation from our observations, and $\kappa$ is thermal diffusivity. The scaled velocity is obtained as $v = \kappa / d v'$.
#
#
#
# ## Computational formulation & model
#
# We implement our model using the Underworld finite element code which makes extensive use of the PETSc compuational framework for efficient parallel computation. Underworld has a python interface and we include the relevant scripts below.
#
# The model consists of a spherical ball divided into am unstructured tetrahedral mesh of quadratic velocity, linear pressure elements with a free slip upper boundary and with a buoyancy force defined using the predicted temperature anomaly from the tomographic model. Since this model does not have good depth information, we make the following assumption:
#
# \\[
# T(r,\theta,\phi) =  T_\textrm{TM}(\theta, \phi) \cdot r  \sin(\pi r)
# \\]
#
# We consider the possibility of a more viscous inner-most inner core that is either defined as a phase change boundary through which material can flow unimpeded (but where the strain rate is low) or, assuming a kinetically slow phase transition or compositional boundary,  as a region where the flow is stagnant.

# ### Model results
#
# The flow pattern is shown in this figure
#
# FIG
#
# The magnitude of the dimensionless flow speed that we obtain with this model is $10^{-4}$ and scales linearly with $\mathrm{Ra}$ which means that we can obtain the prediced flow speed as
#
# \\[
# U = 10^{-4} \frac{\kappa}{d} \mathrm{Ra} = 10^{-4}\frac{g\rho\alpha \Delta T d^2}{\eta}
# \\]
#
# We do not need to re-run models for different $\mathrm{Ra}$ provided we assume a linear viscosity (i.e. independent of strain-rate), and that $\mathrm{Pr}$ is large $(>100)$ so that inertial terms can be neglected.
#
# ### Scaling model results
#
# There is considerable uncertainty for most of the inner core consitutive parameters including thermal expansivity, thermal diffusivity, and viscosity with the last of these being most poorly understood.
#
# For an inner core density of $13000 - 15000 kg/m^3$, thermal conductivity of $10-100 W/m/K$, and heat capacity of $650 J/kg/K$, the range of thermal diffusivity is $5-30 \times 10^{-6} m^2/s$, and a viscosity in the range $10^3-10^{15} Pa.s$, we estimate the Prantdl number to be in the range $10^{3} - 10^{17}$.
#
# This implies a wide range of Rayleigh number, $\mathrm{Ra}$ that we need to consider in our analysis:
#
# \\[
# 10^8 < Ra < 10^{22}
# \\]
#
# Where the lower value assumes the highest viscosity, and a length-scale of half the inner core radius and the higher value assumes the relevant length scale is the whole inner core radius, and a lower viscosity. With consistent application of these assumpitions, we estimate the speed of the flow to be in the range
#
# \\[
# 10^{-6} < U < 10^{6} m/s
# \\]

# To help determine the limit of this wide range for which this analysis is applicable, we calculate the Reynolds number for the flow ($\mathrm{Re} = \rho U d / \eta$) and note that for a Reynolds number above $\sim 1$, the assumption that the flow pattern is Stokes-like will no longer be a reasonable one. This cross-over occurs for viscosities smaller than $\sim 10^9 Pa.s$ which have predicted flow speeds of a few $m/s$
#
# Coriolis forces may play a role for sufficiently large velocities, we can estimate their importance relative to buoyancy forces (Ricard, 2015) as
#
# \\[
# \frac{2 \Omega U}{g \alpha \delta T}
# \\]
#
# $\Omega$ being the angular velocity of the Earth's rotation, and $U$ being the typical
# flow velocity. For flows of $1 cm/yr$, this ratio is $10^{-5}$ and the effects of rotation can be neglected. The two terms are comparable when the flow speed is above about $10m/s$.

# ## Computational script in python

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

visuals = 0
output_dir = f"outputs_free_slip_FK3_ViscGrad5_iic100_QTemp_hr"

# Some gmsh issues, so we'll use a pre-built one
# mesh_file = "Sample_Meshes_Gmsh/test_mesh_sphere_at_res_005_c.msh"

res = uw.options.getReal("resolution", default=0.1)
r_o = uw.options.getReal("radius_o", default=1.0)
r_i = uw.options.getReal("radius_i", default=0.05)
iic_radius = uw.options.getReal("radius_iic", default=0.5)
iic_delta_eta = uw.options.getReal("delta_eta_iic", default=100.0)

Rayleigh = 1.0  # Doesn't actually matter to the solution pattern,
# choose 1 to make re-scaling simple

import os

os.makedirs(output_dir, exist_ok=True)

import petsc4py
from petsc4py import PETSc
import mpi4py
import numpy as np
import sympy


# +
meshball = uw.meshing.SphericalShell(
    radiusInner=r_i,
    radiusOuter=r_o,
    cellSize=res,
    qdegree=2,
)

meshball.dm.view()

# +
passive_swarm = uw.swarm.Swarm(mesh=meshball)

# define particles (globally) and then add them to the mesh locally

# 24 sets of blobs around the equator

blob_fill_param = 5
blob_size = 0.005
blobs_p = 36
blobs_t = 5

particle_coords = np.zeros((blobs_t * blobs_p * blob_fill_param**3, 3))

p = 0

for r in (0.9,):
    for t in range(blobs_t):
        for bl in range(blobs_p):
            r_b = r
            th_b = np.pi / 2 + (t - blobs_t // 2) * np.pi / 18
            ph_b = 2 * np.pi * (bl / blobs_p)

            # This should be part of the Coordinate System functionality
            X0 = r_b * np.cos(ph_b) * np.sin(th_b)
            Y0 = r_b * np.sin(ph_b) * np.sin(th_b)
            Z0 = r_b * np.cos(th_b)

            for ix in range(blob_fill_param):
                for iy in range(blob_fill_param):
                    for iz in range(blob_fill_param):
                        particle_coords[p, 0] = X0 + ix * blob_size / blob_fill_param
                        particle_coords[p, 1] = Y0 + iy * blob_size / blob_fill_param
                        particle_coords[p, 2] = Z0 + iz * blob_size / blob_fill_param

                        p += 1


# Filter out non-local coords

passive_swarm.add_particles_with_coordinates(particle_coords)


# -

v_soln = uw.discretisation.MeshVariable(
    r"u", meshball, meshball.dim, degree=2, vtype=uw.VarType.VECTOR
)
p_soln = uw.discretisation.MeshVariable(r"p", meshball, 1, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=2)
r = uw.discretisation.MeshVariable(r"r", meshball, 1, degree=1)

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

radius_fn = sympy.sqrt(
    meshball.rvec.dot(meshball.rvec)
)  # normalise by outer radius if not 1.0
unit_rvec = meshball.X / (radius_fn + 0.0001)
gravity_fn = radius_fn

# Some useful coordinate stuff

x, y, z = meshball.CoordinateSystem.N
ra, l1, l2 = meshball.CoordinateSystem.R

hw = 1000.0 / res
surface_fn = sympy.exp(-(((r.sym[0] - r_o) / r_o) ** 2) * hw)
base_fn = sympy.exp(-(((r.sym[0] - r_i) / r_o) ** 2) * hw)

x = meshball.N.x
y = meshball.N.y
z = meshball.N.z

with meshball.access(r):
    r.data[:, 0] = uw.function.evaluate(
        sympy.sqrt(x**2 + y**2 + z**2), meshball.data, meshball.N
    )  # cf radius_fn which is 0->1

th = sympy.atan2(y + 1.0e-10, x + 1.0e-10)
ph = sympy.acos(z / (r.fn + 1.0e-10))

hw = 1000.0 / res
surface_fn = sympy.exp(-(((r.fn - r_o) / r_o) ** 2) * hw)

## Inner inner core "mask fn"

iic_fn = 0.5 - 0.5 * sympy.tanh(100.0 * (r.fn - iic_radius))
# -

lons = uw.function.evaluate(th, t_soln.coords, meshball.N)
lats = uw.function.evaluate(ph, t_soln.coords, meshball.N)

# +
ic_raw = np.loadtxt("./qtemp_6000.xyz")  # Data: lon, lat, ?, ?, ?, T
ic_data = ic_raw[:, 5].reshape(181, 361)

## Map heights/ages to the even-mesh grid points


def map_raster_to_mesh(lons, lats, raster):
    latitudes_in_radians = lats
    longitudes_in_radians = lons
    latitudes_in_degrees = np.degrees(latitudes_in_radians)
    longitudes_in_degrees = np.degrees(longitudes_in_radians)

    dlons = longitudes_in_degrees + 180
    dlats = latitudes_in_degrees

    ilons = raster.shape[0] * dlons / 360.0
    ilats = raster.shape[1] * dlats / 180.0

    icoords = np.stack((ilons, ilats))

    from scipy import ndimage

    mvals = ndimage.map_coordinates(raster, icoords, order=3, mode="nearest").astype(
        float
    )

    return mvals


qt_vals = map_raster_to_mesh(lons, lats, ic_data.T)

with meshball.access(t_soln):
    t_soln.data[...] = (
        uw.function.evaluate(
            sympy.sin(radius_fn * np.pi) * radius_fn**2, t_soln.coords, meshball.N
        )
        * qt_vals
    ).reshape(-1, 1)


# +
vtheta = r.sym[0] * sympy.sin(th)

vx = -vtheta * sympy.sin(th)
vy = vtheta * sympy.cos(th)
# +
# Create Stokes solver object

stokes = uw.systems.Stokes(
    meshball,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
    solver_name="stokes",
)

stokes.tolerance = 1.0e-4
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options[
    "snes_max_it"
] = 1  # for timing cases only - force 1 snes iteration for all examples
stokes.penalty = 0.1

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = sympy.sympify(1)

# thermal buoyancy force
buoyancy_force = Rayleigh * gravity_fn * t_soln.sym[0] * 0.001 * (1 - surface_fn)

# Free slip condition by penalizing radial velocity at the surface (non-linear term)
free_slip_penalty_upper = v_soln.sym.dot(unit_rvec) * unit_rvec * surface_fn

stokes.bodyforce = unit_rvec * buoyancy_force
stokes.bodyforce -= 1000000 * free_slip_penalty_upper
stokes.saddle_preconditioner = 1.0

# Velocity boundary conditions
# stokes.add_dirichlet_bc( (0.0, 0.0, 0.0), "Upper", (0,1,2))
stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Centre", (0, 1, 2))
stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Lower", (0, 1, 2))

# +
# stokes._setup_terms()
# -

print(f"Stokes solve ", flush=True)


stokes.solve(zero_init_guess=True)

# +
# speed = v_soln.fn.dot(v_soln.fn)
# _, smean, smin, smax, _, _, srms = meshball.stats(speed)


I = uw.maths.Integral(meshball, sympy.sympify(1))
norm = I.evaluate()

I.fn = v_soln.sym.dot(v_soln.sym)
speed = I.evaluate()

if uw.mpi.rank == 0:
    print("Speed:", speed, norm, speed / norm, flush=True)

I.fn = surface_fn
norm = I.evaluate()

I.fn = v_soln.sym.dot(v_soln.sym) * surface_fn
speed = I.evaluate()

if uw.mpi.rank == 0:
    print("Surface Speed:", speed, norm, speed / norm, flush=True)

# -

ts = 0

## Save velocity / pressure data (step zero only)
meshball.petsc_save_checkpoint(index=0, meshVars=[p_soln, v_soln], outputPath='./output/')


# +
## particle advection in the static field (10 steps)

dt = 1.0 * stokes.estimate_dt()

for step in range(5):
    print(f"Step {ts}", flush=True)
    passive_swarm.advection(v_soln.sym, dt)
    passive_swarm.save(filename=f"{savefile}.passive_swarm.{ts}.h5")

    ts += 1

# +
if mpi4py.MPI.COMM_WORLD.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["S"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(unit_rvec) * (base_fn + surface_fn))
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    
    points = vis.swarm_to_pv_cloud(passive_swarm)
    point_cloud = pv.PolyData(points)
    

    sphere = pv.Sphere(radius=0.9, center=(0.0, 0.0, 0.0))
    clipped = pvmesh.clip_surface(sphere)

    # clipped = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=(0.1, 0, 1), invert=True)
    # -

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="T",
        use_transparency=False,
        opacity=1,
    )

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
    #               use_transparency=False, opacity=1.0)

    pl.add_points(point_cloud, color="Black", point_size=10.0, opacity=0.5)

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=50 / Rayleigh)
    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    pl.show(cpos="xy")

# !ls -trl output | tail
# -


