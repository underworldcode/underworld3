# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Viscous fingering model
#
# Based on Darcy flow and advection-diffusion of two fluids with varying viscosity.
# From Simpson, 2017, and from Homsy, 1987, fingering patterns develop under certain conditions.
#
# ### Darcy model (quasi-static)
#
# The Darcy equation for the steady-state pressure field can be written
#
# $$\nabla \cdot \left( \boldsymbol\kappa \nabla p - \boldsymbol{s} \right) = 0$$
#
# #### Darcy velocity:
#
# The model from Homsy (1987) generally assumes $\nabla \cdot \mathbf{u} = 0$ which is equivalent to the Darcy flow equation with $\boldsymbol{s}=0$
# and using
#
# $$\mathbf{u} = - \frac{k}{\mu_c}\nabla p$$
#
#
# #### viscosity:
# $$\mu_c = \left( \frac{c}{\mu_o^{{1}/{4}}} +  \frac{1-c}{\mu_s^{{1}/{4}}} \right)^{-4}$$
#
# #### Advection-diffusion of material:
#
# $$\varphi \frac{\partial c}{\partial t} + \varphi (\mathbf{u} \cdot \nabla) c= \nabla(\kappa\nabla c)$$
#
# If the diffusion coefficient is small, then it is often more appropriate to assume pure transport
#
# $$\varphi \frac{D c}{D t} = \nabla(\kappa\nabla c) \approx 0$$
#
# ##### Model physical parameters:
#
# | parameter | symbol  | value  | units  |   |
# |---|---|---|---|---|
# | x |  | $$10$$  | $$m$$  |   |
# | y  |  | $$10$$  | $$m$$  |   |
# | permeability  | $$k$$ | $$10^{-13}$$  | $$m^2$$  |   |
# | porosity  | $$\varphi$$ | $$0.1$$ |   |   |
# | diffusivity  | $$\kappa$$  | $$10^{-9}$$  | $$m^2 s^{-1}$$  |   |
# | viscosity (solvent)  | $$\mu{_s}$$ | $$1.33{\cdot}10^{-4}$$  | $$Pa s$$  |   |
# | viscosity (oil)  | $$\mu{_o}$$ | $$20\eta_s$$  | $$Pa s$$  |   |
# | pressure  | $$p$$  | $$10^{5}$$  | $$Pa$$  |   |
#
#
# ## References
#
# Homsy, G. M. (1987). Viscous Fingering in Porous Media. Annual Review of Fluid Mechanics, 19(1), 271–311. https://doi.org/10.1146/annurev.fl.19.010187.001415
#
# [Guy Simpson - Practical Finite Element Modeling in Earth Science using Matlab (2017)](https://www.wiley.com/en-au/Practical+Finite+Element+Modeling+in+Earth+Science+using+Matlab-p-9781119248620)
#
#
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

# from scipy.interpolate import griddata, interp1d

import matplotlib.pyplot as plt

import os

options = PETSc.Options()

# +
outputDir = "./output/viscousFingering_example/"

if uw.mpi.rank == 0:
    ### create folder if required
    os.makedirs(outputDir, exist_ok=True)

# +
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim = uw.scaling.dimensionalise

refLength = 10  ### length and height of box in meters
g = 9.81
eta = 1.33e-4
kappa = 1e-9  ### m^2/s
perm = 1e-13  ### m^2
porosity = 0.1
T_0 = 273.15
T_1 = 1573.15
dT = T_1 - T_0
rho0 = 1e3


refTime = perm / kappa
refViscosity = eta * u.pascal * u.second

KL = refLength * u.meter
KL = 1.0 * u.millimetre
KT = dT * u.kelvin
Kt = refTime * u.seconds
Kt = 0.01 * u.year
KM = refViscosity * KL * Kt

### create unit registry
scaling_coefficients = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"] = KM
scaling_coefficients["[temperature]"] = KT
scaling_coefficients

# +
minX, maxX = 0, nd(10 * u.meter)
minY, maxY = 0, nd(10 * u.meter)

elements = 25

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize=maxY / elements, qdegree=5
)

vizmesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize=0.5 * maxY / elements, qdegree=1
)

p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
mat = uw.discretisation.MeshVariable("mat", mesh, 1, degree=3, continuous=True)

# x and y coordinates
x = mesh.N.x
y = mesh.N.y


# +

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
    )

    pl.show(cpos="xy")

# +
# Create Darcy Solver

darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=p_soln, v_Field=v_soln)
darcy.petsc_options.delValue("ksp_monitor")
darcy.petsc_options[
    "snes_rtol"
] = 1.0e-6  # Needs to be smaller than the contrast in properties
darcy.constitutive_model = uw.constitutive_models.DiffusionModel
# -


darcy

#
# $$
# \color{Green}{\mathbf{f}_{0}}  -
# \nabla \cdot
#         \color{Blue}{{\mathbf{f}_{1}}} =
#         \color{Maroon}{\underbrace{\Bigl[ W \Bigl] }_{\mathbf{f}_{s}}}
# $$
#

# +
swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=5)

material = swarm.add_variable(name="M", size=1, proxy_degree=mat.degree)
conc = swarm.add_variable(name="C", size=1, proxy_degree=mat.degree)

swarm.populate(fill_param=4)

# +
adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh=mesh, u_Field=mat, V_fn=v_soln, #DuDt=conc.sym[0]
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
# -

# ### Random material distribution along the interface

# +
np.random.seed(100)

### on the mesh

with mesh.access(mat):
    x0 = nd(2.5 * u.meter)
    dx = max(mesh.get_min_radius(), nd(0.1 * u.meter))

    fluctuation = nd(0.01 * u.meter) * np.cos(
        mat.coords[:, 1] / nd(0.5 * u.meter) * np.pi
    )
    fluctuation += nd(0.01 * u.meter) * np.cos(
        mat.coords[:, 1] / nd(2.0 * u.meter) * np.pi
    )
    fluctuation += nd(0.05 * u.meter) * np.random.random(size=mat.coords.shape[0])

    mat.data[...] = 0
    mat.data[mat.coords[:, 0] + fluctuation < x0] = 1

# ### on the swarm

with swarm.access(material):
    # material.data[:,0] = mat.rbf_interpolate(new_coords=material.swarm.data, nnn=1)[:,0]
    material.data[:, 0] = uw.function.evalf(mat.sym, swarm.particle_coordinates.data)

# -
if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis
    
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["mat"] = vis.scalar_fn_to_pv_points(pvmesh, mat.sym)

    
    points = vis.swarm_to_pv_cloud(swarm)
    point_cloud = pv.PolyData(points)

    with swarm.access(material):
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=10,
        opacity=0.33,
    )

    pl.show(cpos="xy")


eta_s = nd(1.33e-4 * u.pascal * u.second)
eta_o = 20 * eta_s

# +
### use the mesh var to map composition to viscosity
## eta_fn = (mat.sym[0]/eta_s**0.25+(1-mat.sym[0])/eta_o**0.25)**(-4)

### use the swarm var to map composition to viscosity
eta_fn = (material.sym[0] / eta_s**0.25 + (1 - material.sym[0]) / eta_o**0.25) ** (
    -4
)


# +
nd_perm = nd(perm * u.meter**2)

diffusivity_fn = nd_perm / eta_fn

darcy.constitutive_model.Parameters.diffusivity = diffusivity_fn
# -

# #### Darcy velocity:
# $$ u = - \frac{k}{\mu_c}\nabla p$$

adv_diff.constitutive_model.Parameters.diffusivity = nd(1e-9 * u.meter**2 / u.second)

# +
p0_nd = nd(0.1e6 * u.pascal)
# p_dx = p0_nd * (1 - mesh.X[0])

# with mesh.access(p_soln):
#     p_soln.data[:,0] = uw.function.evaluate(p_dx, p_soln.coords, mesh.N)

# +
## Make sure additional terms are set to zero
darcy.f = 0.0
darcy.s = sympy.Matrix([0, 0]).T

### set up boundary conditions for the Darcy solver
darcy.add_dirichlet_bc(p0_nd, "Left")
darcy.add_dirichlet_bc(0.0, "Right")

### set up boundary conditions for the adv diffusion solver
adv_diff.add_dirichlet_bc(1.0, "Left")
adv_diff.add_dirichlet_bc(0.0, "Right")

# Zero pressure gradient at sides / base (implied bc)

# darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-6
# darcy._v_projector.smoothing = 1e24
# darcy._v_projector.add_dirichlet_bc(0.0, "Left", 0)
# darcy._v_projector.add_dirichlet_bc(0.0, "Right", 0)
# -
darcy.solve()

time = 0
step = 0


# +
finish_time = 0.01 * u.year

# while time < nd(finish_time):

for iteration in range(0, 20):
    if uw.mpi.rank == 0:
        print(f"\n\nstep: {step}, time: {dim(time, u.year)}")

    if step % 5 == 0:
        mesh.write_timestep(
            "viscousFinger",
            meshUpdates=False,
            meshVars=[p_soln, v_soln, mat],
            outputPath=outputDir,
            index=step,
        )

    ### get the Darcy velocity from the darcy solve
    darcy.solve(zero_init_guess=True)

    dt = ndim(0.0002 * u.year)

    ### do the advection-diffusion
    # adv_diff.solve(timestep=dt)

    ### update swarm / swarm variables
    # with swarm.access(material):
    # material.data[:,0] = mat.rbf_interpolate(new_coords=material.swarm.data, nnn=1)[:,0]
    # material.data[:,0] = uw.function.evaluate(mat.sym, swarm.particle_coordinates.data)

    ### advect the swarm
    swarm.advection(
        V_fn=v_soln.sym
        * sympy.Matrix.diag(1 / porosity, 1 / sympy.sympify(1000000000)),
        delta_t=dt,
        order=2,
        evalf=True,
    )

    # with mesh.access(v_soln):
    #     ## Divide by the porosity to get the actual velocity
    #     v_soln.data[:,] *= porosity

    I = uw.maths.Integral(mesh, sympy.sqrt(v_soln.sym.dot(v_soln.sym)))
    Vrms = I.evaluate()
    I.fn = 1.0
    Vrms /= I.evaluate()

    if uw.mpi.rank == 0:
        print(f"V_rms = {Vrms} ... delta t = {dt}.  dL = {Vrms * dt}")

    step += 1
    time += dt

    if time > nd(finish_time):
        break


# -

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis


    pvmesh = vis.mesh_to_pv_mesh(vizmesh)
    pvmesh.point_data["mat"] = vis.scalar_fn_to_pv_points(pvmesh, material.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)/vis.vector_fn_to_pv_points(velocity_points, v_soln.sym).max()
    
    points = vis.swarm_to_pv_cloud(swarm)
    point_cloud = pv.PolyData(points)

    with swarm.access(material):
        point_cloud.point_data["M"] = material.data.copy()


    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh, style="surface", cmap="coolwarm", edge_color="Grey", scalars="P",
                show_edges=False, use_transparency=False, opacity=1)

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=1250, opacity=1)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=2,
        opacity=0.66,
    )

    pl.show(cpos="xy")

mat.stats()
