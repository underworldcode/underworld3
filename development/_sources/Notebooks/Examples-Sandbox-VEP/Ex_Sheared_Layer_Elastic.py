# ---
# jupyter:
#   jupytext:
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

# # Validate constitutive models
#
# Simple shear with material defined by particle swarm (based on inclusion model), position, pressure, strain rate etc.  Check the implementation of the Jacobians using various non-linear terms.
#
# Check elastic stress terms
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
import numpy as np
import sympy
import pyvista as pv
import vtk

from underworld3 import timing

resolution = uw.options.getReal("model_resolution", default=0.05)
mu = uw.options.getInt("mu", default=0.5)
maxsteps = uw.options.getInt("max_steps", default=500)


## Define units here and physical timestep numbers etc.

observation_timescale = 0.01


# +
# Mesh a 2D pipe with a circular hole

mesh1 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.5, -0.5),
    maxCoords=(+1.5, +0.5),
    cellSize=resolution,
)


# +

mesh1.dm.view()

## build periodic mesh (mesh1)
# uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
#     mesh1.dm, [0.1, 0.0], [-1.5, 0.0], [1.5, 0.0])

# mesh1.dm.view()
# -

v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable(
    "P", mesh1, 1, vtype=uw.VarType.SCALAR, degree=1, continuous=True
)
Stress = uw.discretisation.MeshVariable(
    r"Stress",
    mesh1,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    degree=2,
    continuous=True,
    varsymbol=r"{\sigma}",
)
work = uw.discretisation.MeshVariable(
    "W", mesh1, 1, vtype=uw.VarType.SCALAR, degree=2, continuous=True
)
strain_rate_inv2 = uw.discretisation.MeshVariable(
    "eps_dot", mesh1, 1, degree=2, varsymbol=r"{\dot\varepsilon}"
)
strain_rate_inv2_pl = uw.discretisation.MeshVariable(
    "eps_dot_pl", mesh1, 1, degree=2, varsymbol=r"{\dot\varepsilon_{pl}}"
)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=2)

mesh1.view()

# +
swarm = uw.swarm.Swarm(mesh=mesh1, recycle_rate=5)

material = uw.swarm.SwarmVariable(
    "M",
    swarm,
    size=1,
    vtype=uw.VarType.SCALAR,
    proxy_continuous=True,
    proxy_degree=1,
    dtype=int,
)

strain = uw.swarm.SwarmVariable(
    "Strain",
    swarm,
    size=1,
    vtype=uw.VarType.SCALAR,
    proxy_continuous=True,
    proxy_degree=2,
    varsymbol=r"\varepsilon",
    dtype=float,
)

stress_star_p = uw.swarm.SwarmVariable(
    r"stress_p",
    swarm,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    proxy_continuous=True,
    proxy_degree=2,
    varsymbol=r"{\sigma^{*}_{p}}",
)

swarm.populate(fill_param=2)

stress_star_update_dt = uw.swarm.Lagrangian_Updater(
    swarm, Stress.sym, [stress_star_p], dt_physical=observation_timescale
)
# -

# Some useful coordinate stuff
x, y = mesh1.X


with swarm.access(strain, material), mesh1.access():
    XX = swarm.particle_coordinates.data[:, 0]
    YY = swarm.particle_coordinates.data[:, 1]
    mask = (1.0 - (YY * 2) ** 8) * (1 - (2 * XX / 3) ** 6)
    material.data[(XX**2 + YY**2 < 0.01), 0] = 1
    strain.data[:, 0] = (
        0.01 * np.random.random(swarm.particle_coordinates.data.shape[0]) * mask
    )
# +
# Create Solver object

stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
    solver_name="stokes",
)

viscosity_L = sympy.Piecewise(
    (1, material.sym[0] > 0.5),
    (1000, True),
)


# -

stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
    u=v_soln, flux_dt=stress_star_update_dt
)

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_L
stokes.constitutive_model.Parameters.shear_modulus = sympy.sympify(100)
stokes.constitutive_model.Parameters.stress_star = stress_star_p.sym
stokes.constitutive_model.Parameters.dt_elastic = sympy.sympify(observation_timescale)


stokes.constitutive_model

sigma_projector = uw.systems.Tensor_Projection(
    mesh1, tensor_Field=Stress, scalar_Field=work
)
sigma_projector.uw_function = stokes.stress_1d

# +
nodal_strain_rate_inv2 = uw.systems.Projection(
    mesh1, strain_rate_inv2, solver_name="edot_II"
)

nodal_strain_rate_inv2.uw_function = stokes._Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-3

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2, solver_name="stress_II")
nodal_tau_inv2.uw_function = 2 * stokes.constitutive_model.viscosity * stokes._Einv2
nodal_tau_inv2.smoothing = 1.0e-3


# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

stokes.penalty = 1.0
stokes.tolerance = 1.0e-4

# Velocity boundary conditions

stokes.add_dirichlet_bc((0.0, 0.0), "Inclusion", (0, 1))
stokes.add_dirichlet_bc((1.0, 0.0), "Top", (0, 1))
stokes.add_dirichlet_bc((-1.0, 0.0), "Bottom", (0, 1))
stokes.add_dirichlet_bc((0.0), "Left", (1))
stokes.add_dirichlet_bc((0.0), "Right", (1))

# -
stress_star_update_dt.psi_star[0].sym

stokes.solve()

stokes.constitutive_model.Parameters.strainrate_inv_II_min = 0.00001
stokes.constitutive_model.Parameters.yield_stress = 50

stokes

stokes.stress[0, 0]

# +
nodal_strain_rate_inv2.solve()

sigma_projector.uw_function = stokes.stress_deviator
sigma_projector.solve()
# -
with swarm.access(stress_star_p), mesh1.access():
    stress_star_p.data[
        ...
    ] = 0.0  # Stress.rbf_interpolate(swarm.particle_coordinates.data)


timing.reset()
timing.start()

print("Setup terms", flush=True)

stokes._setup_terms()

stokes.stress[0, 0]


stokes.solve(zero_init_guess=False, verbose=True)
timing.print_table(display_fraction=1)
print(stokes._u.max(), stokes._p.max())


# +
nodal_strain_rate_inv2.uw_function = stokes._Einv2
nodal_strain_rate_inv2.solve()

S = stokes.stress_deviator
nodal_tau_inv2.uw_function = stokes.constitutive_model.viscosity * 2 * stokes._Einv2
nodal_tau_inv2.solve()
# -

stokes.constitutive_model.flux_dt

# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvpoints = pvmesh.points[:, 0:2]
    usol = v_soln.rbf_interpolate(pvpoints)

    pvmesh.point_data["P"] = p_soln.rbf_interpolate(pvpoints)
    pvmesh.point_data["Edot"] = strain_rate_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strs"] = dev_stress_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["Mat"] = material.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strn"] = strain._meshVar.rbf_interpolate(pvpoints)
    pvmesh.point_data["SStar"] = stress_star_p._meshVar.rbf_interpolate(pvpoints)

    # Velocity arrows

    v_vectors = np.zeros_like(pvmesh.points)
    v_vectors[:, 0:2] = v_soln.rbf_interpolate(pvpoints)

    # Points (swarm)

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]
        point_cloud = pv.PolyData(points)
        point_cloud.point_data["strain"] = strain.data[:, 0]

    pl = pv.Plotter(window_size=(500, 500))

    pl.add_arrows(pvmesh.points, v_vectors, mag=0.1, opacity=0.75)
    # pl.camera_position = "xy"

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        # clim=[0.0,1.0],
        scalars="Strs",
        use_transparency=False,
        opacity=0.5,
    )

    # pl.add_points(point_cloud, colormap="coolwarm", scalars="strain", point_size=10.0, opacity=0.5)

    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.show()


# -
def return_points_to_domain(coords):
    new_coords = coords.copy()
    new_coords[:, 0] = (coords[:, 0] + 1.5) % 3 - 1.5
    return new_coords


ts = 0

stress_star_update_dt.view()

# +
expt_name = f"shear_band_sw_nonp_{mu}"

for step in range(0, 75):
    stokes.solve(zero_init_guess=False)

    delta_t = 0.01

    nodal_strain_rate_inv2.uw_function = sympy.Max(
        0.0,
        stokes._Einv2
        - 0.5
        * stokes.constitutive_model.Parameters.yield_stress
        / stokes.constitutive_model.Parameters.shear_viscosity_0,
    )
    nodal_strain_rate_inv2.solve()

    with mesh1.access(strain_rate_inv2_pl):
        strain_rate_inv2_pl.data[...] = strain_rate_inv2.data.copy()

    nodal_strain_rate_inv2.uw_function = stokes._Einv2
    nodal_strain_rate_inv2.solve()

    S = stokes.stress_deviator
    nodal_tau_inv2.uw_function = sympy.simplify(sympy.sqrt(((S**2).trace()) / 2))
    nodal_tau_inv2.solve()

    if uw.mpi.rank == 0:
        print(f"Stress Inv II -  {dev_stress_inv2.mean()}")

    sigma_projector.solve()
    stress_star_update_dt.update(dt=delta_t, evalf=True)

    with swarm.access(strain), mesh1.access():
        XX = swarm.particle_coordinates.data[:, 0]
        YY = swarm.particle_coordinates.data[:, 1]
        mask = (2 * XX / 3) ** 4  # * 1.0 - (YY * 2)**8
        strain.data[:, 0] += (
            delta_t * mask * strain_rate_inv2_pl.rbf_interpolate(swarm.data)[:, 0]
            - 0.1 * delta_t
        )
        strain_dat = (
            delta_t * mask * strain_rate_inv2_pl.rbf_interpolate(swarm.data)[:, 0]
        )

        if uw.mpi.rank == 0:
            print(f"Sstar[0,0]     = {(np.sqrt(stress_star_p[0,0].data[:]**2)).mean()}")
            print(f"Sstar[1,0]     = {(np.sqrt(stress_star_p[0,1].data[:]**2)).mean()}")
            print(f"Sstar[1,1]     = {(np.sqrt(stress_star_p[1,1].data[:]**2)).mean()}")

    mesh1.write_timestep(
        expt_name,
        meshUpdates=False,
        meshVars=[p_soln, v_soln, strain_rate_inv2_pl],
        outputPath="output",
        index=ts,
    )

    swarm.save(f"{expt_name}.swarm.{ts}.h5")
    strain.save(f"{expt_name}.strain.{ts}.h5")

    # Update the swarm locations
    swarm.advection(
        v_soln.sym, delta_t=delta_t, restore_points_to_domain_func=None, evalf=True
    )

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))

    ts += 1
# -


stokes.constitutive_model.stress_projection()[0, 0]

stokes.stress[0, 0]

# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvpoints = pvmesh.points[:, 0:2]
    usol = v_soln.rbf_interpolate(pvpoints)

    pvmesh.point_data["P"] = p_soln.rbf_interpolate(pvpoints)
    pvmesh.point_data["Edot"] = strain_rate_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strs"] = dev_stress_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["Mat"] = material.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strn"] = strain._meshVar.rbf_interpolate(pvpoints)
    pvmesh.point_data["SStar"] = uw.function.evalf(stress_star_p[1, 1].sym, pvpoints)

    # Velocity arrows

    v_vectors = np.zeros_like(pvmesh.points)
    v_vectors[:, 0:2] = v_soln.rbf_interpolate(pvpoints)

    # Points (swarm)

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]
        point_cloud = pv.PolyData(points)
        point_cloud.point_data["strain"] = strain.data[:, 0]

    pl = pv.Plotter(window_size=(500, 500))

    pl.add_arrows(pvmesh.points, v_vectors, mag=0.1, opacity=0.75)
    # pl.camera_position = "xy"

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        # clim=[0.0,1.0],
        scalars="Mat",
        use_transparency=False,
        opacity=0.5,
    )

    # pl.add_points(point_cloud, colormap="coolwarm", scalars="strain", point_size=10.0, opacity=0.5)

    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.show()
# -
stress_star_p._meshVar.min()

nodal_tau_inv2.snes.cancelMonitor()


0 / 0

# +
# Adams / Bashforth & Adams Moulton ...

s = sympy.Symbol(r"\sigma")
s1 = sympy.Symbol(r"\sigma^*")
s2 = sympy.Symbol(r"\sigma^**")
dt = sympy.Symbol(r"\Delta t")
mu = sympy.Symbol(r"\mu")
eta = sympy.Symbol(r"\eta")
edot = sympy.Symbol(r"\dot\varepsilon")
tr = sympy.Symbol(r"t_r")

sdot1 = (s - s1) / dt
sdot2 = (3 * s - 4 * s1 + s2) / (2 * dt)
# -


display(sdot1)
display(sdot2)
Seq1 = sympy.Equality(sympy.simplify(sdot1 / (2 * mu) + s / (2 * eta)), edot)
display(Seq1)
sympy.simplify(sympy.solve(Seq1, s)[0])

eta_eff_1 = sympy.simplify(eta * mu * dt / (mu * dt + eta))
display(eta_eff_1)
a = sympy.simplify(2 * eta * sympy.solve(Seq1, s)[0] / (2 * eta_eff_1))
tau_1 = a.subs(eta / mu, tr)
tau_1

Seq2 = sympy.Equality(sympy.simplify(sdot2 / (2 * mu) + s / (2 * eta)), edot)
display(Seq2)
sympy.simplify(sympy.solve(Seq2, s)[0])

eta_eff_2 = sympy.simplify(2 * eta * mu * dt / (2 * mu * dt + 3 * eta))
display(eta_eff_2)
sympy.simplify(2 * eta * sympy.solve(Seq2, s)[0] / (2 * eta_eff_2))

a = sympy.simplify(2 * eta * sympy.solve(Seq2, s)[0] / (2 * eta_eff_2))
tau_2 = a.expand().subs(eta / mu, tr)

tau_2

# +
# 0/0
# -


stokes.constitutive_model.Parameters.yield_stress.subs(
    ((strain.sym[0], 0.25), (y, 0.0))
)

stokes.constitutive_model.Parameters.viscosity


def return_points_to_domain(coords):
    new_coords = coords.copy()
    new_coords[:, 0] = (coords[:, 0] + 1.5) % 3 - 1.5
    return new_coords


ts = 0

# +
expt_name = f"output/shear_band_sw_nonp_{mu}"

for step in range(0, 10):
    stokes.solve(zero_init_guess=False)

    delta_t = stokes.estimate_dt()

    nodal_strain_rate_inv2.uw_function = sympy.Max(
        0.0,
        stokes._Einv2
        - 0.5
        * stokes.constitutive_model.Parameters.yield_stress
        / stokes.constitutive_model.Parameters.shear_viscosity_0,
    )
    nodal_strain_rate_inv2.solve()

    with mesh1.access(strain_rate_inv2_pl):
        strain_rate_inv2_pl.data[...] = strain_rate_inv2.data.copy()

    nodal_strain_rate_inv2.uw_function = stokes._Einv2
    nodal_strain_rate_inv2.solve()

    with swarm.access(strain), mesh1.access():
        XX = swarm.particle_coordinates.data[:, 0]
        YY = swarm.particle_coordinates.data[:, 1]
        mask = (2 * XX / 3) ** 4  # * 1.0 - (YY * 2)**8
        strain.data[:, 0] += (
            delta_t * mask * strain_rate_inv2_pl.rbf_interpolate(swarm.data)[:, 0]
            - 0.1 * delta_t
        )
        strain_dat = (
            delta_t * mask * strain_rate_inv2_pl.rbf_interpolate(swarm.data)[:, 0]
        )
        print(
            f"dStrain / dt = {delta_t * (mask * strain_rate_inv2_pl.rbf_interpolate(swarm.data)[:,0]).mean()}, {delta_t}"
        )

    mesh1.write_timestep_xdmf(
        f"{expt_name}",
        meshUpdates=False,
        meshVars=[p_soln, v_soln, strain_rate_inv2_pl],
        swarmVars=[strain],
        index=ts,
    )

    swarm.save(f"{expt_name}.swarm.{ts}.h5")
    strain.save(f"{expt_name}.strain.{ts}.h5")

    # Update the swarm locations
    swarm.advection(v_soln.sym, delta_t=delta_t, restore_points_to_domain_func=None)

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))

    ts += 1



# +
# nodal_visc_calc.uw_function = sympy.log(stokes.constitutive_model.Parameters.viscosity)
# nodal_visc_calc.solve()

# yield_stress_calc.uw_function = stokes.constitutive_model.Parameters.yield_stress
# yield_stress_calc.solve()

nodal_tau_inv2.uw_function = (
    2 * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
)
nodal_tau_inv2.solve()

# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvpoints = pvmesh.points[:, 0:2]
    usol = v_soln.rbf_interpolate(pvpoints)

    pvmesh.point_data["P"] = p_soln.rbf_interpolate(pvpoints)
    pvmesh.point_data["Edot"] = strain_rate_inv2.rbf_interpolate(pvpoints)
    # pvmesh.point_data["Visc"] = np.exp(node_viscosity.rbf_interpolate(pvpoints))
    pvmesh.point_data["Edotp"] = strain_rate_inv2_pl.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strs"] = dev_stress_inv2.rbf_interpolate(pvpoints)
    # pvmesh.point_data["StrY"] =  yield_stress.rbf_interpolate(pvpoints)
    # pvmesh.point_data["dStrY"] = pvmesh.point_data["StrY"] - 2 *  pvmesh.point_data["Visc"] * pvmesh.point_data["Edot"]
    pvmesh.point_data["Mat"] = material.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strn"] = strain._meshVar.rbf_interpolate(pvpoints)

    # Velocity arrows

    v_vectors = np.zeros_like(pvmesh.points)
    v_vectors[:, 0:2] = v_soln.rbf_interpolate(pvpoints)

    # Points (swarm)

    with swarm.access():
        plot_points = np.where(strain.data > 0.0001)
        strain_data = strain.data.copy()

        points = np.zeros((swarm.data[plot_points].shape[0], 3))
        points[:, 0] = swarm.data[plot_points[0], 0]
        points[:, 1] = swarm.data[plot_points[0], 1]
        point_cloud = pv.PolyData(points)
        point_cloud.point_data["strain"] = strain.data[plot_points]

    pl = pv.Plotter(window_size=(500, 500))

    # pl.add_arrows(pvmesh.points, v_vectors, mag=0.1, opacity=0.75)
    # pl.camera_position = "xy"

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        # clim=[-1.0,1.0],
        scalars="Edotp",
        use_transparency=False,
        opacity=0.5,
    )

    pl.add_points(
        point_cloud,
        colormap="Oranges",
        scalars="strain",
        point_size=10.0,
        opacity=0.0,
        # clim=[0.0,0.2],
    )

    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.show()
# -

pvmesh.point_data["Strn"].shape

import matplotlib.pyplot as plt

plt.scatter(mesh1.data[:, 0], mesh1.data[:, 1], c=pvmesh.point_data["Strn"])


with swarm.access():
    print(strain.data.max())

strain_rate_inv2_pl.rbf_interpolate(mesh1.data).max()


# ##

mesh1._search_lengths
