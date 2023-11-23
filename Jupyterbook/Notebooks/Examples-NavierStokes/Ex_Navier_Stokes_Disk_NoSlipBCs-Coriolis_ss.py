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

# # Cylindrical Stokes with Coriolis term (out of plane)
#
# We implement Stokes/Navier-Stokes flow in a disc with rigid bc's and take into account the effect of Coriolis terms
# $2\Omega \times \mathbf{u}$.
#
# The non-linear implementation in Stokes flow relies on the Newton (SNES) solver but the structure of the solution (with strong Coriolis terms)
# is very different from the Stokes flow pattern (especially the pressure gradients) and so we instead set up a Navier-Stokes problem in which
# we can down-weight the inertial effects but retain a time-evolution terms to approach the solution. In this example, I am using the same
# trick as for the SS benchmark case where I try to suppress the time dependent term in the Navier-Stokes equation and replace timesteps with
# pseudo timesteps.
#
#
# This works best if we spin-up the rotation gradually.
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy

# -


expt_name = "SS_NS_flow_coriolis_10"

# +
import meshio

# meshball = uw.meshes.SphericalShell(
#     dim=2, radius_outer=1.0, radius_inner=0.0, cell_size=0.05, degree=1, verbose=True
# )

meshball = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.0, cellSize=0.05, degree=1, centre=False, verbosity=True)

# +
v_soln = uw.discretisation.MeshVariable("U", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)

v_soln_1 = uw.discretisation.MeshVariable("U_1", meshball, meshball.dim, degree=2)
vorticity = uw.discretisation.MeshVariable("omega", meshball, 1, degree=1)


# +
swarm = uw.swarm.Swarm(mesh=meshball)
v_star = uw.swarm.SwarmVariable("Vs", swarm, meshball.dim, proxy_degree=3)
remeshed = uw.swarm.SwarmVariable("Vw", swarm, 1, proxy_degree=3, dtype="int")
X_0 = uw.swarm.SwarmVariable("X0", swarm, meshball.dim, _proxy=False)

swarm.populate(fill_param=4)

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(
    meshball.rvec.dot(meshball.rvec)
)  # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10 + radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff

x = meshball.N.x
y = meshball.N.y

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

#

Rayleigh = 1.0e2
# +
# Surface-drive flow, use this bc

# vtheta = r * sympy.sin(th)
# vx = -vtheta*sympy.sin(th)
# vy =  vtheta*sympy.cos(th)

# +
# Create NS object

navier_stokes = uw.systems.NavierStokesSwarm(
    meshball,
    velocityField=v_soln,
    pressureField=p_soln,
    velocityStar_fn=v_star.fn,
    u_degree=v_soln.degree,
    p_degree=p_soln.degree,
    rho=1.0,
    theta=0.5,
    verbose=False,
    projection=True,
    solver_name="navier_stokes",
)

navier_stokes.petsc_options.delValue(
    "ksp_monitor"
)  # We can flip the default behaviour at some point
navier_stokes._u_star_projector.petsc_options.delValue("ksp_monitor")
navier_stokes._u_star_projector.petsc_options["snes_rtol"] = 1.0e-2
navier_stokes._u_star_projector.petsc_options["snes_type"] = "newtontr"
navier_stokes._u_star_projector.smoothing = 0.0  # navier_stokes.viscosity * 1.0e-6
navier_stokes._u_star_projector.penalty = 0.0001

# Here we replace the time dependence with the steady state advective transport term
# to lean towards steady state solutions

navier_stokes.UF0 = (
    -navier_stokes.rho * (v_soln.fn - v_soln_1.fn) / navier_stokes.delta_t
)

# Constant visc

navier_stokes.rho = 1000.0
navier_stokes.theta = 0.5
navier_stokes.penalty = 0.0
navier_stokes.viscosity = 1.0
navier_stokes.bodyforce = 1.0e-32 * meshball.N.i
navier_stokes._Ppre_fn = 1.0 / (
    navier_stokes.viscosity + navier_stokes.rho / navier_stokes.delta_t
)

# Velocity boundary conditions

navier_stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Centre", (0, 1))

v_theta = (
    navier_stokes.theta * navier_stokes.u.fn
    + (1.0 - navier_stokes.theta) * navier_stokes.u_star_fn
)
# -

t_init = sympy.cos(3 * th)

# +
# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:, 0] = uw.function.evaluate(t_init, t_soln.coords)
    print(t_soln.data.min(), t_soln.data.max())
# -
navier_stokes.bodyforce = Rayleigh * unit_rvec * t_init  # minus * minus

# +
navier_stokes.solve(timestep=10.0)

with meshball.access():
    v_inertial = v_soln.data.copy()

with swarm.access(v_star, remeshed, X_0):
    v_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data)
    X_0.data[...] = swarm.data[...]

# -

swarm.advection(v_soln.fn, delta_t=navier_stokes.estimate_dt(), corrector=False)


# +
# check the mesh if in a notebook / serial


def plot_V_mesh(filename):

    if uw.mpi.size == 1:

        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshball)
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
        
        velocity_points = vis.meshVariable_to_pv_cloud(navier_stokes.u)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, navier_stokes.u.sym)

        pl = pv.Plotter(window_size=(1000, 750))
        pl.camera.SetPosition(0.0001, 0.0001, 4.0)

        # pl.add_mesh(pvmesh,'Black', 'wireframe')
        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            use_transparency=False,
            opacity=0.5,
        )
        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.05)

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(2560, 2560),
            return_img=False,
        )

        pl.close()

        del pl


# -


ts = 0
swarm_loop = 5


# +

for step in range(0, 50):

    Omega = 10.0 * meshball.N.k * min(ts / 25, 1.0)
    navier_stokes.bodyforce = Rayleigh * unit_rvec * t_init  # minus * minus
    navier_stokes.bodyforce -= (
        2.0 * navier_stokes.rho * sympy.vector.cross(Omega, v_theta)
    )

    delta_t = 10.0 * navier_stokes.estimate_dt()

    navier_stokes.solve(timestep=delta_t, zero_init_guess=False)

    dv_fn = v_soln.fn - v_soln_1.fn
    _, _, _, _, _, _, deltaV = meshball.stats(dv_fn.dot(dv_fn))

    with meshball.access(v_soln_1):
        v_soln_1.data[...] = 0.5 * v_soln_1.data[...] + 0.5 * v_soln.data[...]

    with swarm.access(v_star):
        v_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data)

    swarm.advection(v_soln.fn, delta_t=delta_t, corrector=False)

    # Restore a subset of points to start
    offset_idx = step % swarm_loop

    with swarm.access(swarm.particle_coordinates, remeshed):
        remeshed.data[...] = 0
        remeshed.data[offset_idx::swarm_loop, :] = 1
        swarm.data[offset_idx::swarm_loop, :] = X_0.data[offset_idx::swarm_loop, :]

    # re-calculate v history for remeshed particles
    # Note, they may have moved procs after the access manager closed
    # so we re-index

    with swarm.access(v_star, remeshed):
        idx = np.where(remeshed.data == 1)[0]
        v_star.data[idx] = uw.function.evaluate(v_soln.fn, swarm.data[idx])

    if uw.mpi.rank == 0:
        print(
            "Iteration (pseudo timestep) {}, dt {}, deltaV {}".format(
                ts, delta_t, deltaV
            )
        )

    if ts % 1 == 0:
        # nodal_vorticity_from_v.solve()
        plot_V_mesh(filename="output/{}_step_{}".format(expt_name, ts))

        # savefile = "output/{}_ts_{}.h5".format(expt_name,step)
        # meshball.save(savefile)
        # v_soln.save(savefile)
        # p_soln.save(savefile)
        # vorticity.save(savefile)
        # meshball.generate_xdmf(savefile)

    navier_stokes._u_star_projector.smoothing = navier_stokes.viscosity * 1.0e-6

    ts += 1


# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(navier_stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, navier_stokes.u.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=0.5,
    )
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.033)

    pl.show(cpos="xy")
# -


((v_inertial - usol) ** 2).mean()

v_inertial.max()

#

# #
