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


expt_name = "NS_FS_flow_coriolis_disk_500_iii"

# +
import meshio

# meshball = uw.meshes.SphericalShell(
#     dim=2, radius_outer=1.0, radius_inner=0.0, cell_size=0.075, degree=1, verbose=False
# )

meshball = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.0, cellSize=0.05, degree=1, centre=False, verbosity=True)

# +
v_soln = uw.discretisation.MeshVariable("U", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
r = uw.discretisation.MeshVariable("R", meshball, 1, degree=1)


v_soln_1 = uw.discretisation.MeshVariable("U_1", meshball, meshball.dim, degree=2)
vorticity = uw.discretisation.MeshVariable("omega", meshball, 1, degree=1)


# +
swarm = uw.swarm.Swarm(mesh=meshball)
v_star = uw.swarm.SwarmVariable("Vs", swarm, meshball.dim, proxy_degree=3)
remeshed = uw.swarm.SwarmVariable("Vw", swarm, 1, dtype="int", _proxy=False)
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

# r  = sympy.sqrt(x**2+y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

#
Rayleigh = 1.0e2

#
hw = 1000.0 / 0.075
surface_fn = sympy.exp(-(((r.fn - 1.0) / 1.0) ** 2) * hw)
# -
orientation_wrt_z = sympy.atan2(y + 1.0e-10, x + 1.0e-10)
v_rbm_z_x = -r.fn * sympy.sin(orientation_wrt_z) * meshball.N.i
v_rbm_z_y = r.fn * sympy.cos(orientation_wrt_z) * meshball.N.j
v_rbm_z = v_rbm_z_x + v_rbm_z_y

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
navier_stokes._u_star_projector.penalty = 0.0

# Constant visc

navier_stokes.rho = 1.0
navier_stokes.theta = 0.5
navier_stokes.penalty = 0.0
navier_stokes.viscosity = 1.0

# Free slip condition by penalizing radial velocity at the surface (non-linear term)
free_slip_penalty = 1.0e4 * Rayleigh * v_soln.fn.dot(unit_rvec) * unit_rvec * surface_fn

# Velocity boundary conditions

# navier_stokes.add_dirichlet_bc( (0.0, 0.0), "Upper",  (0,1))
# navier_stokes.add_dirichlet_bc( (0.0, 0.0), "Centre", (0,1))

v_theta = (
    navier_stokes.theta * navier_stokes.u.fn
    + (1.0 - navier_stokes.theta) * navier_stokes.u_star_fn
)

# -

nodal_vorticity_from_v = uw.systems.Projection(meshball, vorticity)
nodal_vorticity_from_v.uw_function = sympy.vector.curl(v_soln.fn).dot(meshball.N.k)
nodal_vorticity_from_v.smoothing = 1.0e-3

t_init = sympy.cos(3 * th)

# +
with meshball.access(r):
    r.data[:, 0] = uw.function.evaluate(
        sympy.sqrt(x**2 + y**2), meshball.data
    )  # cf radius_fn which is 0->1

# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:, 0] = uw.function.evaluate(t_init, t_soln.coords)

# +
navier_stokes.bodyforce = Rayleigh * unit_rvec * t_init  # minus * minus
navier_stokes.bodyforce -= free_slip_penalty  # + solid_body_penalty

v_proj = navier_stokes._u_star_projector.u
free_slip_penalty_p = 100 * v_proj.fn.dot(unit_rvec) * unit_rvec * surface_fn
navier_stokes._u_star_projector.F0 = free_slip_penalty_p  # + solid_body_penalty_p)


# +
navier_stokes.solve(timestep=10.0)
nodal_vorticity_from_v.solve()

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
        pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
        pvmesh.point_data["Om"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)

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
            scalars="Om",
            use_transparency=False,
            opacity=0.5,
        )
        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.03)

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

vorticity.fn

for step in range(0, 10):

    Omega_0 = 50.0 * min(ts / 10, 1.0)
    Coriolis = (
        2.0 * Omega_0 * navier_stokes.rho * sympy.vector.cross(meshball.N.k, v_theta)
    )

    navier_stokes.bodyforce = Rayleigh * unit_rvec * t_init  # minus * minus
    navier_stokes.bodyforce -= free_slip_penalty
    navier_stokes.bodyforce -= Coriolis * (1.0 - surface_fn)

    delta_t = 1.0 * navier_stokes.estimate_dt()

    navier_stokes.solve(timestep=delta_t, zero_init_guess=False)
    nodal_vorticity_from_v.solve()

    _, z_ns, _, _, _, _, _ = meshball.stats(v_soln.fn.dot(v_rbm_z))
    print("Rigid body: {}".format(z_ns))

    dv_fn = v_soln.fn - v_soln_1.fn
    _, _, _, _, _, _, deltaV = meshball.stats(dv_fn.dot(dv_fn))

    with meshball.access(v_soln_1):
        v_soln_1.data[...] = v_soln.data[...]

    with swarm.access(v_star):
        v_star.data[...] = (
            0.5 * uw.function.evaluate(v_soln.fn, swarm.data) + 0.5 * v_star.data[...]
        )

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
        print("Timestep {}, dt {}, deltaV {}".format(ts, delta_t, deltaV))

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


navier_stokes._p_f0


# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["Om"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(navier_stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, navier_stokes.u.sym)

    pl = pv.Plotter(window_size=[1000, 1000])

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Om",
        use_transparency=False,
        opacity=0.5,
    )
    pl.add_arrows(arrow_loc, arrow_length, mag=0.05)

    pl.show(cpos="xy")
# -


meshball.stats(
    sympy.vector.cross(Omega, v_soln.fn).dot(sympy.vector.cross(Omega, v_soln.fn))
)

meshball.stats(v_soln.fn.dot(v_rbm_z))

meshball.stats(v_soln.fn.dot(v_soln.fn))

meshball.stats(v_soln.fn.dot(v_rbm_z))

meshball.stats((v_soln.fn + 0.015 * v_rbm_z).dot(v_rbm_z))

meshball.stats(sympy.vector.cross(Omega, v_soln.fn).dot(v_rbm_z))

sympy.vector.cross(Omega, v_soln.fn)

_, z_ns, _, _, _, _, _ = meshball.stats(v_soln.fn.dot(v_rbm_z))
print("Rigid body: {}".format(z_ns))

x_ns_
