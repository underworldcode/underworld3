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

# # Multiple materials - Drips and Blobs
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

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

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

meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / 24.0,
    regular=True,
    qdegree=2,
)


# +
# meshbox.quadrature.view()
# -

meshbox.dm.view()

# +
# Some useful coordinate stuff

x, y = meshbox.CoordinateSystem.X

# -

v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)


swarm = uw.swarm.Swarm(mesh=meshbox)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=4, proxy_degree=1)
swarm.populate(fill_param=4)


# +
blobs = np.array(
    [
        [0.25, 0.75, 0.1, 1],
        [0.45, 0.70, 0.05, 2],
        [0.65, 0.60, 0.06, 3],
        [0.85, 0.40, 0.06, 1],
        [0.65, 0.20, 0.06, 2],
        [0.45, 0.20, 0.12, 3],
    ]
)


with swarm.access(material):
    material.data[...] = 0

    for i in range(blobs.shape[0]):
        cx, cy, r, m = blobs[i, :]
        inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
        material.data[inside] = m

# -


material.sym

X = meshbox.CoordinateSystem.X

# +
# The material fields are differentiable

sympy.derive_by_array(material.sym, X).reshape(2, 4).tomatrix()
# -

mat_density = np.array([1, 0.1, 0.1, 2])
density = (
    mat_density[0] * material.sym[0]
    + mat_density[1] * material.sym[1]
    + mat_density[2] * material.sym[2]
    + mat_density[3] * material.sym[3]
)

# +
mat_viscosity = np.array([1, 0.1, 10.0, 10.0])
# mat_viscosity = np.array([1, 1, 1.0, 1.0])

viscosity = (
    mat_viscosity[0] * material.sym[0]
    + mat_viscosity[1] * material.sym[1]
    + mat_viscosity[2] * material.sym[2]
    + mat_viscosity[3] * material.sym[3]
)
# -

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    points = vis.swarm_to_pv_cloud(swarm)
    point_cloud = pv.PolyData(points)
    
    pvmesh.point_data["M0"] = vis.scalar_fn_to_pv_points(pvmesh, material.sym[0])
    pvmesh.point_data["M1"] = vis.scalar_fn_to_pv_points(pvmesh, material.sym[1])
    pvmesh.point_data["M2"] = vis.scalar_fn_to_pv_points(pvmesh, material.sym[2])
    pvmesh.point_data["M3"] = vis.scalar_fn_to_pv_points(pvmesh, material.sym[3])
    pvmesh.point_data["M"] = (1.0 * pvmesh.point_data["M1"] 
                              + 2.0 * pvmesh.point_data["M2"]
                              + 3.0 * pvmesh.point_data["M3"])
    pvmesh.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh, density)
    pvmesh.point_data["visc"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.log(viscosity))
    

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)

    pl.add_mesh(
                pvmesh,
                cmap="coolwarm",
                edge_color="Black",
                show_edges=True,
                scalars="visc",
                use_transparency=False,
                opacity=0.95,
               )

    pl.show(cpos="xy")

# +
# Create Stokes object

stokes = uw.systems.Stokes(
    meshbox, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

# Set some things
import sympy
from sympy import Piecewise

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = viscosity

stokes.bodyforce = sympy.Matrix([0, -density])
stokes.saddle_preconditioner = 1.0 / viscosity

# free slip.
# note with petsc we always need to provide a vector of correct cardinality.

stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")


# +
stokes.petsc_options["snes_rtol"] = 1.0e-3
stokes.petsc_options[
    "snes_atol"
] = 1.0e-5  # Not sure why rtol does not do its job when guess is used

# stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None
stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-3
stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 1.0e-2


# -

stokes.solve(zero_init_guess=True)

# +
# check the solution

if uw.mpi.size == 1 and render:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh, density)
    pvmesh.point_data["visc"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.log(viscosity))
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))

    # point sources at cell centres

    cpoints = np.zeros((meshbox._centroids.shape[0] // 4, 3))
    cpoints[:, 0] = meshbox._centroids[::4, 0]
    cpoints[:, 1] = meshbox._centroids[::4, 1]
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
    
    spoints = vis.swarm_to_pv_cloud(swarm)
    spoint_cloud = pv.PolyData(spoints)

    with swarm.access():
        spoint_cloud.point_data["M"] = material.data[...]

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvstream, opacity=1)
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Gray",
        show_edges=True,
        scalars="rho",
        opacity=0.5,
    )

    pl.add_points(
        spoint_cloud,
        cmap="gray_r",
        scalars="M",
        render_points_as_spheres=True,
        point_size=5,
        opacity=0.33,
    )

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -

def plot_mesh(filename):
    if uw.mpi.size == 1:
        
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshbox)
        pvmesh.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh, density)
        pvmesh.point_data["visc"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.log(viscosity))
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
        pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))

        # point sources at cell centres

        cpoints = np.zeros((meshbox._centroids.shape[0], 3))
        cpoints[:, 0] = meshbox._centroids[:, 0]
        cpoints[:, 1] = meshbox._centroids[:, 1]
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

        spoints = vis.swarm_to_pv_cloud(swarm)
        spoint_cloud = pv.PolyData(spoints)

        with swarm.access():
            spoint_cloud.point_data["M"] = material.data[...]

        pl = pv.Plotter()

        # pl.add_mesh(pvmesh, "Gray",  "wireframe")
        # pl.add_arrows(arrow_loc, velocity_field, mag=0.2/vmag, opacity=0.5)

        pl.add_mesh(pvstream, opacity=1)
        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Gray",
            show_edges=True,
            scalars="visc",
            opacity=0.5,
        )

        pl.add_points(
            spoint_cloud,
            cmap="gray_r",
            scalars="M",
            render_points_as_spheres=True,
            point_size=5,
            opacity=0.33,
        )

        # pl.add_points(pdata)

        pl.remove_scalar_bar("M")
        pl.remove_scalar_bar("visc")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1250, 1250),
            return_img=False,
        )

        # pl.show()
        pv.close_all()

        return


t_step = 0

# +
# Update in time

expt_name = "output/blobs"

for step in range(0, 2): # 250
    stokes.solve(zero_init_guess=False)
    delta_t = min(10.0, stokes.estimate_dt())

    # update swarm / swarm variables

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(t_step, delta_t))

    # advect swarm
    print("Swarm Advection")
    swarm.advection(v_soln.fn, delta_t)
    print("Swarm Advection - done")

    if t_step % 1 == 0:
        plot_mesh(filename="{}_step_{}".format(expt_name, t_step))

    t_step += 1

# -
meshbox.petsc_save_checkpoint(index=step, meshVars=[v_soln], outputPath='./output/')


