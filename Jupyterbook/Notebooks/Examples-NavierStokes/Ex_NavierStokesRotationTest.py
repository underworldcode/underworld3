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

# # Navier Stokes test: boundary driven ring with step change in boundary conditions
#
# This should develop a boundary layer with sqrt(t) growth rate

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()



# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3.systems import NavierStokesSLCN

from underworld3 import function

import numpy as np
import sympy


# +

meshball = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5, cellSize=0.05, qdegree=3
)


# +
# Define some functions on the mesh

import sympy

radius_fn = sympy.sqrt(
    meshball.rvec.dot(meshball.rvec)
)  # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10 + radius_fn)

# Some useful coordinate stuff

x = meshball.N.x
y = meshball.N.y

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

# Rigid body rotation v_theta = constant, v_r = 0.0

theta_dot = 2.0 * np.pi  # i.e one revolution in time 1.0
v_x = -1.0 * r * theta_dot * sympy.sin(th)  # * y # to make a convergent / divergent bc
v_y = r * theta_dot * sympy.cos(th)  # * y
# -

v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
vorticity = uw.discretisation.MeshVariable(
    "\omega", meshball, 1, degree=1, continuous=False
)


navier_stokes = NavierStokesSLCN(
    meshball,
    velocityField=v_soln,
    pressureField=p_soln,
    rho=10.0,
    solver_name="navier_stokes",
)




nodal_vorticity_from_v = uw.systems.Projection(meshball, vorticity)
nodal_vorticity_from_v.uw_function = meshball.vector.curl(v_soln.sym)
nodal_vorticity_from_v.smoothing = 0.0


# +
# Constant visc

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
navier_stokes.constitutive_model.Parameters.viscosity = 1.0

# Constant visc

navier_stokes.rho = 250
navier_stokes.penalty = 0.1
navier_stokes.bodyforce = sympy.Matrix([0, 0])

# Velocity boundary conditions
navier_stokes.add_dirichlet_bc((v_x, v_y), "Upper", (0, 1))
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

expt_name = f"Cylinder_NS_rho_{navier_stokes.rho}"

# -

navier_stokes.Unknowns.DuDt

with meshball.access(v_soln):
    v_soln.data[...] = 0.0

navier_stokes.solve(timestep=0.1)
navier_stokes.estimate_dt()

nodal_vorticity_from_v.solve()


# check the mesh if in a notebook / serial
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(
        velocity_points, v_soln.sym
    )


    # point sources at cell centres
    points = np.zeros((meshball._centroids.shape[0], 3))
    points[:, 0] = meshball._centroids[:, 0]
    points[:, 1] = meshball._centroids[:, 1]
    centroid_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        centroid_cloud,
        vectors="V",
        integration_direction="both",
        surface_streamlines=True,
        max_time=0.25,
    )

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh, cmap="RdBu", scalars="Omega", opacity=0.5, show_edges=True)
    pl.add_mesh(pvstream, opacity=0.33)
    pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=1.0e-2,
        opacity=0.75,
    )


    pl.camera.SetPosition(0.75, 0.2, 1.5)
    pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    # pl.remove_scalar_bar("Omega")
    pl.remove_scalar_bar("mag")
    pl.remove_scalar_bar("V")

    pl.show()


def plot_V_mesh(filename):
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshball)
        pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
        pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

        velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(
            velocity_points, v_soln.sym
        )


        # point sources at cell centres
        points = np.zeros((meshball._centroids.shape[0], 3))
        points[:, 0] = meshball._centroids[:, 0]
        points[:, 1] = meshball._centroids[:, 1]
        centroid_cloud = pv.PolyData(points)

        pvstream = pvmesh.streamlines_from_source(
            centroid_cloud,
            vectors="V",
            integration_direction="both",
            surface_streamlines=True,
            max_time=0.25,
        )

        pl = pv.Plotter()

        pl.add_arrows(
            velocity_points.points,
            velocity_points.point_data["V"],
            mag=0.01,
            opacity=0.75,
        )

        # pl.add_points(
        #     point_cloud,
        #     color="Black",
        #     render_points_as_spheres=True,
        #     point_size=2,
        #     opacity=0.66,
        # )

        # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=False,
            scalars="Omega",
            use_transparency=False,
            opacity=0.5,
        )

        pl.add_mesh(
            pvmesh,
            cmap="RdBu",
            scalars="Omega",
            opacity=0.1,  # clim=[0.0, 20.0]
        )

        pl.add_mesh(pvstream, opacity=0.33)

        scale_bar_items = list(pl.scalar_bars.keys())

        for scalar in scale_bar_items:
            pl.remove_scalar_bar(scalar)

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(2560, 2560),
            return_img=False,
        )

        # pl.show()

ts = 0

# +
# Time evolution model / update in time

for step in range(0, 50):  # 250
    delta_t = 3.0 * navier_stokes.estimate_dt()
    navier_stokes.solve(timestep=delta_t, zero_init_guess=False)

    nodal_vorticity_from_v.solve()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(ts, delta_t))

    if ts % 1 == 0:
        plot_V_mesh(filename="output/{}_step_{}".format(expt_name, ts))

    ts += 1
# -




