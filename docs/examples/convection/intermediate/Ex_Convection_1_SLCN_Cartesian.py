# %% [markdown]
"""
# 🔬 Convection 1 SLCN Cartesian

**PHYSICS:** convection  
**DIFFICULTY:** intermediate  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
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
# to fix trame issue
import nest_asyncio

nest_asyncio.apply()
# -

import numpy as np
import petsc4py
import sympy
import underworld3 as uw
from petsc4py import PETSc
from underworld3 import function
from underworld3.systems import Stokes

meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / 12.0,
    regular=False,
    qdegree=3,
)


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    # pv.start_xvfb()

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, edge_color="Black", show_edges=True)

    pl.show(cpos="xy")
# -

v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=3)


# +
# Create Stokes object

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
)

# Constant viscosity

viscosity = 1
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.tolerance = 1.0e-3


# free slip.
# note with petsc we always need to provide a vector of correct cardinality.

stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

# Some useful coordinate stuff

x, y = meshbox.X


# +
# Create adv_diff object

# Set some things
k = 1.0
h = 0.0

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshbox,
    u_Field=t_soln,
    V_fn=v_soln,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.theta = 0.5


# +
# Define T boundary conditions via a sympy function

import sympy

init_t = sympy.sin(5 * sympy.pi * x) * sympy.sin(sympy.pi * y) / 5 + (1.0 - y)

adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

t_0.array[...] = uw.function.evaluate(init_t, t_0.coords)
t_soln.array[...] = t_0.array[...]
# -


buoyancy_force = 1.0e6 * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# check the stokes solve is set up and that it converges
stokes.solve(zero_init_guess=True)

stokes.estimate_dt()

# Check the diffusion part of the solve converges
adv_diff.solve(timestep=2 * stokes.estimate_dt(), zero_init_guess=True)

pvmesh.clear_point_data()


def plot_T_mesh(filename):
    if uw.mpi.size == 1:

        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshbox)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym) / 333
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

        pv_mesh_t = vis.meshVariable_to_pv_mesh_object(t_soln)
        pv_mesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pv_mesh_t, t_soln.sym)

        # point sources at cell centres
        cpoints = np.zeros((meshbox._centroids[::4].shape[0], 3))
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

        points = vis.meshVariable_to_pv_cloud(t_soln)
        points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
        point_cloud = pv.PolyData(points)

        pl = pv.Plotter(window_size=(1000, 750))

        pl.add_mesh(
            pv_mesh_t,
            cmap="coolwarm",
            edge_color="Gray",
            show_edges=False,
            scalars="T",
            use_transparency=False,
            opacity=1,
        )

        pl.add_mesh(
            pv_mesh_t.copy(),
            style="wireframe",
            color="Black",
            use_transparency=False,
            opacity=0.1,
        )

        pl.add_mesh(pvstream, opacity=0.666)

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )
        # pl.show()
        pl.close()

        pvmesh.clear_data()
        pvmesh.clear_point_data()

        pv.close_all()


t_step = 0

# +
# Convection model / update in time

##
## There is a strange interaction here between the solvers if the zero_guess is
## set to False
##

expt_name = "output/Ra1e6"

for step in range(0, 100):
    stokes.solve(zero_init_guess=False)
    delta_t = 2.0 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # stats then loop
    tstats = t_soln.stats()

    uw.pprint(0, "Timestep {}, dt {}".format(step, delta_t))
    #         print(tstats)

    if t_step % 5 == 0:
        plot_T_mesh(filename="{}_step_{}".format(expt_name, t_step))

    t_step += 1
# -


# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshbox.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshbox.generate_xdmf(savefile)


if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, stokes.u.sym)

    pv_mesh_t = vis.meshVariable_to_pv_mesh_object(t_soln)
    pv_mesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pv_mesh_t, t_soln.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=7.5, opacity=0.25)

    pl.add_mesh(
        pv_mesh_t,
        cmap="coolwarm",
        edge_color="Gray",
        show_edges=False,
        scalars="T",
        use_transparency=False,
        opacity=1,
    )

    pl.add_mesh(
        pv_mesh_t.copy(),
        style="wireframe",
        color="Black",
        use_transparency=False,
        opacity=0.05,
    )

    pl.show(cpos="xy")

import weakref

ws = weakref.WeakValueDictionary()

for var in meshbox.vars.values():
    ws.add(var)






