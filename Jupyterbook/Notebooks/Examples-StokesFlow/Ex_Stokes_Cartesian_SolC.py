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

# # Stokes Benchmark SolCx
#


# +
# %%
import petsc4py
from petsc4py import PETSc

import nest_asyncio
nest_asyncio.apply()

# options = PETSc.Options()
# options["help"] = None 

import os
os.environ["UW_TIMING_ENABLE"] = "1"


# +
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
from underworld3 import timing

import numpy as np
import sympy
from sympy import Piecewise

# +
# %%
n_els = 4
refinement = 2

mesh1 = uw.meshing.UnstructuredSimplexBox(regular=True,
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / n_els, 
    qdegree=3, refinement=refinement
)

mesh2 = uw.meshing.StructuredQuadBox(
    elementRes=(n_els, n_els),
    minCoords=(0.0, 0.0), 
    maxCoords=(1.0, 1.0), 
    qdegree=3, 
    refinement=refinement
)

mesh = mesh1
x,y = mesh.X
# -


mesh.view()

stokes = uw.systems.Stokes(mesh, verbose=True)

# +
v = stokes.Unknowns.u
p = stokes.Unknowns.p

stokes.constitutive_model=uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
# %%
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3, continuous=True, varsymbol=r"{T}")
T2 = uw.discretisation.MeshVariable("T2", mesh, 1, degree=3, continuous=True, varsymbol=r"{T_2}")

v0 = stokes.Unknowns.u.clone("V0", r"{v_0}")
v1 = v0.clone("V1", r"{v_1}")
# -


eta_0 = 1.0
x_c = 0.5
f_0 = 1.0


stokes.penalty = 100.0
stokes.bodyforce = sympy.Matrix(
    [
        0,
        Piecewise(
            (f_0, x > x_c),
            (0.0, True),
        ),
    ]
)

# +
# This is the other way to impose no vertical flow

# stokes.add_natural_bc(   [0.0,1e5*v.sym[1]], "Top")              # Top "free slip / penalty"


# +
# free slip.

stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")
# -


# We may need to adjust the tolerance if $\Delta \eta$ is large

stokes.tolerance = 1.0e-6

stokes.petsc_options["snes_monitor"]= None
stokes.petsc_options["ksp_monitor"] = None


# +
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

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

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")


# +
# stokes._setup_pointwise_functions(verbose=True)
# stokes._setup_discretisation(verbose=True)
# stokes.dm.ds.view()
# -

# %%
# Solve time
stokes.solve()

# ### Visualise it !

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, stokes.bodyforce[1])
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v.sym.dot(v.sym))

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Vmag",
        use_transparency=False,
        opacity=1.0,
    )

    arrows = pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=3.0, opacity=1, show_scalar_bar=False)

    pl.show(cpos="xy")


# -
# ## SolCx from the same setup

# +
stokes.bodyforce = sympy.Matrix(
    [0, -sympy.cos(sympy.pi * x) * sympy.sin(2 * sympy.pi * y)]
)

viscosity_fn = sympy.Piecewise(
    (1.0e6, x > x_c),
    (1.0, True),
)

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn
# -


stokes.constitutive_model.Parameters.shear_viscosity_0

stokes.saddle_preconditioner = sympy.simplify(1 / (stokes.constitutive_model.viscosity + stokes.penalty))

# +
timing.reset()
timing.start()
stokes.solve(zero_init_guess=True)
timing.print_table(display_fraction=0.999)

# Save this solution

with mesh.access(v0):
    v0.data[...] = v.data[...]

# +
# reset and re-do with natural bcs

stokes._reset()
stokes.tolerance = 1.0e-6
stokes.add_natural_bc([0.0,1e6*v.sym[1]], "Top") 
stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")
stokes.solve()

with mesh.access(v1):
    v1.data[...] = v.data[...]

# +
# reset and re-do with natural bcs & petsc normals

stokes._reset()
stokes.tolerance = 1.0e-6

Gamma = mesh.Gamma
stokes.add_natural_bc(1e6 * Gamma.dot(v.sym) * Gamma, "Top") 
stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")
stokes.solve()


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v0.sym.dot(v0.sym))
    pvmesh.point_data["Visc"] = vis.scalar_fn_to_pv_points(pvmesh, stokes.constitutive_model.Parameters.shear_viscosity_0)

    pvmesh.point_data["V2"] = vis.vector_fn_to_pv_points(pvmesh, v.sym * stokes.constitutive_model.viscosity)
    pvmesh.point_data["V0"] = vis.vector_fn_to_pv_points(pvmesh, v0.sym * stokes.constitutive_model.viscosity)
    pvmesh.point_data["V1"] = vis.vector_fn_to_pv_points(pvmesh, v1.sym * stokes.constitutive_model.viscosity)
    pvmesh.point_data["dV0"] = pvmesh.point_data["V1"] - pvmesh.point_data["V0"]
    
    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V2"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)
    velocity_points.point_data["V1"] = vis.vector_fn_to_pv_points(velocity_points, v1.sym)
    velocity_points.point_data["V0"] = vis.vector_fn_to_pv_points(velocity_points, v0.sym)
    velocity_points.point_data["dV1"] = velocity_points.point_data["V1"] - velocity_points.point_data["V0"]
    velocity_points.point_data["dV2"] = velocity_points.point_data["V2"] - velocity_points.point_data["V0"]

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Vmag",
        use_transparency=False,
        opacity=1.0,
    )

    arrows0 = pl.add_arrows(velocity_points.points, velocity_points.point_data["V2"], mag=100.0, opacity=1, show_scalar_bar=False)
    arrows1 = pl.add_arrows(velocity_points.points, velocity_points.point_data["dV2"], mag=100000.0, opacity=1, show_scalar_bar=False)

    pl.show(jupyter_backend='client')



# +
# %%
try:
    import underworld as uw2

    solC = uw2.function.analytic.SolC()
    vel_soln_analytic = solC.fn_velocity.evaluate(mesh.data)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    from numpy import linalg as LA

    with mesh.access(v):
        num = function.evaluate(v.fn, mesh.data)  # this appears busted
        if comm.rank == 0:
            print("Diff norm a. = {}".format(LA.norm(v.data - vel_soln_analytic)))
            print("Diff norm b. = {}".format(LA.norm(num - vel_soln_analytic)))
        # if not np.allclose(v.data, vel_soln_analytic, rtol=1):
        #     raise RuntimeError("Solve did not produce expected result.")
    comm.barrier()
except ImportError:
    import warnings

    warnings.warn("Unable to test SolC results as UW2 not available.")

# %%
