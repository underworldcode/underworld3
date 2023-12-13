# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
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
refinement = 3

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
# -


mesh.dm.view()

stokes = uw.systems.Stokes(mesh, verbose=True)

# +
v = stokes.Unknowns.u
p = stokes.Unknowns.p

stokes.constitutive_model=uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns)
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
# %%
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3, continuous=True, varsymbol=r"{T}")
T2 = uw.discretisation.MeshVariable("T2", mesh, 1, degree=3, continuous=True, varsymbol=r"{T_2}")
# -
x,y = mesh.X

mesh.get_min_radius()

hw = 0.1 * mesh.get_min_radius()
surface_fn = 2 * uw.maths.delta_function(y-1, hw) / uw.maths.delta_function(0.0, hw)
base_fn = 2 * uw.maths.delta_function(y, hw)
right_fn = 2 * uw.maths.delta_function(x-1, hw)
left_fn = 2 * uw.maths.delta_function(x, hw)

# +
# options = PETSc.Options()
# options.getAll()
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
# This is the other way to impose no vertical 

# stokes.bodyforce[0] -= 1.0e6 * v.sym[0] * (left_fn + right_fn)
# stokes.bodyforce[1] -= 1.0e3 * v.sym[1] * (surface_fn + base_fn)

# stokes.add_natural_bc( -1.0e10 * v.sym[1], sympy.Matrix((0.0, 0.0)).T , "Top", components=[1])


# +
# free slip.
# note with petsc we always need to provide a vector of correct cardinality.

stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")
# -


# We may need to adjust the tolerance if $\Delta \eta$ is large

# +
# stokes.petsc_options["snes_rtol"] = 1.0e-6
# stokes.petsc_options["ksp_rtol"] = 1.0e-6
# stokes.petsc_options["snes_max_it"] = 10
# -

stokes.tolerance = 1.0e-3

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
    [0, -sympy.cos(sympy.pi * x) * sympy.sin(2 * sympy.pi * y)*(1-(surface_fn + base_fn))]
)

stokes.bodyforce[0] -= 1.0e3* v.sym[0] * (left_fn + right_fn)
stokes.bodyforce[1] -= 1.0e3 * v.sym[1] * (surface_fn + base_fn)

viscosity_fn = sympy.Piecewise(
    (1.0e8, x > x_c),
    (1.0, True),
)

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn
# -


stokes.saddle_preconditioner = sympy.simplify(1 / (stokes.constitutive_model.viscosity + stokes.penalty))

stokes._setup_pointwise_functions()
stokes._setup_discretisation()
stokes._u_f1

# +
timing.reset()
timing.start()

stokes.solve(zero_init_guess=True)

timing.print_table(display_fraction=0.999)

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
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

    arrows = pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=30.0, opacity=1, show_scalar_bar=False)

    pl.show(cpos="xy")

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
# -
mesh.boundaries["Bottom"].value


