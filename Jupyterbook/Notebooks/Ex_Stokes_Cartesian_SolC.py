# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from petsc4py import PETSc
# import underworld3 as uw
# import numpy as np

# options = PETSc.Options()
# # options["help"] = None


# %%

# %%
# n_els = 32
# mesh = uw.util_mesh.StructuredQuadBox(elementRes=(n_els, n_els))
# # mesh = uw.meshes.Box(elementRes=(n_els,n_els))
# v_degree = 1
# stokes = uw.systems.Stokes(mesh, u_degree=v_degree )

# %%
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

# %%
meshbox = uw.util_mesh.StructuredQuadBox(elementRes=(5,)*2)
# meshbox = uw.meshes.Unstructured_Simplex_Box(dim=2, minCoords=(0.0,0.0,0.0), maxCoords=(1.0,1.0,1.0), cell_size=1.0/32.0, regular=True)
meshbox.dm.view()  

# %%
v_degree = 1
stokes = uw.systems.Stokes(meshbox, u_degree=v_degree )

# %%
# Set some things
import sympy
from sympy import Piecewise
N = meshbox.N
eta_0 = 1.
x_c   = 0.5
f_0   = 1.
stokes.viscosity = 1.
stokes.penalty = 0.0
stokes.bodyforce = Piecewise((f_0, N.x>x_c,), (  0.,    True) )*N.j
stokes._Ppre_fn = 1.0 / (stokes.viscosity + stokes.penalty)


# %%
#  free slip.  
# note with petsc we always need to provide a vector of correct cardinality. 
# bnds = meshbox.
stokes.add_dirichlet_bc( (0.,0.), ["Top", "Bottom"], 1 )  # top/bottom: components, function, markers 
stokes.add_dirichlet_bc( (0.,0.), ["Left", "Right"],  0 )  # left/right: components, function, markers

# %%
# stokes.petsc_options["pc_type"]  = "svd"
stokes.petsc_options["ksp_rtol"] =  1.0e-8
# stokes.petsc_options["ksp_monitor_short"] = None
# stokes.petsc_options["snes_type"]  = "fas"
stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor_short"] = None
# stokes.petsc_options["snes_view"]=None
# stokes.petsc_options["snes_test_jacobian"] = None

stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
stokes.petsc_options["snes_max_it"] = 10

# %%
# Solve time
stokes.solve()

# %%
try:
    import underworld as uw2
    solC = uw2.function.analytic.SolC()
    vel_soln_analytic = solC.fn_velocity.evaluate(meshbox.data)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    from numpy import linalg as LA
    with meshbox.access():
        if comm.rank == 0:
            print("Diff norm = {}".format(LA.norm(stokes.u.data - vel_soln_analytic)))
        if not np.allclose(stokes.u.data, vel_soln_analytic, rtol=1.e-2):
            pass #raise RuntimeError("Solve did not produce expected result.")
    comm.barrier()
except ImportError:
    import warnings
    warnings.warn("Unable to test SolC results as UW2 not available.")

# %% [markdown]
# ## Write fields to disk

# %%
meshbox.save("uw3.h5")
stokes.u.save(filename="uw3.h5",name='uw3_v')

# %%
with meshbox.access(stokes.u):
    orig = stokes.u.data
    stokes.u.data[:] = vel_soln_analytic[:]

# %%
meshbox.save("ana.h5")
stokes.u.save(filename="ana.h5",name='analytic')
