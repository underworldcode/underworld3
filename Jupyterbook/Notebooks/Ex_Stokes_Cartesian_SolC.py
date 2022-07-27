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
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

# %%
n_els = 16
mesh = uw.meshing.StructuredQuadBox(elementRes=(n_els,n_els))
mesh.dm.view()  


# %%
v = uw.discretisation.MeshVariable('U',    mesh,  mesh.dim, degree=1 )
p = uw.discretisation.MeshVariable('P',    mesh, 1, degree=0 )

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p )

# %%
# Set some things
import sympy
from sympy import Piecewise
N = mesh.N
eta_0 = 1.
x_c   = 0.5
f_0   = 1.
stokes.viscosity = 1.
stokes.penalty = 0.0
stokes.bodyforce = Piecewise((f_0, N.x>x_c,), (  0.,    True) )*N.j
stokes._Ppre_fn = 1.0 / (stokes.viscosity + stokes.penalty)

# free slip.  
# note with petsc we always need to provide a vector of correct cardinality. 
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
BF = (mesh.vector.to_matrix(stokes.bodyforce))
type(BF)

# %%
# Solve time
stokes.solve()

# %%
try:
    import underworld as uw2
    solC = uw2.function.analytic.SolC()
    vel_soln_analytic = solC.fn_velocity.evaluate(mesh.data)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    from numpy import linalg as LA
    with mesh.access(v):
        num = function.evaluate(v.fn, mesh.data) # this appears busted
        if comm.rank == 0:
            print("Diff norm a. = {}".format(LA.norm(v.data - vel_soln_analytic)))
            print("Diff norm b. = {}".format(LA.norm(num - vel_soln_analytic)))
        # if not np.allclose(v.data, vel_soln_analytic, rtol=1):
        #     raise RuntimeError("Solve did not produce expected result.")
    comm.barrier()
except ImportError:
    import warnings
    warnings.warn("Unable to test SolC results as UW2 not available.")

# %% [markdown]
# ## Write fields to disk

# %%
mesh.save("uw3.h5")
stokes.u.save(filename="uw3.h5",name='uw3_v')

# %%
with mesh.access(stokes.u):
    orig = stokes.u.data
    stokes.u.data[:] = vel_soln_analytic[:]

# %%
mesh.save("ana.h5")
stokes.u.save(filename="ana.h5",name='analytic')
