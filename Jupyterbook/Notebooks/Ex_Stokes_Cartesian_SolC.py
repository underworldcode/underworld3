# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
import numpy as np



# %%
n_els = 32
mesh = uw.util_mesh.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
                                           maxCoords=(1.0,1.0), 
                                           cellSize=1.0/n_els, regular=True)

v_degree = 2

v_soln = uw.mesh.MeshVariable('U',    mesh,  mesh.dim, degree=v_degree )
p_soln = uw.mesh.MeshVariable('P',    mesh,  1, degree=v_degree-1 )

stokes = uw.systems.Stokes(mesh, 
                velocityField=v_soln, 
                pressureField=p_soln, 
                u_degree=v_soln.degree, 
                p_degree=p_soln.degree, 
                solver_name="stokes", 
                verbose=False)



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
stokes.add_dirichlet_bc( (0.,0.), ["Bottom",  "Top"], 1 )  # top/bottom: components, function, markers 
stokes.add_dirichlet_bc( (0.,0.), ["Left", "Right"],  0 )  # left/right: components, function, markers


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
    with mesh.access():
        if comm.rank == 0:
            print("Diff norm = {}".format(LA.norm(stokes.u.data - vel_soln_analytic)))
        if not np.allclose(stokes.u.data, vel_soln_analytic, rtol=1.e-2):
            raise RuntimeError("Solve did not produce expected result.")
    comm.barrier()
except ImportError:
    import warnings
    warnings.warn("Unable to test SolC results as UW2 not available.")

# %%
