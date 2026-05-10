import underworld3 as uw
import numpy as np
import os
import resource
import sympy
import gc
from mpi4py import MPI
import pytest

def get_memory_usage():
    try:
        if os.uname().sysname == 'Darwin':
            rss = int(os.popen('ps -p %d -o rss=' % os.getpid()).read())
            return rss / 1024.0 # MiB
        else:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024.0 # MiB
    except:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def test_stokes_advdiff_memory_leak():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parameters
    res = 32
    n_steps = 100
    warmup_step = 30

    # Setup mesh
    mesh = uw.meshing.StructuredQuadBox(elementRes=(res, res))
    mesh._dminterpolation_cache.max_entries = 10 # CAP CACHE

    # Variables
    v = uw.discretisation.MeshVariable("u", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

    # Swarm for advection
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=2)

    # Stokes
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1.0
    stokes.bodyforce = sympy.Matrix([0, T.sym[0]])
    stokes.add_dirichlet_bc([0.0, 0.0], "Top")
    stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")
    stokes.add_dirichlet_bc([0.0, sympy.oo], "Left")
    stokes.add_dirichlet_bc([0.0, sympy.oo], "Right")

    # AdvDiff
    advdiff = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v)
    advdiff.constitutive_model = uw.constitutive_models.DiffusionModel
    advdiff.constitutive_model.Parameters.diffusivity = 1.0
    advdiff.add_dirichlet_bc([0.0], "Top")
    advdiff.add_dirichlet_bc([1.0], "Bottom")

    # Initial T
    T.data[:] = 0.0
    with mesh.access(T):
        T.data[:, 0] = 1.0 - mesh.data[:, 1]

    mem_warmup = 0.0

    for i in range(n_steps):
        stokes.solve()
        dt = 0.001
        advdiff.solve(timestep=dt)

        with swarm.access(swarm):
            swarm.data[...] = swarm.data[...] + 0.01 * np.random.rand(*swarm.data.shape)
            swarm.data[...] = np.clip(swarm.data[...], 0, 1)
        
        v_at_swarm = uw.function.evaluate(v.sym, swarm.data)

        gc.collect()

        if i == warmup_step:
            local_mem = get_memory_usage()
            mem_warmup = comm.reduce(local_mem, op=MPI.SUM, root=0)

    # Final memory check
    local_mem = get_memory_usage()
    mem_final = comm.reduce(local_mem, op=MPI.SUM, root=0)

    if rank == 0:
        growth = mem_final - mem_warmup
        print(f"\nMemory growth from step {warmup_step} to {n_steps}: {growth:.2f} MiB")
        
        # We allow for some small growth (fragmentation, PETSc pool expansion) 
        # but catch the large unbounded leak (which was ~180 MiB per 100 steps).
        # Stable growth should be < 20 MiB over the last 70 steps for this resolution.
        assert growth < 30.0, f"Significant memory leak detected: {growth:.2f} MiB growth after warm-up."
