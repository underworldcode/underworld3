"""
Reproducer for issue #176 ([BUG] - memory leak), verbatim from @bknight1's
comment 2026-05-10 on GitHub. Linear RSS growth scaling with proc count
(~100 MB/proc baseline, ~5 MiB/step in steps 1-6 after kdtree noise).

Usage:
    # Serial baseline:
    UW_MEMPROBE=1 pixi run python tests/parallel/repro_176_memleak.py

    # Parallel:
    UW_MEMPROBE=1 mpirun -np 4 pixi run python tests/parallel/repro_176_memleak.py

    # With PETSc-side leak dump at finalize:
    UW_MEMPROBE=1 mpirun -np 4 pixi run python tests/parallel/repro_176_memleak.py \
        -malloc_dump -objects_dump

Do NOT modify this file — kept verbatim so measurements compare directly with
those posted on issue #176.
"""
import underworld3 as uw
import numpy as np
import os
import resource
import sympy
import gc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

# Parameters
res = 32
n_steps = 50

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

if rank == 0:
    print(f"Starting Stokes + AdvDiff leak test for {n_steps} steps on {size} ranks...")
    print(f"Resolution: {res}x{res}")
    print(f"Initial Memory = {get_memory_usage():.2f} MiB")

mem_history = []

for i in range(n_steps):
    # 1. Solve Stokes
    # stokes.solve()

    # dt = stokes.estimate_dt()
    dt = 0.001

    # 2. Solve AdvDiff
    advdiff.solve(timestep=dt)

    # 3. Particle advection (triggers interpolation and coordinate caching)
    # We'll use a dummy advection to change coordinates
    with swarm.access(swarm):
        swarm.data[...] = swarm.data[...] + 0.01 * np.random.rand(*swarm.data.shape)
        swarm.data[...] = np.clip(swarm.data[...], 0, 1)

    # Evaluate velocity at swarm to trigger cache population
    v_at_swarm = uw.function.evaluate(v.sym, swarm.data)

    gc.collect()

    if i % 5 == 0:
        local_mem = get_memory_usage()
        total_mem = comm.reduce(local_mem, op=MPI.SUM, root=0)

        if rank == 0:
            dm_fields = mesh.dm.getNumFields()
            print(f"Step {i:3d}: Total Memory = {total_mem:7.2f} MiB | DM Fields = {dm_fields:2d}")
            mem_history.append(total_mem)

if rank == 0:
    print("\nMemory usage summary (Total across ranks):")
    for i, mem in enumerate(mem_history):
        print(f"Step {i*5:3d}: {mem:7.2f} MiB")

    growth = mem_history[-1] - mem_history[0]
    print(f"\nTotal growth: {growth:.2f} MiB")
    if growth > 5.0:
        print("\nRESULT: Memory leak detected!")
    else:
        print("\nRESULT: Memory usage stable.")
