import underworld3 as uw
from mpi4py import MPI

# A simple parallel test of the SemiLagrangian's nodal swarm advection.
# This determines if nodal swarm particles are lost during
# large (going to different processors) advection steps.
# NOTE: Make sure to run this with 4 processors to ensure
# that the test scenario (particles going beyond neighboring processors) happens.

dt = 1
maxsteps = 1
res = 16
velocity = 1.25 # seems to take a long time if a higher velocity is set

xmin, xmax = 0., 4.
ymin, ymax = 0., 1.

mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(xmin,ymin),
        maxCoords=(xmax,ymax),
        cellSize=1 / res, regular=False, qdegree=3, refinement=0)

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree = 2)

# vector being advected
vec_tst = uw.discretisation.MeshVariable("Vn", mesh, mesh.dim, degree = 2)

DuDt = uw.systems.ddt.SemiLagrangian(
                                        mesh,
                                        vec_tst.sym,
                                        v.sym,
                                        vtype = uw.VarType.VECTOR,
                                        degree = 2,
                                        continuous = vec_tst.continuous,
                                        varsymbol = vec_tst.symbol,
                                        verbose = False,
                                        bcs = None,
                                        order = 1,
                                    )

# initialize variables
with mesh.access(v):
    v.data[:, 0] = velocity

# we are only interested in monitoring the number of nodal swarm particles
# before and after advection
with mesh.access(vec_tst):
    vec_tst.data[:, 0] = 1.

# get the number of nodal swarm particles BEFORE advection
with DuDt._nswarm_psi.access(DuDt._nswarm_psi):
    before_adv_swarm_num = len(DuDt._nswarm_psi.data)

comm = uw.mpi.comm

before_total = comm.allreduce(before_adv_swarm_num, op = MPI.SUM)

#print(f"Before advection; rank {uw.mpi.rank} particles: {before_adv_swarm_num}", flush = True)

# do one huge timestep
DuDt.update_pre_solve(dt, verbose = False, evalf = False)
with mesh.access(vec_tst):
    vec_tst.data[...] = DuDt.psi_star[0].data[...]

# get the number of nodal swarm particles AFTER advection
with DuDt._nswarm_psi.access(DuDt._nswarm_psi):
    after_adv_swarm_num = len(DuDt._nswarm_psi.data)

after_total = comm.allreduce(after_adv_swarm_num, op = MPI.SUM)

#print(f"After advection; rank {uw.mpi.rank} particles: {after_adv_swarm_num}", flush = True)

if uw.mpi.rank == 0:
    print(f"Before advection; Total particles in all ranks: {before_total}", flush = True)
    print(f"After advection; Total particles in all ranks: {after_total}", flush = True)

if uw.mpi.rank == 0:
    assert (after_total == before_total), "Error: Nodal swarm particles lost during advection."
