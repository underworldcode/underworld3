"""Collective restrictions of the current native spherical solver adapters."""
from mpi4py import MPI
import underworld3 as uw


def require_native_solver_mesh(mesh):
    r"""Check collective communicator and local-chart restrictions for native adapters.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        Mesh whose communicator and local DM charts are checked collectively.

    Returns
    -------
    None
        Successful return means the native solver setup restrictions are satisfied.

    Raises
    ------
    ValueError
        The communicator is not congruent to UW3's world communicator, or any rank
        has an empty DM chart.

    Notes
    -----
    Trace integrators themselves use the mesh communicator. Native SNES setup
    uses COMM_WORLD and is not empty-chart safe. Empty local interface/source
    strata on nonempty meshes remain supported."""
    comm = mesh.dm.getComm().tompi4py()
    if MPI.Comm.Compare(comm, uw.mpi.comm) not in (MPI.IDENT, MPI.CONGRUENT):
        raise ValueError('Native spherical solver adapters require a mesh communicator congruent to UW3 COMM_WORLD.')
    start, end = mesh.dm.getChart()
    if comm.allreduce(end == start, op=MPI.LOR):
        raise ValueError('Native spherical solver adapters require nonempty mesh charts on every rank; empty interface/source strata are supported.')
