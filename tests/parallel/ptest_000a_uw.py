import underworld3 as uw
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f"{rank} - All done", flush=True)
