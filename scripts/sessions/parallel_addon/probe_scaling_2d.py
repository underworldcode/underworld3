"""2-D analogue: refined box + interior split fault (add_fault), rotated custom-P path, iterations vs ranks."""
import time
import numpy as np, sympy, underworld3 as uw
from mpi4py import MPI
rank0 = MPI.COMM_WORLD.rank == 0
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(3.0, 1.0), cellSize=0.2, regular=False, qdegree=3, refinement=2)
from underworld3.utilities.fault_split import split_fault
surf = uw.meshing.Surface("Fault", base, np.array([[1.0, 0.5, 0.0], [1.5, 0.5, 0.0], [2.0, 0.5, 0.0]]))
surf.discretize()
cut = base.add_conforming_surface(surf)
mesh = split_fault(cut, "Fault")
ncell = MPI.COMM_WORLD.allreduce(int(mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0]), op=MPI.SUM)
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.bodyforce = sympy.Matrix([0, 0])
stokes.tolerance = 1e-4
stokes.add_dirichlet_bc((0.0, 0.0), "Left")
stokes.add_natural_bc(sympy.Matrix([[-0.01, -0.01]]), "Right")
stokes.add_dirichlet_bc((None, 0.0), "Bottom")
stokes.add_fault_bc(0.0, "Fault")
t = time.perf_counter()
stokes.solve(zero_init_guess=True)
info = stokes._rotated_freeslip_info
if rank0:
    print(f"2d fault np {MPI.COMM_WORLD.size} cells {ncell} tail {len(getattr(mesh, '_custom_mg_coarse_meshes', None) or [])}: solve {time.perf_counter()-t:.1f}s: "
          f"velocity_pc {info.get('velocity_pc')} vel its {info.get('vel_its_last')} pres its {info.get('pres_its_last')} converged {info.get('converged')}", flush=True)
