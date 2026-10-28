"""The vectorised node-transfer assembly against the per-row builders it
replaced (custom_mg from the commit before): the same MPIAIJ entries, on a
2-D and a 3-D refined box's coarse-to-fine pair, both builders.

    git show dd1d6eea:src/underworld3/utilities/custom_mg.py > /tmp/parallel-addon/old/custom_mg_old.py
    mpirun -np N python check_transfer_assembly.py
"""
import importlib.util
import numpy as np
import sympy
import underworld3 as uw
from mpi4py import MPI
from underworld3.utilities import custom_mg as new

spec = importlib.util.spec_from_file_location("custom_mg_old", "/tmp/parallel-addon/old/custom_mg_old.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
comm = MPI.COMM_WORLD
bad = []
for dim in (2, 3):
    mesh = (uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1, refinement=1, qdegree=2)
            if dim == 2 else
            uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.3, regular=False, refinement=1, qdegree=2))
    v = uw.discretisation.MeshVariable(f"v{dim}", mesh, dim, degree=2)
    p = uw.discretisation.MeshVariable(f"p{dim}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.add_dirichlet_bc((0.0,) * dim, "Bottom")
    stokes._setup_pointwise_functions(verbose=False) if hasattr(stokes, "_setup_pointwise_functions") else None
    stokes._build(verbose=False) if hasattr(stokes, "_build") else None
    wraps = mesh._coarse_level_meshes()
    cm = wraps[0]
    Xc = np.asarray(cm._get_coords_for_basis(2, True)); Xf = np.asarray(mesh._get_coords_for_basis(2, True))
    lay_c = new._coarse_dof_layout(stokes, cm, 0); lay_f = new._level_dof_layout(stokes.dm, 0)
    for name in ("_build_parallel_transfer", "_build_crosspart_transfer"):
        args = (Xc, Xf, lay_c, lay_f, dim, new.barycentric_prolongation, stokes.dm.comm)
        A = getattr(new, name)(*args)
        args_o = (Xc, Xf, lay_c, lay_f, dim, old.barycentric_prolongation, stokes.dm.comm)
        B = getattr(old, name)(*args_o)
        ia, ja, va = A.getValuesCSR(); ib, jb, vb = B.getValuesCSR()
        same = (A.getSizes() == B.getSizes() and np.array_equal(ia, ib) and np.array_equal(ja, jb)
                and np.array_equal(va, vb))
        same = comm.allreduce(bool(same), op=MPI.LAND)
        if comm.rank == 0:
            print(f"{dim}-D {name}: {'identical' if same else 'DIFFERENT'} ({A.getSize()})", flush=True)
        if not same:
            bad.append((dim, name))
if comm.rank == 0:
    print("FAILURES:", bad or "none", flush=True)
