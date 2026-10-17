"""Build the finest transfer (last coarse wrap -> split finest) with each parallel builder
explicitly and test x-reproduction, zero columns and empty rows."""
import numpy as np, sympy, underworld3 as uw
from mpi4py import MPI
from petsc4py import PETSc
from underworld3.utilities.fault_split import split_fault
from underworld3.utilities import custom_mg as cmg
comm = MPI.COMM_WORLD; rank0 = comm.rank == 0
Lx, Ly, H = 3.0, 2.0, 1.0; h, hf, f = 0.2, 0.1, 4
yf, xa, xb, zb = 1.0, 1.0, 2.0, 0.5; ztop = H - 2 * hf
patch = np.array([[xa, yf, zb], [xb, yf, zb], [xb, yf, ztop], [xa, yf, ztop]])
fine = uw.meshing.BoxInternalPatch(cellSize=f * h, minCoords=(0, 0, 0), maxCoords=(Lx, Ly, H), patch_points=patch, patch_name="Fault",
                                   patch_cellSize=f * hf, grading_distance=8 * hf, refinement=2, qdegree=3)
mesh = split_fault(fine, "Fault")
v = uw.discretisation.MeshVariable("v", mesh, 3, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.bodyforce = sympy.Matrix([0, 0, 0]); stokes.tolerance = 1e-4
stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Left"); stokes.add_natural_bc(sympy.Matrix([[-0.01, -0.01, 0.0]]), "Right")
stokes.add_dirichlet_bc((None, None, 0.0), "Bottom"); stokes.add_fault_bc(0.0, "Fault")
stokes._setup_pointwise_functions(verbose=False) if hasattr(stokes, "_setup_pointwise_functions") else None
stokes._build(verbose=False) if hasattr(stokes, "_build") else None
cm = mesh._custom_mg_coarse_meshes[-1]
Xc = np.asarray(cm._get_coords_for_basis(2, True)); Xf = np.asarray(mesh._get_coords_for_basis(2, True))
lay_c = cmg._coarse_dof_layout(stokes, cm, 0); lay_f = cmg._level_dof_layout(stokes.dm, 0)
pcomm = stokes.dm.comm
def check(P, name):
    xc = P.createVecRight(); xf = P.createVecLeft()
    for i in range(Xc.shape[0]):
        g = int(lay_c.l2g[i * 3]);
        if lay_c.rstart <= g < lay_c.rend: xc.setValue(g, float(Xc[i, 0]))
    for i in range(Xf.shape[0]):
        g = int(lay_f.l2g[i * 3]);
        if lay_f.rstart <= g < lay_f.rend: xf.setValue(g, float(Xf[i, 0]))
    xc.assemble(); xf.assemble(); y = P.createVecLeft(); P.mult(xc, y)
    d = np.abs(y.getArray() - xf.getArray()); mask = np.abs(xf.getArray()) > 0
    nbad = comm.allreduce(int((d[mask] > 1e-6).sum()), op=MPI.SUM); ntot = comm.allreduce(int(mask.sum()), op=MPI.SUM)
    ones = P.createVecRight(); ones.set(1.0); r = P.createVecLeft(); P.mult(ones, r)
    nempty = comm.allreduce(int((np.abs(r.getArray()) < 1e-14).sum()), op=MPI.SUM)
    nzc = cmg._count_zero_columns_parallel(P, pcomm)
    if rank0: print(f"np {comm.size} {name}: x wrong on {nbad}/{ntot}, empty rows {nempty}, zero cols {nzc}", flush=True)
args = (Xc, Xf, lay_c, lay_f, 3, cmg.barycentric_prolongation, pcomm)
check(cmg._build_parallel_transfer(*args), "co-partitioned")
check(cmg._build_crosspart_transfer(*args), "cross-partition")

# --- dissect wrong rows of the cross-partition transfer on rank 0 ---
coords_u, cols_u = cmg._gather_coarse_cloud(Xc, lay_c, 3, pcomm)
Px = cmg._build_crosspart_transfer(*args)
col_coord = {}
for j in range(coords_u.shape[0]):
    g = int(cols_u[j, 0])
    if g >= 0: col_coord[g] = coords_u[j]
indptr, cols, data = Px.getValuesCSR()
fstart = lay_f.rstart
shown = 0
for i in range(Xf.shape[0]):
    g = int(lay_f.l2g[i * 3])
    if not (fstart <= g < lay_f.rend): continue
    r = g - fstart
    sl = slice(indptr[r], indptr[r + 1])
    cs, ws = cols[sl], data[sl]
    xi = Xf[i]
    xc_cols = [col_coord.get(int(c)) for c in cs]
    px = sum(w * cc[0] for w, cc in zip(ws, xc_cols) if cc is not None)
    if abs(px - xi[0]) > 1e-6 and shown < 4:
        dists = [float(np.linalg.norm(cc - xi)) if cc is not None else -1 for cc in xc_cols]
        if comm.rank == 0:
            print(f"rank0 wrong row: fine x {xi.round(3)} -> cols {cs.tolist()} weights {np.round(ws, 3).tolist()} coarse-node distances {np.round(dists, 3).tolist()} (P x = {px:.3f})", flush=True)
        shown += 1
# how many local fine nodes fall outside the LOCAL coarse hull (co-partitioned builder fallback)?
from scipy.spatial import Delaunay
tri = Delaunay(Xc)
outside_local = int((tri.find_simplex(Xf) < 0).sum())
tri_u = Delaunay(coords_u)
outside_union = int((tri_u.find_simplex(Xf) < 0).sum())
per = comm.gather((Xf.shape[0], outside_local, outside_union), root=0)
if comm.rank == 0: print("per rank (fine nodes, outside LOCAL coarse hull, outside UNION hull):", per, flush=True)
