"""Stage 4: case C on the 3-D extruded faulted layer.

Today's extrude tail: lateral levels (unsplit sheet levels, 1 layer) + through-thickness levels
extruded from the UNSPLIT cut sheet. Variant C: the through-thickness levels are extruded from the
SPLIT sheet (they keep the fault), cell-located side-aware transfers, per-level pair rotation.

python stage4_thick.py <refinement> <layers> <variant: split|stock>
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests"))
import numpy as np, scipy.sparse as sp, underworld3 as uw
from petsc4py import PETSc
from underworld3.utilities import custom_mg, rotated_bc
from underworld3.meshing.extrude import extrude_sheet
from cellP import cell_located_prolongation
import test_1083_thin_thick_alignment as T
ref, layers, variant = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
sheet = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1, refinement=ref, qdegree=3)
split = sheet.add_fault([("Flt", np.array([[0.25, 0.5], [0.75, 0.5]]))])
layer_mesh = extrude_sheet(split, T.L, layers=layers)       # stock tail attached
layer = T.model(layer_mesh, uw.systems.Stokes)

def to_scipy(M):
    ai, aj, av = M.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=M.getSize())

def to_petsc(S):
    S = S.tocsr(); S.sort_indices()
    return PETSc.Mat().createAIJ(size=S.shape, csr=(S.indptr.astype(PETSc.IntType), S.indices.astype(PETSc.IntType), S.data),
                                 comm=PETSc.COMM_SELF)

if variant == "split":
    lateral = [extrude_sheet(lev, T.L, layers=1, with_tail=False) for lev in list(sheet._coarse_level_meshes())[:-1]]
    vertical = []
    n = layers // 2
    while n >= 1:
        vertical.append(extrude_sheet(split, T.L, layers=n, with_tail=False)); n //= 2
    levels = lateral + list(reversed(vertical))
    rot = []
    for m in levels:
        if getattr(m, "_fault_point_pairs", None):
            s = T.model(m, uw.systems.Stokes); s._build(); s.snes.setUp()
            Q, Qt, nrows, _ = rotated_bc.build_rotation(s, s._rotated_freeslip_bcs)
            vis = s._subdict["velocity"][0]; vidx = np.asarray(vis.getIndices())
            g2b = {int(g): i for i, g in enumerate(vidx)}
            rot.append((to_scipy(Q.createSubMatrix(vis, vis)), sorted({g2b[g] for g in nrows if g in g2b})))
        else:
            rot.append(None)

    def my_Pl(solver, Q, normal_rows):
        vis = solver._subdict["velocity"][0]; vidx = np.asarray(vis.getIndices())
        g2b = {int(g): i for i, g in enumerate(vidx)}
        Qf = to_scipy(Q.createSubMatrix(vis, vis)); nf = sorted({g2b[g] for g in normal_rows if g in g2b})
        meshes = levels + [solver.mesh]
        rmaps = [custom_mg._coarse_reduced_map(solver, m, 0)[0] for m in levels] + [custom_mg._reduced_map(solver.dm, 0)[0]]
        rots = rot + [(Qf, nf)]
        keep = [np.setdiff1d(np.arange(len(rm)), r[1]) if r is not None else np.arange(len(rm)) for r, rm in zip(rots, rmaps)]
        Pl = []
        for l in range(len(meshes) - 1):
            Pn, inside = cell_located_prolongation(meshes[l], meshes[l + 1])
            P = custom_mg._reduced_from_node_transfer(Pn, rmaps[l], rmaps[l + 1], 3).tocsr()
            if rots[l + 1] is not None: P = rots[l + 1][0] @ P
            if rots[l] is not None: P = P @ rots[l][0].T
            P = P[:, keep[l]]
            if l + 1 == len(meshes) - 1:
                P = P.tolil(); P[nf, :] = 0; P = P.tocsr()
            else:
                P = P[keep[l + 1], :]
            P.eliminate_zeros()
            Pl.append(to_petsc(P))
        if getattr(solver, "_custom_mg", None) is None:
            solver._custom_mg = {"mode": "hierarchy", "hierarchy": None, "verbose": False}
        return Pl
    rotated_bc._build_rotated_custom_Pl = my_Pl

layer.solve()
t0 = time.time(); layer.solve(zero_init_guess=True); wall = time.time() - t0
info = layer._rotated_freeslip_info
print(f"RESULT thick ref={ref} h={0.1/2**ref:.4f} layers={layers} variant={variant} pc={info['velocity_pc']} outer={info['ksp_its']} "
      f"vel_its={info['vel_its_last']} pres_its={info['pres_its_last']} conv={info['converged']} "
      f"umax={np.abs(np.asarray(layer.u.array)).max():.6f} wall={wall:.1f}s", flush=True)
