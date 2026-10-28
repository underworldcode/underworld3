"""Fault multigrid hierarchies, four alternatives through one driver (2-D add-on Stokes, box in
simple shear, split fault). The background mesh (static base, levels L0 h=0.1, L1 h=0.05) is never
cut; add-on level j (j = 1..k) is base.adapt(h_near = 0.05 / 2^j), split by add_fault where the
variant asks. Transfers are cell-located Lagrange prolongations (cellP.py), side-aware on split
levels; every split level carries its own pair rotation with the opening DOFs removed.

  A   : finest split; add-on levels welded; everything Galerkin
  B2  : finest split; add-on levels welded with a TI band, REDISCRETISED; background Galerkin
  C   : all add-on levels split; everything Galerkin
  D   : all add-on levels split (Galerkin); background levels rediscretised with a broad TI zone
  stock: today's hierarchy (mesh.adapt + add_fault, existing transfers) — the A reference

python stage5_variants.py <k> <law frictionless|viscous> <variant> [weak=0.1] [wfac=1.0]
  TI band: Gaussian in distance to the trace, half-width wfac * (level's near-fault h);
  eta_weak = weak*eta (frictionless) or eta_f*w (viscous); geometric blend; director = fault normal.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp, sympy, underworld3 as uw
from petsc4py import PETSc
from underworld3.utilities import custom_mg, rotated_bc
from cellP import cell_located_prolongation
k, law, variant = int(sys.argv[1]), sys.argv[2], sys.argv[3]
weak = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
wfac = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
TIP_A, TIP_B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
FAULT = [("Flt", np.vstack([TIP_A, TIP_B]))]
ETA, ETA_F = 1.0, 1.0
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1, refinement=1, qdegree=2)
pts3 = np.column_stack([np.linspace(TIP_A[0], TIP_B[0], 9), np.linspace(TIP_A[1], TIP_B[1], 9), np.zeros(9)])
surf = uw.meshing.Surface("g", base, pts3); surf.discretize()
h_base = 0.05
AB = TIP_B - TIP_A
NRM = np.array([-AB[1], AB[0]]) / np.linalg.norm(AB)

def addon(j, split):
    ch = base.adapt(surf.refinement_metric_function(h_near=h_base / 2**j, h_far=h_base, width=0.08, profile="linear"),
                    max_levels=j)
    return ch.add_fault(FAULT) if split else ch

def ti_params(m, h_level):
    xc, yc = m.X
    A_, B_ = sympy.Matrix(TIP_A.tolist()), sympy.Matrix(TIP_B.tolist()); ab = B_ - A_
    t = ((xc - A_[0]) * ab[0] + (yc - A_[1]) * ab[1]) / ab.dot(ab)
    tc = sympy.Min(1, sympy.Max(0, t))
    d = sympy.sqrt((xc - A_[0] - tc * ab[0])**2 + (yc - A_[1] - tc * ab[1])**2)
    w = wfac * h_level
    f = sympy.exp(-(d / w)**2)
    eta_w = weak * ETA if law == "frictionless" else ETA_F * w
    return {"eta1": ETA**(1 - f) * eta_w**f, "director": sympy.Matrix([[NRM[0], NRM[1]]])}

def stokes_on(m, fault=True, ti=None):
    x, y = m.X
    s = uw.systems.Stokes(m)
    if ti is None:
        s.constitutive_model = uw.constitutive_models.ViscousFlowModel
        s.constitutive_model.Parameters.shear_viscosity_0 = ETA
    else:
        s.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
        s.constitutive_model.Parameters.shear_viscosity_0 = ETA
        s.constitutive_model.Parameters.shear_viscosity_1 = ti["eta1"]
        s.constitutive_model.Parameters.director = ti["director"]
    s.tolerance = 1e-6
    s.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), side)
    if fault:
        s.add_fault_bc(0 if law == "frictionless" else ETA_F, "Flt")
    return s

def to_scipy(M):
    ai, aj, av = M.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=M.getSize())

def to_petsc(S):
    S = S.tocsr(); S.sort_indices()
    return PETSc.Mat().createAIJ(size=S.shape, csr=(S.indptr.astype(PETSc.IntType), S.indices.astype(PETSc.IntType), S.data),
                                 comm=PETSc.COMM_SELF)

finest = addon(k, True)
stokes = stokes_on(finest)

if variant != "stock":
    background = list(base._coarse_level_meshes())                    # L0, L1
    split_addons = variant in ("C", "D", "Cm")
    addons = [addon(j, split_addons) for j in range(1, k)]
    levels = background + addons
    h_levels = [0.1, 0.05] + [h_base / 2**j for j in range(1, k)]
    # which coarse levels get a rediscretised TI operator
    if variant == "B2":
        ti_levels = set(range(2, len(levels)))                         # every add-on level below the finest
    elif variant == "D":
        ti_levels = {0, 1}                                             # the background levels
    else:
        ti_levels = set()
    rot, redisc = [], {}
    for li, m in enumerate(levels):
        if getattr(m, "_fault_point_pairs", None):
            s = stokes_on(m); s._build()
            Q, Qt, nrows, _ = rotated_bc.build_rotation(s, s._rotated_freeslip_bcs)
            vis = s._subdict["velocity"][0]; vidx = np.asarray(vis.getIndices())
            g2b = {int(g): i for i, g in enumerate(vidx)}
            rot.append((to_scipy(Q.createSubMatrix(vis, vis)), sorted({g2b[g] for g in nrows if g in g2b})))
        else:
            rot.append(None)
        if li in ti_levels:
            s = stokes_on(m, fault=False, ti=ti_params(m, h_levels[li]))
            s.penalty = float(stokes.penalty.sym) if hasattr(stokes.penalty, "sym") else stokes.penalty
            s._build()
            X = s.dm.getGlobalVec(); X.set(0.0)
            J, Jp = s.snes.getJacobian()[:2]
            s.snes.computeJacobian(X, J, Jp)
            vis = s._subdict["velocity"][0]
            redisc[li] = J.createSubMatrix(vis, vis)

    state = {}
    def my_Pl(solver, Q, normal_rows):
        vis = solver._subdict["velocity"][0]; vidx = np.asarray(vis.getIndices())
        g2b = {int(g): i for i, g in enumerate(vidx)}
        Qf = to_scipy(Q.createSubMatrix(vis, vis)); nf = sorted({g2b[g] for g in normal_rows if g in g2b})
        meshes = levels + [solver.mesh]
        rmaps = [custom_mg._coarse_reduced_map(solver, m, 0)[0] for m in levels] + [custom_mg._reduced_map(solver.dm, 0)[0]]
        rots = rot + [(Qf, nf)]
        keep = [np.setdiff1d(np.arange(len(rm)), r[1]) if r is not None else np.arange(len(rm)) for r, rm in zip(rots, rmaps)]
        # Built from the finest level down, so each level's admissible set is final
        # before the level below is prolongated into it. A coarse unknown that no fine
        # unknown reaches (a pair DOF near a tip of non-nested split levels) is dropped:
        # it is a null mode of the Galerkin coarse operator (a zero diagonal under SOR).
        nlev_ = len(meshes)
        Pl = [None] * (nlev_ - 1)
        for l in range(nlev_ - 2, -1, -1):
            Pn, inside = cell_located_prolongation(meshes[l], meshes[l + 1])
            P = custom_mg._reduced_from_node_transfer(Pn, rmaps[l], rmaps[l + 1], 2).tocsr()
            if rots[l + 1] is not None: P = rots[l + 1][0] @ P
            if rots[l] is not None: P = P @ rots[l][0].T
            if l + 1 == nlev_ - 1:
                P = P.tolil(); P[nf, :] = 0; P = P.tocsr()
            else:
                P = P[keep[l + 1], :]
            P = P[:, keep[l]]
            P.eliminate_zeros()
            cn = np.sqrt(np.asarray(P.multiply(P).sum(axis=0)).ravel())
            reached = cn > 0.05 * np.median(cn[cn > 0])
            dropped = int((~reached).sum())
            if dropped:
                print(f"DEBUGP level {l}: dropped {dropped} unreached coarse DOFs", flush=True)
                keep[l] = keep[l][reached]
                P = P[:, np.flatnonzero(reached)]
            Pl[l] = to_petsc(P)
        if getattr(solver, "_custom_mg", None) is None:
            solver._custom_mg = {"mode": "hierarchy", "hierarchy": None, "verbose": False}
        state["keep"] = keep
        return Pl
    rotated_bc._build_rotated_custom_Pl = my_Pl

    if redisc or variant == "Cm":
        orig = custom_mg._configure_pcmg
        def with_redisc(pc, Pl, *a, **kw):
            out = orig(pc, Pl, *a, **kw)
            nlev = len(Pl) + 1
            ops = [None] * nlev
            ops[nlev - 1] = pc.getOperators()[0]
            for l in range(nlev - 2, -1, -1):
                if l in redisc:                       # rediscretised level, on its admissible set
                    kl = state["keep"][l]
                    if len(kl) == redisc[l].getSize()[0]:
                        ops[l] = redisc[l]
                    else:
                        isk = PETSc.IS().createGeneral(np.asarray(kl, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
                        ops[l] = redisc[l].createSubMatrix(isk, isk)
                else:
                    ops[l] = ops[l + 1].PtAP(Pl[l])
            for l in range(nlev):
                d = ops[l].getDiagonal().getArray()
                print(f"DEBUG level {l} size {ops[l].getSize()} type {ops[l].getType()} norm {ops[l].norm():.3e} "
                      f"diag min {np.abs(d).min():.3e} nan {int(np.isnan(d).sum())} P {Pl[l].getSize() if l < nlev - 1 else '-'}", flush=True)
            PETSc.Options()[(pc.getOptionsPrefix() or "") + "pc_mg_galerkin"] = "none"
            pc.setFromOptions()
            for l in range(nlev - 1):
                kk = pc.getMGCoarseSolve() if l == 0 else pc.getMGSmoother(l)
                kk.setOperators(ops[l], ops[l])
            return out
        custom_mg._configure_pcmg = with_redisc

opts = PETSc.Options()
for i in range(1, 6):
    opts[f"rotfs{i}_fieldsplit_vel_ksp_converged_reason"] = None
stokes.solve()
t0 = time.time(); stokes.solve(zero_init_guess=True); wall = time.time() - t0
info = stokes._rotated_freeslip_info
from underworld3.utilities.fault_contact import fault_slip
s_, V, leak = fault_slip(stokes, "Flt", info)
print(f"RESULT V k={k} law={law} variant={variant} weak={weak} wfac={wfac} h_near={h_base / 2**k:.5f} "
      f"pc={info['velocity_pc']} vel_its={info['vel_its_last']} pres_its={info['pres_its_last']} conv={info['converged']} "
      f"peak_slip={np.abs(V).max():.5f} wall={wall:.1f}s", flush=True)
