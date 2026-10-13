"""Moving fault on a static background: per-step cost of add-on meshing, fault surgery,
hierarchy building and solving, for strategies A / C / D / gamg.

Box in simple shear, frictionless straight fault of fixed length and orientation whose centre
migrates x = 0.3 -> 0.7 in `steps` steps. The background (UnstructuredSimplexBox refinement=1) is
built ONCE and never changes; every step re-adapts the add-on part from it (one mesh.adapt call),
splits the finest level, and (C, D) splits each intermediate NVB level of that same adapt call.

python stage6_moving.py <strategy A|C|D|gamg> <k> [steps=8]
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp, sympy, underworld3 as uw
from petsc4py import PETSc
from underworld3.utilities import custom_mg, rotated_bc
from cellP import cell_located_prolongation
strategy, k = sys.argv[1], int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
HALF = np.array([0.20, 0.05])                      # half-vector of the fault (length 0.41)
ETA, WEAK = 1.0, 0.1
h_base = 0.05
T = {}
def tic(name):
    T.setdefault(name, 0.0); T["_t0"] = time.perf_counter(); T["_n"] = name
def toc():
    T[T["_n"]] += time.perf_counter() - T["_t0"]

t_bg = time.perf_counter()
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1, refinement=1, qdegree=2)
background = list(base._coarse_level_meshes())
t_bg = time.perf_counter() - t_bg

def to_scipy(M):
    ai, aj, av = M.getValuesCSR(); return sp.csr_matrix((av, aj, ai), shape=M.getSize())
def to_petsc(S):
    S = S.tocsr(); S.sort_indices()
    return PETSc.Mat().createAIJ(size=S.shape, csr=(S.indptr.astype(PETSc.IntType), S.indices.astype(PETSc.IntType), S.data),
                                 comm=PETSc.COMM_SELF)

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
        s.add_fault_bc(0, "Flt")
    return s

def ti_params(m, A, B, h_level):
    xc, yc = m.X
    A_, B_ = sympy.Matrix(A.tolist()), sympy.Matrix(B.tolist()); ab = B_ - A_
    t = ((xc - A_[0]) * ab[0] + (yc - A_[1]) * ab[1]) / ab.dot(ab)
    tc = sympy.Min(1, sympy.Max(0, t))
    d = sympy.sqrt((xc - A_[0] - tc * ab[0])**2 + (yc - A_[1] - tc * ab[1])**2)
    f = sympy.exp(-(d / h_level)**2)
    nrm = np.array([-(B - A)[1], (B - A)[0]]); nrm /= np.linalg.norm(nrm)
    return {"eta1": ETA**(1 - f) * (WEAK * ETA)**f, "director": sympy.Matrix([[nrm[0], nrm[1]]])}

state = {}
NT = {}
def timed(name, fn):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            NT[name] = NT.get(name, 0.0) + time.perf_counter() - t0
    return wrapper
custom_mg.build_transfers = timed("transfers", custom_mg.build_transfers)
rotated_bc.build_rotation = timed("rotation", rotated_bc.build_rotation)
def my_Pl(solver, Q, normal_rows):
    levels, rot = state["levels"], state["rot"]
    vis = solver._subdict["velocity"][0]; vidx = np.asarray(vis.getIndices())
    g2b = {int(g): i for i, g in enumerate(vidx)}
    Qf = to_scipy(Q.createSubMatrix(vis, vis)); nf = sorted({g2b[g] for g in normal_rows if g in g2b})
    meshes = levels + [solver.mesh]
    rmaps = [custom_mg._coarse_reduced_map(solver, m, 0)[0] for m in levels] + [custom_mg._reduced_map(solver.dm, 0)[0]]
    rots = rot + [(Qf, nf)]
    keep = [np.setdiff1d(np.arange(len(rm)), r[1]) if r is not None else np.arange(len(rm)) for r, rm in zip(rots, rmaps)]
    nlev = len(meshes); Pl = [None] * (nlev - 1)
    for l in range(nlev - 2, -1, -1):
        Pn, _ = cell_located_prolongation(meshes[l], meshes[l + 1])
        P = custom_mg._reduced_from_node_transfer(Pn, rmaps[l], rmaps[l + 1], 2).tocsr()
        if rots[l + 1] is not None: P = rots[l + 1][0] @ P
        if rots[l] is not None: P = P @ rots[l][0].T
        if l + 1 == nlev - 1:
            P = P.tolil(); P[nf, :] = 0; P = P.tocsr()
        else:
            P = P[keep[l + 1], :]
        P = P[:, keep[l]]; P.eliminate_zeros()
        # a coarse unknown reached only weakly (tiny or extrapolated weights near a tip of
        # non-nested split levels) gives a near-zero Galerkin diagonal: drop it with the
        # unreached ones (column 2-norm below 5% of the median column)
        cn = np.sqrt(np.asarray(P.multiply(P).sum(axis=0)).ravel())
        reached = cn > 0
        if not reached.all():
            keep[l] = keep[l][reached]; P = P[:, np.flatnonzero(reached)]
        Pl[l] = to_petsc(P)
    state["keep"] = keep
    if getattr(solver, "_custom_mg", None) is None:
        solver._custom_mg = {"mode": "hierarchy", "hierarchy": None, "verbose": False}
    return Pl

orig_configure = custom_mg._configure_pcmg
def with_redisc(pc, Pl, *a, **kw):
    out = orig_configure(pc, Pl, *a, **kw)
    redisc = state.get("redisc") or {}
    if not redisc:
        return out
    nlev = len(Pl) + 1
    ops = [None] * nlev
    ops[nlev - 1] = pc.getOperators()[0]
    for l in range(nlev - 2, -1, -1):
        if l in redisc:
            kl = state["keep"][l]
            ops[l] = redisc[l] if len(kl) == redisc[l].getSize()[0] else redisc[l].createSubMatrix(
                *(2 * [PETSc.IS().createGeneral(np.asarray(kl, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)]))
        else:
            ops[l] = ops[l + 1].PtAP(Pl[l])
    PETSc.Options()[(pc.getOptionsPrefix() or "") + "pc_mg_galerkin"] = "none"
    pc.setFromOptions()
    for l in range(nlev - 1):
        (pc.getMGCoarseSolve() if l == 0 else pc.getMGSmoother(l)).setOperators(ops[l], ops[l])
    return out
if strategy in ("C", "D"):
    rotated_bc._build_rotated_custom_Pl = timed("transfers", my_Pl)
    custom_mg._configure_pcmg = timed("mg_setup", with_redisc)

records = []
centres = np.linspace(0.3, 0.7, steps)
for step, cx in enumerate(centres):
    for key in [k_ for k_ in T if not k_.startswith("_")]:
        T[key] = 0.0
    A_ = np.array([cx, 0.5]) - HALF; B_ = np.array([cx, 0.5]) + HALF
    trace = np.vstack([A_, B_])
    tic("addon_adapt")
    pts3 = np.column_stack([np.linspace(A_[0], B_[0], 9), np.linspace(A_[1], B_[1], 9), np.zeros(9)])
    surf = uw.meshing.Surface(f"g{step}", base, pts3); surf.discretize()
    child = base.adapt(surf.refinement_metric_function(h_near=h_base / 2**k, h_far=h_base, width=0.08, profile="linear"),
                       max_levels=k)
    toc()
    tic("split_finest")
    finest = child.add_fault([("Flt", trace)])
    toc()
    tail = list(child._custom_mg_coarse_meshes)
    inter = tail[len(background):]
    if strategy in ("C", "D"):
        tic("split_levels")
        split_inter = [lev.add_fault([("Flt", trace)]) for lev in inter]
        toc()
        state["levels"] = background + split_inter
        tic("level_frames")
        rot = [None] * len(background)
        for m in split_inter:
            s = stokes_on(m); s._build(); s.snes.setUp()
            Q, Qt, nrows, _ = rotated_bc.build_rotation(s, s._rotated_freeslip_bcs)
            vis = s._subdict["velocity"][0]; vidx = np.asarray(vis.getIndices())
            g2b = {int(g): i for i, g in enumerate(vidx)}
            rot.append((to_scipy(Q.createSubMatrix(vis, vis)), sorted({g2b[g] for g in nrows if g in g2b})))
        state["rot"] = rot
        toc()
        state["redisc"] = {}
        if strategy == "D":
            tic("background_TI")
            for li, (m, hl) in enumerate(zip(background, (0.1, 0.05))):
                s = stokes_on(m, fault=False, ti=ti_params(m, A_, B_, hl)); s._build(); s.snes.setUp()
                X = s.dm.getGlobalVec(); X.set(0.0)
                J, Jp = s.snes.getJacobian()[:2]; s.snes.computeJacobian(X, J, Jp)
                state["redisc"][li] = J.createSubMatrix(*(2 * [s._subdict["velocity"][0]]))
            toc()
    tic("solver_build")
    stokes = stokes_on(finest)
    if strategy == "gamg":
        stokes.preconditioner = "gamg"
    toc()
    tic("jit_build")
    stokes._build()
    toc()
    NT.clear()
    tic("first_solve")              # assembly, hierarchy transfers, MG setup, Krylov
    stokes.solve()
    toc()
    info = stokes._rotated_freeslip_info
    from underworld3.utilities.fault_contact import fault_slip
    s_, V, leak = fault_slip(stokes, "Flt", info)
    rec = {"step": step, "cx": float(cx), "cells": int(finest.dm.getHeightStratum(0)[1]),
           "levels": len(tail) + 1, "pc": info["velocity_pc"], "vel_its": info["vel_its_last"],
           "pres_its": info["pres_its_last"], "conv": bool(info["converged"]), "peak_slip": float(np.abs(V).max()),
           "times": {kk: round(v, 3) for kk, v in T.items() if not kk.startswith("_")},
           "inside_solve": {kk: round(v, 3) for kk, v in NT.items()}}
    records.append(rec)
    print("STEP", json.dumps(rec), flush=True)
out = os.path.expanduser(f"~/+Simulations/fault_mg_exploration/moving_{strategy}_k{k}{os.environ.get('TAG', '')}.json")
json.dump({"strategy": strategy, "k": k, "background_build": t_bg, "records": records}, open(out, "w"), indent=1)
print("DONE", out, flush=True)
