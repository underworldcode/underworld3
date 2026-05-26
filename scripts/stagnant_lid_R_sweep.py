"""Adapt-strength sweep: how does resolution_ratio affect both
mesh quality and Stokes-solve cost on the adapted mesh?

For each R in {1.0=no-op, 1.2, 1.5, 2.0, 3.0}:
  - adapt the step-125 T snapshot
  - report mesh.quality stats (minA/meanA, area max/min, edge p95/p05)
  - run a cold+warm Stokes solve with default GAMG, record wall+iters
"""
from __future__ import annotations
import os
import time
import argparse
import numpy as np
import sympy

import underworld3 as uw


SRC = os.path.expanduser(
    '~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
STEM = "sl_uniform_res16_Ra1e7_dEta1e4_step00125"
Ra = 1.0e7
theta_FK = float(np.log(1.0e4))

p = argparse.ArgumentParser()
p.add_argument('--R-list', type=str,
               default="1.0,1.2,1.5,2.0,3.0")
args = p.parse_args()

R_list = [float(x) for x in args.R_list.split(',')]


def build_problem(mesh, T, V, P):
    X = mesh.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    unit_r = mesh.CoordinateSystem.unit_e_0
    s = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = (
        sympy.exp(theta_FK * (1 - T.sym[0])))
    s.tolerance = 1.0e-5
    s.penalty = 0.0
    s.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    KFS = 1.0e6
    fs_term = (KFS * V.sym.dot(unit_r) * unit_r)
    s.add_natural_bc(fs_term, mesh.boundaries.Upper.name)
    T_cond = sympy.log(r_sym / 1.0) / sympy.log(0.5 / 1.0)
    s.bodyforce = Ra * (T.sym[0] - T_cond) * unit_r
    return s


def mesh_stats(mesh):
    from underworld3.meshing.smoothing import (
        _tri_cells, _signed_areas)
    tris = _tri_cells(mesh.dm)
    A = np.abs(_signed_areas(np.asarray(mesh.X.coords), tris))
    return dict(minA=A.min(), meanA=A.mean(), maxA=A.max(),
                minA_meanA=A.min() / A.mean(),
                area_ratio=A.max() / A.min())


def run_solve(stokes, V, P, mode):
    if mode == 'cold':
        V.data[...] = 0.0
        P.data[...] = 0.0
        zero = True
    else:
        zero = False
    t0 = time.time()
    try:
        stokes.solve(zero_init_guess=zero)
        wall = time.time() - t0
        reason = int(stokes.snes.getConvergedReason())
        its = int(stokes.snes.getIterationNumber())
    except Exception as e:
        return dict(mode=mode, wall=None, reason=None,
                    its=None, err=str(e))
    vmax = float(np.sqrt(V.data[:, 0] ** 2
                         + V.data[:, 1] ** 2).max())
    return dict(mode=mode, wall=wall, reason=reason,
                its=its, vmax=vmax)


print(f"adapt-strength sweep: R in {R_list}")
print(f"{'R':>5} | {'minA/meanA':>10} {'A max/min':>10} | "
      f"{'cold its':>8} {'cold wall':>10} "
      f"{'warm its':>8} {'warm wall':>10}  {'|v|max':>10}")
print("-" * 90)

results = []
for R in R_list:
    # Fresh mesh load each round (avoid cross-R contamination)
    mesh = uw.discretisation.Mesh(
        os.path.join(SRC, f"{STEM}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    V = uw.discretisation.MeshVariable(
        "V_v2p1", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True)
    P = uw.discretisation.MeshVariable(
        "P_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    T.read_timestep(STEM, "T_v2p1", 0, outputPath=SRC)
    V.read_timestep(STEM, "V_v2p1", 0, outputPath=SRC)
    P.read_timestep(STEM, "P_v2p1", 0, outputPath=SRC)

    # Adapt (skip when R<=1 = no-op, just take uniform mesh stats)
    if R > 1.0:
        rho = uw.meshing.metric_density_from_gradient(
            mesh, T, amp=8.0,
            lo_percentile=50.0, hi_percentile=97.0,
            name=f"R{R:.1f}")
        old_X = np.asarray(mesh.X.coords).copy()
        old_T = np.asarray(T.data).copy()
        old_V = np.asarray(V.data).copy()
        old_P = np.asarray(P.data).copy()
        t_adapt0 = time.time()
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="anisotropic",
            method_kwargs=dict(resolution_ratio=R,
                               relax=0.2, n_outer=12))
        t_adapt = time.time() - t_adapt0
        # Remap
        new_X = np.asarray(mesh.X.coords).copy()
        new_Tx = np.asarray(T.coords).copy()
        new_Vx = np.asarray(V.coords).copy()
        new_Px = np.asarray(P.coords).copy()
        mesh._deform_mesh(old_X)
        T.data[...] = old_T
        V.data[...] = old_V
        P.data[...] = old_P
        rT = np.asarray(uw.function.evaluate(
            T.sym[0], new_Tx)).reshape(-1)
        rV = np.asarray(uw.function.evaluate(V.sym, new_Vx))
        rP = np.asarray(uw.function.evaluate(
            P.sym[0], new_Px)).reshape(-1)
        mesh._deform_mesh(new_X)
        T.data[:, 0] = rT
        V.data[...] = rV.reshape(V.data.shape)
        P.data[:, 0] = rP
        V_remap = np.asarray(V.data).copy()
        P_remap = np.asarray(P.data).copy()
    else:
        t_adapt = 0.0
        V_remap = np.asarray(V.data).copy()
        P_remap = np.asarray(P.data).copy()

    stats = mesh_stats(mesh)
    stokes = build_problem(mesh, T, V, P)

    cold = run_solve(stokes, V, P, 'cold')
    # Warm: feed the remap V,P back
    V.data[...] = V_remap
    P.data[...] = P_remap
    warm = run_solve(stokes, V, P, 'warm')

    line = (f"{R:>5.2f} | {stats['minA_meanA']:>10.4f} "
            f"{stats['area_ratio']:>10.2f} | "
            f"{cold['its']:>8d} {cold['wall']:>9.2f}s "
            f"{warm['its']:>8d} {warm['wall']:>9.2f}s  "
            f"{cold['vmax']:>10.2e}")
    print(line, flush=True)
    results.append(dict(R=R, stats=stats, cold=cold, warm=warm,
                        adapt_wall=t_adapt))

# Save
out = os.path.expanduser(
    '~/+Simulations/StagnantLid/R_sweep_summary.npz')
np.savez(out,
         R=np.asarray([r['R'] for r in results]),
         minA_meanA=np.asarray([r['stats']['minA_meanA']
                                for r in results]),
         area_ratio=np.asarray([r['stats']['area_ratio']
                                for r in results]),
         cold_its=np.asarray([r['cold']['its'] for r in results]),
         cold_wall=np.asarray([r['cold']['wall'] for r in results]),
         warm_its=np.asarray([r['warm']['its'] for r in results]),
         warm_wall=np.asarray([r['warm']['wall'] for r in results]),
         adapt_wall=np.asarray([r['adapt_wall'] for r in results]),
         vmax=np.asarray([r['cold']['vmax'] for r in results]))
print(f"\nsaved {out}")
