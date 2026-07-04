"""GAMG preset sweep on a saturated-R adapted mesh.
Cold-only (V,P=0 IC) since the R_compare snapshots only carry T.
"""
import os
import time
import argparse
import numpy as np
import sympy
import underworld3 as uw


PRESETS = {
    'default':       {},
    'gamg-n1':       {'pc_gamg_agg_nsmooths': 1},
    'gamg-thr':      {'pc_gamg_threshold': 0.02,
                      'pc_gamg_threshold_scale': 0.5},
    'gamg-noagr':    {'pc_gamg_aggressive_coarsening': 0},
    'gamg-sor':      {'mg_levels_ksp_type': 'richardson',
                      'mg_levels_pc_type': 'sor',
                      'mg_levels_ksp_max_it': 2},
    'gamg-noagrsor': {'pc_gamg_aggressive_coarsening': 0,
                      'mg_levels_ksp_type': 'richardson',
                      'mg_levels_pc_type': 'sor',
                      'mg_levels_ksp_max_it': 2},
    'gamg-full':     {'pc_gamg_agg_nsmooths': 1,
                      'pc_gamg_threshold': 0.02,
                      'pc_gamg_threshold_scale': 0.5,
                      'pc_gamg_aggressive_coarsening': 0,
                      'mg_levels_ksp_type': 'richardson',
                      'mg_levels_pc_type': 'sor',
                      'mg_levels_ksp_max_it': 2},
    'direct':        {'ksp_type': 'preonly',
                      'pc_type': 'lu',
                      'pc_factor_mat_solver_type': 'mumps',
                      'mat_mumps_icntl_24': 1},
}


p = argparse.ArgumentParser()
p.add_argument('--R', type=float, default=3.0)
args = p.parse_args()

BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/R_compare')
SRC = os.path.join(BASE, f"R{args.R}")
Ra = 1.0e7
theta_FK = float(np.log(1.0e4))


def build_and_solve(preset_name):
    """Build Stokes with this preset, cold-solve, report."""
    m = uw.discretisation.Mesh(
        os.path.join(SRC, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_{preset_name}_{id(m)}", m,
        vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    V = uw.discretisation.MeshVariable(
        f"V_{preset_name}_{id(m)}", m,
        vtype=uw.VarType.VECTOR, degree=2, continuous=True)
    P = uw.discretisation.MeshVariable(
        f"P_{preset_name}_{id(m)}", m,
        vtype=uw.VarType.SCALAR, degree=1, continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0, outputPath=SRC)
    X = m.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    unit_r = m.CoordinateSystem.unit_e_0
    s = uw.systems.Stokes(m, velocityField=V, pressureField=P)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = (
        sympy.exp(theta_FK * (1 - T.sym[0])))
    s.tolerance = 1.0e-5
    s.penalty = 0.0
    s.add_essential_bc((0.0, 0.0), m.boundaries.Lower.name)
    KFS = 1.0e6
    fs_term = (KFS * V.sym.dot(unit_r) * unit_r)
    s.add_natural_bc(fs_term, m.boundaries.Upper.name)
    T_cond = sympy.log(r_sym / 1.0) / sympy.log(0.5 / 1.0)
    s.bodyforce = Ra * (T.sym[0] - T_cond) * unit_r
    for k, v in PRESETS[preset_name].items():
        s.petsc_options[k] = v
    V.data[...] = 0.0
    P.data[...] = 0.0
    t0 = time.time()
    s.solve(zero_init_guess=True)
    wall = time.time() - t0
    reason = int(s.snes.getConvergedReason())
    its = int(s.snes.getIterationNumber())
    vmax = float(np.sqrt(V.data[:, 0] ** 2
                         + V.data[:, 1] ** 2).max())
    return reason, its, wall, vmax


print(f"GAMG sweep on R={args.R} adapted mesh "
      f"(cold solve, V,P=0)")
print(f"{'preset':>16} {'reason':>6} {'its':>4} {'wall':>8}  "
      f"{'|v|max':>10}")
print("-" * 56)
for ps in PRESETS:
    reason, its, wall, vmax = build_and_solve(ps)
    print(f"{ps:>16} {reason:>6} {its:>4} {wall:>7.2f}s  "
          f"{vmax:>10.3e}", flush=True)
