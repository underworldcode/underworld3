"""Verify GAMG presets actually take effect: dump KSP view +
residual trace for default vs gamg-noagr on the R=3 mesh.
"""
import os
import sys
import numpy as np
import sympy
import underworld3 as uw


BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/R_compare')
SRC = os.path.join(BASE, "R3.0")
Ra = 1.0e7
theta_FK = float(np.log(1.0e4))


def run(preset_name, preset_opts):
    print("=" * 72)
    print(f"PRESET: {preset_name}")
    print("=" * 72)
    m = uw.discretisation.Mesh(
        os.path.join(SRC, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_{preset_name}", m,
        vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    V = uw.discretisation.MeshVariable(
        f"V_{preset_name}", m,
        vtype=uw.VarType.VECTOR, degree=2, continuous=True)
    P = uw.discretisation.MeshVariable(
        f"P_{preset_name}", m,
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
    # Apply preset
    for k, v in preset_opts.items():
        s.petsc_options[k] = v
    # Diagnostics
    s.petsc_options["ksp_monitor"] = None
    s.petsc_options["snes_monitor"] = None
    V.data[...] = 0.0
    P.data[...] = 0.0
    s.solve(zero_init_guess=True)
    # Final solution stats — higher precision
    vmag = np.sqrt(V.data[:, 0] ** 2 + V.data[:, 1] ** 2)
    print(f"\nFINAL: |v|max = {vmag.max():.12e}  "
          f"|v|rms = {np.sqrt(np.mean(vmag**2)):.12e}", flush=True)
    # Print actual petsc_options as registered (probe via getString)
    print(f"\npetsc_options snapshot (subset):")
    keys = ['pc_gamg_aggressive_coarsening',
            'fieldsplit_velocity_pc_gamg_aggressive_coarsening',
            'fieldsplit_velocity_pc_gamg_threshold',
            'fieldsplit_velocity_mg_levels_ksp_type',
            'fieldsplit_velocity_mg_levels_pc_type',
            'fieldsplit_velocity_pc_gamg_agg_nsmooths',
            'pc_type',
            'snes_atol',
            'snes_rtol']
    for k in keys:
        try:
            val = s.petsc_options[k]
            print(f"  {k} = {val}")
        except KeyError:
            print(f"  {k}: <unset>")
    print()


PRESETS = {
    'default':         {},
    # WRONG scope (what we and the catalogue have been doing)
    'gamg-noagr-WRONG': {'pc_gamg_aggressive_coarsening': 0},
    # CORRECT scope — option lives on the velocity Schur subsolve
    'gamg-noagr-CORR':  {'fieldsplit_velocity_pc_gamg_aggressive_coarsening': 0},
    'gamg-thr-CORR':    {
        'fieldsplit_velocity_pc_gamg_threshold': 0.02,
        'fieldsplit_velocity_pc_gamg_threshold_scale': 0.5},
    'gamg-noagrsor-CORR': {
        'fieldsplit_velocity_pc_gamg_aggressive_coarsening': 0,
        'fieldsplit_velocity_mg_levels_ksp_type': 'richardson',
        'fieldsplit_velocity_mg_levels_pc_type': 'sor',
        'fieldsplit_velocity_mg_levels_ksp_max_it': 2},
}

for ps, opts in PRESETS.items():
    run(ps, opts)
