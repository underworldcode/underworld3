"""Side-by-side: two simulations of the same physical problem,
one at dt = DT, one at dt = DT/2, stepped in lockstep.

Compare every comparable quantity at common physical times.
"""

import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


def make_stokes(label):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8),
        minCoords=(-1, -0.5), maxCoords=(1, 0.5))
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=2,
    )
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.shear_modulus = 1.0
    stokes.constitutive_model.Parameters.yield_stress = 0.5
    stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-6
    stokes.constitutive_model._yield_mode = "min"
    V_top = expression(rf"V_{{{label},top}}", sympy.Float(0.5), "Top V")
    stokes.add_dirichlet_bc((V_top, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_top, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_force_iteration"] = True
    return stokes, V_top


def probe(stokes, centre):
    cm = stokes.constitutive_model
    out = {}
    out['sigma']  = float(uw.function.evaluate(stokes.tau.sym[0, 1], centre).flatten()[0])
    out['edot']   = float(uw.function.evaluate(cm.grad_u[0, 1], centre).flatten()[0])
    out['psi0']   = float(uw.function.evaluate(stokes.DFDt.psi_star[0].sym[0, 1], centre).flatten()[0])
    out['psi1']   = float(uw.function.evaluate(stokes.DFDt.psi_star[1].sym[0, 1], centre).flatten()[0])
    out['eta_ve'] = float(uw.function.evaluate(cm.Parameters.ve_effective_viscosity.sym, centre).flatten()[0])
    out['eta_pl'] = float(uw.function.evaluate(cm._plastic_effective_viscosity, centre).flatten()[0])
    out['eta']    = float(uw.function.evaluate(cm.viscosity, centre).flatten()[0])
    out['Eeff']   = float(uw.function.evaluate(cm.E_eff.sym[0, 1], centre).flatten()[0])
    out['EeffII'] = float(uw.function.evaluate(cm.E_eff_inv_II.sym, centre).flatten()[0])
    out['c0'] = float(cm._bdf_c0.sym)
    out['c1'] = float(cm._bdf_c1.sym)
    out['c2'] = float(cm._bdf_c2.sym)
    out['dt_h0'] = stokes.DFDt._dt_history[0]
    dt_e = cm.Parameters.dt_elastic
    if hasattr(dt_e, 'sym'):
        dt_e = dt_e.sym
    try:
        out['dt_e'] = float(dt_e)
    except (TypeError, ValueError):
        out['dt_e'] = float('nan')
    return out


def step_one(stokes, V_top, dt, t_cur):
    V_top.sym = sympy.Float(0.5)  # constant +V0 — no BC flips, just pure loading
    stokes.constitutive_model.Parameters.dt_elastic = dt
    stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=1)


def fmt_row(label, t, p):
    return (f"{label:6s} t={t:.3f} dt_e={p['dt_e']:.3f} dt_h0={str(p['dt_h0']):>5s}  "
            f"σ={p['sigma']:.4f} ε̇={p['edot']:.4f} ψ*0={p['psi0']:.4f} ψ*1={p['psi1']:.4f}  "
            f"E={p['Eeff']:.3f}  η_ve={p['eta_ve']:.4f} η_pl={p['eta_pl']:.4f} η={p['eta']:.4f}  "
            f"c012=[{p['c0']:.3f},{p['c1']:.3f},{p['c2']:.3f}]")


# Build three independent problems
print("Building COARSE simulation (always dt = 0.20)...")
coarse, V_top_c = make_stokes("coarse")
print("Building FINE simulation (always dt = 0.10)...")
fine, V_top_f = make_stokes("fine")
print("Building SWITCH simulation (dt = 0.20 → 0.10 at outer step 4)...")
switch, V_top_s = make_stokes("switch")

centre = np.array([[0.0, 0.0]])
DT_C = 0.20
DT_F = 0.10
N_OUTER = 6  # 6 outer steps; halve in SWITCH starting at outer step 4
HALVE_AT = 4

print(f"\nLockstep: t advances by {DT_C} per outer step.")
print(f"  COARSE: 1 step at dt={DT_C}")
print(f"  FINE:   2 steps at dt={DT_F}")
print(f"  SWITCH: 1 step at dt={DT_C} for outer<{HALVE_AT}; then 2 steps at dt={DT_F}\n")

t_c, t_f, t_s = 0.0, 0.0, 0.0
for k in range(N_OUTER):
    step_one(coarse, V_top_c, DT_C, t_c)
    t_c += DT_C
    step_one(fine, V_top_f, DT_F, t_f);  t_f += DT_F
    step_one(fine, V_top_f, DT_F, t_f);  t_f += DT_F
    if k < HALVE_AT:
        step_one(switch, V_top_s, DT_C, t_s);  t_s += DT_C
    else:
        step_one(switch, V_top_s, DT_F, t_s);  t_s += DT_F
        step_one(switch, V_top_s, DT_F, t_s);  t_s += DT_F

    pc = probe(coarse, centre)
    pf = probe(fine, centre)
    ps = probe(switch, centre)
    marker = " <-- HALVING NOW" if k == HALVE_AT else ""
    print(f"--- outer step {k+1}, t = {t_c:.3f} {marker}")
    print(fmt_row("COARSE", t_c, pc))
    print(fmt_row("FINE",   t_f, pf))
    print(fmt_row("SWITCH", t_s, ps))
    print(f"  σ:   coarse={pc['sigma']:.4f}  fine={pf['sigma']:.4f}  switch={ps['sigma']:.4f}  "
          f"Δ(switch-fine)={ps['sigma']-pf['sigma']:+.4f}  Δ(switch-coarse)={ps['sigma']-pc['sigma']:+.4f}")
    print()
