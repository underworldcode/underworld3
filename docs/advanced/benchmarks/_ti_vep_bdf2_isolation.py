"""Isolate which factor triggers TI-VEP BDF-2 blow-up.

Reference (working): tests/test_1052::test_ti_vep_yield_lock_variable_dt
  - constant V_top, scalar τ_y, min yield, BDF-2 → stable
Failing: bench_ti_vep_harmonic at θ=0°
  - harmonic V_top, spatial τ_y field, softmin yield, BDF-2 → blows up

Variables to flip (3 dimensions, baseline + 3 single-flip variants):

  baseline (failing):  harmonic forcing, spatial τ_y, softmin
  variant A:            const forcing,    spatial τ_y, softmin
  variant B:            harmonic forcing, scalar  τ_y, softmin
  variant C:            harmonic forcing, spatial τ_y, min

Whichever flip stabilises BDF-2 identifies the trigger.
"""

import os
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


V0 = 0.5
OMEGA = np.pi / 2.0
DT = 0.05
T_END = 16.0         # 4 periods — match the original benchmark length
ETA_0 = 1.0; ETA_1 = 1.0; MU = 1.0
TAU_Y = 0.30
TAU_Y_BULK = 200.0
RES = 16


def build(label, *, spatial_yield, yield_mode):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        qdegree=3,
    )
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, 2, degree=2,
                                        vtype=uw.VarType.VECTOR)
    p = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1,
                                        continuous=True, vtype=uw.VarType.SCALAR)
    fault = uw.meshing.Surface(
        f"fault_{label}", mesh,
        np.array([[0.2, 0.5], [0.8, 0.5]]),  # horizontal fault, θ=0
        symbol=f"F{label}",
    )
    fault.discretize()
    if spatial_yield:
        weakness = fault.influence_function(
            width=0.06, value_near=1.0/TAU_Y, value_far=1.0/TAU_Y_BULK,
            profile="gaussian",
        )
        ty = 1.0 / weakness
    else:
        ty = TAU_Y

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
        stokes.Unknowns, order=2,
    )
    stokes.constitutive_model = cm
    cm.Parameters.shear_viscosity_0 = ETA_0
    cm.Parameters.shear_viscosity_1 = ETA_1
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = ty
    cm.Parameters.director = sympy.Matrix([0.0, 1.0])  # θ=0 throughout
    cm.Parameters.strainrate_inv_II_min = 1.0e-6
    cm._yield_mode = yield_mode

    V_top = expression(rf"V_{{top,{label}}}", sympy.Float(0.0), "")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_force_iteration"] = True
    return stokes, V_top


def run(label, *, spatial_yield, yield_mode, harmonic):
    stokes, V_top = build(label, spatial_yield=spatial_yield, yield_mode=yield_mode)
    phi = float(np.arctan(OMEGA))
    n_steps = int(T_END / DT)
    sxy = []
    div = 0
    iters_total = 0
    t0 = time.time()
    for step in range(n_steps):
        t = (step + 1) * DT
        v_now = (V0 * float(np.cos(OMEGA * t + phi))) if harmonic else V0
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = DT
        stokes.solve(zero_init_guess=False, timestep=DT, divergence_retries=2)
        if stokes.snes.getConvergedReason() < 0:
            div += 1
        iters_total += stokes.snes.getIterationNumber()
        # Probe centre
        c = np.array([[0.5, 0.5]])
        td = stokes.tau.data
        idx = int(np.argmin(np.linalg.norm(stokes.tau.coords - c, axis=1)))
        sxy.append(td[idx, 2])
    wall = time.time() - t0
    sxy = np.array(sxy)
    return dict(label=label, spatial_yield=spatial_yield, yield_mode=yield_mode,
                harmonic=harmonic, peak_sxy=float(np.abs(sxy).max()),
                div=div, mean_its=iters_total / n_steps, wall=wall)


def main():
    cases = [
        # baseline (failing at T=16): harmonic + spatial τ_y + softmin
        ("baseline_fail", True,  "softmin", True),
        # B: harmonic + scalar τ_y + softmin (does it blow up at T=16?)
        ("varB_scalarTY", False, "softmin", True),
        # C: harmonic + spatial τ_y + min (regression-test style)
        ("varC_min",      True,  "min",     True),
    ]
    print(f"\n{'label':<18} {'spatial_τy':>11} {'yield':>8} {'forcing':>9} "
          f"{'wall':>6} {'div':>4} {'its':>5} {'peak|σ_xy|':>11}", flush=True)
    for label, sy, ym, harmonic in cases:
        print(f"--- running {label} ---", flush=True)
        r = run(label, spatial_yield=sy, yield_mode=ym, harmonic=harmonic)
        print(f"{r['label']:<18} {str(r['spatial_yield']):>11} {r['yield_mode']:>8} "
              f"{('harmonic' if r['harmonic'] else 'const'):>9} "
              f"{r['wall']:>6.1f} {r['div']:>4d} {r['mean_its']:>5.2f} "
              f"{r['peak_sxy']:>11.4e}", flush=True)


if __name__ == "__main__":
    main()
