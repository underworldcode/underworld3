"""Pin down whether iso BDF-2 instability is from VEP machinery or VE alone.

Three iso cases at the same harmonic forcing, T=8, BDF-2, η=μ=1, on
the same mesh as the TI consistency test:

  pureve         — VE only (yield_stress = sympy.oo, no plastic branch)
  vep_huge_ty    — VEP with yield_stress = 1e8 (yielding effectively off)
  vep_active_ty  — VEP with yield_stress = 0.30 (yielding active)

If pureve is bounded but vep_huge_ty blows up, the BDF-2 instability
is in the VEP softmin/yield expression, not in the BDF-2 method.
"""

import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


V0 = 0.5; OMEGA = np.pi / 2.0; DT = 0.05; T_END = 8.0
ETA = 1.0; MU = 1.0; RES = 16


def build(label, *, yield_stress):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        qdegree=3,
    )
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, 2, degree=2,
                                        vtype=uw.VarType.VECTOR)
    p = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1,
                                        continuous=True, vtype=uw.VarType.SCALAR)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=2,
    )
    cm.Parameters.shear_viscosity_0 = ETA
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = yield_stress
    cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
    cm.Parameters.strainrate_inv_II_min = 1.0e-6
    cm.yield_mode = "softmin"
    stokes.constitutive_model = cm
    stokes.saddle_preconditioner = 1.0 / cm.K
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(rf"V_{{top,{label}}}", sympy.Float(0.0), "")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    return stokes, V_top


def run(label, *, yield_stress):
    stokes, V_top = build(label, yield_stress=yield_stress)
    phi = float(np.arctan(OMEGA))
    n_steps = int(T_END / DT)
    sxy = []
    div = 0; iters_total = 0
    t0 = time.time()
    for step in range(n_steps):
        t = (step + 1) * DT
        v_now = V0 * float(np.cos(OMEGA * t + phi))
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = DT
        stokes.solve(zero_init_guess=False, timestep=DT, divergence_retries=2)
        if stokes.snes.getConvergedReason() < 0:
            div += 1
        iters_total += stokes.snes.getIterationNumber()
        c = np.array([[0.5, 0.5]])
        td = stokes.tau.data
        idx = int(np.argmin(np.linalg.norm(stokes.tau.coords - c, axis=1)))
        sxy.append(td[idx, 2])
    wall = time.time() - t0
    return dict(label=label, yield_stress=str(yield_stress), wall=wall,
                peak=float(np.abs(np.array(sxy)).max()),
                div=div, mean_its=iters_total / max(1, n_steps))


def main():
    cases = [
        ("pureve",         sympy.oo),
        ("vep_huge_ty",    1e8),
        ("vep_active_ty",  0.30),
    ]
    print(f"\n{'label':<14} {'yield_stress':>13} {'wall':>6} {'div':>4} {'its':>5} {'peak|σ_xy|':>11}",
          flush=True)
    for label, ty in cases:
        print(f"--- running {label} ---", flush=True)
        r = run(label, yield_stress=ty)
        print(f"{r['label']:<14} {r['yield_stress']:>13} {r['wall']:>6.1f} "
              f"{r['div']:>4d} {r['mean_its']:>5.2f} {r['peak']:>11.4e}",
              flush=True)


if __name__ == "__main__":
    main()
