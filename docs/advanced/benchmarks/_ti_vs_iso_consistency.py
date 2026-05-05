"""Consistency check: does TI reduce to iso when Δ=0 (no yield, η_0 = η_1)?

The rank-4 TI tensor with η_0 = η_1_eff and Δ = 0 is mathematically
identical to 2·η·I_ijkl (the isotropic Newtonian tensor), regardless of
the director.  At BDF-2, with the SAME ε̇_eff, the resulting stress
should be bit-equal between TI and iso.

If TI matches iso here, the BDF-2 instability is *purely* in the yield
branch (where η_1_eff < η_0 and Δ ≠ 0 in the fault zone).  If TI
diverges from iso even in this trivial case, the bug is more
fundamental — possibly a stray history term, missing factor, or
asymmetric tensor reduction.
"""

import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


V0 = 0.5; OMEGA = np.pi / 2.0; DT = 0.05; T_END = 8.0
ETA = 1.0; MU = 1.0
RES = 16


def build(label, *, ti_model):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        qdegree=3,
    )
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, 2, degree=2,
                                        vtype=uw.VarType.VECTOR)
    p = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1,
                                        continuous=True, vtype=uw.VarType.SCALAR)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    if ti_model:
        cm = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
            stokes.Unknowns, order=2,
        )
        cm.Parameters.shear_viscosity_0 = ETA
        cm.Parameters.shear_viscosity_1 = ETA  # === η_0
        cm.Parameters.shear_modulus = MU
        cm.Parameters.yield_stress = 1e8       # effectively infinite
        cm.Parameters.director = sympy.Matrix([0.0, 1.0])
        cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
        cm._bdf_blend = 1.0  # pure BDF-2
    else:
        cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(
            stokes.Unknowns, order=2,
        )
        cm.Parameters.shear_viscosity_0 = ETA
        cm.Parameters.shear_modulus = MU
        cm.Parameters.yield_stress = 1e8
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


def run(label, *, ti_model):
    stokes, V_top = build(label, ti_model=ti_model)
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
    return dict(label=label, ti=ti_model, wall=wall,
                sxy=np.array(sxy),
                div=div, mean_its=iters_total / max(1, n_steps))


def main():
    print(f"\n{'label':<14} {'ti':>5} {'wall':>6} {'div':>4} {'its':>5} {'peak|σ_xy|':>11}",
          flush=True)
    iso = run("iso_noTY", ti_model=False)
    print(f"{iso['label']:<14} {str(iso['ti']):>5} {iso['wall']:>6.1f} "
          f"{iso['div']:>4d} {iso['mean_its']:>5.2f} "
          f"{float(np.abs(iso['sxy']).max()):>11.4e}", flush=True)
    ti = run("ti_noTY", ti_model=True)
    print(f"{ti['label']:<14} {str(ti['ti']):>5} {ti['wall']:>6.1f} "
          f"{ti['div']:>4d} {ti['mean_its']:>5.2f} "
          f"{float(np.abs(ti['sxy']).max()):>11.4e}", flush=True)

    diff = ti['sxy'] - iso['sxy']
    print(f"\n=== consistency check ===", flush=True)
    print(f"  max|TI - iso|  = {np.abs(diff).max():.6e}", flush=True)
    print(f"  max|iso|       = {np.abs(iso['sxy']).max():.6e}", flush=True)
    print(f"  rel max diff   = {np.abs(diff).max() / np.abs(iso['sxy']).max():.6e}",
          flush=True)
    print(f"  rms TI-iso     = {np.sqrt((diff**2).mean()):.6e}", flush=True)
    if np.abs(diff).max() / np.abs(iso['sxy']).max() < 1e-3:
        print("  → TI ≈ iso (consistent: bug is in yield branch only)",
              flush=True)
    else:
        print("  → TI != iso (deeper inconsistency: BDF-2 TI tensor structure differs)",
              flush=True)


if __name__ == "__main__":
    main()
