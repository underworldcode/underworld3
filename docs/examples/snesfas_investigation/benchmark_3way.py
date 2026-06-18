"""
import os
Three good choices: FMG vs GAMG vs FAS-Vanka on hard Stokes benchmarks.

  FMG       : Newton + fieldsplit-Schur + velocity-block geometric MG (preconditioner="fmg")
  GAMG      : Newton + fieldsplit-Schur + velocity-block algebraic MG (preconditioner="gamg")
  FAS-Vanka : nonlinear multigrid on the full saddle, custom-IS PCASM Vanka smoother

(1) SolCx viscosity step, eta_B = 1 .. 1e6   (linear, discontinuous viscosity)
(2) Viscoplastic yield, tau_y = 10 .. 0.25   (strong nonlinearity)

Open top (no pressure nullspace). refinement=2 => 3-level hierarchy. The point is
robustness with stock settings — no per-problem tuning.

Run:  pixi run -e amr-dev python benchmark_3way.py
"""
import time
import importlib.util
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import analytic as A

fv = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("fv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fas_vanka.py")))
importlib.util.spec_from_file_location("fv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fas_vanka.py")).loader.exec_module(fv)


def make(res=16, refinement=1):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1.0 / res, refinement=refinement, qdegree=3)


def open_top(st):
    st.add_dirichlet_bc((0.0, None), "Left")
    st.add_dirichlet_bc((0.0, None), "Right")
    st.add_dirichlet_bc((0.0, 0.0), "Bottom")


def solcx(mesh, eta_B):
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=eta_B, x_c=0.5, n=1)
    st = uw.systems.Stokes(mesh)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    st.bodyforce = sol.fn_bodyforce
    st._sp = 1.0 / sol.fn_viscosity
    open_top(st)
    return st


def visc(mesh, tau_y):
    x, y = mesh.X
    st = uw.systems.Stokes(mesh)
    st.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.constitutive_model.Parameters.yield_stress = tau_y
    st.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-3
    st.bodyforce = sympy.Matrix([0.0, -50.0 * sympy.exp(-100.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))])
    st._sp = None
    open_top(st)
    return st


def run(build, difficulty, method):
    mesh = make()
    st = build(mesh, difficulty)
    if method in ("fmg", "gamg"):
        st.petsc_options["snes_rtol"] = 1e-7
        st.petsc_options["snes_max_it"] = 25
        st.petsc_options["ksp_max_it"] = 300              # cap inner so failures bail fast
        st.petsc_options["snes_error_if_not_converged"] = 0
        st.preconditioner = method
        if st._sp is not None:
            st.saddle_preconditioner = st._sp
        t0 = time.perf_counter()
        try:
            st.solve(zero_init_guess=True, picard=0)
            r = int(st.snes.getConvergedReason()); its = int(st.snes.getIterationNumber())
        except Exception:
            r, its = -99, None
        dt = time.perf_counter() - t0
        out = dict(reason=r, its=its, t=dt)
    else:  # fas-vanka
        try:
            out = fv.solve_fas_vanka(st, st.dm_hierarchy)
        except Exception as e:  # noqa
            out = dict(reason=-99, its=None, t=float("nan"))
    del st, mesh
    return out


def cell(r):
    ok = "ok  " if (r["reason"] or -1) > 0 else "FAIL"
    return f"{ok} its={str(r['its']):>3} ({str(r['reason']):>3}) {r['t']:5.1f}s"


print(f"underworld3: {uw.__file__}\n")
print("=== (1) SolCx viscosity step (open top) — FMG vs GAMG vs FAS-Vanka ===")
print(f"{'eta_B':>7} | {'FMG':>20} | {'GAMG':>20} | {'FAS-Vanka':>20}")
print("-" * 78)
for eta_B in [1.0, 1.0e3, 1.0e6]:
    rows = {m: run(solcx, eta_B, m) for m in ("fmg", "gamg", "fas-vanka")}
    print(f"{eta_B:>7.0e} | {cell(rows['fmg']):>20} | {cell(rows['gamg']):>20} | {cell(rows['fas-vanka']):>20}")

print("\n=== (2) Viscoplastic yield (open top) — FMG vs GAMG vs FAS-Vanka ===")
print(f"{'tau_y':>7} | {'FMG':>20} | {'GAMG':>20} | {'FAS-Vanka':>20}")
print("-" * 78)
for tau_y in [10.0, 1.0, 0.3]:
    rows = {m: run(visc, tau_y, m) for m in ("fmg", "gamg", "fas-vanka")}
    print(f"{tau_y:>7.2f} | {cell(rows['fmg']):>20} | {cell(rows['gamg']):>20} | {cell(rows['fas-vanka']):>20}")
