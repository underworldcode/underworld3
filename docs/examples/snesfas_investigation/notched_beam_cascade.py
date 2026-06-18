"""
Nested iteration (grid sequencing) on the hard case: viscoplastic compression with
a weak seed -> localizing shear band, which is well-conditioned at low resolution
and increasingly ill-conditioned as you refine (cold Newton convergence collapses).

Tests the classical Brandt "coarse-as-initialization" idea (NOT FAS correction):
  solve coarse (robust) -> interpolate (u,p) up as initial guess -> solve next, ...
each level using a robust saddle-point solver (FMG). No tau-correction, no Vanka,
no coarse-grid correction across the band -> the contrast is handled at the
resolution where it is resolved.

Compares, at each resolution:
  COLD     : Newton+FMG from a zero initial guess (the current default)
  CASCADE  : Newton+FMG warm-started from the interpolated coarser solution

Pure-shear compression (enclosed; UW3 FMG handles the pressure nullspace).
Run:  pixi run -e amr-dev python notched_beam_cascade.py
"""
import time
import numpy as np
import sympy
import underworld3 as uw

TAU_Y = 8.0
SR_MIN = 1.0e-2          # regularization floor -> bounds the effective contrast
SEED_SIG = 0.04
U = 1.0


def build(cellSize, refinement=1):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=cellSize,
        refinement=refinement, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    # weak seed: yield stress dips to 0.1*tau_y in a small central blob -> nucleates the band
    seed = sympy.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * SEED_SIG ** 2))
    st.constitutive_model.Parameters.yield_stress = TAU_Y * (1.0 - 0.9 * seed)
    st.constitutive_model.Parameters.strainrate_inv_II_min = SR_MIN
    st.bodyforce = sympy.Matrix([0.0, 0.0])
    # pure-shear compression: compress in x, extend in y; free-slip tangential
    st.add_dirichlet_bc((+U, None), "Left")
    st.add_dirichlet_bc((-U, None), "Right")
    st.add_dirichlet_bc((None, +U), "Top")
    st.add_dirichlet_bc((None, -U), "Bottom")
    return mesh, st, v, p


def solve_level(st, v, p, warm=None, max_it=40):
    po = st.petsc_options
    po["snes_rtol"] = 1.0e-4
    po["snes_max_it"] = 120
    po["snes_error_if_not_converged"] = 0
    st.preconditioner = "fmg"
    if warm is not None:
        vc, pc = warm
        v.data[:] = np.asarray(uw.function.evaluate(vc.sym, v.coords)).reshape(v.data.shape)
        p.data[:, 0] = np.asarray(uw.function.evaluate(pc.sym, p.coords)).reshape(-1)
        zig = False
    else:
        zig = True
    t0 = time.perf_counter()
    try:
        st.solve(zero_init_guess=zig, picard=40)
        r = int(st.snes.getConvergedReason()); its = int(st.snes.getIterationNumber())
    except Exception:
        r, its = -99, None
    return dict(reason=r, its=its, t=time.perf_counter() - t0)


CELLSIZES = [0.10, 0.05, 0.025]   # coarse -> fine

if __name__ == "__main__":
    print(f"underworld3: {uw.__file__}")
    print(f"viscoplastic pure-shear compression, weak seed, tau_y={TAU_Y}, sr_min={SR_MIN}\n")

    # ---- COLD: Newton+FMG from scratch at each resolution (the current default) ----
    print("COLD  (zero initial guess at each resolution):")
    for cs in CELLSIZES:
        mesh, st, v, p = build(cs)
        ndof = st.snes.getJacobian()[0].getSize()[0] if False else None
        r = solve_level(st, v, p, warm=None)
        tag = "ok  " if (r["reason"] or -1) > 0 else "FAIL"
        print(f"  cellSize={cs:<6}  {tag} newton_its={str(r['its']):>3} ({str(r['reason']):>3})  {r['t']:6.1f}s")
        del st, v, p, mesh

    # ---- CASCADE: coarse -> fine, warm-started by interpolation ----
    print("\nCASCADE  (coarse -> fine, warm-started from the interpolated coarser solution):")
    warm = None
    keep = []
    for cs in CELLSIZES:
        mesh, st, v, p = build(cs)
        r = solve_level(st, v, p, warm=warm)
        tag = "ok  " if (r["reason"] or -1) > 0 else "FAIL"
        src = "cold" if warm is None else "warm"
        print(f"  cellSize={cs:<6}  [{src}] {tag} newton_its={str(r['its']):>3} ({str(r['reason']):>3})  {r['t']:6.1f}s")
        warm = (v, p)
        keep.append((mesh, st, v, p))    # keep alive so warm refs stay valid
