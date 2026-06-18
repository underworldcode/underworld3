"""
The payoff test: custom-IS PCASM Vanka as the per-level smoother of a geometric
multigrid on the FULL Stokes saddle, over UW3's dm_hierarchy (Galerkin coarse
operators — the FMG path, but smoothing the coupled system with Vanka).

Measure outer-KSP iterations vs refinement: flat => mesh-independent => the
scalable Stokes-MG smoother we were after (linear Stokes; this is the FAS-
scalability question in linear form).

Run:  pixi run -e amr-dev python vanka_mg.py
"""
import time
import numpy as np
import sympy
import underworld3 as uw
from petsc4py import PETSc


def build_stokes(refinement, cellSize=0.2):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=cellSize,
        refinement=refinement, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.bodyforce = sympy.Matrix([0.0, -sympy.exp(-100.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))])
    st.add_dirichlet_bc((0.0, 0.0), "Bottom")
    st.add_dirichlet_bc((0.0, None), "Left")
    st.add_dirichlet_bc((0.0, None), "Right")
    return mesh, st, v, p


def pressure_support_patches(dm, J):
    """One patch per pressure DOF = {p} u {velocity DOFs coupled via J row}."""
    names, ises, dms = dm.createFieldDecomposition()
    vset = set(int(i) for i in ises[0].getIndices())
    pidx = [int(i) for i in ises[1].getIndices()]
    patches = []
    for pg in pidx:
        cols, _ = J.getRow(pg)
        mem = {pg} | {int(c) for c in cols if int(c) in vset}
        patches.append(PETSc.IS().createGeneral(np.array(sorted(mem), dtype=PETSc.IntType),
                                                 comm=PETSc.COMM_SELF))
    return patches, len(pidx)


def run(refinement, smoother="vanka"):
    mesh, st, v, p = build_stokes(refinement)
    st.preconditioner = "fmg"                          # the real competitor (velocity-block MG + Schur)
    t0 = time.perf_counter()
    st.solve(zero_init_guess=True, picard=0)           # assemble + FMG linear solve
    t_fmg = time.perf_counter() - t0
    fmg_its = int(st.snes.getLinearSolveIterations())
    Amat = st.snes.getJacobian()[0]
    hier = st.dm_hierarchy                              # coarse -> fine
    nlev = len(hier)
    ndof = Amat.getSize()[0]

    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setOperators(Amat)
    ksp.setType("fgmres")
    ksp.setTolerances(rtol=1e-8, max_it=400)
    ksp.setOptionsPrefix("vmg_")
    pc = ksp.getPC()
    pc.setType("mg")
    pc.setMGLevels(nlev)
    for L in range(1, nlev):                            # interpolation coarse(L-1) -> fine(L)
        interp, _ = hier[L - 1].createInterpolation(hier[L])
        pc.setMGInterpolation(L, interp)

    o = PETSc.Options()
    o["vmg_pc_mg_galerkin"] = "both"                   # coarse operators = R A P
    o["vmg_pc_mg_type"] = "multiplicative"
    if smoother == "lu":                               # control: exact monolithic smoother
        o["vmg_mg_levels_ksp_type"] = "richardson"
        o["vmg_mg_levels_ksp_max_it"] = 4
        o["vmg_mg_levels_pc_type"] = "lu"
    else:
        # Krylov smoother (GMRES) self-stabilises the additive-Schwarz Vanka spectrum
        o["vmg_mg_levels_ksp_type"] = "gmres"
        o["vmg_mg_levels_ksp_max_it"] = 6
        o["vmg_mg_levels_pc_type"] = "asm"
        o["vmg_mg_levels_sub_ksp_type"] = "preonly"
        o["vmg_mg_levels_sub_pc_type"] = "lu"
    o["vmg_mg_coarse_ksp_type"] = "preonly"
    o["vmg_mg_coarse_pc_type"] = "lu"
    ksp.setFromOptions()
    t0 = time.perf_counter()
    ksp.setUp()
    t_setup = time.perf_counter() - t0

    # inject custom pressure-support patches into each fine-ish level smoother
    patch_info = []
    t_patch = 0.0
    if smoother == "vanka":
        for L in range(1, nlev):                        # level 0 is coarse solve (LU)
            sm = pc.getMGSmoother(L)
            Jl = sm.getOperators()[0]
            dmL = hier[L]
            tp = time.perf_counter()
            patches, npres = pressure_support_patches(dmL, Jl)   # Python prototype cost
            t_patch += time.perf_counter() - tp
            pcL = sm.getPC()
            pcL.reset()                                # clear the auto setup so we can set subdomains
            pcL.setOperators(Jl, Jl)                    # reset() drops operators — restore
            pcL.setType("asm")
            pcL.setASMType(PETSc.PC.ASMType.RESTRICT)
            pcL.setASMLocalSubdomains(len(patches), patches)
            patch_info.append(npres)

    x_exact = Amat.createVecRight(); x_exact.setRandom()
    b = Amat.createVecLeft(); Amat.mult(x_exact, b)
    x = b.duplicate(); x.set(0.0)
    t0 = time.perf_counter()
    ksp.solve(b, x)
    t_solve = time.perf_counter() - t0
    reason = ksp.getConvergedReason()
    its = ksp.getIterationNumber()
    err = (x - x_exact).norm() / x_exact.norm()
    print(f"  ndof={ndof:>6} lvl={nlev} | FMG: its={fmg_its:>2} solve={t_fmg:6.2f}s "
          f"| Vanka-MG: its={its:>2} solve={t_solve:6.2f}s (+setup {t_setup:.2f} +patch(py) {t_patch:.2f}) "
          f"err={err:.0e}")
    ksp.destroy()
    return dict(ndof=ndof, fmg_its=fmg_its, t_fmg=t_fmg, its=its, t_solve=t_solve)


if __name__ == "__main__":
    print(f"underworld3: {uw.__file__}")
    print("FMG (velocity-block MG + Schur — the production competitor) vs full-saddle Vanka-MG.")
    print("Linear Stokes; FMG solve = assemble + linear solve; Vanka solve = linear solve on assembled J.\n")
    for R in (1, 2, 3, 4):
        try:
            run(R, "vanka")
        except Exception as e:  # noqa
            print(f"  refinement={R}: EXC {repr(e)[:90]}")
