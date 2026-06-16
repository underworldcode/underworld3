"""
import os
FAS-Vanka: nonlinear multigrid (SNESFAS) with the custom-IS PCASM Vanka smoother.

Turnkey-via-options is not available (PCPATCH won't build pressure-support patches
in this build), so we inject the custom index sets into each FAS level smoother:
  1. solve once with an LU smoother (assembles every level operator),
  2. for each level smoother SNES: build pressure-support patches from its operator,
     install PCASM(RESTRICT)+sub-LU wrapped in a GMRES Krylov smoother,
  3. re-solve, driving the SNES directly.

The patch *structure* depends only on sparsity, so it stays valid as the operator
values change across nonlinear iterations.
"""
import time
import importlib.util
import numpy as np
import sympy
import underworld3 as uw
from petsc4py import PETSc

_spec = importlib.util.spec_from_file_location("va", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vanka_mg.py"))
va = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(va)

FAS_LU = {
    "snes_type": "fas", "snes_fas_type": "full",
    "fas_levels_snes_type": "newtonls", "fas_levels_snes_max_it": 1,
    "fas_levels_snes_linesearch_type": "basic",
    "fas_levels_ksp_type": "preonly", "fas_levels_pc_type": "lu",
    "fas_coarse_snes_type": "newtonls", "fas_coarse_snes_max_it": 30,
    "fas_coarse_ksp_type": "preonly", "fas_coarse_pc_type": "lu",
}


def inject_vanka(snes, hier):
    nlev = snes.getFASLevels()
    opts = PETSc.Options()
    npres_per = []
    for L in range(1, nlev):                       # 0 = coarse solve (LU); smooth on 1..nlev-1
        sm = snes.getFASSmoother(L)                # a SNES
        ksp = sm.getKSP()
        Jl = ksp.getOperators()[0]
        dmL = sm.getDM()                            # the smoother's own level DM
        patches, npres = va.pressure_support_patches(dmL, Jl)
        npres_per.append(npres)
        # GMRES Krylov smoother (stabilises additive-Schwarz Vanka). max_it~15 is a
        # more robust default — it carries moderate viscosity contrast (~1e3); extreme
        # contrast (>=1e6) needs multiplicative Vanka, which additive PCASM can't do.
        ksp.setType("gmres")
        ksp.setTolerances(rtol=1e-2, max_it=15)
        pc = ksp.getPC()
        pc.reset()
        pc.setOperators(Jl, Jl)
        pc.setType("asm")
        pc.setASMType(PETSc.PC.ASMType.RESTRICT)
        pc.setASMLocalSubdomains(len(patches), patches)
        pref = pc.getOptionsPrefix() or ""
        opts[pref + "sub_ksp_type"] = "preonly"
        opts[pref + "sub_pc_type"] = "lu"
        pc.setFromOptions()
    return npres_per


def solve_fas_vanka(stokes, hier, verbose=False):
    po = stokes.petsc_options
    po["snes_rtol"] = 1e-7
    po["snes_error_if_not_converged"] = 0
    for k, v in FAS_LU.items():
        po[k] = v
    # (1) warm-up FAS-LU solve — only needs to ASSEMBLE level operators, not converge
    po["snes_max_it"] = 2
    stokes.solve(zero_init_guess=True, picard=0)
    snes = stokes.snes
    po["snes_max_it"] = 60                          # real solve cap
    snes.setTolerances(max_it=60)
    # (2) inject Vanka smoothers
    npres = inject_vanka(snes, hier)
    # (3) re-solve directly with the Vanka smoothers
    stokes.mesh.update_lvec()
    stokes.dm.setAuxiliaryVec(stokes.mesh.lvec, None)
    stokes._update_constants()
    gvec = stokes.dm.getGlobalVec()
    gvec.set(0.0)
    t0 = time.perf_counter()
    snes.solve(None, gvec)
    dt = time.perf_counter() - t0
    r = int(snes.getConvergedReason())
    its = int(snes.getIterationNumber())
    stokes.dm.restoreGlobalVec(gvec)
    return dict(reason=r, its=its, t=dt, npres=npres)


if __name__ == "__main__":
    # validate on a simple linear blob (open top, no nullspace)
    print(f"underworld3: {uw.__file__}")
    for R in (1, 2, 3):
        mesh, st, v, p = va.build_stokes(R)
        res = solve_fas_vanka(st, st.dm_hierarchy)
        print(f"  FAS-Vanka  refinement={R}  npres/level={res['npres']}  "
              f"->  reason={res['reason']}  fas_its={res['its']}  t={res['t']:.2f}s")
        del st, mesh
