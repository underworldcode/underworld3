"""
Vanka, done right: overlapping Schwarz (PCASM) with patches = support of each
pressure basis function, built from the divergence-block sparsity — per expert
advice (the stock PCPATCH `vanka`/`star` constructs were ineffective on simplex
Taylor-Hood; this is the hand-built-IS route that is standard for P2-P1).

Procedure:
  1. assemble UW3's true saddle Jacobian J = [A B; B^T (0)]  (Amat, not the Schur Pmat)
  2. get the velocity / pressure global DOF index sets from the DM field decomposition
  3. for each pressure DOF p: patch = {p} ∪ {velocity DOFs coupled to p through J's row}
  4. PCASM with those index sets, each sub-block an exact LU mini-Stokes solve
  5. wrap in FGMRES; measure convergence on J x = b

Run:  pixi run -e amr-dev python vanka_asm.py
"""
import numpy as np
import sympy
import underworld3 as uw
from petsc4py import PETSc


def build_stokes(cellSize=0.1):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=cellSize, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.bodyforce = sympy.Matrix([0.0, -sympy.exp(-100.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))])
    st.add_dirichlet_bc((0.0, 0.0), "Bottom")     # open top -> no pressure nullspace
    st.add_dirichlet_bc((0.0, None), "Left")
    st.add_dirichlet_bc((0.0, None), "Right")
    return mesh, st, v, p


def main():
    print(f"underworld3: {uw.__file__}")
    mesh, st, v, p = build_stokes()
    st.solve(zero_init_guess=True, picard=0)        # assembles Amat
    J = st.snes.getJacobian()[0]                    # true saddle Jacobian
    n = J.getSize()[0]
    print(f"saddle matrix: {n} x {n}")

    # field decomposition -> velocity / pressure global DOFs
    names, ises, dms = st.dm.createFieldDecomposition()
    print("fields:", names)
    vIS, pIS = ises[0], ises[1]
    vset = set(int(i) for i in vIS.getIndices())
    pidx = [int(i) for i in pIS.getIndices()]
    print(f"n_vel={len(vset)}  n_pres={len(pidx)}")

    # build one patch per pressure DOF: {p} + coupled velocity DOFs (J row sparsity)
    patches = []
    sizes = []
    for pg in pidx:
        cols, _ = J.getRow(pg)
        members = {pg}
        for c in cols:
            ci = int(c)
            if ci in vset:
                members.add(ci)
        arr = np.array(sorted(members), dtype=PETSc.IntType)
        sizes.append(arr.size)
        patches.append(PETSc.IS().createGeneral(arr, comm=PETSc.COMM_SELF))
    print(f"patches: {len(patches)}  size min/mean/max = "
          f"{min(sizes)}/{np.mean(sizes):.1f}/{max(sizes)}")

    # consistent RHS: b = J x_exact
    x_exact = J.createVecRight(); x_exact.setRandom()
    b = J.createVecLeft(); J.mult(x_exact, b)

    for asm_type, label in [(PETSc.PC.ASMType.RESTRICT, "RAS"),
                            (PETSc.PC.ASMType.BASIC, "additive")]:
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        ksp.setOperators(J)
        ksp.setType("fgmres")
        ksp.setTolerances(rtol=1e-8, max_it=400)
        ksp.setOptionsPrefix(f"vk_{label}_")
        pc = ksp.getPC()
        pc.setType("asm")
        pc.setASMType(asm_type)
        pc.setASMLocalSubdomains(len(patches), patches)
        opts = PETSc.Options()
        opts[f"vk_{label}_sub_ksp_type"] = "preonly"
        opts[f"vk_{label}_sub_pc_type"] = "lu"
        ksp.setFromOptions()
        x = b.duplicate(); x.set(0.0)
        try:
            ksp.solve(b, x)
            reason = ksp.getConvergedReason()
            its = ksp.getIterationNumber()
            err = (x - x_exact).norm() / x_exact.norm()
            print(f"  PCASM-Vanka [{label:8}] reason={reason:>3}  its={its:>4}  rel_err={err:.2e}")
        except Exception as e:  # noqa
            print(f"  PCASM-Vanka [{label:8}] EXC {repr(e)[:70]}")
        ksp.destroy()


if __name__ == "__main__":
    main()
