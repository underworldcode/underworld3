"""Root-cause isolation: is the pure-Neumann + constant_nullspace
scalar Poisson solve itself reliable, independent of all the
Monge-Ampere machinery?

Manufactured radially-symmetric solution on the Annulus with
phi'(R_I)=phi'(R_O)=0 (so natural zero-flux Neumann is exact):

    phi_exact(r) = cos(pi * (r - R_I) / (R_O - R_I))
    phi'_exact(r) = -(pi/L) sin(pi (r-R_I)/L),  L = R_O - R_I
    source s = Laplacian phi = phi'' + phi'/r       (mean-NOT-zero;
        subtract its area mean so the Neumann problem is compatible)

Solve  Laplacian phi = s  with NO essential BC + constant_nullspace,
then compare numeric |grad phi| to the exact profile across mesh
resolution. If the error explodes / grad phi -> 0 as RES rises, the
Neumann nullspace solve is the broken foundation under the MA work.
"""
from __future__ import annotations
import numpy as np
import sympy
import underworld3 as uw

R_I, R_O = 0.5, 1.0
L = R_O - R_I

for RES in (16, 32, 48):
    mesh = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                              cellSize=1.0 / RES, qdegree=3)
    phi = uw.discretisation.MeshVariable(
        f"phi_np{RES}", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    ps = uw.systems.Poisson(mesh, phi)
    ps.constitutive_model = uw.constitutive_models.DiffusionModel
    ps.constitutive_model.Parameters.diffusivity = 1.0
    ps.constant_nullspace = True

    x, y = mesh.X
    rr = sympy.Symbol("rr", positive=True)
    phi_r = sympy.cos(sympy.pi * (rr - R_I) / L)
    fp = sympy.diff(phi_r, rr)
    fpp = sympy.diff(fp, rr)
    s_r = fpp + fp / rr           # radial Laplacian
    r_cart = sympy.sqrt(x ** 2 + y ** 2)
    s = s_r.subs(rr, r_cart)

    # make the source mean-zero (area-weighted) for Neumann
    # compatibility — sample nodally
    coords = np.asarray(mesh.X.coords)
    s_nodal = np.asarray(uw.function.evaluate(s, coords)).reshape(-1)
    rad = np.sqrt((coords ** 2).sum(axis=1))
    s_mean = float(np.mean(s_nodal))
    ps.f = sympy.Matrix([[s - s_mean]])
    ps.solve(zero_init_guess=True)

    # numeric vs exact |grad phi| (radial), via a safe vector proj
    gv = uw.discretisation.MeshVariable(
        f"gphi_np{RES}", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True)
    gp = uw.systems.Vector_Projection(mesh, gv)
    gp.smoothing = 0.0
    gp.uw_function = sympy.Matrix(
        [phi.sym[0].diff(mesh.X[0]), phi.sym[0].diff(mesh.X[1])]).T
    gp.solve()
    g_arr = np.asarray(
        uw.function.evaluate(gv.sym, coords)).reshape(len(coords), -1)
    gnum = np.linalg.norm(g_arr[:, :2], axis=1)
    gex = (np.pi / L) * np.abs(
        np.sin(np.pi * (rad - R_I) / L))

    rel = np.linalg.norm(gnum - gex) / max(
        np.linalg.norm(gex), 1e-30)
    print(f"RES={RES:>2}  max|gradphi| num={gnum.max():.4f} "
          f"exact={gex.max():.4f}  rel-L2(grad)={rel:.3e}")
