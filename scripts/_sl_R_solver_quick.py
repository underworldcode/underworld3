"""Quick: just one Stokes cold solve per high-R adapted mesh
to see if cost continues to climb past R=3 or saturates."""
import os
import time
import numpy as np
import sympy
import underworld3 as uw


BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/R_compare')
R_LIST = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
Ra = 1.0e7
theta_FK = float(np.log(1.0e4))


def build_and_solve(R):
    out = os.path.join(BASE, f"R{R}")
    m = uw.discretisation.Mesh(
        os.path.join(out, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", m, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    V = uw.discretisation.MeshVariable(
        "V_v2p1", m, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True)
    P = uw.discretisation.MeshVariable(
        "P_v2p1", m, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0, outputPath=out)
    X = m.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    unit_r = m.CoordinateSystem.unit_e_0
    s = uw.systems.Stokes(m, velocityField=V, pressureField=P)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = (
        sympy.exp(theta_FK * (1 - T.sym[0])))
    s.tolerance = 1.0e-5
    s.penalty = 0.0
    s.add_essential_bc((0.0, 0.0), m.boundaries.Lower.name)
    KFS = 1.0e6
    fs_term = (KFS * V.sym.dot(unit_r) * unit_r)
    s.add_natural_bc(fs_term, m.boundaries.Upper.name)
    T_cond = sympy.log(r_sym / 1.0) / sympy.log(0.5 / 1.0)
    s.bodyforce = Ra * (T.sym[0] - T_cond) * unit_r
    t0 = time.time()
    V.data[...] = 0.0
    P.data[...] = 0.0
    s.solve(zero_init_guess=True)
    wall = time.time() - t0
    reason = int(s.snes.getConvergedReason())
    its = int(s.snes.getIterationNumber())
    vmax = float(np.sqrt(V.data[:, 0] ** 2
                         + V.data[:, 1] ** 2).max())
    return reason, its, wall, vmax


print(f"{'R':>5} {'reason':>6} {'its':>4} {'wall':>8}  {'|v|max':>10}")
print("-" * 50)
for R in R_LIST:
    reason, its, wall, vmax = build_and_solve(R)
    print(f"{R:>5.1f} {reason:>6} {its:>4} {wall:>7.2f}s  "
          f"{vmax:>10.2e}", flush=True)
