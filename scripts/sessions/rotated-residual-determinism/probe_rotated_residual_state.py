"""Does a rotated free-slip solve's starting residual depend on solver history?

Solver A solves N times with a slowly varying forcing amplitude; solver B is
built fresh on the same mesh and solves once. Both are then given the SAME
velocity, pressure and forcing, and solve once more from that state; the
starting residual |F^0| each reports (``_rotated_freeslip_info["rnorm0"]``) is
compared. Then A's carried state is cleared one piece at a time to see which
piece accounts for the difference:

  cache       solver._rotated_linear_cache = None (the cross-solve cache:
              geometry, the cached operator, the KSP context and its null space)
  alpha       solver._set_newton_alpha(0.0) (the continuation tangent state)

Free slip on both annulus boundaries, so the rotated problem carries the
rigid-rotation mode; a power-law viscosity, so the solve is nonlinear. Run:

    mpirun -np 2 python probe_rotated_residual_state.py -uw_prior_solves 10
"""
import numpy as np
import sympy
import underworld3 as uw

params = uw.Params(uw_prior_solves=10, uw_cell_size=0.15, uw_continuation=1)


def build(mesh, tag, amplitude):
    v = uw.discretisation.MeshVariable(f"V{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    x, y = mesh.X
    g = sympy.Matrix([[v.sym[0].diff(x), v.sym[0].diff(y)],
                      [v.sym[1].diff(x), v.sym[1].diff(y)]])
    e = (g + g.T) / 2
    eII = sympy.sqrt((e[0, 0] ** 2 + e[1, 1] ** 2) / 2 + e[0, 1] ** 2 + 1.0e-12)
    s.constitutive_model.Parameters.shear_viscosity_0 = eII ** (sympy.Rational(1, 3) - 1)
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y, x)
    s.bodyforce = amplitude * sympy.cos(3 * th) * sympy.Matrix([x / r, y / r])
    s.add_rotated_freeslip_bc(0, "Lower")
    s.add_rotated_freeslip_bc(0, "Upper")
    s.petsc_use_pressure_nullspace = True
    if int(params.uw_continuation):
        s.consistent_jacobian = "continuation"
    s.tolerance = 1.0e-6
    return s, v, p


def rnorm0(solver):
    info = solver._rotated_freeslip_info
    return float(info["rnorm0"])


mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=params.uw_cell_size,
                          qdegree=3)
amp_A = uw.function.expression(r"A_{A}", 1.0, "forcing amplitude, solver A")
amp_B = uw.function.expression(r"A_{B}", 1.0, "forcing amplitude, solver B")
sA, vA, pA = build(mesh, "A", amp_A)
sB, vB, pB = build(mesh, "B", amp_B)

for k in range(int(params.uw_prior_solves)):
    amp_A.sym = 1.0 + 0.01 * k
    sA.solve(zero_init_guess=(k == 0))
amp_B.sym = 1.0
sB.solve(zero_init_guess=True)

# the common state: A's last converged solution and A's last amplitude
final_amp = 1.5 * (1.0 + 0.01 * (int(params.uw_prior_solves) - 1))   # a new step: far from converged


# one snapshot of the common state, restored before every evaluation
U0, P0 = np.array(vA.data), np.array(pA.data)


def set_common():
    amp_A.sym = final_amp
    amp_B.sym = final_amp
    for v, p in ((vA, pA), (vB, pB)):
        v.data[...] = U0
        p.data[...] = P0


def compare(label):
    set_common()
    sA.solve(zero_init_guess=False)
    rA = rnorm0(sA)
    set_common()
    sB.solve(zero_init_guess=False)
    rB = rnorm0(sB)
    uw.pprint(f"PROBE {label:>8s}: |F0| A {rA:.10e}  B {rB:.10e}  rel diff {abs(rA - rB) / max(rA, rB):.3e}")


compare("as-is")
sA._rotated_linear_cache = None
sB._rotated_linear_cache = None
compare("cache")
sA._set_newton_alpha(0.0)
sB._set_newton_alpha(0.0)
compare("alpha")
