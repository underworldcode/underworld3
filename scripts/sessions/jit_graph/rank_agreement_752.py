"""The #752 fixture: whether the ranks generate the same C (#823, tier 2).

A power-law transversely isotropic Stokes on an annulus with rotated free-slip, the
law of ``tests/test_0022_rotated_adjoint.py`` (adjoint branches), whose JIT source
differed across ranks in about one np=2 run in five on development. Builds the
kernels and reports, on rank 0, whether ``_agree_source_across_ranks`` saw different
hashes. The route is chosen by the build: run it in a build of this branch
(graph) and in a build of development (tree). Run under ``mpirun -n N``; it was
measured at N = 2, 3 and 4.
"""
import sympy
import underworld3 as uw
import underworld3.utilities._jitextension as jx

R_I, R_O = 0.5, 1.0
seen = []
agree = jx._agree_source_across_ranks


def counted(codeguys, source, source_hash):
    seen.append(len(set(uw.mpi.comm.allgather(source_hash))))
    return agree(codeguys, source, source_hash)


jx._agree_source_across_ranks = counted

mesh = uw.meshing.Annulus(radiusInner=R_I, radiusOuter=R_O, cellSize=0.15, qdegree=3)
x, y = mesh.X
r = sympy.sqrt(x ** 2 + y ** 2)
unit_r = sympy.Matrix([[x / r, y / r]])
th = sympy.atan2(y, x)
v = uw.discretisation.MeshVariable("v_rot_adj", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("p_rot_adj", mesh, 1, degree=1)
eta_1 = uw.expression(r"\eta_1", 0.2, "weak-plane viscosity ratio")
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
edot = mesh.vector.strain_tensor(v.sym)
eII = sympy.sqrt(sympy.Rational(1, 2) * (edot[0, 0] ** 2 + edot[1, 1] ** 2) + edot[0, 1] ** 2)
eta_0 = (sympy.Float(0.01) + eII) ** sympy.Rational(-1, 3)
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1 * eta_0
stokes.constitutive_model.Parameters.director = unit_r
stokes.bodyforce = 1.0e2 * sympy.cos(3 * th) * (r - R_I) / (R_O - R_I) * unit_r
stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
stokes.add_rotated_freeslip_bc(0.0, "Upper")
stokes.consistent_jacobian = True
stokes.petsc_options["snes_max_it"] = 1
stokes.solve(zero_init_guess=True)
uw.pprint(f"[752] getext calls {len(seen)}; calls whose ranks disagreed: "
          f"{sum(1 for k in seen if k > 1)}")
