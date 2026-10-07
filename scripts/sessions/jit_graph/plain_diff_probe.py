"""Whether a route still needs tier 1's real stand-in at the solvers' derivative sites
(#823, tier 2).

Swaps ``diff_wrt_field`` / ``derive_by_array_wrt_field`` in the solvers for plain
``sympy.diff`` / ``sympy.derive_by_array`` and runs the Newton solve that motivated the
stand-in: the Drucker-Prager yield with a yield-stress floor at softness 0, whose floor
``(a + b + sqrt((a - b)**2)) / 2`` becomes an ``Abs`` of the pressure once fields are
real (``test_0023``). The route is chosen by ``UW_JIT_GRAPH``.
"""
import sympy
import underworld3 as uw
import underworld3.cython.generic_solvers as gs
from underworld3.utilities import _jit_graph

gs.diff_wrt_field = lambda e, w: sympy.diff(e, w)
gs.derive_by_array_wrt_field = lambda e, dx: sympy.derive_by_array(e, dx)
route = "graph" if _jit_graph.enabled() else "tree"

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                         cellSize=0.25)
v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
stokes.add_dirichlet_bc((1.0, 0.0), "Top")
stokes.bodyforce = sympy.Matrix([0, -mesh.X[0]])
stokes.tolerance = 1.0e-10
P = stokes.constitutive_model.Parameters
P.shear_viscosity_0 = 1.0
P.yield_stress = (uw.expression(r"C", 100.0, "cohesion")
                  + uw.expression(r"\mu", 0.6, "friction") * p.sym[0])
P.yield_stress_min = uw.expression(r"\tau", 0.01, "yield floor")
stokes.constitutive_model.yield_softness = 0
stokes.consistent_jacobian = True
try:
    stokes.solve()
    print(f"[plain sympy.diff, {route}] solved: {stokes.solve_report.reason_str}, "
          f"nl={stokes.solve_report.nl_its}")
except Exception as failure:
    print(f"[plain sympy.diff, {route}] FAILED: {type(failure).__name__}: "
          f"{str(failure).splitlines()[0][:200]}")
