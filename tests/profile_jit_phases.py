"""Profile JIT compilation phases — isolate where time is spent.

Instruments: sympy derivatives, expression unwrapping, hashing, C code generation,
Cython compilation, and the actual PETSc solve.

Run with: pixi run -e default python tests/profile_jit_phases.py
"""

import time
import sympy
import underworld3 as uw
from underworld3.systems import VE_Stokes

# ── Setup (fast) ──────────────────────────────────────────────────────────────

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=1.0 / 8, qdegree=3,
)

v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True,
                                    vtype=uw.VarType.SCALAR)

stokes = VE_Stokes(mesh, velocityField=v, pressureField=p, order=1)
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.constitutive_model.Parameters.shear_modulus = 1.0
stokes.constitutive_model.Parameters.shear_viscosity_min = 1.0e-3
stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-10
stokes.saddle_preconditioner = 1.0
stokes.tolerance = 1.0e-4

x, y = mesh.X
tau_y = sympy.Piecewise(
    (0.3, (y >= 0.45) & (y <= 0.55)),
    (1.0e6, True),
)
stokes.constitutive_model.Parameters.yield_stress = tau_y

stokes.add_essential_bc(sympy.Matrix([0.5, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
stokes.add_essential_bc((sympy.oo, 0.0), "Left")
stokes.add_essential_bc((sympy.oo, 0.0), "Right")
stokes.bodyforce = sympy.Matrix([0.0, 0.0])
stokes.petsc_options["ksp_type"] = "fgmres"

print("Setup complete.\n")

# ── Phase 1: Sympy derivative computation ─────────────────────────────────────

dim = mesh.dim

# Get residual terms (these are already built by the constitutive model)
F0 = sympy.Array(stokes.F0.sym)
F1 = sympy.Array(stokes.F1.sym)
PF0 = sympy.Array(stokes.PF0.sym)

print(f"F0 has {len(F0.free_symbols)} free symbols, {sum(1 for _ in sympy.preorder_traversal(sympy.Matrix(F0)))} nodes")
print(f"F1 has {len(F1.free_symbols)} free symbols, {sum(1 for _ in sympy.preorder_traversal(sympy.Matrix(F1)))} nodes")

t0 = time.time()
sympy.core.cache.clear_cache()
t_cache_clear = time.time() - t0
print(f"\nsympy.core.cache.clear_cache(): {t_cache_clear:.3f}s")

# UU block derivatives
t0 = time.time()
G0 = sympy.derive_by_array(F0, stokes.u.sym)
t_uu_g0 = time.time() - t0

t0 = time.time()
G1 = sympy.derive_by_array(F0, stokes.Unknowns.L)
t_uu_g1 = time.time() - t0

t0 = time.time()
G2 = sympy.derive_by_array(F1, stokes.u.sym)
t_uu_g2 = time.time() - t0

t0 = time.time()
G3 = sympy.derive_by_array(F1, stokes.Unknowns.L)
t_uu_g3 = time.time() - t0

print(f"\nderive_by_array (UU block):")
print(f"  G0 (dF0/dU):  {t_uu_g0:.3f}s")
print(f"  G1 (dF0/dL):  {t_uu_g1:.3f}s")
print(f"  G2 (dF1/dU):  {t_uu_g2:.3f}s")
print(f"  G3 (dF1/dL):  {t_uu_g3:.3f}s")
print(f"  Total UU:      {t_uu_g0+t_uu_g1+t_uu_g2+t_uu_g3:.3f}s")

# UP block
t0 = time.time()
sympy.derive_by_array(F0, stokes.p.sym)
sympy.derive_by_array(F0, stokes._G)
sympy.derive_by_array(F1, stokes.p.sym)
sympy.derive_by_array(F1, stokes._G)
t_up = time.time() - t0
print(f"\nderive_by_array (UP block): {t_up:.3f}s")

# PU block
t0 = time.time()
sympy.derive_by_array(PF0, stokes.u.sym)
sympy.derive_by_array(PF0, stokes.Unknowns.L)
t_pu = time.time() - t0
print(f"derive_by_array (PU block): {t_pu:.3f}s")

print(f"\nTotal derivative computation: {t_uu_g0+t_uu_g1+t_uu_g2+t_uu_g3+t_up+t_pu:.3f}s")

# ── Phase 2: getext (unwrap + hash + compile) ────────────────────────────────

# Now let the solver do its full setup and time it
print(f"\n--- Full solver._setup_pointwise_functions + getext ---")
stokes.is_setup = False
stokes.constitutive_model._solver_is_setup = False
stokes.DFDt.psi_fn = stokes.constitutive_model.flux.T

t0 = time.time()
stokes._setup_pointwise_functions(verbose=True)
t_setup_pw = time.time() - t0
print(f"_setup_pointwise_functions: {t_setup_pw:.3f}s")

t0 = time.time()
stokes._setup_discretisation(verbose=True)
t_setup_disc = time.time() - t0
print(f"_setup_discretisation: {t_setup_disc:.3f}s")

t0 = time.time()
stokes._setup_solver(verbose=True)
t_setup_solver = time.time() - t0
print(f"_setup_solver: {t_setup_solver:.3f}s")

# ── Phase 3: DFDt update + actual solve ───────────────────────────────────────

print(f"\n--- DFDt + solve ---")
t0 = time.time()
stokes.DFDt.update_pre_solve(0.02, verbose=True)
t_dfdt_pre = time.time() - t0
print(f"DFDt.update_pre_solve: {t_dfdt_pre:.3f}s")

t0 = time.time()
# Call the parent (Stokes) solve directly to skip VE_Stokes overhead
from underworld3.systems.solvers import SNES_Stokes_SaddlePt
SNES_Stokes_SaddlePt.solve(stokes, zero_init_guess=True, _force_setup=False, verbose=True)
t_solve = time.time() - t0
print(f"PETSc SNES solve: {t_solve:.3f}s")

t0 = time.time()
stokes.DFDt.update_post_solve(0.02, verbose=True)
t_dfdt_post = time.time() - t0
print(f"DFDt.update_post_solve: {t_dfdt_post:.3f}s")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"Derivative computation:     {t_uu_g0+t_uu_g1+t_uu_g2+t_uu_g3+t_up+t_pu:.1f}s")
print(f"_setup_pointwise_functions: {t_setup_pw:.1f}s  (includes derivatives + getext)")
print(f"_setup_discretisation:      {t_setup_disc:.1f}s")
print(f"_setup_solver:              {t_setup_solver:.1f}s")
print(f"DFDt pre-solve:             {t_dfdt_pre:.1f}s")
print(f"PETSc solve:                {t_solve:.1f}s")
print(f"DFDt post-solve:            {t_dfdt_post:.1f}s")
