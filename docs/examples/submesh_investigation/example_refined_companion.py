"""
Refined-companion approach: pull a COARSE level out of a refined mesh's
nested hierarchy, solve Stokes on it, map the solution back to the fine
mesh exactly.

This is the sibling of ``test_region_ds_submesh.py``. Both follow the
same pattern -- *get a submesh, build a solver, map back and forth* --
but the submesh here is a different *resolution* of the whole domain
rather than a *subdomain*:

    test_region_ds_submesh.py : extract_region("Inner")   -> subdomain
    example_refined_companion.py : coarsened_companion(...) -> coarse level

The same annulus + radial-buoyancy problem is used so the two examples
are directly comparable.

Design contract (refine-DM mode only): the coarse companion is available
ONLY because a genuine nested refinement hierarchy exists. Transfer
between levels uses PETSc's *nested* interpolator/injector -- exact,
parallel-local, no geometric point location, no KDTree. On a mesh with
no refinement relationship the companion is simply not offered (raises).

Usage:
    pixi run -e amr-dev python -u \
        docs/examples/submesh_investigation/example_refined_companion.py
"""

import numpy as np
import sympy

import underworld3 as uw
from underworld3.systems import Stokes

import refined_pair_prototype as rpp

# --- Parameters ---

r_outer = 1.5
r_inner = 0.5
cellsize = 1 / 16
refine_levels = 2          # fine mesh = 2 uniform refinements of the base
companion_levels = 1       # solve one level coarser than the finest
n = 2
k = 1
stokes_tol = 1.0e-6
vel_penalty = 1.0e6

# --- Fine mesh (the full-resolution domain) ---

uw.pprint(0, "Creating fine annulus mesh (with refinement hierarchy)...")
fine = uw.meshing.Annulus(
    radiusOuter=r_outer,
    radiusInner=r_inner,
    cellSize=cellsize,
    refinement=refine_levels,
)
fS, fE = fine.dm.getHeightStratum(0)
uw.pprint(
    0,
    f"Fine mesh: {fE - fS} cells, "
    f"hierarchy depth {len(fine.dm_hierarchy)}",
)

# --- Pull out the coarse companion (the "submesh") ---

uw.pprint(0, "Pulling coarse level out of the nested hierarchy...")
coarse = rpp.coarsened_companion(fine, levels=companion_levels)
cS, cE = coarse.dm.getHeightStratum(0)
uw.pprint(0, f"Coarse companion: {cE - cS} cells")
uw.pprint(0, f"  parent is fine mesh: {coarse.parent is fine}")
uw.pprint(0, f"  registered with parent: {coarse in fine._registered_submeshes}")
uw.pprint(0, f"  boundaries: {[b.name for b in coarse.boundaries]}")

# --- Variables on the coarse companion ---

v = uw.discretisation.MeshVariable("V", coarse, coarse.dim, degree=2)
p = uw.discretisation.MeshVariable("P", coarse, 1, degree=1, continuous=True)

# --- Coordinate system (same radial-buoyancy problem as the rock/air ex) ---

unit_rvec = coarse.CoordinateSystem.unit_e_0
r, th = coarse.CoordinateSystem.xR
Gamma = coarse.Gamma

# --- Stokes solver on the coarse companion ---

stokes = Stokes(coarse, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.saddle_preconditioner = 1.0

rho = ((r / r_outer) ** k) * sympy.cos(n * th)
stokes.bodyforce = rho * (-1.0 * unit_rvec)

# Free-slip on both annulus boundaries
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Upper")
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Lower")

stokes.tolerance = stokes_tol

uw.pprint(0, "Solving Stokes on the coarse companion...")
stokes.solve(verbose=False)

v_mag = np.sqrt(v.data[:, 0] ** 2 + v.data[:, 1] ** 2)
uw.pprint(
    0,
    f"Coarse solve: |v| max={v_mag.max():.6e} mean={v_mag.mean():.6e}",
)

# --- Map the coarse solution back to the fine mesh ---

uw.pprint(0, "Mapping coarse velocity back to the fine mesh (nested FE)...")
v_fine = uw.discretisation.MeshVariable("Vf", fine, fine.dim, degree=2)
rpp.prolongate(coarse, v, v_fine)

vf_mag = np.sqrt(v_fine.data[:, 0] ** 2 + v_fine.data[:, 1] ** 2)
uw.pprint(
    0,
    f"Prolongated to fine: |v| max={vf_mag.max():.6e} "
    f"mean={vf_mag.mean():.6e}",
)

# Round-trip: sampling the prolongated fine field back to the coarse
# companion must recover the coarse solution exactly (it lives in the
# fine FE space by construction).
v_back = uw.discretisation.MeshVariable("Vb", coarse, coarse.dim, degree=2)
rpp.sample(coarse, v_fine, v_back)
rt_err = np.linalg.norm(v_back.data - v.data) / np.linalg.norm(v.data)

# --- Report ---

uw.pprint(0, "=" * 60)
uw.pprint(0, "Refined-companion approach (coarse level of refined mesh)")
uw.pprint(0, f"  Coarse cells:        {cE - cS}")
uw.pprint(0, f"  Fine cells:          {fE - fS}")
uw.pprint(0, f"  Max |v| (coarse):    {v_mag.max():.10e}")
uw.pprint(0, f"  Max |v| (-> fine):   {vf_mag.max():.10e}")
uw.pprint(
    0,
    f"  Prolongation max|v| rel diff: "
    f"{abs(vf_mag.max() - v_mag.max()) / v_mag.max():.3e}",
)
uw.pprint(0, f"  Round-trip rel error (sample o prolongate): {rt_err:.3e}")
uw.pprint(0, "  -> transfer is exact (nested FE): differences are O(1e-15)")
uw.pprint(0, "=" * 60)

assert rt_err < 1.0e-8, f"nested round-trip not exact: {rt_err:.3e}"
uw.pprint(0, "example_refined_companion: OK")
