import pytest
import sympy
import underworld3 as uw

# Solves small Stokes/Poisson systems to exercise the auto FMG/GAMG switch.
pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]


# A mesh built with refinement carries a real dm_hierarchy (FMG-capable);
# a plain mesh has a single level and must fall back to GAMG.
mesh_refined = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=0.25, refinement=2, qdegree=2,
)
mesh_plain = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=0.2, qdegree=2,
)


def _make_stokes(mesh):
    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.bodyforce = sympy.Matrix([0, 1.0e2 * x])
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, None), "Left")
    stokes.add_dirichlet_bc((0.0, None), "Right")
    return stokes


def _vel_pc(stokes):
    return stokes.petsc_options.getString("fieldsplit_velocity_pc_type")


def test_refinement_mesh_has_hierarchy():
    # Sanity: refinement=2 builds a 3-level hierarchy; plain mesh has one level.
    assert len(mesh_refined.dm_hierarchy) > 1
    assert len(mesh_plain.dm_hierarchy) == 1


def test_auto_selects_geometric_fmg_on_refinement_mesh():
    stokes = _make_stokes(mesh_refined)
    assert stokes.preconditioner == "auto"  # default
    stokes.solve()
    assert _vel_pc(stokes) == "mg"
    assert stokes.snes.getConvergedReason() > 0


def test_auto_falls_back_to_gamg_without_hierarchy():
    stokes = _make_stokes(mesh_plain)
    stokes.solve()
    assert _vel_pc(stokes) == "gamg"
    assert stokes.snes.getConvergedReason() > 0


def test_explicit_gamg_override():
    stokes = _make_stokes(mesh_refined)
    stokes.preconditioner = "gamg"
    stokes.solve()
    assert _vel_pc(stokes) == "gamg"
    assert stokes.snes.getConvergedReason() > 0


def test_toggle_back_to_fmg():
    # Switching gamg -> fmg must cleanly re-engage geometric multigrid.
    stokes = _make_stokes(mesh_refined)
    stokes.preconditioner = "gamg"
    stokes.solve()
    stokes.preconditioner = "fmg"
    stokes.solve()
    assert _vel_pc(stokes) == "mg"
    assert stokes.snes.getConvergedReason() > 0


def test_fmg_without_hierarchy_falls_back():
    # Asking for fmg on a mesh with no hierarchy warns and uses GAMG.
    stokes = _make_stokes(mesh_plain)
    stokes.preconditioner = "fmg"
    with pytest.warns(UserWarning, match="falling back to GAMG"):
        stokes.solve()
    assert _vel_pc(stokes) == "gamg"
    assert stokes.snes.getConvergedReason() > 0


def test_invalid_preconditioner_raises():
    stokes = _make_stokes(mesh_plain)
    with pytest.raises(ValueError):
        stokes.preconditioner = "wibble"


def test_scalar_poisson_auto_falls_back_to_gamg():
    # #276: native geometric FMG is locked out for single-field (scalar/vector)
    # solvers — DMCreateInjection is not reliably constructible on a refined
    # DMPlex for a single field (fails on curved shells and some high-degree flat
    # cases). So a scalar solver on a refined hierarchy falls back to GAMG rather
    # than crashing; robust geometric MG for scalars is via custom_mg.set_custom_fmg.
    poisson = uw.systems.Poisson(mesh_refined)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.solve()
    assert poisson.petsc_options.getString("pc_type") == "gamg"
    assert poisson.snes.getConvergedReason() > 0


def test_explicit_velocity_options_survive_rebuild():
    # Regression: in "auto" mode the helper must NOT clobber user-set
    # velocity-block PC options when it re-runs. _apply_preconditioner_options()
    # is called on EVERY _build (so "auto" re-resolves after a remesh); a
    # mesh-mover deform triggers exactly such a rebuild. Previously the override
    # was respected only on the first call — the second overwrote the user's
    # tuned coarse-solver / smoother with the framework FMG bundle. Drive the
    # helper directly (twice) to test the option latch in isolation.
    stokes = _make_stokes(mesh_refined)
    vp = "fieldsplit_velocity_"
    stokes.petsc_options[vp + "pc_type"] = "mg"
    stokes.petsc_options[vp + "mg_coarse_pc_type"] = "redundant"
    stokes.petsc_options[vp + "mg_coarse_redundant_pc_type"] = "lu"
    stokes.petsc_options[vp + "mg_levels_ksp_type"] = "richardson"

    stokes._apply_preconditioner_options()  # first build: adopt + respect
    assert stokes.petsc_options.getString(vp + "mg_coarse_pc_type") == "redundant"
    stokes._apply_preconditioner_options()  # rebuild: must STILL respect them
    assert stokes.petsc_options.getString(vp + "pc_type") == "mg"
    assert stokes.petsc_options.getString(vp + "mg_coarse_pc_type") == "redundant"
    assert stokes.petsc_options.getString(vp + "mg_levels_ksp_type") == "richardson"


def test_geometric_mg_without_galerkin_is_repaired():
    # UW3's geometric MG REQUIRES Galerkin RAP (no coarse-DM operator callbacks
    # are installed, so PETSc cannot re-discretise on the coarse levels). A user
    # who selects pc_type=mg but omits pc_mg_galerkin must be repaired (forced to
    # "both") and warned — not left to fail as PETSc error 73 (serial) or
    # DMCoarsen->ParMmg (parallel).
    stokes = _make_stokes(mesh_refined)
    vp = "fieldsplit_velocity_"
    stokes.petsc_options[vp + "pc_type"] = "mg"
    # deliberately DO NOT set pc_mg_galerkin
    with pytest.warns(UserWarning, match="requires Galerkin"):
        stokes.solve()
    assert stokes.petsc_options.getString(vp + "pc_mg_galerkin") == "both"
    assert stokes.snes.getConvergedReason() > 0


def test_default_fmg_bundle_is_parallel_safe():
    # The property's OWN default FMG bundle must be usable at np>1 unaided: a
    # parallel-safe coarse solver (redundant+lu, not bare serial lu) and a
    # smoother sized for a DEEP hierarchy — gmres+sor, not eigen-estimate-fragile
    # chebyshev and not stationary richardson, which degrades on the non-symmetric
    # consistent-Newton operator (measured: per-V-cycle contraction 0.75 richardson
    # vs 0.56 gmres over 4 nested levels on the Spiegelman notch).
    stokes = _make_stokes(mesh_refined)
    stokes.preconditioner = "fmg"
    stokes.solve()
    vp = "fieldsplit_velocity_"
    assert stokes.petsc_options.getString(vp + "mg_coarse_pc_type") == "redundant"
    assert stokes.petsc_options.getString(vp + "mg_coarse_redundant_pc_type") == "lu"
    assert stokes.petsc_options.getString(vp + "mg_levels_ksp_type") == "gmres"
    assert stokes.petsc_options.getString(vp + "mg_levels_pc_type") == "sor"
    # Fixed-cost V-cycle: exactly mg_levels_ksp_max_it smoother iterations, no
    # residual-norm computation and no early exit.
    assert stokes.petsc_options.getString(vp + "mg_levels_ksp_norm_type") == "none"
    assert stokes.snes.getConvergedReason() > 0
