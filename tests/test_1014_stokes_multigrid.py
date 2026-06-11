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


def test_scalar_poisson_auto_geometric_mg():
    poisson = uw.systems.Poisson(mesh_refined)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.solve()
    assert poisson.petsc_options.getString("pc_type") == "mg"
    assert poisson.snes.getConvergedReason() > 0
