import os
import pytest
import underworld3 as uw

# Persisting + restoring the geometric-multigrid (FMG) hierarchy across a
# mesh checkpoint round-trip (serial). See
# docs/developer/design/fmg-checkpoint-hierarchy.md
pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]


def _refined_box(cellSize=0.3, refinement=1):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cellSize, refinement=refinement, qdegree=2,
    )


def test_refinement_mesh_writes_single_coarsest_sidecar(tmp_path):
    m = _refined_box(refinement=2)
    assert len(m.dm_hierarchy) == 3              # coarse, mid, fine
    fn = str(tmp_path / "mesh.h5")
    m.write(fn)
    # Only the coarsest level is stored; intermediate levels are rebuilt by
    # refinement on reload.
    assert os.path.isfile(str(tmp_path / "mesh.hierarchy.L0.h5"))
    assert not os.path.isfile(str(tmp_path / "mesh.hierarchy.L1.h5"))


def test_reload_restores_hierarchy(tmp_path):
    m = _refined_box(refinement=2)
    fn = str(tmp_path / "mesh.h5")
    m.write(fn)
    m2 = uw.discretisation.Mesh(fn)
    assert len(m2.dm_hierarchy) == len(m.dm_hierarchy) == 3


def test_no_hierarchy_writes_no_sidecar(tmp_path):
    # A plain mesh (no refinement) must not write sidecars and must reload
    # exactly as before — regression guard for existing checkpoints.
    m = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=2)
    assert len(m.dm_hierarchy) == 1
    fn = str(tmp_path / "plain.h5")
    m.write(fn)
    assert not os.path.isfile(str(tmp_path / "plain.hierarchy.L0.h5"))
    m2 = uw.discretisation.Mesh(fn)
    assert len(m2.dm_hierarchy) == 1


def test_reloaded_hierarchy_drives_geometric_mg(tmp_path):
    # The restored hierarchy must actually work as a geometric-multigrid
    # preconditioner on the reloaded mesh.
    m = _refined_box(refinement=1)
    fn = str(tmp_path / "mesh.h5")
    m.write(fn)
    m2 = uw.discretisation.Mesh(fn)
    assert len(m2.dm_hierarchy) == 2

    poisson = uw.systems.Poisson(m2)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    for k, v in {
        "pc_type": "mg", "pc_mg_type": "full", "pc_mg_galerkin": "both",
        "mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "sor",
        "mg_coarse_pc_type": "lu",
    }.items():
        poisson.petsc_options[k] = v
    poisson.solve()
    assert poisson.petsc_options.getString("pc_type") == "mg"
    assert poisson.snes.getConvergedReason() > 0
