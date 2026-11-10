"""A solver the user no longer holds is freed, with its SNES and KSP tree (#1041).

- Each solver registers itself on its mesh, so that a mesh move or a DM change
  can mark it for rebuild, and so that a snapshot can capture its step state.
  The register was a plain list, which kept every solver ever built on the mesh
  alive for as long as the mesh: about 4.6 MB per dropped solver, and its PETSc
  options stayed in the global database. It now holds its solvers weakly, in
  the order they were built; so does a material distribution, which outlives
  the solvers it is handed to.
- A collected solver's KSPs stayed: the solve instrumentation's callbacks on
  each KSP hold the instrumentation, which holds the KSP, a cycle through PETSc
  that the collector cannot see (#1012's cycle). It is now broken when the
  solver is collected, as it already was when the SNES is torn down.
- Like the model's registries of history managers and transcript parts, which
  were already weak, the register does not extend a solver's lifetime: a
  snapshot holding a dropped solver's history cannot be restored, and says so.
  Before, it restored into the dead solver, which the register kept alive.
"""
import gc
import weakref

import numpy as np
import pytest
from petsc4py import PETSc

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _stokes(mesh):
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.bodyforce = (0.0, 1.0)
    stokes.petsc_use_pressure_nullspace = True
    return stokes


def _poisson(mesh, name):
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    return poisson


def test_a_dropped_solver_is_collected_with_its_options():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.25, qdegree=2)
    refs, prefixes = [], []
    for _ in range(3):
        stokes = _stokes(mesh)
        stokes.solve()
        refs.append(weakref.ref(stokes))
        prefixes.append(stokes.petsc_options_prefix)
        del stokes
        gc.collect()
    # development: all three alive, held by mesh._equation_systems_register
    assert [r() for r in refs] == [None, None, None]
    assert len(list(mesh._equation_systems_register)) == 0
    options = PETSc.Options().getAll()
    assert not [key for key in options if any(key.startswith(p) for p in prefixes)]


def _live_ksps():
    gc.collect()
    return sum(isinstance(o, PETSc.KSP) for o in gc.get_objects())


def test_a_dropped_solver_leaves_no_ksp_behind():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.25, qdegree=2)
    before = _live_ksps()
    for _ in range(3):
        stokes = _stokes(mesh)
        stokes.solve()
        del stokes
    # development: 3 KSPs more per dropped Stokes solver (9 here); with the
    # register weak but the instrumentation cycle intact, still 9
    assert _live_ksps() == before


def test_materials_do_not_keep_a_dropped_solver():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.25, qdegree=2)
    materials = uw.swarm.MaterialSwarm(mesh, fill_param=2, name="M1154")
    materials.add("lower", shear_viscosity_0=1.0)
    materials.add("upper", shear_viscosity_0=10.0)
    materials["upper"] = mesh.X[1] > 0.5
    stokes = _stokes(mesh)
    stokes.materials = materials
    stokes.solve()
    ref = weakref.ref(stokes)
    del stokes
    gc.collect()
    # the distribution's list of the solvers it pushes to kept it
    assert ref() is None
    stokes = _stokes(mesh)
    stokes.materials = materials
    assert materials._live_solvers() == [stokes]


def _advection_diffusion(mesh, name):
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, mesh.dim, degree=1)
    T = uw.discretisation.MeshVariable(f"T{name}", mesh, 1, degree=2)
    v.array[:, 0, 0] = 1.0
    T.array[:, 0, 0] = 1.0
    adv_diff = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v)
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = 0.01
    adv_diff.add_dirichlet_bc(0.0, "Left")
    adv_diff.solve(timestep=0.01)
    return adv_diff


def test_a_snapshot_of_a_dropped_solver_says_it_cannot_be_restored(tmp_path):
    from underworld3.checkpoint.snapshot import SnapshotInvalidatedError
    uw.reset_default_model()
    orchestration_model = uw.get_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0.0, 0.0),
                                        maxCoords=(1.0, 1.0), qdegree=3)
    adv_diff = _advection_diffusion(mesh, "1154s")
    token = orchestration_model.save_state()
    path = str(tmp_path / "dropped.snap.h5")
    orchestration_model.save_state(file=path)

    # while the solver is held, both restore
    orchestration_model.load_state(token)
    orchestration_model.load_state(path)

    del adv_diff
    gc.collect()
    # development: both "restored" into the dropped solver's history, which
    # the register kept alive
    with pytest.raises(SnapshotInvalidatedError, match="collected"):
        orchestration_model.load_state(token)
    with pytest.raises(ValueError, match="collected"):
        orchestration_model.load_state(path)
    uw.reset_default_model()


def test_the_register_still_reaches_the_live_solvers_in_build_order():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.25, qdegree=2)
    first, dropped, last = _poisson(mesh, "T1154a"), _poisson(mesh, "T1154b"), _poisson(mesh, "T1154c")
    first.solve()
    last.solve()
    del dropped
    gc.collect()
    assert list(mesh._equation_systems_register) == [first, last]

    # a move marks every live solver for rebuild, so the next solve reads the
    # moved coordinates (#122)
    assert first.is_setup and last.is_setup
    coords = np.asarray(mesh._coords).copy()
    coords[:, 1] *= 1.1
    mesh.deform(coords)
    assert not first.is_setup and not last.is_setup
