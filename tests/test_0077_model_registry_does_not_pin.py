"""The default model's registries do not keep a dropped mesh alive.

Every mesh, mesh variable and solver registers with the default model. The
registries held them by strong reference, so a mesh the caller had dropped
-- with its DM hierarchy, its variables' vectors and its solver's matrices --
lived for the rest of the process. A driver that rebuilds a faulted mesh
each time a fault moves therefore kept every mesh it ever built: measured on
the moving-fault fixture (scripts/sessions/rebuild_leak), 9 meshes, 21 DMs,
12 matrices, one KSP and one SNES more after each rebuild, 53 MB per rebuild
at k = 3 in serial, and about 0.4 GB per rank per rebuild in the kaiju
production runs (an out-of-memory kill after 24 rebuilds). The swarms and
state-bearers were already held weakly; the meshes, variables and solvers
now are too.
"""
import gc
import weakref

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(autouse=True)
def _fresh_model():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _dropped(make):
    """Weak references to what ``make`` returns, after it is dropped."""
    refs = {name: weakref.ref(obj) for name, obj in make().items()}
    gc.collect()
    return {name: ref() is None for name, ref in refs.items()}


def test_a_dropped_mesh_with_a_variable_and_a_solver_is_freed():
    def make():
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
        variable = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
        poisson = uw.systems.Poisson(mesh, u_Field=variable)
        return {"mesh": mesh, "variable": variable, "solver": poisson}

    freed = _dropped(make)
    assert all(freed.values()), freed


def test_a_dropped_faulted_rebuild_is_freed():
    """The production pattern: adapt, add_fault, solve, drop."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
        refinement=1, qdegree=2)

    def make():
        tip_a, tip_b = np.array([0.30, 0.45]), np.array([0.70, 0.55])
        samples = np.column_stack([np.linspace(tip_a[0], tip_b[0], 9),
                                   np.linspace(tip_a[1], tip_b[1], 9),
                                   np.zeros(9)])
        guide = uw.meshing.Surface("guide", base, samples)
        guide.discretize()
        metric = guide.refinement_metric_function(
            h_near=0.0125, h_far=0.05, width=0.08, profile="linear")
        child = base.adapt(metric, max_levels=2)
        mesh = child.add_fault([("Flt", np.vstack([tip_a, tip_b]))])
        _x, y = mesh.X
        stokes = uw.systems.Stokes(mesh)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        stokes.petsc_use_pressure_nullspace = True
        for wall in ("Top", "Bottom", "Left", "Right"):
            stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
        stokes.add_fault_bc(0, "Flt")
        stokes.solve()
        return {"finest": mesh, "child": child, "solver": stokes}

    freed = _dropped(make)
    assert all(freed.values()), freed


def test_a_variable_keeps_its_mesh():
    """A helper may return only the variable: its mesh stays usable."""
    def make():
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
        return uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

    variable = make()
    gc.collect()
    variable.array[...] = 1.0
    assert variable.mesh is not None
    assert float(variable.array.max()) == 1.0


def test_a_solver_that_solved_in_a_step_is_freed(monkeypatch, tmp_path):
    """The step transcript names each part's object; it must not keep it."""
    monkeypatch.setenv("UW_TRANSCRIPT", "on")
    monkeypatch.chdir(tmp_path)
    uw.reset_default_model()

    def make():
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
        poisson = uw.systems.Poisson(mesh)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel
        poisson.constitutive_model.Parameters.diffusivity = 1.0
        poisson.f = 1.0
        poisson.add_dirichlet_bc(0.0, "Bottom")
        with uw.get_default_model().step(0.1):
            poisson.solve()
        return {"mesh": mesh, "solver": poisson}

    freed = _dropped(make)
    assert all(freed.values()), freed


def test_the_primary_mesh_is_released_and_replaced():
    """A freed primary mesh is not reported, and the next mesh takes over."""
    def make():
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
        return {"mesh": mesh}

    freed = _dropped(make)
    assert freed["mesh"]
    assert uw.get_default_model().mesh is None
    replacement = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    assert uw.get_default_model().mesh is replacement
