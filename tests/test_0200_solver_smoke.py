"""
Quick solver smoke tests for early regression detection.

These are minimal tests that verify solvers can be created and run without errors.
They do NOT validate correctness - that's for level 3 physics tests.

Test numbering: 0200 (level 1 - quick tests)
Purpose: Catch obvious breakages before running full test suite.
"""

import pytest

# Level 1: Quick tests for CI
pytestmark = pytest.mark.level_1


class TestSolverSmoke:
    """Minimal smoke tests - just verify solvers don't crash."""

    def test_poisson_solver_runs(self):
        """Verify Poisson solver can be created and solved."""
        import underworld3 as uw
        import numpy as np

        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
        u = uw.discretisation.MeshVariable("u", mesh, 1, degree=2)

        poisson = uw.systems.Poisson(mesh, u_Field=u)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel
        poisson.constitutive_model.Parameters.diffusivity = 1.0
        poisson.f = 0.0

        poisson.add_dirichlet_bc(0.0, "Bottom")
        poisson.add_dirichlet_bc(1.0, "Top")

        poisson.solve()

        # Just verify solution is in expected range
        assert u.data.min() >= -0.1
        assert u.data.max() <= 1.1

    def test_stokes_solver_runs(self):
        """Verify Stokes solver can be created and solved."""
        import underworld3 as uw
        import sympy

        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
        v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)

        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.viscosity = 1.0
        stokes.bodyforce = sympy.Matrix([0, -1])

        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0), "Top")
        stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
        stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

        stokes.solve()

        # Just verify solver completed
        assert v.data is not None
        assert p.data is not None

    def test_darcy_solver_runs(self):
        """Verify Darcy solver can be created and solved."""
        import underworld3 as uw

        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

        # SteadyStateDarcy creates its own fields if not provided
        darcy = uw.systems.SteadyStateDarcy(mesh)
        darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
        darcy.constitutive_model.Parameters.permeability = 1.0
        darcy.f = 0.0

        darcy.add_dirichlet_bc(0.0, "Bottom")
        darcy.add_dirichlet_bc(1.0, "Top")

        darcy.solve()

        assert darcy.u.data is not None

    def test_advection_diffusion_solver_runs(self):
        """Verify advection-diffusion solver can be created and solved."""
        import underworld3 as uw
        import numpy as np

        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
        v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)

        # Initialize
        with uw.synchronised_array_update():
            T.array[:, 0, 0] = 0.5
            v.array[:, 0, :] = 0.0

        adv_diff = uw.systems.AdvDiffusion(
            mesh,
            u_Field=T,
            V_fn=v.sym,
        )
        adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
        adv_diff.constitutive_model.Parameters.diffusivity = 1.0

        adv_diff.add_dirichlet_bc(0.0, "Bottom")
        adv_diff.add_dirichlet_bc(1.0, "Top")

        # Just one timestep
        adv_diff.solve(timestep=0.001)

        assert T.data is not None


class TestMeshSmoke:
    """Quick mesh creation tests."""

    def test_structured_quad_mesh(self):
        """Verify StructuredQuadBox mesh creation."""
        import underworld3 as uw

        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
        assert mesh.dim == 2
        assert mesh.X.coords.shape[0] > 0

    def test_unstructured_simplex_mesh(self):
        """Verify UnstructuredSimplexBox mesh creation."""
        import underworld3 as uw

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25
        )
        assert mesh.dim == 2
        assert mesh.X.coords.shape[0] > 0

    def test_annulus_mesh(self):
        """Verify Annulus mesh creation."""
        import underworld3 as uw

        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
        assert mesh.dim == 2
        assert mesh.X.coords.shape[0] > 0


class TestSwarmSmoke:
    """Quick swarm tests."""

    def test_swarm_creation_and_advection(self):
        """Verify swarm can be created and advected."""
        import underworld3 as uw
        import numpy as np

        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
        v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)

        # Constant velocity
        with uw.synchronised_array_update():
            v.array[:, 0, 0] = 0.1
            v.array[:, 0, 1] = 0.0

        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=2)

        initial_count = swarm.data.shape[0]

        # Advect
        swarm.advection(v.sym, delta_t=0.01, order=1)

        # Swarm should still exist
        assert swarm.data.shape[0] == initial_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
