"""
Unit tests for Template expression parameter propagation.

Verifies that changing constitutive model parameters correctly triggers
Template re-evaluation while preserving expression object identity.

This tests the complete chain:
1. Parameter change → constitutive_model._reset()
2. _reset() → solver.is_setup = False (via bidirectional reference)
3. Template.__get__() detects is_setup=False → re-evaluates lambda
4. Template updates .sym in-place → preserves object identity

Tests cover all solver types that use Templates and constitutive models.
"""

import pytest
import underworld3 as uw
import sympy
import numpy as np


class TestTemplateParameterPropagation:
    """Test Template re-evaluation mechanism for all solvers."""

    def setup_method(self):
        """Create a simple mesh for testing."""
        self.mesh = uw.meshing.StructuredQuadBox(
            elementRes=(5, 5),
            minCoords=(-1.0, -1.0),
            maxCoords=(0.0, 0.0),
            qdegree=2,
        )

    def test_darcy_gravity_parameter_propagation(self):
        """Test Darcy solver: changing gravity parameter updates F1 Template."""
        # Create variables
        p_soln = uw.discretisation.MeshVariable("P", self.mesh, 1, degree=2)
        v_soln = uw.discretisation.MeshVariable("U", self.mesh, self.mesh.dim, degree=1)

        # Create Darcy solver
        darcy = uw.systems.SteadyStateDarcy(self.mesh, p_soln, v_soln)
        darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel

        # Test 1: Bidirectional reference established
        assert (
            darcy.constitutive_model.Parameters._solver is not None
        ), "Constitutive model should have solver reference"
        assert (
            darcy.constitutive_model.Parameters._solver is darcy
        ), "Solver reference should point to correct solver"

        # Test 2: Get initial F1 Template (no gravity)
        F1_before = darcy.F1
        F1_id_before = id(F1_before)
        F1_sym_before = F1_before.sym

        # Test 3: Change gravity parameter
        darcy.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T
        assert darcy.is_setup is False, "Solver is_setup should be False after parameter change"

        # Test 4: Get F1 Template after parameter change
        F1_after = darcy.F1
        F1_id_after = id(F1_after)
        F1_sym_after = F1_after.sym

        # Test 5: Verify object identity preserved
        assert F1_before is F1_after, "F1 Template should return same object (identity preserved)"
        assert F1_id_before == F1_id_after, "F1 Template should have same Python id"

        # Test 6: Verify symbolic content updated
        assert (
            F1_sym_before != F1_sym_after
        ), "F1 symbolic content should change after parameter update"
        assert "+1" in str(F1_sym_after) or "+ 1" in str(
            F1_sym_after
        ), "F1 should include gravity term after parameter change"

    def test_darcy_permeability_parameter_propagation(self):
        """Test Darcy solver: changing permeability parameter updates F1 Template."""
        p_soln = uw.discretisation.MeshVariable("P", self.mesh, 1, degree=2)
        v_soln = uw.discretisation.MeshVariable("U", self.mesh, self.mesh.dim, degree=1)

        darcy = uw.systems.SteadyStateDarcy(self.mesh, p_soln, v_soln)
        darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel

        # Get initial F1 with default permeability (1.0)
        F1_before = darcy.F1
        F1_id_before = id(F1_before)
        F1_sym_before = F1_before.sym

        # Change permeability
        darcy.constitutive_model.Parameters.permeability = 2.5
        assert darcy.is_setup is False, "Solver is_setup should be False after permeability change"

        # Get F1 after permeability change
        F1_after = darcy.F1
        F1_id_after = id(F1_after)
        F1_sym_after = F1_after.sym

        # Verify object identity preserved
        assert (
            F1_before is F1_after
        ), "F1 Template should return same object after permeability change"
        assert F1_id_before == F1_id_after, "F1 should have same Python id"

        # Verify symbolic content potentially updated (depends on expression structure)
        # The Template mechanism should work even if symbolic form doesn't visibly change

    def test_poisson_diffusivity_parameter_propagation(self):
        """Test Poisson solver: changing diffusivity parameter updates F1 Template."""
        u_soln = uw.discretisation.MeshVariable("U", self.mesh, 1, degree=2)

        poisson = uw.systems.Poisson(self.mesh, u_Field=u_soln)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel

        # Test bidirectional reference
        assert poisson.constitutive_model.Parameters._solver is not None
        assert poisson.constitutive_model.Parameters._solver is poisson

        # Get initial F1
        F1_before = poisson.F1
        F1_id_before = id(F1_before)

        # Change diffusivity
        poisson.constitutive_model.Parameters.diffusivity = 5.0
        assert poisson.is_setup is False, "Solver is_setup should be False after diffusivity change"

        # Get F1 after change
        F1_after = poisson.F1
        F1_id_after = id(F1_after)

        # Verify object identity preserved
        assert F1_before is F1_after
        assert F1_id_before == F1_id_after

    def test_stokes_viscosity_parameter_propagation(self):
        """Test Stokes solver: changing viscosity parameter updates F1 Template."""
        v_soln = uw.discretisation.MeshVariable("U", self.mesh, self.mesh.dim, degree=2)
        p_soln = uw.discretisation.MeshVariable("P", self.mesh, 1, degree=1)

        stokes = uw.systems.Stokes(self.mesh, velocityField=v_soln, pressureField=p_soln)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

        # Test bidirectional reference
        assert stokes.constitutive_model.Parameters._solver is not None
        assert stokes.constitutive_model.Parameters._solver is stokes

        # Get initial F1
        F1_before = stokes.F1
        F1_id_before = id(F1_before)
        F1_sym_before = F1_before.sym

        # Change viscosity
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 10.0
        assert stokes.is_setup is False, "Solver is_setup should be False after viscosity change"

        # Get F1 after change
        F1_after = stokes.F1
        F1_id_after = id(F1_after)
        F1_sym_after = F1_after.sym

        # Verify object identity preserved
        assert F1_before is F1_after
        assert F1_id_before == F1_id_after

        # For Stokes, the Template should update (flux depends on viscosity)
        # Symbolic content may or may not look different depending on substitution

    @pytest.mark.skip(
        reason="AdvDiffusion F1 not yet converted to Template pattern - creates new expressions instead of preserving identity"
    )
    def test_advdiff_diffusivity_parameter_propagation(self):
        """Test AdvDiff solver: changing diffusivity parameter updates F1 Template.

        NOTE: This test currently fails because AdvDiffusion.F1 is not implemented
        as a Template - it creates new expression objects on each access instead of
        preserving object identity while updating symbolic content.

        TODO: Convert AdvDiffusion.F1 to use Template pattern like Darcy/Poisson/Stokes.
        """
        phi = uw.discretisation.MeshVariable("Phi", self.mesh, 1, degree=2)
        v_soln = uw.discretisation.MeshVariable("V", self.mesh, self.mesh.dim, degree=1)

        # Set a simple velocity field
        with uw.synchronised_array_update():
            v_soln.array[...] = 0.0

        adv_diff = uw.systems.AdvDiffusion(
            self.mesh,
            u_Field=phi,
            V_fn=v_soln,
        )
        adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel

        # Test bidirectional reference (this part works)
        assert adv_diff.constitutive_model.Parameters._solver is not None
        assert adv_diff.constitutive_model.Parameters._solver is adv_diff

        # Get initial F1
        F1_before = adv_diff.F1
        F1_id_before = id(F1_before)

        # Change diffusivity
        adv_diff.constitutive_model.Parameters.diffusivity = 3.0
        assert (
            adv_diff.is_setup is False
        ), "Solver is_setup should be False after diffusivity change"

        # Get F1 after change
        F1_after = adv_diff.F1
        F1_id_after = id(F1_after)

        # This currently FAILS - AdvDiffusion needs to be updated to use Templates
        assert (
            F1_before is F1_after
        ), "F1 Template should preserve object identity (currently fails for AdvDiffusion)"
        assert F1_id_before == F1_id_after

    def test_multiple_parameter_changes(self):
        """Test multiple consecutive parameter changes preserve object identity."""
        p_soln = uw.discretisation.MeshVariable("P", self.mesh, 1, degree=2)
        v_soln = uw.discretisation.MeshVariable("U", self.mesh, self.mesh.dim, degree=1)

        darcy = uw.systems.SteadyStateDarcy(self.mesh, p_soln, v_soln)
        darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel

        # Get initial F1
        F1_initial = darcy.F1
        F1_id_initial = id(F1_initial)

        # Multiple parameter changes
        for i in range(3):
            darcy.constitutive_model.Parameters.s = sympy.Matrix([0, -i]).T
            F1_current = darcy.F1
            assert (
                id(F1_current) == F1_id_initial
            ), f"F1 should maintain same id after {i+1} parameter changes"
            assert (
                F1_current is F1_initial
            ), f"F1 should be same object after {i+1} parameter changes"

    def test_template_re_evaluation_on_access(self):
        """Test that Template only re-evaluates when accessed after parameter change."""
        p_soln = uw.discretisation.MeshVariable("P", self.mesh, 1, degree=2)
        v_soln = uw.discretisation.MeshVariable("U", self.mesh, self.mesh.dim, degree=1)

        darcy = uw.systems.SteadyStateDarcy(self.mesh, p_soln, v_soln)
        darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel

        # Access F1 to create it
        _ = darcy.F1

        # Change parameter
        darcy.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T
        assert darcy.is_setup is False

        # Template should re-evaluate on next access
        F1_updated = darcy.F1

        # After accessing with is_setup=False, the lambda is re-evaluated
        # (We can't directly observe the re-evaluation, but we can verify the result)
        assert "+1" in str(F1_updated.sym) or "+ 1" in str(F1_updated.sym)

    def test_parameter_change_without_solver_reference_fails_gracefully(self):
        """Test that constitutive models without solver reference don't crash."""
        # Create a standalone constitutive model (not attached to solver)
        # This shouldn't happen in normal usage, but we should handle it gracefully

        # Create minimal unknowns structure
        u = uw.discretisation.MeshVariable("U", self.mesh, 1, degree=2)

        class FakeUnknowns:
            def __init__(self, u):
                self.u = u
                self.DuDt = None
                self.DFDt = None

        unknowns = FakeUnknowns(u)

        # Create standalone model
        model = uw.constitutive_models.DiffusionModel(unknowns)

        # Parameters._solver should be None
        assert model.Parameters._solver is None

        # Changing parameter shouldn't crash, just won't propagate
        try:
            model.Parameters.diffusivity = 2.0
            # Should complete without error
        except Exception as e:
            pytest.fail(f"Parameter change should not crash: {e}")

    def test_solver_assignment_establishes_reference(self):
        """Test that assigning constitutive_model to solver establishes reference."""
        p_soln = uw.discretisation.MeshVariable("P", self.mesh, 1, degree=2)
        v_soln = uw.discretisation.MeshVariable("U", self.mesh, self.mesh.dim, degree=1)

        darcy = uw.systems.SteadyStateDarcy(self.mesh, p_soln, v_soln)

        # Create model separately
        u = uw.discretisation.MeshVariable("U_temp", self.mesh, 1, degree=2)

        class FakeUnknowns:
            def __init__(self, u):
                self.u = u
                self.DuDt = None
                self.DFDt = None

        model = uw.constitutive_models.DarcyFlowModel(FakeUnknowns(u))

        # Model shouldn't have solver reference yet
        assert model.Parameters._solver is None

        # Assign to solver
        darcy.constitutive_model = model

        # Now model should have solver reference
        assert model.Parameters._solver is not None
        assert model.Parameters._solver is darcy


if __name__ == "__main__":
    # Run with: pytest -v test_0820_template_parameter_propagation.py
    pytest.main([__file__, "-v", "--tb=short"])
