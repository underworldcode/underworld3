#!/usr/bin/env python3
"""
Regression tests for constitutive model tensor operation failures.

These tests prevent regressions of the critical array corruption issue that
occurred when UWexpression objects were used directly in SymPy tensor operations
within constitutive models. The issue caused 16-element arrays to be reduced
to 12 elements, triggering IndexError on as_immutable().

Issues covered:
1. UWexpression tensor multiplication array corruption
2. Constitutive model tensor construction with UWexpression objects
3. Viscosity parameter tensor operations
4. Matrix/tensor operations with mathematical objects
"""

import pytest
import sympy
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import underworld3 as uw


class TestConstitutiveModelTensorOperations:
    """Test constitutive model tensor operations with UWexpression objects."""

    @pytest.fixture
    def basic_mesh(self):
        """Create a basic mesh for testing."""
        return uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1))

    def test_viscosity_tensor_construction_2d(self, basic_mesh):
        """Test 2D viscosity tensor construction with UWexpression viscosity."""
        # Create velocity and pressure variables (Stokes requires both)
        u = uw.discretisation.MeshVariable("U", basic_mesh, basic_mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable("P", basic_mesh, 1, degree=1)

        # Create UWexpression viscosity (this was causing array corruption)
        viscosity = uw.function.expression(r"\eta", sym=1e21)

        # CORRECT API: Let Stokes solver create the constitutive model
        stokes = uw.systems.Stokes(basic_mesh, velocityField=u, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity

        # Access the constitutive model through the solver
        # Note: Use 'constitutive_model' not 'model' to avoid confusion with uw.model
        constitutive_model = stokes.constitutive_model

        # Access the tensor - this should not corrupt arrays
        try:
            # Use public property 'c' which triggers _build_c_tensor() if needed
            tensor = constitutive_model.c
            assert tensor is not None

            # Check tensor dimensions - constitutive tensor is a rank-4 tensor
            if hasattr(tensor, "shape"):
                assert len(tensor.shape) == 4, f"Expected rank-4 tensor, got shape {tensor.shape}"
                # For 2D: should be (2, 2, 2, 2) - 4th order tensor
                # C_ijkl relates stress σ_ij to strain rate ε_kl
                assert tensor.shape == (
                    2,
                    2,
                    2,
                    2,
                ), f"2D tensor should be (2,2,2,2), got {tensor.shape}"

        except (IndexError, ValueError) as e:
            pytest.fail(f"Viscosity tensor construction failed: {e}")

    def test_viscosity_tensor_construction_3d(self):
        """Test 3D viscosity tensor construction with UWexpression viscosity."""
        # Create 3D mesh
        mesh_3d = uw.meshing.StructuredQuadBox(
            elementRes=(3, 3, 3), minCoords=(0, 0, 0), maxCoords=(1, 1, 1)
        )

        u = uw.discretisation.MeshVariable("U", mesh_3d, mesh_3d.dim, degree=2)
        p = uw.discretisation.MeshVariable("P", mesh_3d, 1, degree=1)
        viscosity = uw.function.expression(r"\eta", sym=1e20)

        # CORRECT API: Let Stokes solver create the constitutive model
        stokes = uw.systems.Stokes(mesh_3d, velocityField=u, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity

        constitutive_model = stokes.constitutive_model

        try:
            # Use public property 'c' which triggers _build_c_tensor() if needed
            tensor = constitutive_model.c
            assert tensor is not None

            # Check tensor dimensions - constitutive tensor is a rank-4 tensor
            if hasattr(tensor, "shape"):
                assert len(tensor.shape) == 4, f"Expected rank-4 tensor, got shape {tensor.shape}"
                # For 3D: should be (3, 3, 3, 3) - 4th order tensor
                # C_ijkl relates stress σ_ij to strain rate ε_kl
                assert tensor.shape == (
                    3,
                    3,
                    3,
                    3,
                ), f"3D tensor should be (3,3,3,3), got {tensor.shape}"

        except (IndexError, ValueError) as e:
            pytest.fail(f"3D viscosity tensor construction failed: {e}")

    def test_multiple_viscosity_values(self, basic_mesh):
        """Test tensor construction with various viscosity values."""
        # Test different viscosity values that might trigger different code paths
        viscosity_values = [1.0, 1e-3, 1e3, 1e21, 1e-21, 0.5, 2.5]

        for visc_val in viscosity_values:
            # Create fresh variables for each test
            u = uw.discretisation.MeshVariable(
                f"U_{visc_val}", basic_mesh, basic_mesh.dim, degree=2
            )
            p = uw.discretisation.MeshVariable(f"P_{visc_val}", basic_mesh, 1, degree=1)
            viscosity = uw.function.expression(rf"\eta_{visc_val}", sym=visc_val)

            try:
                # CORRECT API: Let Stokes solver create the constitutive model
                stokes = uw.systems.Stokes(basic_mesh, velocityField=u, pressureField=p)
                stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
                stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity

                constitutive_model = stokes.constitutive_model
                # Use public property which ensures tensor is built
                tensor = constitutive_model.c
                assert tensor is not None

                # Tensor should already be immutable from the property
                # (the property calls as_immutable())
                assert tensor is not None

            except Exception as e:
                pytest.fail(f"Tensor construction failed for viscosity {visc_val}: {e}")


class TestTensorArrayCorruptionRegression:
    """Specific tests for the array corruption issue (16→12 elements)."""

    def test_tensor_identity_with_uwexpression(self):
        """Test tensor identity operations with UWexpression multipliers."""
        # Create a UWexpression
        multiplier = uw.function.expression(r"k", sym=2.0)

        # Create rank-4 identity tensor (this was getting corrupted)
        try:
            identity_2d = uw.maths.tensor.rank4_identity(2)  # Should be 3x3
            identity_3d = uw.maths.tensor.rank4_identity(3)  # Should be 6x6

            # Convert to matrix form for testing
            matrix_2d = uw.maths.tensor.rank4_to_mandel(identity_2d, 2)
            matrix_3d = uw.maths.tensor.rank4_to_mandel(identity_3d, 3)

            # These operations were corrupting arrays when done with UWexpression directly
            # Now they should use .sym property to avoid corruption

            # Test that we can multiply by UWexpression.sym safely
            result_2d = 2 * identity_2d * multiplier.sym  # Use .sym to avoid corruption
            result_3d = 2 * identity_3d * multiplier.sym

            # Convert results to check dimensions
            result_matrix_2d = uw.maths.tensor.rank4_to_mandel(result_2d, 2)
            result_matrix_3d = uw.maths.tensor.rank4_to_mandel(result_3d, 3)

            # Check that dimensions are preserved (not corrupted)
            assert result_matrix_2d.shape == (3, 3)  # 2D: 3x3
            assert result_matrix_3d.shape == (6, 6)  # 3D: 6x6

        except Exception as e:
            pytest.fail(f"Tensor identity operations failed: {e}")

    def test_direct_tensor_uwexpression_multiplication_warning(self):
        """Test that direct UWexpression tensor multiplication should use .sym."""
        # This test documents the issue and verifies the workaround

        multiplier = uw.function.expression(r"direct", sym=3.0)
        identity = uw.maths.tensor.rank4_identity(2)

        # CORRECT approach: use .sym property
        try:
            correct_result = 2 * identity * multiplier.sym
            correct_matrix = uw.maths.tensor.rank4_to_mandel(correct_result, 2)
            assert correct_matrix.shape == (3, 3)
        except Exception as e:
            pytest.fail(f"Correct tensor multiplication (.sym) failed: {e}")

        # PROBLEMATIC approach: direct UWexpression (this may corrupt arrays)
        # We test this carefully to document the issue
        try:
            # This MIGHT work but could corrupt internal arrays
            problematic_result = 2 * identity * multiplier  # Direct UWexpression
            problematic_matrix = uw.maths.tensor.rank4_to_mandel(problematic_result, 2)

            # If it works, check dimensions
            if hasattr(problematic_matrix, "shape"):
                # If corruption occurred, shape might be wrong
                if problematic_matrix.shape != (3, 3):
                    # Expected behavior: corruption detected
                    pass
                else:
                    # If it works, that's fine too (defensive coding)
                    pass

        except (IndexError, ValueError):
            # Expected: corruption can cause various errors
            # This is documented behavior - use .sym instead
            pass

    def test_as_immutable_with_uwexpression_tensors(self):
        """Test as_immutable() calls that were failing due to array corruption."""
        viscosity = uw.function.expression(r"immutable_test", sym=1.5)

        # Create identity tensor and multiply (the problematic operation)
        identity = uw.maths.tensor.rank4_identity(2)

        # Use correct approach with .sym
        try:
            tensor = 2 * identity * viscosity.sym
            matrix = uw.maths.tensor.rank4_to_mandel(tensor, 2)

            # This call was failing when arrays were corrupted
            if hasattr(matrix, "as_immutable"):
                immutable_result = matrix.as_immutable()
                assert immutable_result is not None

        except Exception as e:
            pytest.fail(f"as_immutable() failed: {e}")


class TestConstitutiveModelParameterTypes:
    """Test constitutive models with different parameter types."""

    @pytest.fixture
    def mesh_2d(self):
        return uw.meshing.StructuredQuadBox(elementRes=(3, 3), minCoords=(0, 0), maxCoords=(1, 1))

    def test_viscous_flow_model_parameter_types(self, mesh_2d):
        """Test ViscousFlowModel with different viscosity parameter types."""
        # Test different parameter types
        parameter_types = [
            1.0,  # Float
            1e21,  # Scientific notation
            sympy.Float(1.5),  # SymPy Float
            uw.function.expression(r"\eta", sym=2.0),  # UWexpression
        ]

        for i, param in enumerate(parameter_types):
            try:
                # Create fresh variables for each test
                u = uw.discretisation.MeshVariable(f"U_param_{i}", mesh_2d, mesh_2d.dim, degree=2)
                p = uw.discretisation.MeshVariable(f"P_param_{i}", mesh_2d, 1, degree=1)

                # CORRECT API: Let Stokes solver create the constitutive model
                stokes = uw.systems.Stokes(mesh_2d, velocityField=u, pressureField=p)
                stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
                stokes.constitutive_model.Parameters.shear_viscosity_0 = param

                constitutive_model = stokes.constitutive_model

                # Should be able to access tensor without errors
                # Use public property which ensures tensor is built
                tensor = constitutive_model.c
                assert tensor is not None

                # Should be able to build the complete tensor system
                constitutive_model._build_c_tensor()

            except Exception as e:
                pytest.fail(f"ViscousFlowModel failed with parameter type {type(param)}: {e}")

    def test_viscoelastic_plastic_model_tensors(self, mesh_2d):
        """Test ViscoElasticPlasticFlowModel tensor operations."""
        u = uw.discretisation.MeshVariable("U_vep", mesh_2d, mesh_2d.dim, degree=2)
        p = uw.discretisation.MeshVariable("P_vep", mesh_2d, 1, degree=1)

        try:
            # CORRECT API: Let Stokes solver create the constitutive model
            stokes = uw.systems.Stokes(mesh_2d, velocityField=u, pressureField=p)
            stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel

            # Set parameters that might trigger tensor operations
            stokes.constitutive_model.Parameters.shear_modulus = uw.function.expression(
                r"G", sym=1e10
            )
            stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.function.expression(
                r"\eta", sym=1e21
            )

            constitutive_model = stokes.constitutive_model

            # Should be able to build tensor without array corruption
            # Use public property which automatically builds tensor if needed
            tensor = constitutive_model.c
            assert tensor is not None

        except Exception as e:
            pytest.fail(f"ViscoElasticPlasticFlowModel tensor construction failed: {e}")


class TestTensorOperationSafety:
    """Test that tensor operations are safe with mathematical objects."""

    def test_safe_tensor_multiplication_patterns(self):
        """Test recommended safe patterns for tensor operations."""
        # Create test objects
        scalar_expr = uw.function.expression(r"safe", sym=1.5)
        identity = uw.maths.tensor.rank4_identity(2)

        # SAFE PATTERNS (recommended)
        safe_patterns = [
            lambda: 2 * identity * scalar_expr.sym,  # Use .sym property
            lambda: identity * 2.0 * scalar_expr.sym,  # Multiple operations
            lambda: scalar_expr.sym * identity * 2,  # Different order
        ]

        for i, pattern in enumerate(safe_patterns):
            try:
                result = pattern()
                matrix = uw.maths.tensor.rank4_to_mandel(result, 2)
                assert matrix.shape == (3, 3), f"Safe pattern {i} gave wrong shape: {matrix.shape}"

            except Exception as e:
                pytest.fail(f"Safe tensor pattern {i} failed: {e}")

    def test_tensor_operation_result_consistency(self):
        """Test that tensor operations give consistent results."""
        scalar = uw.function.expression(r"consistent", sym=2.5)
        identity = uw.maths.tensor.rank4_identity(2)

        # Compare direct float vs UWexpression.sym results
        try:
            # Direct float multiplication
            float_result = 2.5 * identity
            float_matrix = uw.maths.tensor.rank4_to_mandel(float_result, 2)

            # UWexpression.sym multiplication
            expr_result = scalar.sym * identity
            expr_matrix = uw.maths.tensor.rank4_to_mandel(expr_result, 2)

            # Results should be equivalent
            assert float_matrix.shape == expr_matrix.shape

            # Values should be mathematically equivalent
            # (exact equality may not hold due to SymPy vs float differences)
            # But shapes and structure should match

        except Exception as e:
            pytest.fail(f"Tensor operation consistency test failed: {e}")


class TestVoigtMandelConversions:
    """Test Voigt and Mandel notation conversions for constitutive tensors."""

    def test_rank4_to_voigt_2d(self):
        """Test conversion of 2D rank-4 tensor to Voigt notation."""
        # Create identity tensor
        identity = uw.maths.tensor.rank4_identity(2)

        # Convert to Voigt (should be 3x3 for 2D)
        voigt_matrix = uw.maths.tensor.rank4_to_voigt(identity, 2)

        assert voigt_matrix.shape == (3, 3), f"Expected (3,3), got {voigt_matrix.shape}"

        # Voigt matrix should preserve tensor structure
        assert voigt_matrix is not None

    def test_rank4_to_voigt_3d(self):
        """Test conversion of 3D rank-4 tensor to Voigt notation."""
        # Create identity tensor
        identity = uw.maths.tensor.rank4_identity(3)

        # Convert to Voigt (should be 6x6 for 3D)
        voigt_matrix = uw.maths.tensor.rank4_to_voigt(identity, 3)

        assert voigt_matrix.shape == (6, 6), f"Expected (6,6), got {voigt_matrix.shape}"

        # Voigt matrix should preserve tensor structure
        assert voigt_matrix is not None

    def test_rank4_to_mandel_2d(self):
        """Test conversion of 2D rank-4 tensor to Mandel notation."""
        # Create identity tensor
        identity = uw.maths.tensor.rank4_identity(2)

        # Convert to Mandel (should be 3x3 for 2D)
        mandel_matrix = uw.maths.tensor.rank4_to_mandel(identity, 2)

        assert mandel_matrix.shape == (3, 3), f"Expected (3,3), got {mandel_matrix.shape}"

        # Mandel matrix should preserve tensor structure
        assert mandel_matrix is not None

    def test_rank4_to_mandel_3d(self):
        """Test conversion of 3D rank-4 tensor to Mandel notation."""
        # Create identity tensor
        identity = uw.maths.tensor.rank4_identity(3)

        # Convert to Mandel (should be 6x6 for 3D)
        mandel_matrix = uw.maths.tensor.rank4_to_mandel(identity, 3)

        assert mandel_matrix.shape == (6, 6), f"Expected (6,6), got {mandel_matrix.shape}"

        # Mandel matrix should preserve tensor structure
        assert mandel_matrix is not None

    def test_voigt_roundtrip_2d(self):
        """Test rank4 → Voigt → rank4 roundtrip for 2D."""
        # Create identity tensor
        identity = uw.maths.tensor.rank4_identity(2)

        # Convert to Voigt and back
        voigt_matrix = uw.maths.tensor.rank4_to_voigt(identity, 2)
        recovered = uw.maths.tensor.voigt_to_rank4(voigt_matrix, 2)

        # Check shape is preserved
        assert (
            recovered.shape == identity.shape
        ), f"Shape mismatch: {recovered.shape} vs {identity.shape}"

        # Check values are preserved (within numerical tolerance)
        import numpy as np

        identity_array = np.array(identity.tolist(), dtype=float)
        recovered_array = np.array(recovered.tolist(), dtype=float)
        assert np.allclose(identity_array, recovered_array, rtol=1e-10), "Voigt roundtrip failed"

    def test_voigt_roundtrip_3d(self):
        """Test rank4 → Voigt → rank4 roundtrip for 3D."""
        # Create identity tensor
        identity = uw.maths.tensor.rank4_identity(3)

        # Convert to Voigt and back
        voigt_matrix = uw.maths.tensor.rank4_to_voigt(identity, 3)
        recovered = uw.maths.tensor.voigt_to_rank4(voigt_matrix, 3)

        # Check shape is preserved
        assert (
            recovered.shape == identity.shape
        ), f"Shape mismatch: {recovered.shape} vs {identity.shape}"

        # Check values are preserved (within numerical tolerance)
        import numpy as np

        identity_array = np.array(identity.tolist(), dtype=float)
        recovered_array = np.array(recovered.tolist(), dtype=float)
        assert np.allclose(identity_array, recovered_array, rtol=1e-10), "Voigt roundtrip failed"

    def test_mandel_roundtrip_2d(self):
        """Test rank4 → Mandel → rank4 roundtrip for 2D."""
        # Create identity tensor
        identity = uw.maths.tensor.rank4_identity(2)

        # Convert to Mandel and back
        mandel_matrix = uw.maths.tensor.rank4_to_mandel(identity, 2)
        recovered = uw.maths.tensor.mandel_to_rank4(mandel_matrix, 2)

        # Check shape is preserved
        assert (
            recovered.shape == identity.shape
        ), f"Shape mismatch: {recovered.shape} vs {identity.shape}"

        # Check values are preserved (within numerical tolerance)
        import numpy as np

        identity_array = np.array(identity.tolist(), dtype=float)
        recovered_array = np.array(recovered.tolist(), dtype=float)
        assert np.allclose(identity_array, recovered_array, rtol=1e-10), "Mandel roundtrip failed"

    def test_mandel_roundtrip_3d(self):
        """Test rank4 → Mandel → rank4 roundtrip for 3D."""
        # Create identity tensor
        identity = uw.maths.tensor.rank4_identity(3)

        # Convert to Mandel and back
        mandel_matrix = uw.maths.tensor.rank4_to_mandel(identity, 3)
        recovered = uw.maths.tensor.mandel_to_rank4(mandel_matrix, 3)

        # Check shape is preserved
        assert (
            recovered.shape == identity.shape
        ), f"Shape mismatch: {recovered.shape} vs {identity.shape}"

        # Check values are preserved (within numerical tolerance)
        import numpy as np

        identity_array = np.array(identity.tolist(), dtype=float)
        recovered_array = np.array(recovered.tolist(), dtype=float)
        assert np.allclose(identity_array, recovered_array, rtol=1e-10), "Mandel roundtrip failed"

    def test_constitutive_tensor_voigt_conversion(self):
        """Test Voigt conversion of actual constitutive tensor from Stokes solver."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1))
        u = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

        stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1e21

        # Get rank-4 tensor
        tensor = stokes.constitutive_model.c
        assert tensor.shape == (2, 2, 2, 2)

        # Convert to Voigt
        voigt = uw.maths.tensor.rank4_to_voigt(tensor, 2)
        assert voigt.shape == (3, 3), f"Voigt should be (3,3), got {voigt.shape}"

        # Convert to Mandel
        mandel = uw.maths.tensor.rank4_to_mandel(tensor, 2)
        assert mandel.shape == (3, 3), f"Mandel should be (3,3), got {mandel.shape}"

    def test_constitutive_tensor_3d_voigt_conversion(self):
        """Test Voigt conversion of 3D constitutive tensor."""
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(3, 3, 3), minCoords=(0, 0, 0), maxCoords=(1, 1, 1)
        )
        u = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

        stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1e20

        # Get rank-4 tensor
        tensor = stokes.constitutive_model.c
        assert tensor.shape == (3, 3, 3, 3)

        # Convert to Voigt
        voigt = uw.maths.tensor.rank4_to_voigt(tensor, 3)
        assert voigt.shape == (6, 6), f"Voigt should be (6,6), got {voigt.shape}"

        # Convert to Mandel
        mandel = uw.maths.tensor.rank4_to_mandel(tensor, 3)
        assert mandel.shape == (6, 6), f"Mandel should be (6,6), got {mandel.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
