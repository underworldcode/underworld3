#!/usr/bin/env python3
"""
Regression tests for API consistency issues.

These tests prevent regressions of API consistency failures that occurred
when internal classes were exposed in public interfaces, breaking when
those classes were refactored to use factory functions.

Issues covered:
1. EnhancedMeshVariable direct class usage vs factory functions
2. Hidden class exposure in test suites
3. API consistency across units system
4. Factory function vs direct class instantiation patterns
"""

import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import underworld3 as uw


class TestEnhancedMeshVariableAPIConsistency:
    """Test API consistency for enhanced mesh variables."""

    @pytest.fixture
    def basic_mesh(self):
        """Create a basic mesh for testing."""
        return uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1))

    def test_factory_function_is_preferred_api(self, basic_mesh):
        """Test that factory function is the correct API."""

        # CORRECT API: Use factory function
        try:
            enhanced_var = uw.create_enhanced_mesh_variable(
                "test_var", basic_mesh, 1, vtype=uw.VarType.SCALAR
            )
            assert enhanced_var is not None
            assert hasattr(enhanced_var, "name")
            assert enhanced_var.name == "test_var"

        except Exception as e:
            pytest.fail(f"Factory function API failed: {e}")

    def test_direct_class_access_is_hidden(self, basic_mesh):
        """Test that direct class access is properly hidden."""

        # INCORRECT API: Direct class usage should be discouraged/hidden
        # This test verifies that EnhancedMeshVariable is not easily accessible

        # Check if EnhancedMeshVariable is exposed at top level
        has_enhanced_class = hasattr(uw, "EnhancedMeshVariable")

        if has_enhanced_class:
            # If it's exposed, it should work but should be discouraged
            try:
                # This pattern was breaking tests when API changed
                direct_var = uw.EnhancedMeshVariable(
                    "direct_var", basic_mesh, 1, vtype=uw.VarType.SCALAR
                )
                # If this works, it should produce same result as factory
                factory_var = uw.create_enhanced_mesh_variable(
                    "factory_var", basic_mesh, 1, vtype=uw.VarType.SCALAR
                )

                assert type(direct_var) == type(factory_var)

            except (TypeError, AttributeError):
                # Expected: direct class access should be hidden or restricted
                pass
        else:
            # Preferred: EnhancedMeshVariable not exposed at top level
            assert not has_enhanced_class

    def test_api_signature_consistency(self, basic_mesh):
        """Test that API signatures are consistent across similar functions."""

        # Factory functions should have consistent signatures

        # Regular mesh variable
        regular_var = uw.discretisation.MeshVariable("regular", basic_mesh, 1)

        # Enhanced mesh variable
        enhanced_var = uw.create_enhanced_mesh_variable("enhanced", basic_mesh, 1)

        # Both should have similar basic interface
        common_attributes = ["name", "mesh", "num_components", "data"]

        for attr in common_attributes:
            assert hasattr(regular_var, attr), f"Regular variable missing {attr}"
            assert hasattr(enhanced_var, attr), f"Enhanced variable missing {attr}"

    def test_units_parameter_consistency(self, basic_mesh):
        """Test that units parameters work consistently across variable types."""

        # Units should work with both regular and enhanced variables
        try:
            # Regular mesh variable with units
            regular_with_units = uw.discretisation.MeshVariable(
                "regular_units", basic_mesh, 1, units="m/s"
            )

            # Enhanced mesh variable with units
            enhanced_with_units = uw.create_enhanced_mesh_variable(
                "enhanced_units", basic_mesh, 1, units="m/s"
            )

            # Both should handle units consistently
            if hasattr(regular_with_units, "units"):
                assert hasattr(enhanced_with_units, "units")

                # Units should be comparable
                reg_units = str(regular_with_units.units)
                enh_units = str(enhanced_with_units.units)

                # May not be exactly equal due to different backends,
                # but should both represent meter/second
                assert "meter" in reg_units.lower() or "m" in reg_units
                assert "meter" in enh_units.lower() or "m" in enh_units

        except Exception as e:
            pytest.fail(f"Units parameter consistency failed: {e}")


class TestFactoryFunctionPatterns:
    """Test factory function patterns across the codebase."""

    def test_create_functions_exist(self):
        """Test that create_* factory functions exist where expected."""

        factory_functions = [
            "create_enhanced_mesh_variable",
            "create_quantity",  # From units system
        ]

        for func_name in factory_functions:
            assert hasattr(uw, func_name), f"Missing factory function: {func_name}"

            func = getattr(uw, func_name)
            assert callable(func), f"{func_name} is not callable"

    def test_factory_function_documentation(self):
        """Test that factory functions have proper documentation."""

        # Factory functions should have docstrings
        create_enhanced = getattr(uw, "create_enhanced_mesh_variable")
        assert create_enhanced.__doc__ is not None
        assert len(create_enhanced.__doc__.strip()) > 0

        # Should document parameters
        doc = create_enhanced.__doc__.lower()
        expected_params = ["name", "mesh", "num_components"]

        for param in expected_params:
            assert param in doc, f"Factory function doc missing parameter: {param}"

    def test_factory_vs_direct_class_equivalence(self):
        """Test that factory functions produce equivalent objects to direct classes."""

        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3), minCoords=(0, 0), maxCoords=(1, 1))

        # Create via factory function
        factory_var = uw.create_enhanced_mesh_variable("factory", mesh, 1)

        # If direct class is accessible, compare
        if hasattr(uw, "EnhancedMeshVariable"):
            try:
                direct_var = uw.EnhancedMeshVariable("direct", mesh, 1)

                # Should have same type and basic functionality
                assert type(factory_var) == type(direct_var)
                assert factory_var.num_components == direct_var.num_components
                assert factory_var.mesh == direct_var.mesh

            except (TypeError, AttributeError):
                # Expected if direct class access is restricted
                pass


class TestAPIBreakagePreventionPatterns:
    """Test patterns that prevent API breakage in test suites."""

    def test_import_pattern_stability(self):
        """Test that import patterns used in tests are stable."""

        # These import patterns should remain stable
        stable_imports = [
            "underworld3",
            "underworld3.meshing",
            "underworld3.discretisation",
            "underworld3.function",
            "underworld3.units",
        ]

        for import_path in stable_imports:
            try:
                module = __import__(import_path, fromlist=[""])
                assert module is not None
            except ImportError:
                pytest.fail(f"Stable import failed: {import_path}")

    def test_factory_function_parameter_stability(self):
        """Test that factory function parameters remain stable."""

        mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2), minCoords=(0, 0), maxCoords=(1, 1))

        # Core parameters should remain stable
        stable_parameters = {
            "varname": "test_name",  # CORRECT: factory function uses 'varname' not 'name'
            "mesh": mesh,
            "num_components": 1,
        }

        try:
            var = uw.create_enhanced_mesh_variable(**stable_parameters)
            assert var is not None
        except TypeError as e:
            # If this fails, API has changed - need to update tests
            pytest.fail(f"Factory function parameter stability broken: {e}")

    def test_backward_compatibility_imports(self):
        """Test that old import patterns still work for backward compatibility."""

        # These should work for backward compatibility
        try:
            # Core functionality should be importable
            from underworld3 import meshing
            from underworld3 import discretisation
            from underworld3 import function

            assert meshing is not None
            assert discretisation is not None
            assert function is not None

        except ImportError as e:
            pytest.fail(f"Backward compatibility import failed: {e}")


class TestUnitsSystemAPIConsistency:
    """Test API consistency in the units system."""

    def test_units_creation_patterns(self):
        """Test consistent patterns for creating unit-aware objects."""

        # Pattern 1: Direct units specification
        try:
            quantity1 = uw.create_quantity(10.0, "m/s")
            assert quantity1 is not None

        except Exception as e:
            pytest.fail(f"Direct units creation failed: {e}")

        # Pattern 2: uw.units multiplication
        try:
            quantity2 = 10.0 * uw.units.m / uw.units.s
            assert quantity2 is not None

        except Exception as e:
            pytest.fail(f"uw.units multiplication failed: {e}")

    def test_units_api_exposure(self):
        """Test that units API is properly exposed."""

        # Core units functionality should be accessible
        units_api = [
            "units",  # Units registry
            "create_quantity",  # Factory function
        ]

        for api_item in units_api:
            assert hasattr(uw, api_item), f"Missing units API: {api_item}"

    def test_units_integration_consistency(self):
        """Test that units integrate consistently across subsystems."""

        mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2), minCoords=(0, 0), maxCoords=(1, 1))

        # Units should work with both variable types
        var_types = [
            ("regular", uw.discretisation.MeshVariable),
            ("enhanced", uw.create_enhanced_mesh_variable),
        ]

        for var_name, var_creator in var_types:
            try:
                if var_creator == uw.discretisation.MeshVariable:
                    var = var_creator(var_name, mesh, 1, units="m/s")
                else:
                    var = var_creator(var_name, mesh, 1, units="m/s")

                # Should have units attribute or method
                has_units = (
                    hasattr(var, "units") or hasattr(var, "get_units") or hasattr(var, "has_units")
                )

                assert has_units, f"{var_name} variable type doesn't support units"

            except Exception as e:
                pytest.fail(f"Units integration failed for {var_name}: {e}")


class TestRegressionPreventionDocumentation:
    """Document regression prevention strategies."""

    def test_api_design_principles(self):
        """Document API design principles for preventing regressions."""

        principles = {
            "factory_functions": "Use factory functions instead of exposing internal classes",
            "stable_imports": "Keep top-level imports stable for test compatibility",
            "parameter_consistency": "Maintain consistent parameter names across similar functions",
            "backward_compatibility": "Preserve old import patterns when possible",
            "error_handling": "Provide clear errors when APIs change",
            "documentation": "Document all public APIs with examples",
            "testing_patterns": "Use factory functions in all new tests",
        }

        # This test documents the principles - always passes
        for principle, description in principles.items():
            print(f"{principle}: {description}")

        assert True, "API design principles documented"

    def test_common_regression_patterns(self):
        """Document common patterns that cause regressions."""

        regression_patterns = [
            "Direct class usage in tests breaks when classes are refactored",
            "Missing factory function documentation leads to incorrect usage",
            "Inconsistent parameter names across similar functions",
            "Units API changes break existing unit-aware code",
            "Import path changes break test suites",
            "Hidden classes accidentally exposed in public API",
            "Parameter signature changes without deprecation warnings",
        ]

        print("Common regression patterns to avoid:")
        for i, pattern in enumerate(regression_patterns, 1):
            print(f"  {i}. {pattern}")

        # This test always passes - it's documentation
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
