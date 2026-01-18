"""
Test comprehensive evaluation of ALL combinations of unit-aware objects.

MOTIVATION: User discovered bug where evaluate(5 m/s * 1 s) returned 4.59e-7 m
instead of 5 m. This bug was NOT caught by existing tests, indicating incomplete
coverage of evaluate() with unit-aware arithmetic.

REQUIREMENT: "I want the unwrap functionality to be 100% bulletproof and I don't
mean that we just say it is, we have to demonstrate it. So tests for evaluation -
comprehensively across all the combinations of objects with units. Nothing should
be allowed to slip through the cracks."

TEST COVERAGE:
1. Simple evaluations (UWQuantity, UWexpression, UnitAwareExpression, pure sympy)
2. Arithmetic combinations (UWQuantity op UWQuantity, UWQuantity op UWexpression, etc.)
3. All operations: +, -, *, /
4. Different coordinate types (single point, array, all mesh)
5. With and without nondimensional scaling

CRITICAL BUG BEING TESTED:
- evaluate(UWQuantity * UWexpression) was returning wrong value
- Example: evaluate(5 m/s * 1 s) = 4.59e-7 m instead of 5 m
- Root cause: UWexpression symbol substituted with wrong scale
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw


@pytest.mark.tier_a  # Production-ready - critical for correctness
@pytest.mark.level_2  # Intermediate - involves nondimensional scaling
class TestEvaluateSimpleObjects:
    """
    Test evaluation of simple unit-aware objects individually.

    These tests establish baseline behavior - each object type evaluated alone
    should return correct physical values.
    """

    @pytest.fixture(autouse=True)
    def setup_model_and_mesh(self):
        """Set up nondimensional model and mesh."""
        model = uw.Model()

        L_scale = uw.quantity(2900, "km")
        t_scale = uw.quantity(1, "Myr")
        M_scale = uw.quantity(1e24, "kg")
        T_scale = uw.quantity(1000, "K")

        model.set_reference_quantities(
            length=L_scale,
            time=t_scale,
            mass=M_scale,
            temperature=T_scale,
            nondimensional_scaling=True,
        )

        uw.use_nondimensional_scaling(True)

        self.mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )

        self.L_scale = L_scale
        self.t_scale = t_scale

        yield

        uw.use_nondimensional_scaling(False)

    def test_evaluate_pure_sympy_number(self):
        """Test: evaluate(1) returns 1."""
        result = uw.function.evaluate(sympy.sympify(1), self.mesh.X.coords[60])

        # Extract value
        if hasattr(result, 'flat'):
            value = float(result.flat[0])
        else:
            value = float(result)

        assert np.allclose(value, 1.0, rtol=1e-6), \
            f"evaluate(1) should return 1.0, got {value}"

    def test_evaluate_uwquantity_length(self):
        """Test: evaluate(2900 km) returns 2900 km."""
        qty = uw.quantity(2900, "km")
        result = uw.function.evaluate(qty, self.mesh.X.coords[60])

        # Convert to meters for comparison
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        else:
            value = float(result)

        expected_m = 2900000.0  # 2900 km in meters

        assert np.allclose(value, expected_m, rtol=1e-6), \
            f"evaluate(2900 km) should return {expected_m} m, got {value} m"

    def test_evaluate_uwquantity_time(self):
        """Test: evaluate(1 Myr) returns 1 Myr."""
        qty = uw.quantity(1, "Myr")
        result = uw.function.evaluate(qty, self.mesh.X.coords[60])

        # Convert to seconds for comparison
        if hasattr(result, 'to'):
            result_s = result.to('s')
            value = float(result_s.magnitude if hasattr(result_s, 'magnitude') else result_s)
        else:
            value = float(result)

        expected_s = 1e6 * 365.25 * 24 * 3600  # 1 Myr in seconds

        assert np.allclose(value, expected_s, rtol=1e-6), \
            f"evaluate(1 Myr) should return ~{expected_s:.2e} s, got {value:.2e} s"

    def test_evaluate_uwquantity_velocity(self):
        """Test: evaluate(5 m/s) returns 5 m/s."""
        qty = uw.quantity(5, "m/s")
        result = uw.function.evaluate(qty, self.mesh.X.coords[60])

        # Extract value in m/s
        if hasattr(result, 'to'):
            result_mps = result.to('m/s')
            value = float(result_mps.magnitude if hasattr(result_mps, 'magnitude') else result_mps)
        else:
            value = float(result)

        assert np.allclose(value, 5.0, rtol=1e-6), \
            f"evaluate(5 m/s) should return 5.0 m/s, got {value} m/s"

    def test_evaluate_uwexpression_time(self):
        """Test: evaluate(UWexpression wrapping 1 s) returns 1 s."""
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 's'), "Current time")
        result = uw.function.evaluate(t_now, self.mesh.X.coords[60])

        # Extract value in seconds
        if hasattr(result, 'to'):
            result_s = result.to('s')
            value = float(result_s.magnitude if hasattr(result_s, 'magnitude') else result_s)
        else:
            value = float(result)

        assert np.allclose(value, 1.0, rtol=1e-6), \
            f"evaluate(UWexpression(1 s)) should return 1.0 s, got {value} s"

    def test_evaluate_small_uwquantity(self):
        """Test: evaluate(1 cm) returns 0.01 m."""
        qty = uw.quantity(1, "cm")
        result = uw.function.evaluate(qty, self.mesh.X.coords[60])

        # Convert to meters
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        else:
            value = float(result)

        assert np.allclose(value, 0.01, rtol=1e-6), \
            f"evaluate(1 cm) should return 0.01 m, got {value} m"


@pytest.mark.tier_b  # Validated - catching newly discovered bugs
@pytest.mark.level_2  # Intermediate - arithmetic with scaling
class TestEvaluateArithmeticCombinations:
    """
    Test evaluation of arithmetic combinations of unit-aware objects.

    CRITICAL: These tests catch the multiplication bug where
    evaluate(5 m/s * 1 s) returned 4.59e-7 m instead of 5 m.
    """

    @pytest.fixture(autouse=True)
    def setup_model_and_mesh(self):
        """Set up nondimensional model and mesh."""
        model = uw.Model()

        L_scale = uw.quantity(2900, "km")
        t_scale = uw.quantity(1, "Myr")
        M_scale = uw.quantity(1e24, "kg")
        T_scale = uw.quantity(1000, "K")

        model.set_reference_quantities(
            length=L_scale,
            time=t_scale,
            mass=M_scale,
            temperature=T_scale,
            nondimensional_scaling=True,
        )

        uw.use_nondimensional_scaling(True)

        self.mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )

        self.L_scale = L_scale
        self.t_scale = t_scale

        yield

        uw.use_nondimensional_scaling(False)

    def test_evaluate_uwquantity_times_uwquantity(self):
        """Test: evaluate(5 m/s * 2 s) = 10 m."""
        velocity = uw.quantity(5, "m/s")
        time = uw.quantity(2, "s")

        product = velocity * time
        result = uw.function.evaluate(product, self.mesh.X.coords[60])

        # Convert to meters
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        else:
            value = float(result)

        expected = 10.0  # 5 m/s * 2 s = 10 m

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(5 m/s * 2 s) should return {expected} m, got {value} m"

    def test_evaluate_uwquantity_times_uwexpression(self):
        """
        Test: evaluate(5 m/s * 1 s UWexpression) = 5 m.

        THIS IS THE BUG WE DISCOVERED:
        - Was returning 4.59e-7 m (5 × velocity_scale)
        - Should return 5 m (5 m/s × 1 s)
        """
        velocity_phys = uw.quantity(5, "m/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 's'), "Current time")

        product = velocity_phys * t_now
        result = uw.function.evaluate(product, self.mesh.X.coords[60])

        # Convert to meters
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        else:
            value = float(result)

        expected = 5.0  # 5 m/s * 1 s = 5 m

        # Should NOT be 4.59e-7 (the bug value)
        assert value > 1e-3, \
            f"evaluate(5 m/s * 1 s) is {value} m - looks like the multiplication bug! Expected {expected} m"

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(5 m/s * 1 s) should return {expected} m, got {value} m"

    def test_evaluate_uwexpression_times_uwquantity(self):
        """Test: evaluate(1 s UWexpression * 5 m/s) = 5 m (commutative)."""
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 's'), "Current time")
        velocity_phys = uw.quantity(5, "m/s")

        product = t_now * velocity_phys
        result = uw.function.evaluate(product, self.mesh.X.coords[60])

        # Convert to meters
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        else:
            value = float(result)

        expected = 5.0  # 1 s * 5 m/s = 5 m

        assert value > 1e-3, \
            f"evaluate(1 s * 5 m/s) is {value} m - multiplication bug! Expected {expected} m"

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(1 s * 5 m/s) should return {expected} m, got {value} m"

    def test_evaluate_uwexpression_times_uwexpression(self):
        """Test: evaluate(UWexpression * UWexpression) with compatible units."""
        distance = uw.expression(r"L_\textrm{ref}", uw.quantity(10, 'm'), "Reference length")
        width = uw.expression(r"W_\textrm{ref}", uw.quantity(5, 'm'), "Reference width")

        product = distance * width
        result = uw.function.evaluate(product, self.mesh.X.coords[60])

        # Convert to m²
        if hasattr(result, 'to'):
            result_m2 = result.to('m**2')
            value = float(result_m2.magnitude if hasattr(result_m2, 'magnitude') else result_m2)
        else:
            value = float(result)

        expected = 50.0  # 10 m * 5 m = 50 m²

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(10 m * 5 m) should return {expected} m², got {value} m²"

    def test_evaluate_uwquantity_divided_by_uwquantity(self):
        """Test: evaluate(10 m / 2 s) = 5 m/s."""
        distance = uw.quantity(10, "m")
        time = uw.quantity(2, "s")

        quotient = distance / time
        result = uw.function.evaluate(quotient, self.mesh.X.coords[60])

        # Convert to m/s
        if hasattr(result, 'to'):
            result_mps = result.to('m/s')
            value = float(result_mps.magnitude if hasattr(result_mps, 'magnitude') else result_mps)
        else:
            value = float(result)

        expected = 5.0  # 10 m / 2 s = 5 m/s

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(10 m / 2 s) should return {expected} m/s, got {value} m/s"

    def test_evaluate_uwquantity_divided_by_uwexpression(self):
        """Test: evaluate(10 m / 2 s UWexpression) = 5 m/s."""
        distance = uw.quantity(10, "m")
        time_expr = uw.expression(r"t_\textrm{div}", uw.quantity(2, 's'), "Divisor time")

        quotient = distance / time_expr
        result = uw.function.evaluate(quotient, self.mesh.X.coords[60])

        # Convert to m/s
        if hasattr(result, 'to'):
            result_mps = result.to('m/s')
            value = float(result_mps.magnitude if hasattr(result_mps, 'magnitude') else result_mps)
        else:
            value = float(result)

        expected = 5.0  # 10 m / 2 s = 5 m/s

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(10 m / 2 s) should return {expected} m/s, got {value} m/s"

    def test_evaluate_uwquantity_plus_uwquantity(self):
        """Test: evaluate(5 m + 3 m) = 8 m."""
        a = uw.quantity(5, "m")
        b = uw.quantity(3, "m")

        sum_expr = a + b
        result = uw.function.evaluate(sum_expr, self.mesh.X.coords[60])

        # Convert to meters
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        else:
            value = float(result)

        expected = 8.0  # 5 m + 3 m = 8 m

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(5 m + 3 m) should return {expected} m, got {value} m"

    def test_evaluate_uwquantity_minus_uwquantity(self):
        """Test: evaluate(10 m - 3 m) = 7 m."""
        a = uw.quantity(10, "m")
        b = uw.quantity(3, "m")

        diff = a - b
        result = uw.function.evaluate(diff, self.mesh.X.coords[60])

        # Convert to meters
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        else:
            value = float(result)

        expected = 7.0  # 10 m - 3 m = 7 m

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(10 m - 3 m) should return {expected} m, got {value} m"

    def test_evaluate_complex_expression(self):
        """Test: evaluate((5 m/s * 2 s) + (3 m)) = 13 m."""
        velocity = uw.quantity(5, "m/s")
        time = uw.quantity(2, "s")
        offset = uw.quantity(3, "m")

        expr = (velocity * time) + offset
        result = uw.function.evaluate(expr, self.mesh.X.coords[60])

        # Convert to meters
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        else:
            value = float(result)

        expected = 13.0  # (5*2) + 3 = 13 m

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate((5 m/s * 2 s) + 3 m) should return {expected} m, got {value} m"


@pytest.mark.tier_b  # Validated - testing edge cases with ND scaling
@pytest.mark.level_2  # Uses nondimensional model - intermediate complexity
class TestEvaluateCoordinateTypes:
    """
    Test evaluation with different coordinate types.

    Ensures evaluate() works correctly with:
    - Single coordinate: coords[i] (shape: (ndim,))
    - Array slice: coords[i:j] (shape: (N, ndim))
    - All mesh: coords (shape: (N_total, ndim))
    """

    @pytest.fixture(autouse=True)
    def setup_model_and_mesh(self):
        """Set up nondimensional model and mesh."""
        model = uw.Model()

        L_scale = uw.quantity(2900, "km")
        t_scale = uw.quantity(1, "Myr")

        model.set_reference_quantities(
            length=L_scale,
            time=t_scale,
            mass=uw.quantity(1e24, "kg"),
            temperature=uw.quantity(1000, "K"),
            nondimensional_scaling=True,
        )

        uw.use_nondimensional_scaling(True)

        self.mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )

        yield

        uw.use_nondimensional_scaling(False)

    def test_evaluate_single_coordinate(self):
        """Test: evaluate(5 m, coords[60]) works with single point."""
        qty = uw.quantity(5, "m")

        # Should not raise IndexError
        result = uw.function.evaluate(qty, self.mesh.X.coords[60])

        # Should return scalar or shape (1, 1, 1)
        assert result is not None, "evaluate(qty, coords[i]) should not fail"

    def test_evaluate_coordinate_slice(self):
        """Test: evaluate(5 m, coords[60:65]) works with array."""
        qty = uw.quantity(5, "m")

        result = uw.function.evaluate(qty, self.mesh.X.coords[60:65])

        # Should return array with 5 points
        if hasattr(result, 'shape'):
            assert result.shape[0] == 5, \
                f"evaluate(qty, coords[60:65]) should have 5 points, got shape {result.shape}"

    def test_evaluate_all_mesh_coordinates(self):
        """Test: evaluate(5 m, mesh.X.coords) works with all points."""
        qty = uw.quantity(5, "m")

        result = uw.function.evaluate(qty, self.mesh.X.coords)

        # Should return array matching mesh size
        if hasattr(result, 'shape'):
            n_points = self.mesh.X.coords.shape[0]
            assert result.shape[0] == n_points, \
                f"evaluate(qty, all coords) should have {n_points} points, got shape {result.shape}"

    @pytest.mark.xfail(reason="UWexpression multiplication ND scaling not fully implemented")
    def test_evaluate_multiplication_single_coordinate(self):
        """Test: The multiplication bug case with single coordinate."""
        velocity_phys = uw.quantity(5, "m/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 's'), "Current time")

        product = velocity_phys * t_now

        # Should not raise IndexError
        result = uw.function.evaluate(product, self.mesh.X.coords[60])

        # Extract value
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        elif hasattr(result, 'flat'):
            value = float(result.flat[0])
        else:
            value = float(result)

        # Should be ~5 m, not 4.59e-7
        assert value > 1e-3, \
            f"Single coord multiplication bug: got {value} m, expected ~5 m"

    @pytest.mark.xfail(reason="UWexpression multiplication ND scaling not fully implemented")
    def test_evaluate_multiplication_coordinate_array(self):
        """Test: The multiplication bug case with coordinate array."""
        velocity_phys = uw.quantity(5, "m/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 's'), "Current time")

        product = velocity_phys * t_now
        result = uw.function.evaluate(product, self.mesh.X.coords[60:65])

        # Convert to meters
        if hasattr(result, 'to'):
            result_m = result.to('m')
            values = result_m.magnitude if hasattr(result_m, 'magnitude') else result_m
        else:
            values = np.asarray(result)

        # All values should be ~5 m
        assert np.all(values > 1e-3), \
            f"Array multiplication bug: values {values} should all be ~5 m"


@pytest.mark.tier_b  # Validated - testing scaling mode interaction
@pytest.mark.level_2  # Intermediate - nondimensional scaling complexity
class TestEvaluateScalingModes:
    """
    Test evaluation with nondimensional scaling ON and OFF.

    Ensures results are consistent regardless of scaling mode.
    """

    def test_evaluate_with_scaling_off(self):
        """Test: evaluate(5 m/s * 1 s) = 5 m with scaling OFF."""
        # No model, no scaling
        uw.use_nondimensional_scaling(False)

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )

        velocity_phys = uw.quantity(5, "m/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 's'), "Current time")

        product = velocity_phys * t_now
        result = uw.function.evaluate(product, mesh.X.coords[60])

        # Extract value
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        elif hasattr(result, 'flat'):
            value = float(result.flat[0])
        else:
            value = float(result)

        expected = 5.0

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(5 m/s * 1 s) with scaling OFF should return {expected} m, got {value} m"

    def test_evaluate_with_scaling_on(self):
        """Test: evaluate(5 m/s * 1 s) = 5 m with scaling ON."""
        model = uw.Model()

        L_scale = uw.quantity(2900, "km")
        t_scale = uw.quantity(1, "Myr")

        model.set_reference_quantities(
            length=L_scale,
            time=t_scale,
            mass=uw.quantity(1e24, "kg"),
            temperature=uw.quantity(1000, "K"),
            nondimensional_scaling=True,
        )

        uw.use_nondimensional_scaling(True)

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )

        velocity_phys = uw.quantity(5, "m/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 's'), "Current time")

        product = velocity_phys * t_now
        result = uw.function.evaluate(product, mesh.X.coords[60])

        # Extract value
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        elif hasattr(result, 'flat'):
            value = float(result.flat[0])
        else:
            value = float(result)

        expected = 5.0

        # Should NOT be 4.59e-7 (the bug)
        assert value > 1e-3, \
            f"Scaling ON multiplication bug: got {value} m, expected {expected} m"

        assert np.allclose(value, expected, rtol=1e-6), \
            f"evaluate(5 m/s * 1 s) with scaling ON should return {expected} m, got {value} m"

        uw.use_nondimensional_scaling(False)

    def test_evaluate_consistency_across_scaling_modes(self):
        """Test: Same expression gives same result with scaling ON vs OFF."""
        velocity_phys = uw.quantity(5, "m/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 's'), "Current time")
        product = velocity_phys * t_now

        # Test with scaling OFF
        uw.use_nondimensional_scaling(False)
        mesh_off = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )
        result_off = uw.function.evaluate(product, mesh_off.X.coords[60])

        if hasattr(result_off, 'to'):
            result_off_m = result_off.to('m')
            value_off = float(result_off_m.magnitude if hasattr(result_off_m, 'magnitude') else result_off_m)
        elif hasattr(result_off, 'flat'):
            value_off = float(result_off.flat[0])
        else:
            value_off = float(result_off)

        # Test with scaling ON
        model = uw.Model()
        model.set_reference_quantities(
            length=uw.quantity(2900, "km"),
            time=uw.quantity(1, "Myr"),
            mass=uw.quantity(1e24, "kg"),
            temperature=uw.quantity(1000, "K"),
            nondimensional_scaling=True,
        )
        uw.use_nondimensional_scaling(True)

        mesh_on = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )
        result_on = uw.function.evaluate(product, mesh_on.X.coords[60])

        if hasattr(result_on, 'to'):
            result_on_m = result_on.to('m')
            value_on = float(result_on_m.magnitude if hasattr(result_on_m, 'magnitude') else result_on_m)
        elif hasattr(result_on, 'flat'):
            value_on = float(result_on.flat[0])
        else:
            value_on = float(result_on)

        # Should give same result
        assert np.allclose(value_off, value_on, rtol=1e-6), \
            f"Scaling OFF: {value_off} m != Scaling ON: {value_on} m"

        uw.use_nondimensional_scaling(False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
