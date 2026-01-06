"""Tests for adaptivity metric creation functions.

Tests the public API functions:
- create_metric: Create metric from h-field
- metric_from_gradient: Create metric from scalar field gradient
- metric_from_field: Create metric from indicator field

These tests verify the metric creation logic without requiring
MMG (no actual mesh adaptation).
"""
import numpy as np
import pytest

import underworld3 as uw


@pytest.fixture
def simple_mesh():
    """Create a simple 2D mesh for testing."""
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.1,
        regular=False,
    )


class TestCreateMetric:
    """Tests for create_metric function."""

    def test_basic_creation(self, simple_mesh):
        """Test basic metric creation from h-field."""
        n_nodes = simple_mesh.X.coords.shape[0]
        h_values = np.full(n_nodes, 0.05)

        metric = uw.adaptivity.create_metric(simple_mesh, h_values)

        assert metric is not None
        with simple_mesh.access(metric):
            # Check metric = 1/h²
            expected = 1.0 / (0.05 ** 2)
            np.testing.assert_allclose(metric.data[:, 0], expected, rtol=1e-10)

    def test_variable_h_field(self, simple_mesh):
        """Test metric with spatially varying h-field."""
        with simple_mesh.access():
            coords = simple_mesh.X.coords
            # h varies linearly from 0.02 to 0.1 across domain
            h_values = 0.02 + 0.08 * coords[:, 0]

        metric = uw.adaptivity.create_metric(simple_mesh, h_values, name="test_metric")

        with simple_mesh.access(metric):
            h_recovered = 1.0 / np.sqrt(metric.data[:, 0])
            np.testing.assert_allclose(h_recovered, h_values, rtol=1e-10)


class TestMetricFromGradient:
    """Tests for metric_from_gradient function."""

    def test_gaussian_gradient(self, simple_mesh):
        """Test metric from gradient of Gaussian field."""
        T = uw.discretisation.MeshVariable("T", simple_mesh, 1, degree=1)

        with simple_mesh.access(T):
            x, y = T.coords[:, 0], T.coords[:, 1]
            T.data[:, 0] = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (2 * 0.1**2))

        metric = uw.adaptivity.metric_from_gradient(
            T, h_min=0.02, h_max=0.15, profile="linear"
        )

        with simple_mesh.access(metric, T):
            h_values = 1.0 / np.sqrt(metric.data[:, 0])
            # h should be in specified range
            assert h_values.min() >= 0.02 - 1e-10
            assert h_values.max() <= 0.15 + 1e-10

            # Near center (high gradient) should have smaller h
            coords = T.coords
            center_mask = ((coords[:, 0] - 0.5)**2 + (coords[:, 1] - 0.5)**2) < 0.15**2
            h_center = h_values[center_mask].mean()
            h_far = h_values[~center_mask].mean()
            assert h_far > h_center, "High gradient region should have smaller h"

    def test_uniform_field_uses_h_max(self, simple_mesh):
        """Test that uniform field (zero gradient) uses h_max everywhere."""
        T = uw.discretisation.MeshVariable("T_uniform", simple_mesh, 1, degree=1)
        with simple_mesh.access(T):
            T.data[:, 0] = 1.0  # Constant field

        metric = uw.adaptivity.metric_from_gradient(T, h_min=0.01, h_max=0.1)

        with simple_mesh.access(metric):
            h_values = 1.0 / np.sqrt(metric.data[:, 0])
            # Should be h_max everywhere (no gradient)
            np.testing.assert_allclose(h_values, 0.1, rtol=0.1)

    def test_profiles(self, simple_mesh):
        """Test different interpolation profiles."""
        T = uw.discretisation.MeshVariable("T_profiles", simple_mesh, 1, degree=1)
        with simple_mesh.access(T):
            x = T.coords[:, 0]
            T.data[:, 0] = x  # Linear field - constant gradient

        for profile in ["linear", "smoothstep", "power"]:
            metric = uw.adaptivity.metric_from_gradient(
                T, h_min=0.02, h_max=0.1, profile=profile,
                name=f"metric_{profile}"
            )
            with simple_mesh.access(metric):
                h_values = 1.0 / np.sqrt(metric.data[:, 0])
                assert h_values.min() >= 0.02 - 1e-10
                assert h_values.max() <= 0.1 + 1e-10


class TestMetricFromField:
    """Tests for metric_from_field function."""

    def test_basic_mapping(self, simple_mesh):
        """Test basic indicator-to-metric mapping."""
        indicator = uw.discretisation.MeshVariable("indicator", simple_mesh, 1, degree=1)

        with simple_mesh.access(indicator):
            x, y = indicator.coords[:, 0], indicator.coords[:, 1]
            # High indicator at center
            indicator.data[:, 0] = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (2 * 0.2**2))

        metric = uw.adaptivity.metric_from_field(
            indicator, h_min=0.02, h_max=0.1
        )

        with simple_mesh.access(metric, indicator):
            h_values = 1.0 / np.sqrt(metric.data[:, 0])
            # High indicator should map to small h
            high_mask = indicator.data[:, 0] > 0.8
            h_high = h_values[high_mask].mean() if high_mask.any() else 0
            h_low = h_values[~high_mask].mean()
            assert h_low > h_high, "High indicator should have smaller h"

    def test_invert_option(self, simple_mesh):
        """Test that invert=True swaps the mapping."""
        indicator = uw.discretisation.MeshVariable("ind_inv", simple_mesh, 1, degree=1)

        with simple_mesh.access(indicator):
            x = indicator.coords[:, 0]
            indicator.data[:, 0] = x  # Linear from 0 to 1

        # Without invert: high indicator (right side) -> small h
        metric_normal = uw.adaptivity.metric_from_field(
            indicator, h_min=0.02, h_max=0.1, invert=False, name="metric_normal"
        )

        # With invert: high indicator (right side) -> large h
        metric_inv = uw.adaptivity.metric_from_field(
            indicator, h_min=0.02, h_max=0.1, invert=True, name="metric_inv"
        )

        with simple_mesh.access(metric_normal, metric_inv, indicator):
            h_normal = 1.0 / np.sqrt(metric_normal.data[:, 0])
            h_inv = 1.0 / np.sqrt(metric_inv.data[:, 0])

            right_mask = indicator.coords[:, 0] > 0.5

            # Normal: right (high indicator) should have smaller h
            assert h_normal[right_mask].mean() < h_normal[~right_mask].mean()

            # Inverted: right (high indicator) should have larger h
            assert h_inv[right_mask].mean() > h_inv[~right_mask].mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
