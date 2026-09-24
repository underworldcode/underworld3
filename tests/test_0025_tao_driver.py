"""uw.adjoint.minimise drives an objective with PETSc TAO: a quadratic, bounded and not."""
import numpy as np
import pytest

import underworld3 as uw


def _quadratic(x):
    c = np.array([1.0, -0.5, 2.0])
    return float(np.sum((x - c) ** 2 * [1, 2, 3])), 2 * (x - c) * [1, 2, 3]


@pytest.mark.level_1
@pytest.mark.tier_a
def test_lmvm_finds_the_minimum():
    x, info = uw.adjoint.minimise(_quadratic, np.zeros(3), gradient_tolerance=1e-12)
    assert np.allclose(x, [1.0, -0.5, 2.0], atol=1e-6)
    assert info["reason"] > 0
    assert len(info["history"]) == len(info["history"]) and info["history"][0][0] > info["history"][-1][0]


@pytest.mark.level_1
@pytest.mark.tier_a
def test_blmvm_honours_bounds():
    x, info = uw.adjoint.minimise(_quadratic, np.zeros(3), method="blmvm",
                                  bounds=(np.array([0.0, 0.0, 0.0]), np.array([0.5, 1.0, 1.0])),
                                  gradient_tolerance=1e-12)
    assert np.allclose(x, [0.5, 0.0, 1.0], atol=1e-6)
    assert info["reason"] > 0
