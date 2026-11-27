"""Offline smoke test for the example's headless figure generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE))

from gwb_plots import make_figures  # noqa: E402


def test_all_three_scientific_figures_are_written(tmp_path):
    """Ensure that the archived plotting path runs without a display server."""
    particle_coords = np.array([
        [1, 1.5], [2, 1.4], [3, 1.3], [4, 1.2],
    ], dtype=float)
    labels = np.array([0, 1, 2, 3])
    profile = np.array([[3500, 0], [3400, 100], [3300, 200]])
    coords = np.array([
        [0, 0], [8, 0], [0, 1.6], [8, 1.6], [4, 0.8]
    ], dtype=float)
    velocity = np.array([
        [0, 0], [0, 0], [0, 0], [0, 0], [0.1, 0.05]
    ])
    solutions = {
        "coupled": (coords, velocity),
        "weak_interface": (coords, velocity * 1.5),
    }
    records = {
        "coupled": {"strain": {
            "band_rms_per_s": 1.0e-14, "collar_rms_per_s": 1.0e-14,
        }},
        "weak_interface": {"strain": {
            "band_rms_per_s": 7.5e-14, "collar_rms_per_s": 1.5e-14,
        }},
    }
    make_figures(particle_coords, labels, profile, solutions, 1.0, records, tmp_path)
    for filename in (
        "gwb_four_material_geometry.png",
        "stokes_velocity_comparison.png",
        "interface_strain_comparison.png",
    ):
        picture = tmp_path / filename
        assert picture.is_file()
        assert picture.stat().st_size > 1000
        assert picture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
