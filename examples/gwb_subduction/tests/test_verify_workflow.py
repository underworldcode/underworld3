"""Offline checks of the post-run audit and its error detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE))

import verify_workflow as audit  # noqa: E402


def test_postrun_audit_detects_corrupted_material_storage(tmp_path, monkeypatch):
    """Verify the audit accepts consistent fields and rejects corrupted labels."""
    output = tmp_path / "run"
    output.mkdir()
    coords = np.array([
        [0, 0], [4, 0], [8, 0],
        [0, 0.8], [4, 0.8], [8, 0.8],
        [0, 1.6], [4, 1.6], [8, 1.6],
    ], dtype=float)
    coupled = np.zeros_like(coords)
    coupled[4, 0] = 0.1
    weak = coupled * 2.0
    particles = np.column_stack((np.linspace(0, 8, 12), np.zeros(12)))
    labels = np.tile(np.arange(4), 3)
    interface = np.array([1] * 11 + [0])
    scale = 1e-10
    speed = 0.1 * scale * 100 * audit.YEAR_SECONDS

    strain = {
        "band_rms_per_s": 1e-14,
        "collar_rms_per_s": 1e-14,
        "band_nodes": 20,
        "collar_nodes": 30,
    }
    monkeypatch.setattr(audit, "strain_rms", lambda *args, **kwargs: strain)

    cases = {
        key: {
            "converged": True,
            "ksp_reason": 2,
            "velocity_pc": "custom-FMG",
            "velocity_pc_type": "mg",
            "boundary": {
                "left_normal": 0.0,
                "right_normal": 0.0,
                "top_normal": 0.0,
                "bottom_speed": 0.0,
            },
            "maximum_speed_cm_year": speed * multiplier,
            "strain": strain,
        }
        for key, multiplier in (("coupled", 1.0), ("weak_interface", 2.0))
    }
    summary = {
        "gwb_version": "v1.1.0",
        "domain_km": [8000, 1600],
        "finest_cells": [160, 80],
        "temperature_preservation_max_error_K": 0.0,
        "tag_preservation_max_error": 0.0,
        "material_mask_mismatches": 0,
        "weak_particles": 11,
        "temperature_dofs": 3,
        "swarm_particles": 12,
        "material_counts": {
            "Mantle": 3,
            "Incoming plate": 3,
            "Descending slab": 3,
            "Overriding plate": 3,
        },
        "cases": cases,
        "relative_velocity_field_change": 1.0,
    }

    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(audit, "REFERENCE", ref)

    model = json.loads(audit.GWB_REFERENCE.read_text(encoding="utf-8"))
    (output / "four_material_subduction.wb").write_text(
        json.dumps(model), encoding="utf-8"
    )
    (output / "verification.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    np.savetxt(
        output / "slab_top_km.csv", np.array([[0, 0], [8e3, 200]]),
        delimiter=",", header="x_km,depth_km", comments="",
    )
    for figure in (
        "gwb_four_material_geometry.png",
        "stokes_velocity_comparison.png",
        "interface_strain_comparison.png",
    ):
        (output / figure).write_bytes(b"fixture")

    def write_arrays(current_labels):
        """Write a minimal stored-field fixture for the audit."""
        np.savez_compressed(
            output / "demo_fields.npz", velocity_coords=coords,
            coupled_velocity=coupled, weak_velocity=weak,
            swarm_coords=particles, material_labels=current_labels,
            weak_labels=interface, temperature_coords=coords[:3],
            temperature_K=np.ones(3) * 1573,
            velocity_scale_mps=scale,
        )

    write_arrays(labels)
    result = audit.verify(output)
    assert result["status"] == "PASS"
    assert result["swarm_particles"] == 12

    broken = labels.copy()
    broken[0] = 1
    write_arrays(broken)
    with pytest.raises(AssertionError, match="counts"):
        audit.verify(output)
