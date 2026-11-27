#!/usr/bin/env python3
"""Audit a completed GWB to Underworld3 run against its archived reference.

This is a post-run validation utility. It does not import Underworld3 or launch
GWB. Run the simulation first, then pass its output directory as the argument.
The historical reference is illustrative and applies only to the fixed mesh,
physics, solver configuration, and GWB v1.1.0 used by this example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gwb_core import boundary_check, strain_rms


REFERENCE = Path(__file__).with_name("reference_verification.json")
GWB_REFERENCE = Path(__file__).with_name("four_material_subduction.wb")
YEAR_SECONDS = 365.25 * 86400.0


def require(condition: bool, message: str) -> None:
    """Raise a descriptive numerical-validation error when a check fails."""
    if not condition:
        raise AssertionError(message)


def check_close(name: str, value: float, reference: float, rtol=0.01) -> None:
    """Compare a measured scalar with the archived reference at a stated tolerance."""
    require(np.isfinite(value), f"{name}: value is not finite")
    require(
        np.isclose(value, reference, rtol=rtol, atol=1e-20),
        f"{name}: measured {value}, archived {reference}, tolerance {rtol:.1%}",
    )


def verify(output_dir: Path) -> dict:
    """Verify model, transferred fields, two solutions, and diagnostic files."""
    output_dir = Path(output_dir)
    summary_path = output_dir / "verification.json"
    arrays_path = output_dir / "demo_fields.npz"
    profile_path = output_dir / "slab_top_km.csv"
    generated_model_path = output_dir / "four_material_subduction.wb"
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = json.loads(generated_model_path.read_text(encoding="utf-8"))
    expected_model = json.loads(GWB_REFERENCE.read_text(encoding="utf-8"))

    require(model == expected_model, "Generated GWB model changed from reference")
    require(summary["gwb_version"] == reference["gwb_version"],
            "GWB version changed")
    require(summary["domain_km"] == reference["domain_km"],
            "Domain geometry changed")
    require(summary["finest_cells"] == reference["finest_cells"],
            "Mesh resolution changed")
    require(summary["temperature_preservation_max_error_K"] <= 1e-5,
            "Temperature transfer does not preserve original GWB values")
    require(summary["tag_preservation_max_error"] <= 1e-8,
            "Tag transfer does not preserve original GWB values")
    require(summary["material_mask_mismatches"] == 0,
            "Material masks differ from the original model")
    require(summary["weak_particles"] >= 10, "Weak band has too few particles")

    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}

    velocity_coords = arrays["velocity_coords"]
    coupled = arrays["coupled_velocity"]
    weak = arrays["weak_velocity"]
    particle_coords = arrays["swarm_coords"]
    material = arrays["material_labels"]
    interface = arrays["weak_labels"]
    temperature = arrays["temperature_K"]
    speed_scale = float(arrays["velocity_scale_mps"])
    profile = np.loadtxt(profile_path, delimiter=",", skiprows=1)

    require(velocity_coords.shape == coupled.shape == weak.shape,
            "Velocity coordinate or field shape mismatch")
    require(velocity_coords.ndim == 2 and velocity_coords.shape[1] == 2,
            "Expected two-component velocity fields")
    require(particle_coords.shape == (len(material), 2),
            "Particle coordinate count mismatch")
    require(interface.shape == material.shape,
            "Material and interface masks differ in size")
    require(len(temperature) == summary["temperature_dofs"],
            "Temperature DOF count mismatch")
    require(len(material) == summary["swarm_particles"],
            "Swarm particle count mismatch")
    require(int(np.count_nonzero(interface)) == summary["weak_particles"],
            "Weak particle count mismatch")
    require(np.isin(material, [0, 1, 2, 3]).all(),
            "Material labels contain invalid values")
    require(np.isin(interface, [0, 1]).all(),
            "Weak-zone labels contain invalid values")
    require(all(np.isfinite(arr).all() for arr in arrays.values()),
            "Stored fields contain non-finite values")

    names = [
        "Mantle", "Incoming plate", "Descending slab", "Overriding plate"
    ]
    counts = {name: int((material == index).sum())
              for index, name in enumerate(names)}
    require(counts == summary["material_counts"],
            "Stored material counts disagree with verification.json")
    require(counts == reference["material_counts"],
            "Material counts changed from archived baseline")

    comparison = float(
        np.linalg.norm(weak - coupled) / np.linalg.norm(coupled)
    )
    check_close("full-field change", comparison,
                summary["relative_velocity_field_change"], rtol=1e-10)
    check_close("archived full-field change", comparison,
                reference["relative_velocity_field_change"])

    checks = {}
    for key, field in (("coupled", coupled), ("weak_interface", weak)):
        record = summary["cases"][key]
        require(record["converged"], f"{key} did not converge")
        require(record["ksp_reason"] > 0, f"{key} failed KSP convergence")
        require(record["velocity_pc"] == "custom-FMG", f"{key} changed FMG")
        require(record["velocity_pc_type"] == "mg", f"{key} changed velocity PC")

        boundaries = boundary_check(velocity_coords, field)
        for label, value in boundaries.items():
            check_close(f"{key}.{label}", value,
                        record["boundary"][label], rtol=1e-6)
        speed = np.linalg.norm(field, axis=1).max() * speed_scale
        maximum_speed = float(speed * 100.0 * YEAR_SECONDS)
        check_close(f"{key}.maximum_speed", maximum_speed,
                    record["maximum_speed_cm_year"], rtol=1e-10)
        check_close(f"{key}.archived_speed", maximum_speed,
                    reference["cases"][key]["maximum_speed_cm_year"])
        strain = strain_rms(velocity_coords, field, profile, speed_scale)
        for metric, value in strain.items():
            check_close(f"{key}.strain.{metric}", float(value),
                        float(record["strain"][metric]), rtol=1e-9)
        checks[key] = {
            "maximum_speed_cm_year": maximum_speed,
            "strain": strain,
            "boundary": boundaries,
        }

    for figure in (
        "gwb_four_material_geometry.png",
        "stokes_velocity_comparison.png",
        "interface_strain_comparison.png",
    ):
        require((output_dir / figure).is_file(), f"Missing figure: {figure}")

    return {
        "status": "PASS",
        "temperature_dofs": len(temperature),
        "swarm_particles": len(material),
        "material_counts": counts,
        "weak_particles": int(np.count_nonzero(interface)),
        "relative_velocity_field_change": comparison,
        "cases": checks,
    }


def main() -> None:
    """Parse an output directory and emit a machine-readable audit result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir", type=Path, nargs="?",
        default=Path.cwd() / "outputs" / "gwb_subduction",
    )
    args = parser.parse_args()
    result = verify(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
